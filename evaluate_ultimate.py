import os
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from train_ultimate import load_all_data, TransPINN_DIO

print("Loading validation datasets (this may take a moment)...")
features, labels = load_all_data()
split_idx = int(len(features) * 0.8)
val_feats, val_lbls = features[split_idx:], labels[split_idx:]

print("Loading Ultimate Trans-PINN DIO model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransPINN_DIO(in_channels=6, d_model=128).to(device)
# Make sure to set weights_only=True for security, but allow it if older PyTorch
try:
    model.load_state_dict(torch.load('ultimate_pinn_model.pth', map_location=device, weights_only=True))
except TypeError:
    model.load_state_dict(torch.load('ultimate_pinn_model.pth', map_location=device))
model.eval()

def get_realtime_predictions(session_feats):
    window_size = 50
    if len(session_feats) <= window_size:
        return np.zeros(len(session_feats))
        
    x_windows = []
    for i in range(window_size, len(session_feats)):
        x_windows.append(session_feats[i-window_size:i])
        
    x_tensor = torch.tensor(np.array(x_windows, dtype=np.float32)).to(device)
    preds = np.zeros(len(session_feats))
    
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(x_tensor), batch_size):
            batch_x = x_tensor[i:i+batch_size]
            v_pred, p_stop = model(batch_x)
            
            # Apply ZUPT (Zero Velocity Update)
            # If the network is >90% confident we are stopped, force speed to 0
            v_pred = torch.where(p_stop > 0.9, torch.zeros_like(v_pred), v_pred)
            
            # Extract the prediction for the CURRENT (last) frame of the sequence
            last_preds = v_pred[:, -1].cpu().numpy()
            preds[window_size + i : window_size + i + len(last_preds)] = last_preds
            
    # Pad the beginning
    preds[:window_size] = preds[window_size]
    return np.maximum(0, preds) # Prevent negative speeds

print("\nRunning ISRO Hackathon Benchmarks across all unseen validation sessions...")

err_50_list, drift_50_list = [], []
err_1000_list, drift_1000_list = [], []
dt = 0.1

for session_idx, (feats, true_speeds) in enumerate(zip(val_feats, val_lbls)):
    ai_speeds = get_realtime_predictions(feats)
    phone_gyro_yaw = feats[:, 4] # Index 4 is Gyro Pitch, which maps to Vehicle Yaw
    
    def evaluate_session(target_distance_m):
        errors, drift_rates = [], []
        i = 50
        while i < len(feats):
            true_x, true_y, true_h = 0.0, 0.0, 0.0
            ai_x, ai_y, ai_h = 0.0, 0.0, 0.0
            dist_traveled = 0.0
            reached_target = False
            
            while i < len(feats):
                # Update headings (simulating an IMU EKF)
                true_h += phone_gyro_yaw[i] * dt
                ai_h += phone_gyro_yaw[i] * dt
                
                # Update true position
                v_t = true_speeds[i]
                step_dist = v_t * dt
                true_x += step_dist * np.cos(true_h)
                true_y += step_dist * np.sin(true_h)
                dist_traveled += step_dist
                
                # Update AI position
                v_ai = ai_speeds[i]
                ai_x += (v_ai * dt) * np.cos(ai_h)
                ai_y += (v_ai * dt) * np.sin(ai_h)
                
                i += 1
                
                if dist_traveled >= target_distance_m:
                    reached_target = True
                    break
                    
            if reached_target:
                error_m = np.sqrt((true_x - ai_x)**2 + (true_y - ai_y)**2)
                errors.append(error_m)
                drift_rates.append((error_m / dist_traveled) * 100)
            else:
                break
        return errors, drift_rates

    e50, d50 = evaluate_session(50.0)
    e1000, d1000 = evaluate_session(1000.0)
    err_50_list.extend(e50)
    drift_50_list.extend(d50)
    err_1000_list.extend(e1000)
    drift_1000_list.extend(d1000)

print("\n" + "="*40)
print("🏆 ULTIMATE ISRO BENCHMARK RESULTS 🏆")
print("="*40)

print("\n--- BENCHMARK 1: < 5m drift over 50m ---")
print(f"Total 50m segments tested: {len(err_50_list)}")
print(f"Average Position Error: {np.mean(err_50_list):.2f} meters")
print(f"Pass Rate (<5m error): {np.mean(np.array(err_50_list) < 5.0)*100:.1f}%")

print("\n--- BENCHMARK 2: < 100m drift over 1km ---")
if len(err_1000_list) > 0:
    print(f"Total 1km segments tested: {len(err_1000_list)}")
    print(f"Average Position Error: {np.mean(err_1000_list):.2f} meters")
    print(f"Pass Rate (<100m error): {np.mean(np.array(err_1000_list) < 100.0)*100:.1f}%")
else:
    print("Validation set isn't long enough for 1km segments.")

print("\n--- BENCHMARK 3: Target < 10% Drift Rate ---")
print(f"Average Drift Rate (50m segments): {np.mean(drift_50_list):.2f}%")
if len(drift_1000_list) > 0:
    print(f"Average Drift Rate (1km segments): {np.mean(drift_1000_list):.2f}%")

print("\n" + "="*40)
