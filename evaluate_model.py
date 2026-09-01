import os
import torch
import numpy as np
import pandas as pd
from train_tcn import preprocess_iovnb_pair
from train_advanced import RoNIN_ResNet_LSTM
import warnings
warnings.filterwarnings('ignore')

print("Loading data and model for benchmark evaluation...")
v_path = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv"
s_path = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv"

# Re-use our preprocessing to ensure exact same conditions
v_df, s_df = preprocess_iovnb_pair(v_path, s_path)

# Evaluate on the VALIDATION SET (the 20% of data the model NEVER saw during training)
# This proves it actually learned, and didn't just memorize the data.
split_idx = int(len(v_df) * 0.8)
v_val = v_df.iloc[split_idx:].reset_index(drop=True)
s_val = s_df.iloc[split_idx:].reset_index(drop=True)

print(f"Validation set size: {len(v_val)} frames (~{len(v_val)/10/60:.1f} minutes of driving)")

model = RoNIN_ResNet_LSTM(in_channels=6)
model.load_state_dict(torch.load("advanced_best_model.pth", map_location='cpu', weights_only=True))
model.eval()

print("Generating AI predictions for the entire validation set...")
window_size = 50
dt = 0.1
features = s_val[['Vehicle_Accel_X', 'Vehicle_Accel_Y', 'Vehicle_Accel_Z',
                 'Vehicle_Gyro_Roll', 'Vehicle_Gyro_Pitch', 'Vehicle_Gyro_Yaw']].values
# Apply same clipping as training
features = np.clip(features, -49.0, 49.0)

true_speeds_ms = v_val['Velocity (km/hr)'].values / 3.6
# Use phone gyro pitch (which maps to Yaw) for heading changes
phone_gyro_yaw = s_val['Vehicle_Gyro_Pitch'].values

ai_speeds_ms = np.zeros(len(s_val))
with torch.no_grad():
    for i in range(window_size - 1, len(s_val)):
        x_window = features[i - window_size + 1 : i + 1]
        x_tensor = torch.tensor(x_window, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
        pred = model(x_tensor).item()
        ai_speeds_ms[i] = max(0, pred)

print("\nRunning ISRO Hackathon Benchmarks...")

def run_benchmark(target_distance_m):
    errors = []
    drift_rates = []
    
    i = window_size
    while i < len(s_val):
        # Start new segment
        true_x, true_y, true_h = 0.0, 0.0, 0.0
        ai_x, ai_y, ai_h = 0.0, 0.0, 0.0
        dist_traveled = 0.0
        
        reached_target = False
        
        while i < len(s_val):
            # Update headings
            true_h += phone_gyro_yaw[i] * dt
            ai_h += phone_gyro_yaw[i] * dt
            
            # Update true position
            v_t = true_speeds_ms[i]
            step_dist = v_t * dt
            true_x += step_dist * np.cos(true_h)
            true_y += step_dist * np.sin(true_h)
            dist_traveled += step_dist
            
            # Update AI position
            v_ai = ai_speeds_ms[i]
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
            break # Reached end of dataset
            
    return errors, drift_rates

print("\n--- BENCHMARK 1: < 5m drift over 50m ---")
err_50, drift_50 = run_benchmark(50.0)
print(f"Total 50m segments evaluated: {len(err_50)}")
print(f"Average Position Error: {np.mean(err_50):.2f}m")
print(f"Pass Rate (<5m error): {np.mean(np.array(err_50) < 5.0)*100:.1f}%")

print("\n--- BENCHMARK 2: < 100m drift over 1km ---")
err_1000, drift_1000 = run_benchmark(1000.0)
if len(err_1000) > 0:
    print(f"Total 1km segments evaluated: {len(err_1000)}")
    print(f"Average Position Error: {np.mean(err_1000):.2f}m")
    print(f"Pass Rate (<100m error): {np.mean(np.array(err_1000) < 100.0)*100:.1f}%")
else:
    print("Validation set isn't long enough for a full 1km continuous segment.")

print("\n--- BENCHMARK 3: Drift rate < 10% ---")
print(f"Average Drift Rate (50m segments): {np.mean(drift_50):.2f}%")
if len(drift_1000) > 0:
    print(f"Average Drift Rate (1km segments): {np.mean(drift_1000):.2f}%")

# Save results for markdown artifact
with open("benchmark_results.txt", "w") as f:
    f.write(f"50m_err={np.mean(err_50):.2f}\n")
    f.write(f"50m_pass={np.mean(np.array(err_50) < 5.0)*100:.1f}\n")
    if len(err_1000) > 0:
        f.write(f"1km_err={np.mean(err_1000):.2f}\n")
        f.write(f"1km_pass={np.mean(np.array(err_1000) < 100.0)*100:.1f}\n")
        f.write(f"overall_drift={np.mean(drift_1000):.2f}\n")
    else:
        f.write(f"overall_drift={np.mean(drift_50):.2f}\n")

print("\nEvaluation complete! Results saved.")
