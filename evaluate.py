import os
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from train import RoNIN_ResNet_LSTM

print("Loading data for Absolute Heading Benchmark...")

v_path = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv"
s_path = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv"

v_df = pd.read_csv(v_path)
s_df = pd.read_csv(s_path, encoding='latin-1')
v_df.columns = v_df.columns.str.strip()
s_df.columns = s_df.columns.str.strip()

# Lag correction (9 samples)
lag_samples = 9
v_aligned = v_df.iloc[:-lag_samples].reset_index(drop=True)
s_aligned = s_df.iloc[lag_samples:].reset_index(drop=True)

split_idx = int(len(v_aligned) * 0.8)
v_val = v_aligned.iloc[split_idx:].reset_index(drop=True)
s_val = s_aligned.iloc[split_idx:].reset_index(drop=True)

def extract_features(s_data):
    gx, gy, gz = s_data['GRAVITY X (m/s²)'].values, s_data['GRAVITY Y (m/s²)'].values, s_data['GRAVITY Z (m/s²)'].values
    ax, ay, az = s_data['ACCELEROMETER X (m/s²)'].values, s_data['ACCELEROMETER Y (m/s²)'].values, s_data['ACCELEROMETER Z (m/s²)'].values
    groll, gpitch, gyaw = s_data['GYROSCOPE Roll (rad/s)'].values, s_data['GYROSCOPE Pitch (rad/s)'].values, s_data['GYROSCOPE Yaw (rad/s)'].values
    
    rot_ax, rot_ay, rot_az = np.zeros_like(ax), np.zeros_like(ay), np.zeros_like(az)
    rot_groll, rot_gpitch, rot_gyaw = np.zeros_like(groll), np.zeros_like(gpitch), np.zeros_like(gyaw)
    
    for i in range(len(s_data)):
        g_vec = np.array([gx[i], gy[i], gz[i]])
        g_norm = np.linalg.norm(g_vec)
        if g_norm < 0.1: continue
        g_normed = g_vec / g_norm
        target_g = np.array([0, 0, 1])
        v = np.cross(g_normed, target_g)
        c = np.dot(g_normed, target_g)
        s = np.linalg.norm(v)
        if s < 1e-6: rot_matrix = np.eye(3)
        else:
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        rot_ax[i], rot_ay[i], rot_az[i] = rot_matrix.dot(np.array([ax[i], ay[i], az[i]]))
        rot_groll[i], rot_gpitch[i], rot_gyaw[i] = rot_matrix.dot(np.array([groll[i], gpitch[i], gyaw[i]]))

    return np.column_stack((rot_ax, rot_ay, rot_az, rot_groll, rot_gpitch, rot_gyaw))

print("Extracting AI features...")
val_features = extract_features(s_val)
val_features = np.clip(val_features, -49.0, 49.0)

print("Loading Champion AI Model (ResNet-BiLSTM)...")
model = RoNIN_ResNet_LSTM(in_channels=6)
# I will checkout the main branch first or just load it from disk since we are on main
model.load_state_dict(torch.load("champion_model.pth", map_location='cpu', weights_only=True))
model.eval()

print("Generating AI Speed Predictions...")
window_size = 50
dt = 0.1
ai_speeds_ms = np.zeros(len(s_val))

with torch.no_grad():
    for i in range(window_size - 1, len(s_val)):
        x_window = val_features[i - window_size + 1 : i + 1]
        x_tensor = torch.tensor(x_window, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
        pred = model(x_tensor).item()
        ai_speeds_ms[i] = max(0, pred)

true_speeds_ms = v_val['Velocity (km/hr)'].values / 3.6

# Absolute Phone Orientation (Fused Gyro + Magnetometer)
absolute_heading = np.radians(s_val.iloc[:, 21].values)

print("\nRunning Sensor-Fused Benchmark (< 5% Target)...")

def run_fused_benchmark(target_distance_m):
    errors, drift_rates = [], []
    i = window_size
    while i < len(s_val):
        true_x, true_y = 0.0, 0.0
        ai_x, ai_y = 0.0, 0.0
        dist_traveled = 0.0
        reached_target = False
        
        while i < len(s_val):
            # Use Absolute Hardware Heading
            current_h = absolute_heading[i]
            
            step_dist = true_speeds_ms[i] * dt
            true_x += step_dist * np.cos(current_h)
            true_y += step_dist * np.sin(current_h)
            dist_traveled += step_dist
            
            ai_step = ai_speeds_ms[i] * dt
            ai_x += ai_step * np.cos(current_h)
            ai_y += ai_step * np.sin(current_h)
            
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

print("\n--- BENCHMARK 1: < 5m drift over 50m ---")
err_50, drift_50 = run_fused_benchmark(50.0)
print(f"Total 50m segments evaluated: {len(err_50)}")
print(f"Average Position Error: {np.mean(err_50):.2f}m")
print(f"Pass Rate (<5m error): {np.mean(np.array(err_50) < 5.0)*100:.1f}%")

print("\n--- BENCHMARK 2: < 100m drift over 1km ---")
err_1000, drift_1000 = run_fused_benchmark(1000.0)
if len(err_1000) > 0:
    print(f"Total 1km segments evaluated: {len(err_1000)}")
    print(f"Average Position Error: {np.mean(err_1000):.2f}m")
    print(f"Pass Rate (<100m error): {np.mean(np.array(err_1000) < 100.0)*100:.1f}%")

print("\n--- BENCHMARK 3: Drift rate ---")
print(f"Average Drift Rate (50m segments): {np.mean(drift_50):.2f}%")
if len(drift_1000) > 0:
    print(f"Average Drift Rate (1km segments): {np.mean(drift_1000):.2f}%")
