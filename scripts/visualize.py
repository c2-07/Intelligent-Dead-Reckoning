import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

from dead_reckoning.model import RoNIN_ResNet_LSTM


def extract_features(s_data):
    """Applies gravity rotation matrix and normalizes IMU inputs."""
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

import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize Dead Reckoning Path")
    parser.add_argument("--model", type=str, default="models/resnet_bilstm_v1.pth", help="Path to model weights")
    args = parser.parse_args()

    # Load 60 seconds of True GPS + Smartphone data
    v_path = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv"
    s_path = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv"

    if not os.path.exists(v_path):
        print(f"Error: Could not find {v_path}. Make sure the dataset is downloaded.")
        return

    v_df = pd.read_csv(v_path)
    s_df = pd.read_csv(s_path, encoding='latin-1')
    v_df.columns = v_df.columns.str.strip()
    s_df.columns = s_df.columns.str.strip()

    lag_samples = 9
    v_aligned = v_df.iloc[:-lag_samples].reset_index(drop=True)
    s_aligned = s_df.iloc[lag_samples:].reset_index(drop=True)

    # Pick a 60-second window during a straight+turn segment (e.g., sample 1500 to 2100)
    start_idx = 1500
    end_idx = 2100
    if end_idx > len(s_aligned):
        start_idx = 0
        end_idx = min(600, len(s_aligned))

    v_val = v_aligned.iloc[start_idx:end_idx].reset_index(drop=True)
    s_val = s_aligned.iloc[start_idx:end_idx].reset_index(drop=True)
    duration = end_idx - start_idx

    # Extract Features for the window
    print("Extracting features for 60s blackout visualization...")
    val_features = extract_features(s_val)
    val_features = np.clip(val_features, -49.0, 49.0)

    # Load Model
    print(f"Loading Model from {args.model}...")
    model = RoNIN_ResNet_LSTM(in_channels=6)
    try:
        model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
    except Exception:
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
    
    model.eval()

    print("4. Simulating a 60-second GPS Tunnel Blackout...")
    window_size = 50
    dt = 0.1
    ai_speeds_ms = np.zeros(len(s_val))
    
    # We also simulate raw IMU integration to show the error
    phone_accel_forward = s_val['ACCELEROMETER Y (m/s²)'].values

    with torch.no_grad():
        # Pre-pad the first window_size frames with 0s for visualization simplicity
        for i in range(window_size - 1, len(s_val)):
            x_window = val_features[i - window_size + 1 : i + 1]
            x_tensor = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            pred = model(x_tensor).item()
            ai_speeds_ms[i] = max(0, pred)
            
    # Fill the start frames with actual speed so graph isn't broken at 0
    true_speeds_ms = v_val['Velocity (km/hr)'].values / 3.6
    ai_speeds_ms[:window_size-1] = true_speeds_ms[:window_size-1]

    # --- Simulate the Trajectories (Dead Reckoning) ---
    pos_true = {'x': [0], 'y': [0]}
    pos_ai = {'x': [0], 'y': [0]}
    pos_raw = {'x': [0], 'y': [0]}

    # Use Absolute Heading from Smartphone Compass for fusion
    absolute_heading = np.radians(s_val.iloc[:, 21].values)
    
    raw_speed = true_speeds_ms[0]

    for i in range(duration - 1):
        current_h = absolute_heading[i]
        
        # 1. Ground Truth Path
        v_t = true_speeds_ms[i]
        pos_true['x'].append(pos_true['x'][-1] + v_t * np.cos(current_h) * dt)
        pos_true['y'].append(pos_true['y'][-1] + v_t * np.sin(current_h) * dt)
        
        # 2. AI Dead Reckoning Path
        v_ai = ai_speeds_ms[i]
        pos_ai['x'].append(pos_ai['x'][-1] + v_ai * np.cos(current_h) * dt)
        pos_ai['y'].append(pos_ai['y'][-1] + v_ai * np.sin(current_h) * dt)
        
        # 3. Raw Phone IMU Path (What happens without AI - Double Integration)
        # Note: ACCELEROMETER Y is forward/backward on phones typically
        raw_speed += phone_accel_forward[i] * dt
        pos_raw['x'].append(pos_raw['x'][-1] + raw_speed * np.cos(current_h) * dt)
        pos_raw['y'].append(pos_raw['y'][-1] + raw_speed * np.sin(current_h) * dt)

    # --- Plot the Results ---
    plt.figure(figsize=(12, 10))

    # Subplot 1: Map / Trajectory
    plt.subplot(2, 1, 1)
    plt.title("GPS Tunnel Blackout Simulation (60 Seconds)", fontsize=14, fontweight='bold')
    plt.plot(pos_true['x'], pos_true['y'], 'g-', linewidth=3, label="Actual Path (Ground Truth)")
    plt.plot(pos_ai['x'], pos_ai['y'], 'b--', linewidth=2.5, label="AI Nav System (ResNet-BiLSTM)")
    plt.plot(pos_raw['x'], pos_raw['y'], 'r:', linewidth=2, label="Raw Phone Sensors (Massive Drift)")

    plt.scatter(0, 0, c='black', s=100, marker='o', label="Tunnel Enter (GPS Lost)")
    plt.scatter(pos_true['x'][-1], pos_true['y'][-1], c='green', s=100, marker='x', label="Actual Tunnel Exit")
    plt.scatter(pos_ai['x'][-1], pos_ai['y'][-1], c='blue', s=100, marker='x')

    plt.xlabel("Meters East")
    plt.ylabel("Meters North")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal') 

    # Subplot 2: Speed Comparison over time
    plt.subplot(2, 1, 2)
    time_axis = np.arange(0, duration) * dt
    plt.title("Speed Estimation During Blackout", fontsize=14, fontweight='bold')
    plt.plot(time_axis, [s * 3.6 for s in true_speeds_ms], 'g-', linewidth=2, label="Actual Speed")
    plt.plot(time_axis, [s * 3.6 for s in ai_speeds_ms], 'b--', linewidth=2, label="AI Estimated Speed")

    # Calculate raw IMU speed plot
    raw_speeds_plot = []
    rs = true_speeds_ms[0]
    for a in phone_accel_forward:
        rs += a * dt
        raw_speeds_plot.append(rs * 3.6)
    plt.plot(time_axis, raw_speeds_plot, 'r:', linewidth=1.5, label="Raw Sensor Speed (Double Integration Error)")

    plt.xlabel("Time in Tunnel (seconds)")
    plt.ylabel("Speed (km/h)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-10, max(max(true_speeds_ms)*3.6, 60) + 20) 

    plt.tight_layout()
    output_file = "blackout_simulation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved successfully to: {output_file}")

if __name__ == "__main__":
    main()
