import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_tcn import preprocess_iovnb_pair, SimpleTCN
import warnings
warnings.filterwarnings('ignore')

print("1. Loading the aligned dataset...")
v_path = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv"
s_path = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv"

v_df, s_df = preprocess_iovnb_pair(v_path, s_path)

print("2. Loading the trained AI model...")
model = SimpleTCN(num_channels=6)
# Load on CPU since we are just doing a small inference test locally
model.load_state_dict(torch.load("best_tcn_model.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

print("3. Simulating a 60-second GPS Tunnel Blackout...")
# Let's pick a segment where the car is actively driving and turning
# Frame 10,000 to 10,600 (for 60 seconds)
start_idx = 10000
duration = 600 
window_size = 20
dt = 0.1 # 10Hz sampling = 0.1 seconds per frame

# --- Extract raw data for this segment ---
# Ground Truth Speed (from VBOX GPS on car roof)
true_speeds_kmh = v_df['Velocity (km/hr)'].values[start_idx : start_idx + duration]
true_speeds_ms = true_speeds_kmh / 3.6

# Phone Accelerometer (Forward direction)
phone_accel_x = s_df['Vehicle_Accel_X'].values[start_idx : start_idx + duration]

# Phone Gyroscope (Yaw / Turning direction) - Remember our audit found it maps to 'Pitch'
phone_gyro_yaw = s_df['Vehicle_Gyro_Pitch'].values[start_idx : start_idx + duration]

# --- Run the AI Model ---
predicted_speeds_ms = []
features = s_df[['Vehicle_Accel_X', 'Vehicle_Accel_Y', 'Vehicle_Accel_Z',
                 'Vehicle_Gyro_Roll', 'Vehicle_Gyro_Pitch', 'Vehicle_Gyro_Yaw']].values

# The model needs 20 frames of history to guess the current speed
with torch.no_grad():
    for i in range(duration):
        current_frame = start_idx + i
        # Get past 20 frames
        x_window = features[current_frame - window_size + 1 : current_frame + 1]
        x_window = np.clip(x_window, -49.0, 49.0) # Clip outliers like we did in training
        
        # Format for PyTorch (1 batch, 6 channels, 20 sequence length)
        x_tensor = torch.tensor(x_window, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
        
        # Predict speed
        pred_speed = model(x_tensor).item()
        # Prevent AI from predicting negative speeds
        pred_speed = max(0, pred_speed)
        predicted_speeds_ms.append(pred_speed)

# --- Simulate the Trajectories (Dead Reckoning) ---
# We will track X, Y coordinates starting at (0,0)
# We also track Heading (angle) starting at 0
pos_true = {'x': [0], 'y': [0], 'heading': 0}
pos_ai = {'x': [0], 'y': [0], 'heading': 0}
pos_raw = {'x': [0], 'y': [0], 'heading': 0}

raw_speed = true_speeds_ms[0] # Raw IMU starts with the exact correct speed before GPS drops

for i in range(duration):
    # Update heading using the phone's gyroscope (rad/s)
    # This is the "EKF" part simplified - we integrate gyro to get angle
    pos_true['heading'] += phone_gyro_yaw[i] * dt
    pos_ai['heading'] += phone_gyro_yaw[i] * dt
    pos_raw['heading'] += phone_gyro_yaw[i] * dt
    
    # 1. Ground Truth Path
    v_t = true_speeds_ms[i]
    pos_true['x'].append(pos_true['x'][-1] + v_t * np.cos(pos_true['heading']) * dt)
    pos_true['y'].append(pos_true['y'][-1] + v_t * np.sin(pos_true['heading']) * dt)
    
    # 2. AI Dead Reckoning Path
    v_ai = predicted_speeds_ms[i]
    pos_ai['x'].append(pos_ai['x'][-1] + v_ai * np.cos(pos_ai['heading']) * dt)
    pos_ai['y'].append(pos_ai['y'][-1] + v_ai * np.sin(pos_ai['heading']) * dt)
    
    # 3. Raw Phone IMU Path (What happens without AI)
    # Speed = previous speed + acceleration * time
    raw_speed += phone_accel_x[i] * dt
    pos_raw['x'].append(pos_raw['x'][-1] + raw_speed * np.cos(pos_raw['heading']) * dt)
    pos_raw['y'].append(pos_raw['y'][-1] + raw_speed * np.sin(pos_raw['heading']) * dt)

# --- Plot the Results ---
plt.figure(figsize=(12, 10))

# Subplot 1: Map / Trajectory
plt.subplot(2, 1, 1)
plt.title("GPS Tunnel Blackout Simulation (60 Seconds)", fontsize=14, fontweight='bold')
plt.plot(pos_true['x'], pos_true['y'], 'g-', linewidth=3, label="Actual Path (Ground Truth)")
plt.plot(pos_ai['x'], pos_ai['y'], 'b--', linewidth=2.5, label="AI Nav System (Our Solution)")
plt.plot(pos_raw['x'], pos_raw['y'], 'r:', linewidth=2, label="Raw Phone Sensors (Drift)")

# Mark start and end points
plt.scatter(0, 0, c='black', s=100, marker='o', label="Tunnel Enter (GPS Lost)")
plt.scatter(pos_true['x'][-1], pos_true['y'][-1], c='green', s=100, marker='x', label="Actual Tunnel Exit")
plt.scatter(pos_ai['x'][-1], pos_ai['y'][-1], c='blue', s=100, marker='x')

plt.xlabel("Meters East")
plt.ylabel("Meters North")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal') # Keep map proportions realistic

# Subplot 2: Speed Comparison over time
plt.subplot(2, 1, 2)
time_axis = np.arange(0, 60, dt)
plt.title("Speed Estimation During Blackout", fontsize=14, fontweight='bold')
plt.plot(time_axis, [s * 3.6 for s in true_speeds_ms], 'g-', linewidth=2, label="Actual Speed")
plt.plot(time_axis, [s * 3.6 for s in predicted_speeds_ms], 'b--', linewidth=2, label="AI Estimated Speed")

# Calculate how crazy the raw IMU speed gets
raw_speeds_plot = []
rs = true_speeds_ms[0]
for a in phone_accel_x:
    rs += a * dt
    raw_speeds_plot.append(rs * 3.6)
plt.plot(time_axis, raw_speeds_plot, 'r:', linewidth=1.5, label="Raw Sensor Speed (Massive Error)")

plt.xlabel("Time in Tunnel (seconds)")
plt.ylabel("Speed (km/h)")
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(-10, max(max(true_speeds_ms)*3.6, 60) + 20) # Cap Y axis so it doesn't get ruined by raw error

plt.tight_layout()
output_file = "blackout_simulation.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved successfully to: {output_file}")
