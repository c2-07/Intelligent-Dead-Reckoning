import os
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

from dead_reckoning.model import RoNIN_ResNet_LSTM

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

def main():
    model_path = "models/resnet_bilstm_v4.pth"
    print("Loading data for comprehensive SIH Submission Benchmark (STRICT HELD-OUT v4)...")

    v_path = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv"
    s_path = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv"

    v_df = pd.read_csv(v_path)
    s_df = pd.read_csv(s_path, encoding='latin-1')
    v_df.columns = v_df.columns.str.strip()
    s_df.columns = s_df.columns.str.strip()

    lag_samples = 9
    v_aligned = v_df.iloc[:-lag_samples].reset_index(drop=True)
    s_aligned = s_df.iloc[lag_samples:].reset_index(drop=True)

    # Note: Sequential split, meaning early records are train, late records are val.
    split_idx = int(len(v_aligned) * 0.8)
    v_val = v_aligned.iloc[split_idx:].reset_index(drop=True)
    s_val = s_aligned.iloc[split_idx:].reset_index(drop=True)

    val_features = extract_features(s_val)
    val_features = np.clip(val_features, -49.0, 49.0)

    model = RoNIN_ResNet_LSTM(in_channels=6)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    window_size = 50
    dt = 0.1
    ai_speeds_ms = np.zeros(len(s_val))
    
    print("Running Inference...")
    with torch.no_grad():
        for i in range(window_size - 1, len(s_val)):
            x_window = val_features[i - window_size + 1 : i + 1]
            x_tensor = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            pred = model(x_tensor).item()
            ai_speeds_ms[i] = max(0, pred)

    true_speeds_ms = v_val['Velocity (km/hr)'].values / 3.6
    absolute_heading = np.radians(s_val.iloc[:, 21].values)

    csv_data = []
    
    # 5s, 10s, 20s, 30s, 60s, 120s
    time_intervals = [5, 10, 20, 30, 60, 120]
    
    results = {}

    for t_sec in time_intervals:
        frames = int(t_sec / dt)
        drifts_m = []
        drifts_pct = []
        baseline_drifts_pct = []
        
        # We start evaluating after the initial 50-frame warmup
        # Use a 10-frame (1 second) stride to generate overlapping windows for a statistically robust sample size
        stride = 10
        for start_idx in range(window_size, len(s_val) - frames, stride):
            end_idx = start_idx + frames
            
            true_x, true_y = 0.0, 0.0
            ai_x, ai_y = 0.0, 0.0
            base_x, base_y = 0.0, 0.0
            
            dist_traveled = 0.0
            
            # Baseline: Constant Last-Known Smartphone GPS Speed (not Vbox truth)
            last_known_speed = s_val['GPS SPEED (Kmh)'].values[start_idx-1] / 3.6
            
            for i in range(start_idx, end_idx):
                h = absolute_heading[i]
                
                # True
                t_step = true_speeds_ms[i] * dt
                true_x += t_step * np.cos(h)
                true_y += t_step * np.sin(h)
                dist_traveled += t_step
                
                # AI
                ai_step = ai_speeds_ms[i] * dt
                ai_x += ai_step * np.cos(h)
                ai_y += ai_step * np.sin(h)
                
                # Baseline
                base_step = last_known_speed * dt
                base_x += base_step * np.cos(h)
                base_y += base_step * np.sin(h)
            
            if dist_traveled > 1.0: # Skip stationary segments for drift %
                error_m = np.sqrt((true_x - ai_x)**2 + (true_y - ai_y)**2)
                drift_p = (error_m / dist_traveled) * 100
                base_err_m = np.sqrt((true_x - base_x)**2 + (true_y - base_y)**2)
                base_drift_p = (base_err_m / dist_traveled) * 100
                
                drifts_m.append(error_m)
                drifts_pct.append(drift_p)
                baseline_drifts_pct.append(base_drift_p)
                
                csv_data.append({
                    'Trip_ID': 'M_Driver_B_Val',
                    'Blackout_Start_s': start_idx * dt,
                    'Blackout_End_s': end_idx * dt,
                    'Blackout_Duration_s': t_sec,
                    'True_Distance_m': round(dist_traveled, 2),
                    'True_X': round(true_x, 2),
                    'True_Y': round(true_y, 2),
                    'Predicted_X': round(ai_x, 2),
                    'Predicted_Y': round(ai_y, 2),
                    'Drift_Meters': round(error_m, 2),
                    'Drift_Percentage': round(drift_p, 2),
                    'Baseline_Constant_Speed_Drift_Percentage': round(base_drift_p, 2)
                })
                
        if len(drifts_pct) > 0:
            results[t_sec] = {
                'count': len(drifts_pct),
                'mean_m': np.mean(drifts_m),
                'median_m': np.median(drifts_m),
                'p90_m': np.percentile(drifts_m, 90),
                'mean_pct': np.mean(drifts_pct),
                'median_pct': np.median(drifts_pct),
                'p90_pct': np.percentile(drifts_pct, 90),
                'p95_pct': np.percentile(drifts_pct, 95),
                'max_pct': np.max(drifts_pct),
                'pass_under_10': np.mean(np.array(drifts_pct) < 10.0) * 100,
                'baseline_mean_pct': np.mean(baseline_drifts_pct)
            }
            
    df = pd.DataFrame(csv_data)
    df.to_csv("submission_predictions.csv", index=False)
    print("Saved detailed predictions to submission_predictions.csv")
    
    print("\n" + "="*50)
    print("DETAILED SIH SUBMISSION METRICS (STRICT HELD-OUT v4)")
    print("="*50)
    for t_sec in time_intervals:
        if t_sec in results:
            r = results[t_sec]
            print(f"\n--- {t_sec}s Blackout Window (N={r['count']}) ---")
            print(f"Mean Drift %:      {r['mean_pct']:.2f}%")
            print(f"Median Drift %:    {r['median_pct']:.2f}%")
            print(f"P90 Drift %:       {r['p90_pct']:.2f}%")
            print(f"P95 Drift %:       {r['p95_pct']:.2f}%")
            print(f"Worst-Case Drift:  {r['max_pct']:.2f}%")
            print(f"Pass Rate (<10%):  {r['pass_under_10']:.1f}%")
            print(f"Median Drift (m):  {r['median_m']:.2f}m")
            print(f"P90 Drift (m):     {r['p90_m']:.2f}m")
            print(f"BASELINE Constant-Speed Mean Drift: {r['baseline_mean_pct']:.2f}%")

if __name__ == "__main__":
    main()
