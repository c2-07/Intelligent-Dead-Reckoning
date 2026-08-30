import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

def preprocess_iovnb_pair(v_path, s_path, output_dir, lag_samples=9):
    print(f"Processing:\n  V: {v_path}\n  S: {s_path}")
    
    # 1. Load data
    v_df = pd.read_csv(v_path)
    s_df = pd.read_csv(s_path, encoding='latin-1')
    
    # ---------------------------------------------------------
    # FIX 1: Timestamp Restart (Use DATE column)
    # ---------------------------------------------------------
    print("Fixing timestamps...")
    # The date format is '2019-09-07 09:13:29:506' - pandas expects a dot for ms
    date_str = s_df['DATE (YYYY-MO-DD HH-MI-SS_SSS)'].str.replace(r':(\d{3})$', r'.\1', regex=True)
    s_df['Datetime'] = pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S.%f')
    
    # Calculate continuous milliseconds from the first sample
    s_df['Continuous_Time_ms'] = (s_df['Datetime'] - s_df['Datetime'].iloc[0]).dt.total_seconds() * 1000
    
    # ---------------------------------------------------------
    # FIX 2: Time Synchronization Lag
    # ---------------------------------------------------------
    print(f"Applying time shift (lag = {lag_samples} samples)...")
    # V leads S by 9 samples. So V[0] corresponds to S[9].
    # We drop the first 9 of S and last 9 of V.
    v_aligned = v_df.iloc[:-lag_samples].reset_index(drop=True)
    s_aligned = s_df.iloc[lag_samples:].reset_index(drop=True)
    
    # ---------------------------------------------------------
    # FIX 3: Dynamic Axis Re-alignment (Virtual Flattening)
    # ---------------------------------------------------------
    print("Virtually rotating phone sensors to align with vehicle...")
    
    # Extract gravity vectors
    gx = s_aligned['GRAVITY X (m/s²)'].values
    gy = s_aligned['GRAVITY Y (m/s²)'].values
    gz = s_aligned['GRAVITY Z (m/s²)'].values
    
    # Raw IMU vectors
    ax = s_aligned['ACCELEROMETER X (m/s²)'].values
    ay = s_aligned['ACCELEROMETER Y (m/s²)'].values
    az = s_aligned['ACCELEROMETER Z (m/s²)'].values
    
    gyaw = s_aligned['GYROSCOPE Yaw (rad/s)'].values
    gpitch = s_aligned['GYROSCOPE Pitch (rad/s)'].values
    groll = s_aligned['GYROSCOPE Roll (rad/s)'].values
    
    # Arrays to hold rotated data
    rot_ax, rot_ay, rot_az = np.zeros_like(ax), np.zeros_like(ay), np.zeros_like(az)
    rot_gyaw, rot_gpitch, rot_groll = np.zeros_like(gyaw), np.zeros_like(gpitch), np.zeros_like(groll)
    
    # Calculate rotation matrix for every row based on gravity
    for i in range(len(s_aligned)):
        g_vec = np.array([gx[i], gy[i], gz[i]])
        g_norm = np.linalg.norm(g_vec)
        if g_norm < 0.1:
            continue
            
        g_normed = g_vec / g_norm
        target_g = np.array([0, 0, 1]) # We want gravity pointing straight down (Z)
        
        # Calculate rotation axis and angle to align gravity with Z axis
        v = np.cross(g_normed, target_g)
        c = np.dot(g_normed, target_g)
        s = np.linalg.norm(v)
        
        if s < 1e-6:
            rot_matrix = np.eye(3)
        else:
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        
        # Apply rotation to Accelerometer
        a_vec = np.array([ax[i], ay[i], az[i]])
        a_rot = rot_matrix.dot(a_vec)
        rot_ax[i], rot_ay[i], rot_az[i] = a_rot
        
        # Apply rotation to Gyroscope
        g_gyro_vec = np.array([groll[i], gpitch[i], gyaw[i]]) # Map X/Y/Z to Roll/Pitch/Yaw
        g_rot = rot_matrix.dot(g_gyro_vec)
        rot_groll[i], rot_gpitch[i], rot_gyaw[i] = g_rot

    # Add corrected columns to dataframe
    s_aligned['Vehicle_Accel_X'] = rot_ax
    s_aligned['Vehicle_Accel_Y'] = rot_ay
    s_aligned['Vehicle_Accel_Z'] = rot_az
    s_aligned['Vehicle_Gyro_Roll'] = rot_groll
    s_aligned['Vehicle_Gyro_Pitch'] = rot_gpitch
    s_aligned['Vehicle_Gyro_Yaw'] = rot_gyaw
    
    # ---------------------------------------------------------
    # Save Cleaned Data
    # ---------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_v = os.path.join(output_dir, os.path.basename(v_path).replace('.csv', '_cleaned.csv'))
    out_s = os.path.join(output_dir, os.path.basename(s_path).replace('.csv', '_cleaned.csv'))
    
    v_aligned.to_csv(out_v, index=False)
    s_aligned.to_csv(out_s, index=False)
    
    print(f"✅ Success! Cleaned data saved to:\n  {out_v}\n  {out_s}\n")

if __name__ == "__main__":
    # Example usage for the files we already downloaded
    v_file = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv"
    s_file = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv"
    
    if os.path.exists(v_file) and os.path.exists(s_file):
        preprocess_iovnb_pair(v_file, s_file, "IO-VNBD/Cleaned_Data/")
    else:
        print("Please run `git lfs pull` to download the CSV files first.")
