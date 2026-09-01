import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import subprocess

def pull_all_data():
    """Downloads the IO-VNBD dataset from GitHub LFS if not present."""
    if not os.path.exists("IO-VNBD"):
        print("Cloning IO-VNBD Dataset...")
        subprocess.run("git clone https://github.com/onyekpeu/IO-VNBD", shell=True)
        print("Pulling LFS files...")
        subprocess.run("cd IO-VNBD && git lfs install && git lfs pull --include='**/*.csv'", shell=True)

def process_single_pair(v_path, s_path, lag_samples=9):
    """Processes a synchronized Vehicle and Smartphone CSV pair."""
    try:
        v_df = pd.read_csv(v_path)
        s_df = pd.read_csv(s_path, encoding='latin-1')
        v_df.columns = v_df.columns.str.strip()
        s_df.columns = s_df.columns.str.strip()
        
        v_aligned = v_df.iloc[:-lag_samples].reset_index(drop=True)
        s_aligned = s_df.iloc[lag_samples:].reset_index(drop=True)
        
        gx, gy, gz = s_aligned['GRAVITY X (m/s²)'].values, s_aligned['GRAVITY Y (m/s²)'].values, s_aligned['GRAVITY Z (m/s²)'].values
        ax, ay, az = s_aligned['ACCELEROMETER X (m/s²)'].values, s_aligned['ACCELEROMETER Y (m/s²)'].values, s_aligned['ACCELEROMETER Z (m/s²)'].values
        groll, gpitch, gyaw = s_aligned['GYROSCOPE Roll (rad/s)'].values, s_aligned['GYROSCOPE Pitch (rad/s)'].values, s_aligned['GYROSCOPE Yaw (rad/s)'].values
        
        rot_ax, rot_ay, rot_az = np.zeros_like(ax), np.zeros_like(ay), np.zeros_like(az)
        rot_groll, rot_gpitch, rot_gyaw = np.zeros_like(groll), np.zeros_like(gpitch), np.zeros_like(gyaw)
        
        # Apply Gravity Rotation Matrix to align phone axes to Earth's gravity
        for i in range(len(s_aligned)):
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

        features = np.column_stack((rot_ax, rot_ay, rot_az, rot_groll, rot_gpitch, rot_gyaw))
        labels = v_aligned['Velocity (km/hr)'].values / 3.6
        return features, labels
    except Exception as e:
        # Fails silently for corrupted/LFS pointer files during dataset loading
        return None, None

def load_all_data(base_dir="IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/"):
    """Loads all V and S CSV pairs from the dataset directory."""
    folders = glob.glob(os.path.join(base_dir, "*"))
    all_features, all_labels = [], []
    for folder in folders:
        v_files = sorted(glob.glob(os.path.join(folder, "**", "V-*.csv"), recursive=True))
        s_files = sorted(glob.glob(os.path.join(folder, "**", "S-*.csv"), recursive=True))
        for v_path, s_path in zip(v_files, s_files):
            feats, lbls = process_single_pair(v_path, s_path)
            if feats is not None:
                all_features.append(feats)
                all_labels.append(lbls)
    return all_features, all_labels

class AdvancedIDRDataset(Dataset):
    """
    PyTorch Dataset for Dead Reckoning.
    Creates sliding windows of length `window_size` across the time series.
    """
    def __init__(self, feature_list, label_list, window_size=50, step=25):
        self.x, self.y = [], []
        for feats, lbls in zip(feature_list, label_list):
            # Clip outlier values
            feats = np.clip(feats, -49.0, 49.0)
            for i in range(0, len(feats) - window_size, step):
                self.x.append(feats[i:i+window_size])
                self.y.append(lbls[i+window_size-1])
                
        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self): 
        return len(self.x)

    def __getitem__(self, idx):
        x_win = self.x[idx].copy()
        # Basic Data Augmentation: Scale injection
        scale = np.random.uniform(0.9, 1.1, size=(6,))
        x_win *= scale
        x_win += np.random.normal(0, 0.05, x_win.shape)
        # Transpose for PyTorch 1D convolutions: (Channels, Sequence_Length)
        return torch.tensor(x_win).transpose(0, 1), torch.tensor(self.y[idx])
