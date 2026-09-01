import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import subprocess
import time

# ==========================================
# 1. ROBUST DATA PIPELINE
# ==========================================
def pull_all_data():
    if not os.path.exists("IO-VNBD"):
        subprocess.run("git clone https://github.com/onyekpeu/IO-VNBD", shell=True)
        subprocess.run("cd IO-VNBD && git lfs install && git lfs pull --include='**/*.csv'", shell=True)

def process_single_pair(v_path, s_path, lag_samples=9):
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
    except:
        return None, None

def load_all_data():
    base_dir = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/"
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

class KinematicDataset(Dataset):
    def __init__(self, feature_list, label_list, window_size=50):
        self.x, self.y = [], []
        step = window_size // 2
        for feats, lbls in zip(feature_list, label_list):
            feats = np.clip(feats, -49.0, 49.0)
            for i in range(0, len(feats) - window_size, step):
                self.x.append(feats[i:i+window_size])
                self.y.append(lbls[i:i+window_size])
                
        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

# ==========================================
# 2. SOTA ARCHITECTURE (DUET Bias + Suspension Decoupling)
# ==========================================
class SOTA_2025_Net(nn.Module):
    def __init__(self, in_channels=6, d_model=128):
        super().__init__()
        
        # Stream 1: Suspension Decoupler (Low Pass / High Pass Simulation via Dilated Convs)
        self.high_freq_conv = nn.Conv1d(in_channels, d_model//2, kernel_size=3, dilation=1, padding=1)
        self.low_freq_conv = nn.Conv1d(in_channels, d_model//2, kernel_size=7, dilation=2, padding=6)
        
        self.fusion = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # Temporal Core (LSTM as lightweight Mamba substitute for standard PyTorch)
        self.temporal = nn.LSTM(d_model, d_model//2, num_layers=2, batch_first=True, bidirectional=True)
        
        # Output Heads
        # Head 1: Forward Speed (vx)
        self.vel_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        
        # Head 2: DUET Dynamic Bias Estimator (outputs a continuously updating bias vector)
        self.bias_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, in_channels))

    def forward(self, x):
        # x: (Batch, Seq, Channels)
        x_t = x.transpose(1, 2) # (Batch, Channels, Seq)
        
        hf = torch.relu(self.high_freq_conv(x_t))
        lf = torch.relu(self.low_freq_conv(x_t))
        
        fused = torch.cat((hf, lf), dim=1) # (Batch, d_model, Seq)
        fused = self.fusion(fused).transpose(1, 2) # (Batch, Seq, d_model)
        
        features, _ = self.temporal(fused) # (Batch, Seq, d_model)
        
        v_pred = self.vel_head(features).squeeze(-1) # (Batch, Seq)
        bias_pred = self.bias_head(features) # (Batch, Seq, 6)
        
        return v_pred, bias_pred

# ==========================================
# 3. PHYSICS-INFORMED KINEMATIC LOSS
# ==========================================
def kinematic_loss(v_pred, v_true, bias_pred, raw_inputs, lambda_ackermann=0.1, lambda_bias=0.01):
    criterion = nn.SmoothL1Loss()
    
    # 1. Base Velocity Loss
    loss_vel = criterion(v_pred, v_true)
    
    # 2. DUET Bias Smoothness Loss (Random Walk constraint)
    # Bias shouldn't jump wildly between frames
    bias_diff = bias_pred[:, 1:, :] - bias_pred[:, :-1, :]
    loss_bias = torch.mean(bias_diff ** 2)
    
    # 3. Ackermann Kinematic Centrifugal Constraint
    # In a turn, Lateral Accel (ay) = v_forward * yaw_rate
    # Raw Inputs: [ax, ay, az, roll, pitch, yaw]
    # NOTE: Assuming ay is index 1, yaw_rate is index 5 in the rotated frame
    ay = raw_inputs[:, :, 1]
    yaw_rate = raw_inputs[:, :, 5]
    
    # Expected lateral acceleration based on AI's predicted speed
    expected_ay = v_pred * yaw_rate
    
    # Only enforce this strongly when actually turning (yaw_rate > 0.1)
    turn_mask = (torch.abs(yaw_rate) > 0.1).float()
    loss_ackermann = torch.mean(turn_mask * (ay - expected_ay)**2)
    
    total_loss = loss_vel + (lambda_bias * loss_bias) + (lambda_ackermann * loss_ackermann)
    return total_loss, loss_vel, loss_ackermann

def main():
    print("=== SOTA 2025 KINEMATIC TRAINING PIPELINE ===")
    pull_all_data()
    features, labels = load_all_data()
    
    split_idx = int(len(features) * 0.8)
    train_feats, train_lbls = features[:split_idx], labels[:split_idx]
    val_feats, val_lbls = features[split_idx:], labels[split_idx:]
    
    train_dataset = KinematicDataset(train_feats, train_lbls)
    val_dataset = KinematicDataset(val_feats, val_lbls)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SOTA_2025_Net().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    epochs = 40
    best_loss = float('inf')
    
    print("Beginning Training...")
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss, train_v_loss = 0.0, 0.0
        for x, y_v in train_loader:
            x, y_v = x.to(device), y_v.to(device)
            optimizer.zero_grad()
            
            v_pred, bias_pred = model(x)
            loss, l_vel, l_ack = kinematic_loss(v_pred, y_v, bias_pred, x)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_v_loss += l_vel.item()
            
        # Validation
        model.eval()
        val_loss, val_v_loss = 0.0, 0.0
        with torch.no_grad():
            for x, y_v in val_loader:
                x, y_v = x.to(device), y_v.to(device)
                v_pred, bias_pred = model(x)
                loss, l_vel, l_ack = kinematic_loss(v_pred, y_v, bias_pred, x)
                val_loss += loss.item()
                val_v_loss += l_vel.item()
                
        train_rmse = np.sqrt(train_v_loss / len(train_loader) * 2) * 3.6
        val_rmse = np.sqrt(val_v_loss / len(val_loader) * 2) * 3.6
        
        print(f"Epoch {epoch+1:02d} | Train RMSE: {train_rmse:.2f} km/h | Val RMSE: {val_rmse:.2f} km/h")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'sota_2025_model.pth')
            
    print(f"Done in {(time.time()-start_time)/60:.1f} mins. Saved sota_2025_model.pth")

if __name__ == "__main__":
    main()
