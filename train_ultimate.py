import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import subprocess
import time

# ==========================================
# 1. DATA PREPROCESSING (Same robust loader)
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

class UltimateDataset(Dataset):
    def __init__(self, feature_list, label_list, window_size=50):
        self.x, self.y, self.is_stopped = [], [], []
        step = window_size // 2
        for feats, lbls in zip(feature_list, label_list):
            feats = np.clip(feats, -49.0, 49.0)
            for i in range(0, len(feats) - window_size, step):
                # Using Seq2Seq - predicting the whole window
                x_win = feats[i:i+window_size]
                y_win = lbls[i:i+window_size]
                stop_win = (y_win < 0.2).astype(np.float32) # < 0.2 m/s means stopped
                
                self.x.append(x_win)
                self.y.append(y_win)
                self.is_stopped.append(stop_win)
                
        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        self.is_stopped = np.array(self.is_stopped, dtype=np.float32)

    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        # Augmentation
        x_win = self.x[idx].copy()
        scale = np.random.uniform(0.9, 1.1, size=(6,))
        x_win *= scale
        x_win += np.random.normal(0, 0.1, x_win.shape)
        
        return torch.tensor(x_win), torch.tensor(self.y[idx]), torch.tensor(self.is_stopped[idx])

# ==========================================
# 2. ULTIMATE ARCHITECTURE (Transformer + FFT + ZUPT)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransPINN_DIO(nn.Module):
    def __init__(self, in_channels=6, d_model=128):
        super().__init__()
        # Spatial/Temporal Extraction
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 1. TRANSFORMER ENCODER
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 2. FFT SPECTROGRAM BRANCH
        # FFT of 50 samples gives 26 frequency bins. 6 channels * 26 bins = 156
        self.fft_fc = nn.Sequential(
            nn.Linear(156, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        # 3. OUTPUT HEADS
        self.vel_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        # 4. ZUPT (Zero Velocity) Classification Head
        self.stop_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        # x: (Batch, Seq, Channels)
        batch_size = x.size(0)
        
        # FFT Branch
        # Apply rfft over the sequence dimension
        x_fft = torch.abs(torch.fft.rfft(x, dim=1)) # (Batch, 26, 6)
        x_fft_flat = x_fft.reshape(batch_size, -1) # (Batch, 156)
        fft_feats = self.fft_fc(x_fft_flat).unsqueeze(1).repeat(1, x.size(1), 1) # (Batch, Seq, d_model)
        
        # Transformer Branch
        x_proj = self.input_proj(x)
        x_pos = self.pos_encoder(x_proj)
        trans_out = self.transformer(x_pos) # (Batch, Seq, d_model)
        
        # Fusion
        fused = torch.cat((trans_out, fft_feats), dim=-1)
        features = torch.relu(self.fusion(fused))
        
        # Predictions
        v_pred = self.vel_head(features).squeeze(-1) # (Batch, Seq)
        p_stop = self.stop_head(features).squeeze(-1) # (Batch, Seq)
        
        return v_pred, p_stop

# ==========================================
# 3. PHYSICS-INFORMED LOSS & TRAINING
# ==========================================
def physics_informed_loss(v_pred, v_true, p_stop, stop_true, dt=0.1, lambda_dist=0.5, alpha_stop=0.2):
    criterion = nn.SmoothL1Loss()
    bce = nn.BCELoss()
    
    # Standard Velocity Loss
    loss_vel = criterion(v_pred, v_true)
    
    # 1. Physics Trajectory Integration Loss (Distance Drift)
    # distance = sum(velocity * dt)
    dist_pred = torch.cumsum(v_pred * dt, dim=1)
    dist_true = torch.cumsum(v_true * dt, dim=1)
    loss_dist = criterion(dist_pred, dist_true)
    
    # 2. ZUPT Classification Loss
    loss_stop = bce(p_stop, stop_true)
    
    # Total PINN Loss
    total_loss = loss_vel + (lambda_dist * loss_dist) + (alpha_stop * loss_stop)
    return total_loss, loss_vel, loss_dist

def main():
    print("=== ULTIMATE Trans-PINN DIO PIPELINE ===")
    pull_all_data()
    features, labels = load_all_data()
    
    split_idx = int(len(features) * 0.8)
    train_feats, train_lbls = features[:split_idx], labels[:split_idx]
    val_feats, val_lbls = features[split_idx:], labels[split_idx:]
    
    train_dataset = UltimateDataset(train_feats, train_lbls)
    val_dataset = UltimateDataset(val_feats, val_lbls)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransPINN_DIO().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    epochs = 50 # We will keep it to 50 for time constraints, PINNs converge faster anyway
    best_loss = float('inf')
    
    print("Beginning Ultimate Physics-Informed Training...")
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss, train_v_loss = 0.0, 0.0
        for x, y_v, y_stop in train_loader:
            x, y_v, y_stop = x.to(device), y_v.to(device), y_stop.to(device)
            optimizer.zero_grad()
            v_pred, p_stop = model(x)
            
            loss, l_vel, l_dist = physics_informed_loss(v_pred, y_v, p_stop, y_stop)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_v_loss += l_vel.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss, val_v_loss = 0.0, 0.0
        with torch.no_grad():
            for x, y_v, y_stop in val_loader:
                x, y_v, y_stop = x.to(device), y_v.to(device), y_stop.to(device)
                v_pred, p_stop = model(x)
                
                # ZUPT Activation (Force zero velocity if network is 90% confident car is stopped)
                v_pred = torch.where(p_stop > 0.9, torch.zeros_like(v_pred), v_pred)
                
                loss, l_vel, l_dist = physics_informed_loss(v_pred, y_v, p_stop, y_stop)
                val_loss += loss.item()
                val_v_loss += l_vel.item()
                
        train_rmse = np.sqrt(train_v_loss / len(train_loader) * 2) * 3.6
        val_rmse = np.sqrt(val_v_loss / len(val_loader) * 2) * 3.6
        
        print(f"Epoch {epoch+1:02d} | Train RMSE: {train_rmse:.2f} km/h | Val RMSE: {val_rmse:.2f} km/h (w/ PINN & ZUPT)")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'ultimate_pinn_model.pth')
            
    print(f"Done in {(time.time()-start_time)/60:.1f} mins. Saved ultimate_pinn_model.pth")

if __name__ == "__main__":
    main()
