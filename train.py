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
# 1. ROBUST DATA PREPROCESSING
# ==========================================
def pull_all_data():
    print("Pulling entire IO-VNBD dataset (This will take a moment)...")
    subprocess.run("sudo apt-get update && sudo apt-get install -y git-lfs", shell=True, check=False)
    if not os.path.exists("IO-VNBD"):
        subprocess.run("git clone https://github.com/onyekpeu/IO-VNBD", shell=True, check=True)
    
    pull_cmd = "cd IO-VNBD && git lfs install && git lfs pull --include='**/*.csv'"
    subprocess.run(pull_cmd, shell=True, check=True)
    print("Data download complete.")

def process_single_pair(v_path, s_path, lag_samples=9):
    try:
        v_df = pd.read_csv(v_path)
        s_df = pd.read_csv(s_path, encoding='latin-1')
        v_df.columns = v_df.columns.str.strip()
        s_df.columns = s_df.columns.str.strip()
        
        # Lag correction
        v_aligned = v_df.iloc[:-lag_samples].reset_index(drop=True)
        s_aligned = s_df.iloc[lag_samples:].reset_index(drop=True)
        
        # Virtual flattening using gravity
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
            
            if s < 1e-6:
                rot_matrix = np.eye(3)
            else:
                kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rot_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
            
            rot_ax[i], rot_ay[i], rot_az[i] = rot_matrix.dot(np.array([ax[i], ay[i], az[i]]))
            rot_groll[i], rot_gpitch[i], rot_gyaw[i] = rot_matrix.dot(np.array([groll[i], gpitch[i], gyaw[i]]))

        features = np.column_stack((rot_ax, rot_ay, rot_az, rot_groll, rot_gpitch, rot_gyaw))
        labels = v_aligned['Velocity (km/hr)'].values / 3.6  # Convert to m/s
        return features, labels
    except Exception as e:
        print(f"Error processing {v_path}: {e}")
        return None, None

def load_all_data():
    base_dir = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/"
    folders = glob.glob(os.path.join(base_dir, "*"))
    
    all_features = []
    all_labels = []
    
    for folder in folders:
        v_files = sorted(glob.glob(os.path.join(folder, "**", "V-*.csv"), recursive=True))
        s_files = sorted(glob.glob(os.path.join(folder, "**", "S-*.csv"), recursive=True))
        
        for v_path, s_path in zip(v_files, s_files):
            print(f"Processing pair in {os.path.basename(folder)}...")
            feats, lbls = process_single_pair(v_path, s_path)
            if feats is not None:
                all_features.append(feats)
                all_labels.append(lbls)
                
    return all_features, all_labels

# ==========================================
# 2. DATASET & AUGMENTATION
# ==========================================
class AdvancedIDRDataset(Dataset):
    def __init__(self, feature_list, label_list, window_size=50, augment=False):
        self.window_size = window_size
        self.augment = augment
        
        self.x = []
        self.y = []
        
        # Create sliding windows with 50% overlap for more data
        step = window_size // 2
        for feats, lbls in zip(feature_list, label_list):
            feats = np.clip(feats, -49.0, 49.0) # Clip extreme outliers
            for i in range(0, len(feats) - window_size, step):
                self.x.append(feats[i:i+window_size])
                self.y.append(lbls[i+window_size-1])
                
        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_win = self.x[idx].copy()
        
        # Data Augmentation (only during training)
        if self.augment:
            # Random scaling (90% to 110%)
            scale = np.random.uniform(0.9, 1.1, size=(6,))
            x_win *= scale
            # Add Gaussian noise
            noise = np.random.normal(0, 0.1, x_win.shape)
            x_win += noise
            
        x_win = torch.tensor(x_win).transpose(0, 1) # (Channels, SeqLen)
        y_val = torch.tensor(self.y[idx])
        return x_win, y_val

# ==========================================
# 3. STATE-OF-THE-ART ARCHITECTURE (ResNet + BiLSTM)
# ==========================================
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

class RoNIN_ResNet_LSTM(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = ResBlock1D(64, 64)
        self.layer2 = ResBlock1D(64, 128, stride=2)
        
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, bidirectional=True)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Prep for LSTM (Batch, Seq, Channels)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        
        # Take the output of the last sequence step
        x = x[:, -1, :]
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)

# ==========================================
# 4. ADVANCED TRAINING LOOP
# ==========================================
def main():
    print("=== STARTING ADVANCED SOTA TRAINING PIPELINE ===")
    pull_all_data()
    
    print("Loading and preprocessing all datasets...")
    features, labels = load_all_data()
    
    # 80/20 Train/Val Split by sessions (not random rows, to prevent data leakage)
    split_idx = int(len(features) * 0.8)
    train_feats, train_lbls = features[:split_idx], labels[:split_idx]
    val_feats, val_lbls = features[split_idx:], labels[split_idx:]
    
    print(f"Train sessions: {len(train_feats)} | Val sessions: {len(val_feats)}")
    
    # Window size 50 = 5 seconds of context
    train_dataset = AdvancedIDRDataset(train_feats, train_lbls, window_size=50, augment=True)
    val_dataset = AdvancedIDRDataset(val_feats, val_lbls, window_size=50, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RoNIN_ResNet_LSTM().to(device)
    
    # Huber Loss is robust to extreme pothole/sensor outliers
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    epochs = 100
    patience = 15
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"Total training windows: {len(train_dataset)}")
    print("Beginning Training...")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        # Convert SmoothL1 to approximate RMSE in km/h for intuition
        train_rmse_kmh = np.sqrt(train_loss * 2) * 3.6 
        val_rmse_kmh = np.sqrt(val_loss * 2) * 3.6
        
        print(f"Epoch {epoch+1:03d}/{epochs} | LR: {scheduler.get_last_lr()[0]:.6f} | Train Error: ~{train_rmse_kmh:.2f} km/h | Val Error: ~{val_rmse_kmh:.2f} km/h")
        
        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'advanced_best_model.pth')
            print("  [Saved new best SOTA model]")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs! No improvement for {patience} epochs.")
                break

    elapsed = (time.time() - start_time) / 60
    print(f"Advanced training complete in {elapsed:.1f} minutes! Best model saved as advanced_best_model.pth")

if __name__ == "__main__":
    main()
