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

class AdvancedIDRDataset(Dataset):
    # ==========================================
    # DATA MULTIPLIER TRICK
    # Step reduced from 25 to 5. Generates 5x more training pairs!
    # ==========================================
    def __init__(self, feature_list, label_list, window_size=50, step=5):
        self.x, self.y = [], []
        for feats, lbls in zip(feature_list, label_list):
            feats = np.clip(feats, -49.0, 49.0)
            for i in range(0, len(feats) - window_size, step):
                self.x.append(feats[i:i+window_size])
                self.y.append(lbls[i+window_size-1])
                
        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        x_win = self.x[idx].copy()
        scale = np.random.uniform(0.95, 1.05, size=(6,))
        x_win *= scale
        x_win += np.random.normal(0, 0.05, x_win.shape)
        return torch.tensor(x_win).transpose(0, 1), torch.tensor(self.y[idx])

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class RoNIN_ResNet_LSTM(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(32, 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = [BasicBlock1D(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

def main():
    print("=== CHAMPION BOOST: DATA MULTIPLIER TRAINING ===")
    pull_all_data()
    features, labels = load_all_data()
    
    split_idx = int(len(features) * 0.8)
    train_feats, train_lbls = features[:split_idx], labels[:split_idx]
    val_feats, val_lbls = features[split_idx:], labels[split_idx:]
    
    # Using Data Multiplier Dataset (5x Data)
    print("Generating 5x Expanded Dataset...")
    train_dataset = AdvancedIDRDataset(train_feats, train_lbls, step=5)
    val_dataset = AdvancedIDRDataset(val_feats, val_lbls, step=25) # Keep validation same for fair comparison
    
    print(f"Total Training Windows: {len(train_dataset)} (Massive Increase!)")
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RoNIN_ResNet_LSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 5x data means 1 epoch takes 5x longer. We train for 15 epochs.
    epochs = 15
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    criterion = nn.HuberLoss(delta=1.0)
    
    best_loss = float('inf')
    start_time = time.time()
    
    print("Beginning Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                
        train_rmse = np.sqrt(train_loss / len(train_loader) * 2) * 3.6
        val_rmse = np.sqrt(val_loss / len(val_loader) * 2) * 3.6
        print(f"Epoch {epoch+1:02d} | Train RMSE: {train_rmse:.2f} km/h | Val RMSE: {val_rmse:.2f} km/h")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'champion_boost_model.pth')
            
    print(f"Done in {(time.time()-start_time)/60:.1f} mins. Saved champion_boost_model.pth")

if __name__ == "__main__":
    main()
