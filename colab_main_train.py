"""
Main branch training + evaluation on Colab.
Self-contained: embeds the model, dataset, train, and evaluate logic.
"""
import os
import glob
import subprocess
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
warnings.filterwarnings('ignore')

# ==========================================
# DATA PIPELINE (from src/dead_reckoning/dataset.py)
# ==========================================
def pull_all_data():
    check_dir = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/"
    if os.path.exists(check_dir) and len(glob.glob(os.path.join(check_dir, "**", "*.csv"), recursive=True)) > 10:
        print("IO-VNBD dataset already present, skipping download.")
        return
    print("Cloning IO-VNBD Dataset...")
    subprocess.run("git clone https://github.com/onyekpeu/IO-VNBD", shell=True)
    subprocess.run("cd IO-VNBD && git lfs install --force && git lfs pull --include='**/*.csv'", shell=True)

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
    except Exception as e:
        return None, None

def load_all_data(exclude_trip=None, include_trip=None):
    base_dir = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/"
    folders = sorted(glob.glob(os.path.join(base_dir, "*")))
    all_features, all_labels = [], []
    for folder in folders:
        folder_name = os.path.basename(folder)
        if exclude_trip and exclude_trip in folder_name: continue
        if include_trip and include_trip not in folder_name: continue
        
        v_files = sorted(glob.glob(os.path.join(folder, "**", "V-*.csv"), recursive=True))
        s_files = sorted(glob.glob(os.path.join(folder, "**", "S-*.csv"), recursive=True))
        for v_path, s_path in zip(v_files, s_files):
            feats, lbls = process_single_pair(v_path, s_path)
            if feats is not None:
                all_features.append(feats)
                all_labels.append(lbls)
    return all_features, all_labels

# ==========================================
# DATASET (from src/dead_reckoning/dataset.py)
# ==========================================
class AdvancedIDRDataset(Dataset):
    def __init__(self, feature_list, label_list, window_size=50, augment=False):
        self.augment = augment
        self.x, self.y = [], []
        step = window_size // 2
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
        if self.augment:
            scale = np.random.uniform(0.9, 1.1, size=(6,))
            x_win *= scale
            x_win += np.random.normal(0, 0.05, x_win.shape)
        return torch.tensor(x_win).transpose(0, 1), torch.tensor(self.y[idx])

# ==========================================
# MODEL (from src/dead_reckoning/model.py)
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
                nn.BatchNorm1d(out_channels))
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)

# ==========================================
# TRAIN (from scripts/train.py)
# ==========================================
def train_model():
    print("=== DETERMINISTIC TRAINING: Colab T4 GPU (STRICT HELD-OUT SPLIT) ===")
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    pull_all_data()
    
    # STRICT TRIP-LEVEL SPLIT to prevent data leakage
    print("Loading Training Data (Excluding M (Driver B))...")
    train_feats, train_lbls = load_all_data(exclude_trip="M (Driver B)")
    print("Loading Validation Data (Only M (Driver B))...")
    val_feats, val_lbls = load_all_data(include_trip="M (Driver B)")
    
    print(f"Loaded {len(train_feats)} training trips, {len(val_feats)} validation trips.")

    if len(train_feats) == 0:
        print("ERROR: No data loaded.")
        return None

    train_dataset = AdvancedIDRDataset(train_feats, train_lbls, augment=True)
    val_dataset = AdvancedIDRDataset(val_feats, val_lbls, augment=False)

    g = torch.Generator()
    g.manual_seed(seed)

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=2, generator=g, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                            num_workers=2, worker_init_fn=worker_init_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = RoNIN_ResNet_LSTM().to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    epochs = 100
    patience = 15
    best_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

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
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        train_rmse = np.sqrt(train_loss * 2) * 3.6
        val_rmse = np.sqrt(val_loss * 2) * 3.6
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Error: ~{train_rmse:.2f} km/h | Val Error: ~{val_rmse:.2f} km/h")

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'colab_trained_model_v4.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break

    print(f"Done in {(time.time()-start_time)/60:.1f} mins.")
    model.load_state_dict(torch.load('colab_trained_model_v4.pth', map_location=device, weights_only=True))
    return model

if __name__ == "__main__":
    model = train_model()
    print("Training finished. File saved as colab_trained_model_v4.pth")
