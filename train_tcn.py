import os
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

# ==========================================
# 1. PREPROCESSING (In Colab)
# ==========================================
def preprocess_iovnb_pair(v_path, s_path, lag_samples=9):
    print(f"Loading data:\n  {v_path}\n  {s_path}")
    v_df = pd.read_csv(v_path)
    s_df = pd.read_csv(s_path, encoding='latin-1')
    v_df.columns = v_df.columns.str.strip()
    s_df.columns = s_df.columns.str.strip()
    
    # Lag correction
    v_aligned = v_df.iloc[:-lag_samples].reset_index(drop=True)
    s_aligned = s_df.iloc[lag_samples:].reset_index(drop=True)
    
    # Virtual flattening (simplified for speed)
    gx = s_aligned['GRAVITY X (m/s²)'].values
    gy = s_aligned['GRAVITY Y (m/s²)'].values
    gz = s_aligned['GRAVITY Z (m/s²)'].values
    ax = s_aligned['ACCELEROMETER X (m/s²)'].values
    ay = s_aligned['ACCELEROMETER Y (m/s²)'].values
    az = s_aligned['ACCELEROMETER Z (m/s²)'].values
    gyaw = s_aligned['GYROSCOPE Yaw (rad/s)'].values
    gpitch = s_aligned['GYROSCOPE Pitch (rad/s)'].values
    groll = s_aligned['GYROSCOPE Roll (rad/s)'].values
    
    rot_ax, rot_ay, rot_az = np.zeros_like(ax), np.zeros_like(ay), np.zeros_like(az)
    rot_gyaw, rot_gpitch, rot_groll = np.zeros_like(gyaw), np.zeros_like(gpitch), np.zeros_like(groll)
    
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

    s_aligned['Vehicle_Accel_X'] = rot_ax
    s_aligned['Vehicle_Accel_Y'] = rot_ay
    s_aligned['Vehicle_Accel_Z'] = rot_az
    s_aligned['Vehicle_Gyro_Roll'] = rot_groll
    s_aligned['Vehicle_Gyro_Pitch'] = rot_gpitch
    s_aligned['Vehicle_Gyro_Yaw'] = rot_gyaw
    
    return v_aligned, s_aligned

# ==========================================
# 2. DATASET DEFINITION
# ==========================================
class IDRDataset(Dataset):
    def __init__(self, s_df, v_df, window_size=20):
        self.window_size = window_size
        
        # Features: Accel X,Y,Z and Gyro R,P,Y
        features = s_df[['Vehicle_Accel_X', 'Vehicle_Accel_Y', 'Vehicle_Accel_Z',
                         'Vehicle_Gyro_Roll', 'Vehicle_Gyro_Pitch', 'Vehicle_Gyro_Yaw']].values
        
        # Labels: Velocity (km/hr) -> m/s
        labels = (v_df['Velocity (km/hr)'].values) / 3.6
        
        # Clip outliers to 5g
        features[:, 0:3] = np.clip(features[:, 0:3], -49.0, 49.0)
        
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.x) - self.window_size

    def __getitem__(self, idx):
        # Shape: (channels, seq_len) -> required for Conv1d
        x_window = self.x[idx:idx+self.window_size].transpose(0, 1)
        y_label = self.y[idx+self.window_size - 1]
        return x_window, y_label

# ==========================================
# 3. TCN MODEL
# ==========================================
class SimpleTCN(nn.Module):
    def __init__(self, num_channels=6):
        super(SimpleTCN, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, dilation=2, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, dilation=4, padding=4)
        self.relu3 = nn.ReLU()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1) # Output: Velocity in m/s

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.global_pool(x).squeeze(2)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(1)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def main():
    print("Starting pipeline...")
    v_path = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv"
    s_path = "IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv"
    
    if not os.path.exists(v_path):
        print(f"ERROR: Cannot find {v_path}")
        return
        
    v_df, s_df = preprocess_iovnb_pair(v_path, s_path)
    
    # Split 80/20
    split_idx = int(len(v_df) * 0.8)
    train_dataset = IDRDataset(s_df.iloc[:split_idx], v_df.iloc[:split_idx])
    val_dataset = IDRDataset(s_df.iloc[split_idx:], v_df.iloc[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleTCN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    best_val_loss = float('inf')
    
    print("Starting training...")
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
        
        # RMSE in km/h
        train_rmse_kmh = np.sqrt(train_loss) * 3.6
        val_rmse_kmh = np.sqrt(val_loss) * 3.6
        
        print(f"Epoch {epoch+1}/{epochs} | Train RMSE: {train_rmse_kmh:.2f} km/h | Val RMSE: {val_rmse_kmh:.2f} km/h")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_tcn_model.pth')
            print("  [Saved best model]")

    print("Training complete! Model saved as best_tcn_model.pth")

if __name__ == "__main__":
    main()
