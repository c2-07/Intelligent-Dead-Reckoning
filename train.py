import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import time

from dead_reckoning.model import RoNIN_ResNet_LSTM
from dead_reckoning.dataset import AdvancedIDRDataset, load_all_data, pull_all_data

def main():
    print("Starting Training Pipeline...")
    pull_all_data()
    features, labels = load_all_data()
    
    if len(features) == 0:
        print("Error: No data loaded. Did Git LFS pull successfully?")
        return

    split_idx = int(len(features) * 0.8)
    train_feats, train_lbls = features[:split_idx], labels[:split_idx]
    val_feats, val_lbls = features[split_idx:], labels[split_idx:]
    
    train_dataset = AdvancedIDRDataset(train_feats, train_lbls, step=25)
    val_dataset = AdvancedIDRDataset(val_feats, val_lbls, step=25)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    model = RoNIN_ResNet_LSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    epochs = 43
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.HuberLoss(delta=1.0)
    
    best_loss = float('inf')
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
            torch.save(model.state_dict(), 'models/resnet_bilstm_v1.pth')
            
    print(f"Done in {(time.time()-start_time)/60:.1f} mins. Saved models/resnet_bilstm_v1.pth")

if __name__ == "__main__":
    main()
