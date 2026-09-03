import time

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from dead_reckoning.dataset import AdvancedIDRDataset, load_all_data, pull_all_data
from dead_reckoning.model import RoNIN_ResNet_LSTM


def main():
    print("Starting Training Pipeline...")
    
    # Full deterministic mode for bit-exact reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    pull_all_data()
    features, labels = load_all_data()
    
    if len(features) == 0:
        print("Error: No data loaded. Did Git LFS pull successfully?")
        return

    split_idx = int(len(features) * 0.8)
    train_feats, train_lbls = features[:split_idx], labels[:split_idx]
    val_feats, val_lbls = features[split_idx:], labels[split_idx:]
    
    train_dataset = AdvancedIDRDataset(train_feats, train_lbls, augment=True)
    val_dataset = AdvancedIDRDataset(val_feats, val_lbls, augment=False)
    
    # Seeded generator for deterministic shuffle order
    g = torch.Generator()
    g.manual_seed(seed)
    
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=2, generator=g, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                            num_workers=2, worker_init_fn=worker_init_fn)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    model = RoNIN_ResNet_LSTM().to(device)
    
    # Huber Loss is robust to extreme pothole/sensor outliers
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
            torch.save(model.state_dict(), 'models/resnet_bilstm_latest.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break
            
    print(f"Done in {(time.time()-start_time)/60:.1f} mins. Saved models/resnet_bilstm_latest.pth")

if __name__ == "__main__":
    main()
