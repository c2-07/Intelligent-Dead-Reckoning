import torch
import os
from scripts.train import RoNIN_ResNet_LSTM

def export_to_onnx():
    print("Loading PyTorch model...")
    model = RoNIN_ResNet_LSTM(in_channels=6)
    
    # Load weights safely
    try:
        model.load_state_dict(torch.load("models/champion_model.pth", map_location='cpu', weights_only=True))
    except:
        model.load_state_dict(torch.load("models/champion_model.pth", map_location='cpu'))
    
    model.eval()

    # Create a dummy input tensor with the exact shape the model expects
    # Shape: (Batch_Size, Sequence_Length, Channels) -> (1, 50, 6)
    print("Creating dummy input tensor of shape (1, 50, 6)...")
    dummy_input = torch.randn(1, 6, 50)

    onnx_path = "models/champion_model.onnx"
    print(f"Exporting to {onnx_path}...")
    
    # Export the model
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14,  # Standard version for modern mobile inference
        do_constant_folding=True,
        input_names=['imu_window'],
        output_names=['forward_speed'],
        dynamic_axes={
            'imu_window': {0: 'batch_size'},
            'forward_speed': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Successfully exported PyTorch model to ONNX format!")
    print("This .onnx file can now be loaded natively in Flutter/Android.")

if __name__ == "__main__":
    export_to_onnx()
