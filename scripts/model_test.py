import torch 
from src.data.dataloading import EnavippH5Dataset
from src.models.eGoNavi import eGoNavi
import matplotlib.pyplot as plt

# Default configurations for the model components
encoder_cfg = {
    "in_channels": 5, 
    "out_dim": 512
}

vint_cfg = {
    "token_dim": 512, 
    "num_tokens": 6, 
    "num_layers": 4, 
    "num_heads": 4, 
    "ff_dim": 2048
}

diffusion_cfg = {
    "context_dim": 512, 
    "action_dim": 3,   # Matches your H5 (271, 8, 3)
    "traj_len": 8,     # Matches your H5 (271, 8, 3)
    "hidden_dim": 512
}

# Instantiate the model
model = eGoNavi(encoder_cfg, vint_cfg, diffusion_cfg)

print("Model instantiated successfully!")
print(model)

# Example: Loading one H5 for testing
ds = EnavippH5Dataset("../h5_test/data_collect_20260219_170113.h5", load_rgb=False)
print(f"Dataset keys: {ds[10].keys()}")
