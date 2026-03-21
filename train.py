import h5py
import torch 
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


from src.data.dataloading import EnavippH5Dataset
from src.models.eGoNavi import eGoNavi
from src.data.preprocessing import collate_enavi
from src.utils.helper import run_profiler

from src.utils.engine import train_step, eval_step
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

def load_nomad_weights(model, nomad_path):
    print(f"Attempting to load SOTA NoMaD weights from {nomad_path}...")
    if not Path(nomad_path).exists():
        print(f"Warning: {nomad_path} not found. Skipping weight load.")
        return model
    
    checkpoint = torch.load(nomad_path, map_location='cpu')
    sota_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model_dict = model.state_dict()
    
    load_dict = {}
    
    # Mapping logic for SOTA -> Our Model
    for k, v in sota_dict.items():
        new_k = None
        
        # 1. Vision Encoder Mapping (EfficientNet)
        if k.startswith("vision_encoder.obs_encoder."):
            new_k = k.replace("vision_encoder.obs_encoder.", "vision_encoder.encoder.")
            if "_conv_stem.weight" in k: continue # Skip first layer (5/10 channels vs 3)

        # 2. Transformer Mapping (SA Encoder)
        elif k.startswith("vision_encoder.sa_encoder."):
            # SOTA: vision_encoder.sa_encoder.layers.0.self_attn.in_proj_weight
            # Ours: transformer.transformer.layers.0.self_attn.in_proj_weight
            new_k = k.replace("vision_encoder.sa_encoder.", "transformer.transformer.")
        elif k == "vision_encoder.pos_embed":
            new_k = "transformer.pos_embed"

        # 3. Action Head Mapping (Diffusion)
        elif k.startswith("noise_pred_net."):
            new_k = k.replace("noise_pred_net.", "action_head.unet_block.")

        # Final Check and Load
        if new_k and new_k in model_dict:
            if model_dict[new_k].shape == v.shape:
                load_dict[new_k] = v
            else:
                print(f"Shape mismatch for {new_k}: SOTA {v.shape} vs Model {model_dict[new_k].shape}")

    msg = model.load_state_dict(load_dict, strict=False)
    print(f"Successfully loaded {len(load_dict)} layers from NoMaD.")
    print(f"Missing keys (intended): {len([k for k in msg.missing_keys if 'fusion' not in k and 'conv_stem' not in k])}")
    return model

def main():
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_epochs = 500
    batch_size   = 32
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    experiment_name = "eGoNavi_nomad_v256_early_fusion"
    load_nomad = True 
    freeze_backbone = True # Freeze pre-trained parts for small data
    nomad_path = "./nomad.pth"
    
    max_patience  = 20

    # 1. Architecture (256-D SOTA)
    encoder_cfg = {"in_channels": 5, "out_dim": 256}
    vint_cfg = {"token_dim": 256, "num_tokens": 5, "num_layers": 4, "num_heads": 4, "ff_dim": 1024}
    diffusion_cfg = {"context_dim": 256, "action_dim": 3, "traj_len": 8, "hidden_dim": 256}

    data_dir = Path("../h5_test") 
    h5_files = sorted(list(data_dir.glob("*.h5")))
    print(f"Loading {len(h5_files)} trajectories...")

    # 2. Update to Max-Scaling Statistics
    print(f"Calculating global action max stats across {len(h5_files)} files...")
    all_actions_list = []
    for f_path in h5_files:
        with h5py.File(f_path, 'r') as f:
            all_actions_list.append(np.abs(f['actions'][()]))
    
    all_actions_concat = np.concatenate(all_actions_list, axis=0)
    global_max_xy = torch.from_numpy(all_actions_concat[:, :, 0:2].max(axis=(0, 1)).astype(np.float32))
    global_max_theta = torch.from_numpy(all_actions_concat[:, :, 2:3].max(axis=(0, 1)).astype(np.float32))
    
    global_stats = {
        'action_max_xy': torch.clamp(global_max_xy, min=1e-6),
        'action_max_theta': torch.clamp(global_max_theta, min=1e-6)
    }
    print(f"Global Max XY: {global_max_xy}, Max Theta: {global_max_theta}")

    datasets = []
    for f in h5_files:
        if f.exists():
            datasets.append(EnavippH5Dataset(f, load_rgb=False, stats=global_stats))      

    if not datasets:
        raise FileNotFoundError("No valid datasets were loaded.")

    combined_dataset = ConcatDataset(datasets)
    dataset_size = len(combined_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size)) 

    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset   = Subset(combined_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_enavi)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_enavi)
    
    model  = eGoNavi(encoder_cfg, vint_cfg, diffusion_cfg).to(device)

    # 3. Optional SOTA Weight Loading
    if load_nomad:
        model = load_nomad_weights(model, nomad_path)

    param_groups = [
        {'params': model.vision_encoder.parameters(), 'lr': 1e-4}, # Lower LR for pre-trained parts
        {'params': model.fusion_layer.parameters(),   'lr': 1e-4}, # New layer, higher LR
        {'params': model.transformer.parameters(),    'lr': 1e-4}, 
        {'params': model.action_head.parameters(),    'lr': 1e-4},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-6)

    warmup_epochs = 5
    linear_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scaler = GradScaler()
    
    best_val_loss = float('inf')
    patience      = 0

    model_path = Path(f"./model/{experiment_name}")
    model_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, total_epochs + 1):
        train_loss = train_step(model, train_loader, optimizer, device, scaler, epoch)
        val_loss = eval_step(model, val_loader, epoch, device)
        scheduler.step()
        
        print(f"Epoch {epoch}/{total_epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'val_loss': val_loss}, model_path / 'best_model.pth')
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'val_loss': val_loss}, model_path / f'checkpoint_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
