import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from src.models.diffusionhead import compute_ddpm_loss

def train_step(model, loader, optimizer, device, scaler, epoch):
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(progress_bar):
        voxel       = batch["voxel"].to(device)       # (B, 5, C, H, W)
        goal_voxel  = batch["goal_voxel"].to(device)  # (B, C, H, W)
        action      = batch["action"].to(device)      # (B, 8, 3)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast(device_type="cuda"):
            features = model.encode(voxel, goal_voxel)            # (B, 512)
            loss     = compute_ddpm_loss(model, features, action)
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping — important for diffusion stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        # progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / n_batches



@torch.no_grad()
def eval_step(model, loader, epoch, device):
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for batch_idx, batch in enumerate(progress_bar):
        voxel  = batch['voxel'].to(device)
        action = batch['action'].to(device)
        
        if voxel.dim() == 4:
            voxel = voxel.unsqueeze(1).repeat(1, 5, 1, 1, 1)
        goal_voxel = voxel[:, -1]

        with autocast(device_type="cuda"):
            features = model.encode(voxel, goal_voxel)
            loss     = compute_ddpm_loss(model, features, action)
        
        total_loss += loss.item()
    
    return total_loss / len(loader)