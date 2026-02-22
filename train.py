import torch 
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


from src.data.dataloading import EnavippH5Dataset
from src.models.eGoNavi import eGoNavi
from src.data.preprocessing import collate_enavi
from src.utils.helper import compute_ddpm_loss, run_profiler

from src.utils.engine import train_step, eval_step
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np


def main():
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_epochs = 1
    batch_size   = 32
    np.random.seed(42)
    experiment_name = "eGoNavi_test01"
    max_patience  = 10

    
    encoder_cfg = {"in_channels": 5, "out_dim": 512}
    vint_cfg = {"token_dim": 512, "num_tokens": 6, "num_layers": 4, "num_heads": 4, "ff_dim": 2048}
    diffusion_cfg = {"context_dim": 512, "action_dim": 3, "traj_len": 8, "hidden_dim": 512}

    data_dir = Path("../h5_test") 

    h5_files = sorted(list(data_dir.glob("*.h5")))
    print(f"Loading {len(h5_files)} trajectories...")

    datasets = []
    for f in h5_files:
        if f.exists():
            datasets.append(EnavippH5Dataset(f, load_rgb=False))      

    if not datasets:
        raise FileNotFoundError("No valid datasets were loaded.")

    combined_dataset = ConcatDataset(datasets)
    dataset_size = len(combined_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size)) # Split indices for 80/20 train/val

    
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(combined_dataset, train_indices)
    val_dataset   = Subset(combined_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_enavi
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_enavi
    )
    
    model  = eGoNavi(encoder_cfg, vint_cfg, diffusion_cfg).to(device)

    param_groups = [
        {'params': model.vision_encoder.parameters(), 'lr': 1e-4},
        {'params': model.transformer.parameters(),    'lr': 5e-4},
        {'params': model.action_head.parameters(),    'lr': 5e-4},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-6)

    warmup_epochs = 5 # for the cosine scheduler to ramp up to full LR
    linear_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR( optimizer, schedulers=[ linear_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scaler    = GradScaler()
    
    print("Running profiler...")
    run_profiler(model, train_loader, optimizer, device)
    

    best_val_loss = float('inf')
    patience      = 0

    model_path = Path(f"./model/{experiment_name}")
    model_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, total_epochs + 1):

        train_loss = train_step(model, train_loader, optimizer, device, scaler, epoch)
        val_loss = eval_step(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch}/{total_epochs} | "
              f"Train loss: {train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | "
              f"LR (encoder, transformer, action_head): {optimizer.param_groups[0]['lr']:.2e} {optimizer.param_groups[1]['lr']:.2e} {optimizer.param_groups[2]['lr']:.2e}")
        

        
        torch.save({
                'epoch':      epoch,
                'model':      model.state_dict(),
                'val_loss':   val_loss,
            }, model_path / f'/checkpoint_epoch_{epoch}.pth')
        
        print(f" Saved {epoch} model (val_loss={val_loss:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience      = 0
        else:
            # Early stopping patience
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break



if __name__ == '__main__':
    main()
