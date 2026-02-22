import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import torch.nn.functional as F


def run_profiler(model, loader, optimizer, device, output_dir='./profiler_logs'):
    """
    Run profiler for 10 batches and save trace for TensorBoard.
    View with: tensorboard --logdir=./profiler_logs
    """
    model.train()
    batch_iter = iter(loader)
    
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,       # skip first batch (warmup)
            warmup=1,     # warmup 1 batch
            active=8,     # profile 8 batches
            repeat=1
        ),
        on_trace_ready=tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        for step in range(10):
            try:
                batch  = next(batch_iter)
            except StopIteration:
                break
                
            voxel  = batch['voxel'].to(device)
            action = batch['action'].to(device)
            
            if voxel.dim() == 4:
                voxel = voxel.unsqueeze(1).repeat(1, 5, 1, 1, 1)
            goal_voxel = voxel[:, -1]

            optimizer.zero_grad()
            
            with record_function("encode"):
                features = model.encode(voxel, goal_voxel)
            
            with record_function("ddpm_loss"):
                loss = compute_ddpm_loss(model, features, action)
            
            with record_function("backward"):
                loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            prof.step()
    
    # Print top 15 operations by CUDA time
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15
    ))


def compute_ddpm_loss(model, features, gt_actions):
    """
    model:      full eGoNavi model
    features:   (B, 512) transformer output
    gt_actions: (B, 8, 3) normalized ground truth actions
    
    Returns: scalar loss
    """
    B = gt_actions.shape[0]
    device = gt_actions.device
    
    # Flatten actions: (B, 8, 3) → (B, 24)
    gt_flat = gt_actions.reshape(B, -1)
    
    # Sample random diffusion timesteps for each sample in batch
    t = torch.randint(
        0,
        model.action_head.T,
        (B,),
        device=device
    )
    
    # Sample gaussian noise — same shape as flattened actions
    noise = torch.randn_like(gt_flat)
    
    # Forward diffusion: add noise to clean actions
    noisy_actions_flat = model.action_head.add_noise(gt_flat, noise, t)
    
    # Reshape for action head: (B, 24) -> (B, 8, 3)
    noisy_actions = noisy_actions_flat.reshape(B, model.action_head.traj_len, model.action_head.action_dim)
    
    # Predict the noise that was added
    pred_noise = model.action_head(noisy_actions, t, features)
    
    # Loss: how well did we predict the noise
    loss = F.mse_loss(pred_noise.reshape(B, -1), noise)
    
    return loss
