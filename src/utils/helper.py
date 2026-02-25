import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from src.models.diffusionhead import compute_ddpm_loss

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



