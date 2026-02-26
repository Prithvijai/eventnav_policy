import torch 
import torch.nn.functional as F


def collate_enavi(batch):
    """
    Custom collate function to batch samples and resize event voxels and RGB.
    - Mode: 'bilinear' for both to preserve scene structure.
    """
    # Target sizes maintaining 16:9 aspect ratio
    RGB_SIZE = (126, 224)   # (Height, Width)

    # 1. Extract and stack basic tensors (Voxels are already resized in dataloader)
    voxels = torch.stack([s['voxel'].float() for s in batch])  # (B, 5, C, 72, 128)
    goal_voxels = torch.stack([s['goal_voxel'].float() for s in batch]) # (B, C, 72, 128)
    actions = torch.stack([s['action'] for s in batch])        # (B, 8, 3)
    timestamps = torch.tensor([s['timestamp_ns'] for s in batch])

    batch_out = {
        'voxel': voxels, 
        'goal_voxel': goal_voxels,
        'action': actions,
        'timestamp_ns': timestamps
    }
    
    # 2. Handle optional RGB if present in the batch
    if 'rgb' in batch[0]:
        rgbs = torch.stack([s['rgb'].float() for s in batch])
        # Resize to 16:9 target without cropping
        rgbs = F.interpolate(rgbs, size=RGB_SIZE, mode='bilinear', align_corners=False)
        batch_out['rgb'] = rgbs
        
    return batch_out
