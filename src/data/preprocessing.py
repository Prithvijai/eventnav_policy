import torch 
import torch.nn.functional as F


def collate_enavi(batch):
    """
    Custom collate function to batch samples and resize event voxels and RGB.
    - Voxels: 128x72 (Full 16:9 aspect ratio, no cropping)
    - RGB: 224x126 (Full 16:9 aspect ratio, no cropping)
    - Mode: 'bilinear' for both to preserve scene structure.
    """
    # Target sizes maintaining 16:9 aspect ratio
    VOXEL_SIZE = (72, 128)  # (Height, Width)
    RGB_SIZE = (126, 224)   # (Height, Width)

    # 1. Extract and stack basic tensors
    voxels = torch.stack([s['voxel'].float() for s in batch])  # (B, 5, 720, 1280)
    actions = torch.stack([s['action'] for s in batch])        # (B, 8, 3)
    timestamps = torch.tensor([s['timestamp_ns'] for s in batch])
    
    # 2. Resize Voxels (Full View)
    # We remove the cropping logic to keep the full 16:9 field of view
    voxels = F.interpolate(voxels, size=VOXEL_SIZE, mode='bilinear', align_corners=False)
    
    batch_out = {
        'voxel': voxels,
        'action': actions,
        'timestamp_ns': timestamps
    }
    
    # 3. Handle optional RGB if present in the batch
    if 'rgb' in batch[0]:
        rgbs = torch.stack([s['rgb'].float() for s in batch])
        # Resize to 16:9 target without cropping
        rgbs = F.interpolate(rgbs, size=RGB_SIZE, mode='bilinear', align_corners=False)
        batch_out['rgb'] = rgbs
        
    return batch_out
