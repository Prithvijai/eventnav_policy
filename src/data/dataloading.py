import torch.nn.functional as F
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class EnavippH5Dataset(Dataset):
    """
    Dataset loader for ENavipp H5 files.
    Loads data from disk only when requested (Lazy Loading).
    Normalizes actions to mean 0, std 1 based on dataset statistics.
    """
    def __init__(self, h5_path, load_rgb=False, stats=None):
        self.h5_path = str(h5_path)
        self.load_rgb = load_rgb
        self._h5f = None # File handle stays None until first access
        
        if not Path(self.h5_path).exists():
            raise FileNotFoundError(f"H5 file not found at {self.h5_path}")

        # Open briefly to read metadata and shapes
        with h5py.File(self.h5_path, 'r') as f:
            self._available_keys = list(f.keys())
            if 'voxels' not in f:
                raise KeyError(f"Dataset 'voxels' not found in {h5_path}")
            
            self.length = f['voxels'].shape[0]
            self.has_rgb_data = 'rgb_images' in f and 'rgb_indices' in f and 'rgb_mask' in f
            
            if self.load_rgb and not self.has_rgb_data:
                print(f"Warning: load_rgb=True but RGB datasets not found in {h5_path}.")
                self.load_rgb = False

            # Calculate or use provided action normalization stats (Max Scaling)
            if stats is None:
                all_actions = f['actions'][()] # (N, 8, 3)
                # Max absolute value for XY and Theta for scaling to [-1, 1]
                self.action_max_xy = torch.from_numpy(np.abs(all_actions[:, :, 0:2]).max(axis=(0, 1)).astype(np.float32))
                self.action_max_theta = torch.from_numpy(np.abs(all_actions[:, :, 2:3]).max(axis=(0, 1)).astype(np.float32))
                # Avoid division by zero
                self.action_max_xy = torch.clamp(self.action_max_xy, min=1e-6)
                self.action_max_theta = torch.clamp(self.action_max_theta, min=1e-6)
            else:
                self.action_max_xy = stats['action_max_xy']
                self.action_max_theta = stats['action_max_theta']

    @property
    def h5f(self):
        """
        Lazy loader for the H5 file. 
        Important for PyTorch multi-processing (num_workers > 0).
        """
        if self._h5f is None:
            # Open in Single-Writer Multiple-Reader (SWMR) mode for stability
            self._h5f = h5py.File(self.h5_path, 'r', swmr=True)
        return self._h5f

    def __len__(self):
        return self.length

    def keys(self):
        """Returns the keys available in the H5 file."""
        return self._available_keys

    def get_stats(self):
        return {
            'action_max_xy': self.action_max_xy,
            'action_max_theta': self.action_max_theta
        }

    def __getitem__(self, idx):
        f = self.h5f
        VOXEL_SIZE = (72, 128)  # Target size
        
        # 1. Load and Resize History (5 frames)
        history_indices = [max(0, idx - i) for i in range(4, -1, -1)]
        history_voxels = []
        for h_idx in history_indices:
            v = torch.from_numpy(f['voxels'][h_idx].astype(np.float32)).unsqueeze(0) # (1, C, H, W)
            # Resize immediately to low-res to save RAM
            v = F.interpolate(v, size=VOXEL_SIZE, mode='area').squeeze(0)
            history_voxels.append(v)
        voxels = torch.stack(history_voxels, dim=0) # (5, C, 72, 128)
        
        # 2. Load and Resize Goal
        max_goal_dist = min(20, self.length - 1 - idx)
        if max_goal_dist > 0:
            goal_dist = np.random.randint(1, max_goal_dist + 1)
        else:
            goal_dist = 0
        goal_idx = min(idx + goal_dist, self.length - 1)
        
        goal_voxel = torch.from_numpy(f['voxels'][goal_idx].astype(np.float32)).unsqueeze(0)
        goal_voxel = F.interpolate(goal_voxel, size=VOXEL_SIZE, mode='area').squeeze(0)
        
        # 3. Load and normalize action (Max Scaling to [-1, 1])
        raw_action = torch.from_numpy(f['actions'][idx].astype(np.float32))
        raw_action_xy = raw_action[:,0:2]
        raw_action_theta = raw_action[:,2:3]
        
        action_xy = raw_action_xy / self.action_max_xy
        action_theta = raw_action_theta / self.action_max_theta
        
        action = torch.concat([action_xy, action_theta], dim=-1)

        
        sample = {
            'voxel': voxels,
            'goal_voxel': goal_voxel,
            'action': action,
            'timestamp_ns': f['timestamps_ns'][idx]
        }
        
        if self.load_rgb and self.has_rgb_data:
            if f['rgb_mask'][idx]:
                img_idx = f['rgb_indices'][idx]
                rgb = torch.from_numpy(f['rgb_images'][img_idx].astype(np.float32)).permute(2, 0, 1) / 255.0
                sample['rgb'] = rgb
            else:
                _, h, w = voxels.shape[-2:] 
                sample['rgb'] = torch.zeros((3, h, w), dtype=torch.float32)
                
        return sample

    def close(self):
        if self._h5f is not None:
            try:
                self._h5f.close()
            except:
                pass
            self._h5f = None

    def __del__(self):
        self.close()
