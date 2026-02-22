import torch
import torch.nn as nn
from .eGoNavi_vint import eGoNavi_ViNT
from .event_voxel_encoder import EventVoxelEncoder
from .diffusionhead import DiffusionPolicyAction

class eGoNavi(nn.Module):
    def __init__(self, encoder_cfg, vint_cfg, diffusion_cfg):
        super(eGoNavi, self).__init__()

        self.vision_encoder = EventVoxelEncoder(**encoder_cfg)
        self.transformer = eGoNavi_ViNT(**vint_cfg)
        self.action_head = DiffusionPolicyAction(**diffusion_cfg)
    
    def encode(self, obs_voxels, goal_voxel):
        """
        obs_voxels: (B, 5, C, H, W)
        goal_voxel: (B, C, H, W)
        Returns: context (B, 512)
        """
        B = obs_voxels.shape[0]
        obs_tokens = []

        # Encode current observations (history of 5)
        for i in range(5):
            obs_tokens.append(self.vision_encoder(obs_voxels[:, i]))

        # Encode goal observation
        goal_token = self.vision_encoder(goal_voxel)
        
        # Combine tokens for transformer (5 obs tokens + 1 goal token)
        tokens = torch.stack(obs_tokens + [goal_token], dim=1)  # (B, 6, 512)
        
        # Transformer cross-attention context
        context = self.transformer(tokens)  # (B, 512)
        return context

    def forward(self, obs_voxels, goal_voxel, noisy_action, timestep):
        """
        obs_voxels: (B, 5, C, H, W)   # Current observation history
        goal_voxel: (B, C, H, W)      # Goal observation
        noisy_action: (B, 8, 3)       # Noisy action sequence
        timestep: (B,)                # Diffusion timesteps
        """
        context = self.encode(obs_voxels, goal_voxel)
        
        # Predict noise added to actions
        pred_noise = self.action_head(noisy_action, timestep, context)
        
        return pred_noise
