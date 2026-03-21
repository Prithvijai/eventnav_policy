import torch
import torch.nn as nn
from .eGoNavi_vint import eGoNavi_ViNT
from .event_voxel_encoder import EventVoxelEncoder
from .diffusionhead import DiffusionPolicyAction

class eGoNavi(nn.Module):
    def __init__(self, encoder_cfg, vint_cfg, diffusion_cfg):
        super(eGoNavi, self).__init__()

        self.vision_encoder = EventVoxelEncoder(**encoder_cfg)
        
        # SOTA Early Fusion: Concatenate current obs + goal (2 * in_channels)
        # and compress back to in_channels using a 1x1 convolution
        in_channels = encoder_cfg.get("in_channels", 5)
        self.fusion_layer = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        
        self.transformer = eGoNavi_ViNT(**vint_cfg)
        self.action_head = DiffusionPolicyAction(**diffusion_cfg)
    
    def encode(self, obs_voxels, goal_voxel):
        """
        obs_voxels: (B, 5, C, H, W) - History of 5 observations
        goal_voxel: (B, C, H, W)    - Goal observation
        Returns: context (B, 512)
        """
        B, T, C, H, W = obs_voxels.shape
        obs_tokens = []

        # 1. Encode past observations (0 to 3) individually
        for i in range(4):
            obs_tokens.append(self.vision_encoder(obs_voxels[:, i]))

        # 2. SOTA Early Fusion for "Relative" Goal Encoding
        # Concatenate current observation (index 4) and goal voxel
        current_obs = obs_voxels[:, 4] # (B, C, H, W)
        relative_goal_input = torch.cat([current_obs, goal_voxel], dim=1) # (B, 2*C, H, W)
        
        # Compress fusion to original channel size
        fused_goal_voxel = self.fusion_layer(relative_goal_input) # (B, C, H, W)
        
        # Encode the relative goal information
        relative_goal_token = self.vision_encoder(fused_goal_voxel) # (B, 512)
        
        # 3. Combine tokens for transformer (4 past tokens + 1 relative goal token)
        # Note: We now have 5 tokens total instead of 6. 
        # Token 4 (last one) now represents "Where am I relative to the goal?"
        tokens = torch.stack(obs_tokens + [relative_goal_token], dim=1)  # (B, 5, 512)
        
        # Transformer cross-attention context
        context = self.transformer(tokens)  # (B, 512)
        return context

    @torch.no_grad()
    def sample(self, obs_voxels, goal_voxel, device):
        """
        Sample actions given observations.
        """
        context = self.encode(obs_voxels, goal_voxel)
        return self.action_head.sample(context, device)

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
