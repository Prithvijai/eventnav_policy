import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionPolicyAction(nn.Module):
    """
    DDPM Action Head for Diffusion Policy.
    Predicts the noise added to a sequence of actions.
    """
    def __init__(self, context_dim=512, action_dim=3, traj_len=8, hidden_dim=512, T=100):
        super().__init__()

        self.traj_len = traj_len
        self.action_dim = action_dim
        self.T = T # Number of diffusion steps
        
        # Precompute DDPM schedule for noise addition
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        # Timestep Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Main MLP
        input_dim = (traj_len * action_dim) + context_dim + hidden_dim
        
        self.mid_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, traj_len * action_dim)
        )

    def add_noise(self, x_0, noise, t):
        """Add noise according to the DDPM forward process."""
        # x_0: (B, 24) or (B, 8, 3)
        # noise: (B, 24) or (B, 8, 3)
        # t: (B,)
        
        # Ensure correct shapes for broadcasting
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1)
        if x_0.dim() == 2: # flattened
             sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1)
        
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1)
        if x_0.dim() == 2: # flattened
             sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1)

        return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise

    def forward(self, noisy_action, timestep, context):
        """
        noisy_action: (B, traj_len, action_dim)
        timestep: (B,) 
        context: (B, context_dim)
        Returns: predicted noise (B, traj_len, action_dim)
        """
        B = noisy_action.shape[0]
        
        # Flatten noisy action
        a_flat = noisy_action.reshape(B, -1)
        
        # Get time embedding
        t_emb = self.time_mlp(timestep)
        
        # Concatenate everything
        x = torch.cat([a_flat, context, t_emb], dim=1)
        
        # Predict noise
        out = self.mid_mlp(x)
        return out.reshape(B, self.traj_len, self.action_dim)
