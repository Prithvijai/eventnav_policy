import torch
import torch.nn as nn
import math
from .diffusionpolicy import ConditionalUnet1D, SinusoidalPosEmb
import torch.nn.functional as F

    
def squared_noise_sheduler(t, s=0.008):
    level = torch.Tensor(range(t+1))
    
    alpha_noise = torch.cos( (((level/ t) + s) / (1 + s)) * (torch.pi/2)) ** 2
    alpha_noise_norm = alpha_noise / alpha_noise[0]
    return alpha_noise_norm

def compute_ddpm_loss(model, features, gt_actions):
    """
    features:   (B, 512) transformer output
    gt_actions: (B, 8, 3) normalized ground truth actions
    """
    B = gt_actions.shape[0]
    device = gt_actions.device

    t = torch.randint(
        0,
        model.action_head.step,
        (B,),
        device=device
    ).long()

    noisy_actions, noise = model.action_head.add_noise(gt_actions, t)

    pred_noise = model.action_head(noisy_actions, t, features)
    loss = F.mse_loss(pred_noise, noise)

    return loss

class DiffusionPolicyAction(nn.Module):
    def __init__ (self, context_dim=512, action_dim=3, traj_len=8, step=10, hidden_dim=256):
        super().__init__()

        self.register_buffer('alpha_noise_norm', squared_noise_sheduler(step))

        self.step = step

        self.proj_context = nn.Linear(context_dim, hidden_dim)

        self.unet_block = ConditionalUnet1D(input_dim=action_dim,
                global_cond_dim=hidden_dim,
                down_dims= [64, 128, 256],
                cond_predict_scale=False)
        
    def add_noise(self, action, t):
        alpha_t = self.alpha_noise_norm[t]
        sqrt_alpha_t = torch.sqrt(alpha_t).view(-1, 1, 1)
        sqrt_one_miuns_alpha_t = torch.sqrt(1 - alpha_t).view(-1, 1, 1)

        # print(sqrt_alpha_t, sqrt_one_miuns_alpha_t, action)
        noise = torch.randn_like(action)
        noise_action = sqrt_alpha_t * action + sqrt_one_miuns_alpha_t * noise

        return noise_action, noise
    
    def forward(self, action, timestep, context):
        return self.unet_block(action, timestep, global_cond=self.proj_context(context))
    

    @torch.no_grad()
    def sample(self, context, device):
        s = 1e-10
        B = context.shape[0] 
        action = torch.randn(B, 8, 3).to(device)

        for t in reversed(range(self.step)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self.forward(action, t_batch, context)

            alpha_t = self.alpha_noise_norm[t].to(device)
            alpha_t1  = self.alpha_noise_norm[t - 1].to(device) if t > 0 else torch.tensor(1.0, device=device)
            
            sqrt_alpha_t = torch.sqrt(alpha_t).view(-1, 1, 1)
            sqrt_one_miuns_alpha_t = torch.sqrt(1 - alpha_t + s).view(-1, 1, 1)

            sqrt_alpha_t1 = torch.sqrt(alpha_t1).view(-1, 1, 1)
            curr_alpha_t = alpha_t / alpha_t1
            beta_t = 1 - curr_alpha_t

            pred_action = (action - sqrt_one_miuns_alpha_t * pred_noise) / sqrt_alpha_t
            pred_action = pred_action.clamp(-1, 1)

            mu_t1 = (sqrt_alpha_t1 * beta_t * pred_action) / (1 - alpha_t + s) + (torch.sqrt(curr_alpha_t) * (1 - alpha_t1 + s) * action) / (1 - alpha_t + s)

            if t > 0:
                noise = torch.randn_like(action)
                sigma = torch.sqrt((1 - alpha_t1 + s) / (1 - alpha_t + s) * beta_t)
                action = mu_t1 + sigma * noise
            else:
                action = mu_t1

        return action


     
        
