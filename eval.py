import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.dataloading import EnavippH5Dataset
from src.models.eGoNavi import eGoNavi
from src.data.preprocessing import collate_enavi
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm

def compute_trajectory(actions):
    """
    Computes (x, y, theta) trajectory from relative actions (dx, dy, dtheta).
    actions: (N, 3) where columns are [dx, dy, dtheta]
    Returns: (N+1, 3) where columns are [x, y, theta]
    """
    traj = np.zeros((actions.shape[0] + 1, 3))
    x, y, theta = 0.0, 0.0, 0.0
    for i in range(actions.shape[0]):
        dx, dy, dtheta = actions[i]
        # Body frame to world frame
        x += dx * np.cos(theta) - dy * np.sin(theta)
        y += dx * np.sin(theta) + dy * np.cos(theta)
        theta += dtheta
        traj[i+1] = [x, y, theta]
    return traj

def denormalize_action(action_norm, action_mean_xy, action_std_xy):
    """
    Denormalizes action: [dx, dy] using mean/std, [dtheta] using pi.
    """
    # 1. Denormalize XY (first 2 columns)
    xy_denorm = action_norm[..., :2] * action_std_xy + action_mean_xy
    # 2. Denormalize Theta (last column) using pi
    theta_denorm = action_norm[..., 2:3] * torch.pi
    return torch.cat([xy_denorm, theta_denorm], dim=-1)

def plot_full_trajectory(gt_actions, pred_actions, file_name, save_path=None):
    """
    Plots the full sequence of actions and the resulting integrated trajectory.
    """
    gt_traj = compute_trajectory(gt_actions)
    pred_traj = compute_trajectory(pred_actions)
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Top-down view (XY Plane)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', label='Ground Truth', alpha=0.8, linewidth=2)
    ax1.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', label='Predicted', alpha=0.8, linewidth=2)
    ax1.scatter(gt_traj[0, 0], gt_traj[0, 1], c='blue', marker='o', s=100, label='Start')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top-down Trajectory (XY Plane)')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True)
    
    # 2. Action Components (dx, dy, dtheta)
    steps = np.arange(len(gt_actions))
    labels = ['dx (Forward)', 'dy (Lateral)', 'dtheta (Rotation)']
    for i in range(3):
        ax = fig.add_subplot(3, 2, 2 * (i + 1))
        ax.plot(steps, gt_actions[:, i], 'g-', alpha=0.6, label='GT' if i==0 else "")
        ax.plot(steps, pred_actions[:, i], 'r--', alpha=0.6, label='Pred' if i==0 else "")
        ax.set_ylabel(labels[i])
        ax.grid(True)
        if i == 0: ax.legend()
        if i == 2: ax.set_xlabel('Time Step')

    plt.suptitle(f"Full Trajectory Evaluation: {file_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_actions(gt_action, pred_action, idx, save_path=None):
    """
    Visualizes an 8-step action chunk.
    """
    gt_traj = compute_trajectory(gt_action)
    pred_traj = compute_trajectory(pred_action)
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-o', label='GT')
    ax1.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--s', label='Pred')
    ax1.set_title(f"Sample {idx} - XY")
    ax1.legend(); ax1.axis('equal'); ax1.grid(True)
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(gt_action[:, 2], 'g-o', label='GT dTheta')
    ax2.plot(pred_action[:, 2], 'r--s', label='Pred dTheta')
    ax2.set_title("Rotation")
    ax2.legend(); ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate eGoNavi DDPM Model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/media/saitama/Games1/Documents_ubuntu/eGoNavi_bag_test/")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--full_traj", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Model Configuration
    encoder_cfg = {"in_channels": 5, "out_dim": 512}
    vint_cfg = {"token_dim": 512, "num_tokens": 6, "num_layers": 4, "num_heads": 4, "ff_dim": 2048}
    diffusion_cfg = {"context_dim": 512, "action_dim": 3, "traj_len": 8, "hidden_dim": 256}

    model = eGoNavi(encoder_cfg, vint_cfg, diffusion_cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)
    model.eval()

    h5_files = sorted(list(Path(args.data_dir).glob("*.h5")))
    dataset = EnavippH5Dataset(h5_files[0], load_rgb=False)
    stats = dataset.get_stats()
    action_mean_xy = stats['action_mean_xy'].to(device)
    action_std_xy = stats['action_std_xy'].to(device)

    if args.full_traj:
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_enavi)
        gt_list, pred_list = [], []
        
        with torch.no_grad():
            for batch in tqdm(loader):
                voxels = batch["voxel"].to(device)
                goal_voxel = batch["goal_voxel"].to(device)
                
                pred_action_norm = model.sample(voxels, goal_voxel, device)

                gt_unnorm = denormalize_action(batch["action"].to(device), action_mean_xy, action_std_xy).cpu().numpy()[0]
                pred_unnorm = denormalize_action(pred_action_norm, action_mean_xy, action_std_xy).cpu().numpy()[0]

                gt_list.append(gt_unnorm[0])
                pred_list.append(pred_unnorm[0])

        plot_full_trajectory(np.array(gt_list), np.array(pred_list), h5_files[0].name)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_enavi)
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.num_samples: break
                voxels = batch['voxel'].to(device)
                goal_voxel = batch['goal_voxel'].to(device)
                
                pred_action_norm = model.sample(voxels, goal_voxel, device)
                
                gt_action = denormalize_action(batch['action'].to(device), action_mean_xy, action_std_xy).cpu().numpy()[0]
                pred_action = denormalize_action(pred_action_norm, action_mean_xy, action_std_xy).cpu().numpy()[0]
                
                plot_actions(gt_action, pred_action, i)

if __name__ == "__main__":
    evaluate()
