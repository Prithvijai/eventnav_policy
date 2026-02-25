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
        # Body frame to world frame (assuming standard X-forward, Y-left)
        x += dx * np.cos(theta) - dy * np.sin(theta)
        y += dx * np.sin(theta) + dy * np.cos(theta)
        theta += dtheta
        traj[i+1] = [x, y, theta]
    return traj

def plot_full_trajectory(gt_actions, pred_actions, file_name, save_path=None):
    """
    Plots the full sequence of actions and the resulting integrated trajectory.
    gt_actions, pred_actions: (N, 3)
    """
    gt_traj = compute_trajectory(gt_actions)
    pred_traj = compute_trajectory(pred_actions)
    
    fig = plt.figure(figsize=(18, 10))
    
    # Top-down view (XY Plane)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', label='Ground Truth', alpha=0.8, linewidth=2)
    ax1.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', label='Predicted', alpha=0.8, linewidth=2)
    
    # Start and end points
    ax1.scatter(gt_traj[0, 0], gt_traj[0, 1], c='blue', marker='o', s=100, label='Start', zorder=5)
    ax1.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c='green', marker='X', s=150, label='GT End', zorder=5)
    ax1.scatter(pred_traj[-1, 0], pred_traj[-1, 1], c='red', marker='X', s=150, label='Pred End', zorder=5)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top-down Trajectory (XY Plane)')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True)
    
    #dx, dy, dtheta plots
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
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate eGoNavi DDPM Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--data_dir", type=str, default="../h5_test", help="Directory containing H5 test files")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize (if not --full_traj)")
    parser.add_argument("--full_traj", action="store_true", help="Evaluate and plot the entire first H5 file sequentially")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Model Configurations
    encoder_cfg = {"in_channels": 5, "out_dim": 512}
    vint_cfg = {"token_dim": 512, "num_tokens": 6, "num_layers": 1, "num_heads": 4, "ff_dim": 2048}
    diffusion_cfg = {"context_dim": 512, "action_dim": 3, "traj_len": 8, "hidden_dim": 256}

    model = eGoNavi(encoder_cfg, vint_cfg, diffusion_cfg).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()

    data_dir = Path(args.data_dir)
    h5_files = sorted(list(data_dir.glob("*.h5")))
    if not h5_files:
        print(f"No H5 files found in {args.data_dir}")
        return

    # Process the first file
    dataset = EnavippH5Dataset(h5_files[0], load_rgb=False)
    stats = dataset.get_stats()
    action_mean = stats['action_mean'].to(device)
    action_std = stats['action_std'].to(device)

    if args.full_traj:
        # Sequential loading for full trajectory
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_enavi)
        print(f"Processing full trajectory sequentially: {h5_files[0].name}...")
        
        gt_list = []
        pred_list = []
        
        history = []
        with torch.no_grad():
            for batch in tqdm(loader):
                voxel = batch["voxel"].to(device) # (1, 5, H, W)
                gt_action_norm = batch["action"].to(device) # (1, 8, 3)
                
                history.append(voxel)
                if len(history) > 5:
                    history.pop(0)
                
                if len(history) < 5:
                    continue
                
                input_voxels = torch.stack(history, dim=1)
                goal_voxel = voxel
                
                pred_action_norm = model.sample(input_voxels, goal_voxel, device) # (1, 8, 3)
                
                gt_unnorm = (gt_action_norm * action_std + action_mean).cpu().numpy()[0]
                pred_unnorm = (pred_action_norm * action_std + action_mean).cpu().numpy()[0]
                
                gt_list.append(gt_unnorm[0])
                pred_list.append(pred_unnorm[0])
        plot_full_trajectory(np.array(gt_list), np.array(pred_list), h5_files[0].name)
        
    else:
        # Random sample visualization
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_enavi)
        print(f"Visualizing {args.num_samples} random samples...")
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.num_samples: break
                voxel = batch['voxel'].to(device)
                gt_action_norm = batch['action'].to(device)
                if voxel.dim() == 4:
                    voxel = voxel.unsqueeze(1).repeat(1, 5, 1, 1, 1)
                goal_voxel = voxel[:, -1]
                pred_action_norm = model.sample(voxel, goal_voxel, device)
                gt_action = (gt_action_norm * action_std + action_mean).cpu().numpy()[0]
                pred_action = (pred_action_norm * action_std + action_mean).cpu().numpy()[0]
                
                from eval import plot_actions # self-import or local helper
                plot_actions(gt_action, pred_action, i)

def plot_actions(gt_action, pred_action, idx, save_path=None):
    """
    Visualizes an 8-step action chunk as a 2D trajectory.
    """
    gt_traj = compute_trajectory(gt_action)
    pred_traj = compute_trajectory(pred_action)
    
    fig = plt.figure(figsize=(18, 6))
    
    # 2D Plot
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-o', label='Ground Truth')
    ax1.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--s', label='Predicted')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('8-Step Chunk Trajectory')
    ax1.axis('equal')
    ax1.legend()
    ax1.grid(True)
    
    # Rotation plot
    ax2 = fig.add_subplot(1, 2, 2)
    steps = np.arange(8)
    ax2.plot(steps, gt_action[:, 2], 'g-o', label='GT dTheta')
    ax2.plot(steps, pred_action[:, 2], 'r--s', label='Pred dTheta')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('dTheta (rad)')
    ax2.set_title('Rotation Change per Step')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(f"Action Chunk Visualization - Sample {idx}")
    plt.tight_layout()
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show()

if __name__ == "__main__":
    evaluate()
