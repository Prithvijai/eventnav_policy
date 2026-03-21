import torch
from src.models.eGoNavi import eGoNavi
import os

def check_weights(checkpoint_path):
    print(f"Loading SOTA weights from: {checkpoint_path}")
    
    # Load state dict
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Check if checkpoint is a dict or just state_dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    print("\n--- SOTA Checkpoint Structure ---")
    sota_keys = list(state_dict.keys())
    
    # Group keys by component to understand structure
    encoder_keys = [k for k in sota_keys if "vision_encoder" in k]
    transformer_keys = [k for k in sota_keys if "transformer" in k]
    action_keys = [k for k in sota_keys if "action_head" in k or "noise_pred_net" in k or "dist_pred_net" in k]
    
    print(f"Encoder keys: {len(encoder_keys)}")
    print(f"Transformer keys: {len(transformer_keys)}")
    print(f"Action Head keys: {len(action_keys)}")
    
    if transformer_keys:
        print("\nTransformer Example Key:")
        print(f"  {transformer_keys[0]} - {state_dict[transformer_keys[0]].shape}")
        
    if action_keys:
        print("\nAction Head Example Key:")
        print(f"  {action_keys[0]} - {state_dict[action_keys[0]].shape}")

    # Define our current model architecture
    encoder_cfg = {"in_channels": 5, "out_dim": 512}
    vint_cfg = {"token_dim": 512, "num_tokens": 5, "num_layers": 4, "num_heads": 4, "ff_dim": 2048}
    diffusion_cfg = {"context_dim": 512, "action_dim": 3, "traj_len": 8, "hidden_dim": 256}
    
    model = eGoNavi(encoder_cfg, vint_cfg, diffusion_cfg)
    model_dict = model.state_dict()
    
    print("\n--- Potential Mapping ---")
    # Try to find common parts
    # Our transformer: transformer.transformer.layers.0.self_attn.in_proj_weight
    # SOTA transformer: transformer.layers.0.self_attn.in_proj_weight (likely)
    
    our_trans_key = "transformer.transformer.layers.0.self_attn.in_proj_weight"
    if our_trans_key in model_dict:
        print(f"Our Trans Key: {our_trans_key} ({model_dict[our_trans_key].shape})")
        # Look for partial match in SOTA
        match = [k for k in sota_keys if "layers.0.self_attn.in_proj_weight" in k]
        if match:
            print(f"SOTA Match: {match[0]} ({state_dict[match[0]].shape})")

if __name__ == "__main__":
    check_weights("/home/saitama/Documents/Event_based_Navigation/nomad.pth")
