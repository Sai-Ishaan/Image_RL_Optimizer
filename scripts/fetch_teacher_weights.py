import os
import torch
from lightweight_rl.env_client import NatureCNN
from huggingface_hub import hf_hub_download
from lightweight_rl.env_client import NatureCNN

def fetch_and_remap_weights():
    repo_id = "cleanrl/PongNoFrameskip-v4-dqn_atari-seed1"
    filename = "dqn_atari.cleanrl_model"
    # Load the downloaded file directly as a TorchScript module
    #checkpoint_path = r"C:\Users\Admin\.cache\huggingface\hub\models--cleanrl--PongNoFrameskip-v4-dqn_atari-seed1\snapshots\7abf8b0255fad67c925b9bf4b28cbaa5691228ee\agent.pt"    
    print("1. Downloading raw PyTorch state dict from HuggingFace Hub({filename})")
    try:
        # Load the entire model object
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
        ##DEBUG Print keys found in checkpoint'
        print(f"....Downloaded to: {checkpoint_path}")
    except Exception as e:
        print(f"[ERROR] Could not download: {e}")
        return
    print("\n2. Loading State Dictionary....")
    # Define your local NatureCNN
    external_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    local_model = NatureCNN(in_channels=4, num_actions=6)
    
    print("3. Mapping CleanRL network keys to NatureCNN...")
    # CleanRL dqn_atari.py typically maps to specific names. 
    mapping = {
        "network.0.weight": "conv.0.weight", "network.0.bias": "conv.0.bias",
        "network.2.weight": "conv.2.weight", "network.2.bias": "conv.2.bias",
        "network.4.weight": "conv.4.weight", "network.4.bias": "conv.4.bias",
        "network.7.weight": "fc.0.weight",   "network.7.bias": "fc.0.bias",
        "network.9.weight": "fc.2.weight",   "network.9.bias": "fc.2.bias",
    }
    # If standard mapping fails, we manually align the layers.
    new_state_dict = {}
    # Mapping logic: CleanRL often stores layers as 'network.X.weight'

    for ext_key, local_key in mapping.items():
        if ext_key in external_state_dict:
            new_state_dict[local_key] = external_state_dict[ext_key]
            print(f" -> Successfully mapped {ext_key} to {local_key} | Shape: {list(external_state_dict[ext_key].shape)}")
        else:
            print(f" -> [WARNING] {ext_key} not found in agent!")

    # Load and save
    local_model.load_state_dict(new_state_dict, strict=False)
    
    os.makedirs("checkpoints", exist_ok=True)
    target_path = os.path.join("checkpoints", "nature_cnn_pong.pt")
    torch.save(local_model.state_dict(), target_path)
    print("\n[SUCCESS] Checkpoint saved to checkpoints/nature_cnn_pong.pt")

if __name__ == "__main__":
    fetch_and_remap_weights()