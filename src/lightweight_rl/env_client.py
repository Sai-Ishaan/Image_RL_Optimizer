import torch
import torch.nn as nn
import numpy as np

class NatureCNN(nn.Module):
    ##Std NatureCNN architecture 
    # Expected input shape: (batch, channels, height, width) -> typically(B,4,84,84)
    def __init__(self, in_channels=4, num_actions=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))

class WeightMappingParser:
    ##Handles loading pre-trained weights and matching tensor shapes.
    def __init__(self, model: nn.Module):
        self.model = model

    def load_and_map_weights(self, checkpoint_path: str):
        print(f"Parsing checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        current_model_dict = self.model.state_dict()
        filtered_dict = {k:v for k, v in state_dict.items() if k in current_model_dict and v.shape == current_model_dict[k].shape}
        assert(len(filtered_dict) >0, "NO overlapping weights found. Check state dict naming schema.")

        current_model_dict.update(filtered_dict)
        self.model.load_state_dict(current_model_dict)
        print(f"Successfully mapped {len(filtered_dict)} layers into Teacher Network")
