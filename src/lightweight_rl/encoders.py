###MobileNetV3-Small and Shallow ResNet definitions
###Begin Phase 2
import torch
import torch.nn as nn 

class StudentEncoder(nn.Module):
    ##Light-weight spatial feature extractor for Atari Frames
    ##Aim: To reduce channel capacity (16X32X32) to minimize parameters and accelerate inference
    def __init__(self, in_channels: int=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16,32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.feature_dim = 32 * 7 *7 #1568 flattened features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x.view(x.size(0), -1)

class LightweightStudent(nn.Module):
    ### Compressed Student policy model (~200k parameters vs 1.7M in NatureCNN)
    def __init__(self, in_channels: int=4, num_actions: int=6, hidden_dim: int = 128):
        super().__init__()
        self.encoder = StudentEncoder(in_channels=in_channels)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)
    def get_features_and_logits(self, x: torch.Tensor):
        ## Returns both latent feature representation and action logits for loss matching...
        features = self.encoder(x)
        logits = self.head(features)
        return features, logits