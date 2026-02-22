import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EventVoxelEncoder(nn.Module):
    def __init__(self, in_channels=5, out_dim=512):
        super().__init__()
        
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
        # print(self.encoder)
        first_conv = self.encoder._conv_stem
        self.encoder._conv_stem = nn.Conv2d(
            in_channels, first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        
        in_features = self.encoder._fc.in_features
        self.encoder._fc = nn.Linear(in_features, out_dim)

    def forward(self, x):
        return self.encoder(x)
    


# event_encoder = EventVoxelEncoder()