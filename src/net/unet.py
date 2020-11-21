import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import Conv1D, Deconv1D

class Up(nn.Module):
   
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = Deconv1D(in_ch, in_ch // 2)
        self.conv = Conv1D(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        # Pad x1 to the size of x2
        diff = x2.shape[2] - x1.shape[2]
        x1 = F.pad(x1, [diff// 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetCNN1D(nn.Module):
   
    def __init__(
            self, 
            num_layers: int = 5,
            features_start: int = 8,
            n_channels: int =1
    ):
        super().__init__()
        self.num_layers = num_layers
        layers = [Conv1D(n_channels, features_start)]
        feats = features_start
        for i in range(num_layers - 1):
            layers.append(Conv1D(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2))
            feats //= 2
        
       
        conv = nn.Conv1d(feats, num_classes, kernel_size=1)
        conv = nn.utils.weight_norm(conv)
        nn.init.xavier_uniform_(conv.weight)
        layers.append(conv)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]
        
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
            
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
            
        out = self.layers[-1](xi[-1])
        return out

