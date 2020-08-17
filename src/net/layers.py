import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MLPLayer(nn.Module):
    def __init__(self, in_size, 
                 hidden_arch=[128, 512, 1024], 
                 output_size=None, 
                 activation=nn.PReLU(),
                 batch_norm=True):
        
        super(MLPLayer, self).__init__()
        self.in_size = in_size
        self.output_size = output_size
        layer_sizes = [in_size] + [x for x in hidden_arch]
        self.layers = []
        
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
                    
            if batch_norm and i!=0:
                bn = nn.BatchNorm1d(layer_sizes[i+1])
                self.layers.append(bn)
     
            self.layers.append(activation)
           
        if output_size is not None:
            layer = nn.Linear(layer_sizes[-1], output_size)
            self.layers.append(layer)
            self.layers.append(activation)
            
        self.init_weights()
        self.mlp_network =  nn.Sequential(*self.layers)
        
    def forward(self, z):
        return self.mlp_network(z)
        
    def init_weights(self):
        for layer in self.layers:
            try:
                if isinstance(layer, nn.Linear):
                    nn.utils.weight_norm(layer)
                    init.xavier_uniform_(layer.weight)
            except: pass


class Conv1D(nn.Module):
   
    def __init__(self, 
                 n_channels, 
                 n_kernels,
                 kernel_size=3, 
                 stride=2, 
                 padding=1, 
                 last=False, 
                 activation=nn.PReLU()):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(
            n_channels, n_kernels,
            kernel_size, stride, padding
        )
        if not last:
            self.net = nn.Sequential(
                self.conv,
                nn.BatchNorm1d(n_kernels),
                activation)
        else:
            self.net = self.conv
        nn.utils.weight_norm(self.conv)    
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.net(x)  
        
class Deconv1D(nn.Module):
   
    def __init__(self, 
                 n_channels, 
                 n_kernels,
                 kernel_size=3, 
                 stride=2, 
                 padding=1, 
                 last=False, 
                 activation=nn.PReLU()):
        super(Deconv1D, self).__init__()
        self.deconv = nn.ConvTranspose1d(
            n_channels, n_kernels,
            kernel_size, stride, padding
        )
        if not last:
            self.net = nn.Sequential(
                self.deconv,
                nn.BatchNorm1d(n_kernels),
                activation
            )
        else:
            self.net = self.deconv
        nn.init.xavier_uniform_(self.deconv.weight)

    def forward(self, x):
        return self.net(x)          

class Encoder(nn.Module):
    def __init__(self, 
                 n_channels=10, 
                 n_kernels=16, 
                 n_layers=3, 
                 seq_size=50):
        super(Encoder, self).__init__()
        self.feat_size = (seq_size-1) // 2**n_layers +1
        self.feat_dim = self.feat_size * n_kernels
        self.conv_stack = nn.Sequential(
            *([Conv1D(n_channels, n_kernels // 2**(n_layers-1))] +
              [Conv1D(n_kernels//2**(n_layers-l),
                         n_kernels//2**(n_layers-l-1))
               for l in range(1, n_layers-1)] +
              [Conv1D(n_kernels // 2, n_kernels, last=True)])
        )
    def forward(self, x):
        assert len(x.size())==3
        feats = self.conv_stack(x)
        return feats
        

