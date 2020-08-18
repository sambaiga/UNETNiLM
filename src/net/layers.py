import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class BNN1D(nn.Module):
    #https://github.com/hmi88/mcbn/blob/master/MCBN_src/model/mcbn.py
    def __init__(self, n_in_feat):
        super().__init__()
        self.n_in_feat = n_in_feat
        self.bn1d = nn.BatchNorm1d(n_in_feat, eps=1e-5)
        self.bn1d.weight.data.fill_(1)
        self.bn1d.bias.data.zero_()
        

    def forward(self, x):
        if self.training:
            y = self.bn1d(x)
        else:
            x_size = x.size()
            half_batch = x_size[0]//2
            self.train()
            self.bn1d.running_mean = torch.zeros(self.n_in_feat).to(x.device)
            self.bn1d.running_var = torch.ones(self.n_in_feat).to(x.device)
            _ = self.bn1d(x[half_batch:])
            self.eval()
            y = self.bn1d(x)
        return y

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
                bn = BNN1D(layer_sizes[i+1])
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
                BNN1D(n_kernels),
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
                BNN1D(n_kernels),
                activation
            )
        else:
            self.net = self.deconv
        nn.utils.weight_norm(self.deconv)        
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
        

class AttentionLayer(nn.Module):
    #https://fleuret.org/git-extract/pytorch/attentiontoy1d.py
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size = 1, bias = False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size = 1, bias = False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size = 1, bias = False)
        nn.utils.weight_norm(self.conv_Q)    
        nn.init.xavier_uniform_(self.conv_Q.weight)
        nn.utils.weight_norm(self.conv_K)    
        nn.init.xavier_uniform_(self.conv_K.weight)
        nn.utils.weight_norm(self.conv_V)    
        nn.init.xavier_uniform_(self.conv_V.weight)

    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def __repr__(self):
        return self._get_name() + \
            '(in_channels={}, out_channels={}, key_channels={})'.format(
                self.conv_Q.in_channels,
                self.conv_V.out_channels,
                self.conv_K.out_channels
            )

    def attention(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        return Q.permute(0, 2, 1).matmul(K).softmax(2)