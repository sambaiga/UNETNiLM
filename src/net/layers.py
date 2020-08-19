import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, seq_length=50):
        super().__init__()
        self.c = math.ceil(math.log(seq_length) / math.log(2.0))
        self.row_embed = nn.Embedding(seq_length, self.c)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
    
    def forward(self, x):
        size = x.size(-1)
        i = torch.arange(size, device=x.device)
        x_emb = self.row_embed(i)
        x_emb.unsqueeze(0).repeat(size, 1, 1)
        return x_emb.unsqueeze(0).repeat(x.shape[0], 1, 1).permute(0,2,1)

class Dropout1D(nn.Module):
    #https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/MC_dropout/model.py
    def __init__(self, dropout, mc=False):
        super().__init__()
        self.mc = mc
        self.dropout = dropout

    def forward(self, x):
        if not self.mc:
            if self.training:
                return F.dropout(x, p=self.dropout, training=True)
            else:
                return F.dropout(x, p=self.dropout, training=False)
        else:
            return F.dropout(x, p=self.dropout, training=True)
        
class Dropout2D(nn.Module):
    #https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/MC_dropout/model.py
    def __init__(self, dropout, mc=False):
        super().__init__()
        self.mc = mc
        self.dropout = dropout
        
    def forward(self, x):
        if not self.mc:
            if self.training:
                return F.dropout2d(x, p=self.dropout, training=True)
            else:
                return F.dropout2d(x, p=self.dropout, training=False)
        else:
            return F.dropout2d(x, p=self.dropout, training=True)


class MLPLayer(nn.Module):
    def __init__(self, in_size, 
                 hidden_arch=[128, 512, 1024], 
                 output_size=None, 
                 activation=nn.PReLU(),
                 batch_norm=True,
                 dropout=0.25,
                 mc = False):
        
        super(MLPLayer, self).__init__()
        self.in_size = in_size
        self.mc = mc
        self.output_size = output_size
        layer_sizes = [in_size] + [x for x in hidden_arch]
        self.layers = []
        self.dropout = Dropout1D(dropout, self.mc)
        
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            nn.utils.weight_norm(layer)
            nn.init.xavier_uniform_(layer.weight)
            self.layers.append(layer)
            self.layers.append(self.dropout)
            #self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            self.layers.append(activation)        
            
            
        if output_size is not None:
            layer = nn.Linear(layer_sizes[-1], output_size)
            self.layers.append(layer)
        self.mlp_network =  nn.Sequential(*self.layers)
        
    def forward(self, z):
        return self.mlp_network(z)
        
 

class Conv1D(nn.Module):
   
    def __init__(self, 
                 n_channels, 
                 n_kernels,
                 kernel_size=3, 
                 stride=1, 
                 last=False, 
                 dropout=0.25,
                 activation=nn.PReLU(),
                 mc=False):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(
            n_channels, n_kernels,
            kernel_size, stride, kernel_size//2
        )
        nn.utils.weight_norm(self.conv)    
        nn.init.xavier_uniform_(self.conv.weight)
        self.mc = mc
        self.dropout = Dropout1D(dropout, self.mc )
        if not last:
            self.net = nn.Sequential(
                self.conv,
                self.dropout,
                #nn.BatchNorm1d(n_kernels),
                activation
                )
        else:
            self.net = self.conv
        
    def forward(self, x):
        return self.net(x)  
        
class Deconv1D(nn.Module):
   
    def __init__(self, 
                 n_channels, 
                 n_kernels,
                 kernel_size=3, 
                 stride=1, 
                 last=False, 
                 activation=nn.PReLU(),
                 dropout=0.25,
                 mc=False):
        super(Deconv1D, self).__init__()
        self.mc = mc
        self.dropout = Dropout1D(dropout, mc)
        self.deconv = nn.ConvTranspose1d(
            n_channels, n_kernels,
            kernel_size, stride, kernel_size//2
        )
        if not last:
            self.net = nn.Sequential(
                self.deconv,
                self.dropout,
                #nn.BatchNorm1d(n_kernels),
                activation
                
                
            )
        else:
            self.net = self.deconv
        nn.utils.weight_norm(self.deconv)        
        nn.init.xavier_uniform_(self.deconv.weight)

    def forward(self, x):
        return self.net(x)          

class ConvBlock(nn.Module):
    def __init__(self, 
                 n_channels=1, 
                 n_kernels=16, 
                 n_layers=3, 
                 seq_size=50,
                 dropout=0.1,
                 pos_enconding=True,
                 mc=False):
        super().__init__()
        self.pos_enconding= pos_enconding
        if pos_enconding:
            self.positional_emb = PositionEmbeddingLearned(seq_length=seq_size)
            n_channels = n_channels + self.positional_emb.c
        
        self.feat_size = (seq_size-1) // 2**n_layers +1
        self.feat_dim = self.feat_size * n_kernels
        
        self.conv_stack = nn.Sequential(
            *([Conv1D(n_channels, n_kernels // 2**(n_layers-1), dropout=dropout, mc=mc)] +
              [Conv1D(n_kernels//2**(n_layers-l),
                      n_kernels//2**(n_layers-l-1), dropout=dropout, mc=mc)
               for l in range(1, n_layers-1)] +
              [Conv1D(n_kernels // 2, n_kernels, dropout=dropout, mc=mc)])
        )
       
    def forward(self, x):
        assert len(x.size())==3
        if self.pos_enconding:
            positional_input = self.positional_emb(x)
            x = torch.cat((x, positional_input), 1)
        feats = self.conv_stack(x)
        return feats

class DeconvBlock(nn.Module):
       
    def __init__(self, in_ch: int, out_ch: int, dropout=0.1, mc=False):
        super().__init__()
        self.upsample = Deconv1D(in_ch, in_ch // 2, dropout=dropout)
        self.conv = Conv1D(in_ch, out_ch, dropout=dropout,  mc=mc)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        # Pad x1 to the size of x2
        diff = x2.shape[2] - x1.shape[2]
        x1 = F.pad(x1, [diff// 2, diff - diff // 2])
        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        

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