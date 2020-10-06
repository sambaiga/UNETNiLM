import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MLPLayer,  Encoder
from .unet import UNetBaseline
import torch.nn.init as init
import math


class CNN1DModel(nn.Module):
    def __init__(self, in_size=1, 
                 output_size=5,
                 d_model=128,
                 dropout=0.01, 
                 seq_len=9,  
                 n_layers=5, 
                 n_quantiles=3, 
                 pool_filter=16):
        super(CNN1DModel, self).__init__()
        self.enc_net = Encoder(n_channels=in_size, n_kernels=d_model, n_layers=n_layers, seq_size=seq_len)
        self.pool_filter = pool_filter
        self.mlp_layer = MLPLayer(in_size=d_model*pool_filter, hidden_arch=[1024], output_size=None)
        self.dropout = nn.Dropout(dropout)
        self.pool_filter = pool_filter
        self.n_quantiles = n_quantiles
        
        self.fc_out_state  = nn.Linear(1024, output_size*2)
        self.fc_out_power  = nn.Linear(1024, output_size*n_quantiles)
        nn.init.xavier_normal_(self.fc_out_state.weight)
        nn.init.xavier_normal_(self.fc_out_power.weight)
        self.fc_out_state.bias.data.fill_(0)
        self.fc_out_power.bias.data.fill_(0)
        
   

    def forward(self, x):
        x = x.permute(0,2,1)
        B = x.size(0)
        conv_out = self.dropout(self.enc_net(x))
        conv_out = F.adaptive_avg_pool1d(conv_out, self.pool_filter).reshape(x.size(0), -1)
        mlp_out  = self.dropout(self.mlp_layer(conv_out))
        states_logits   = self.fc_out_state(mlp_out).reshape(B, 2, -1)
        power_logits    = self.fc_out_power(mlp_out)
        if self.n_quantiles>1:
            power_logits = power_logits.reshape(B, self.n_quantiles, -1)
        return  states_logits,  power_logits    
        


class UNETNiLM(nn.Module):
    def __init__(self, in_size=1, 
                 output_size=5,
                 d_model=128, 
                 dropout=0.1, 
                 seq_len=99, 
                 features_start=16,  
                 n_layers=4, 
                 n_quantiles=3, 
                 pool_filter=16):
        super().__init__()
        self.unet = UNetBaseline(num_classes=output_size, num_layers=n_layers, features_start=features_start, n_channels=in_size)
        self.conv_layer = Encoder(n_channels=output_size, n_kernels=d_model, n_layers=n_layers//2, seq_size=seq_len)
        self.mlp_layer = MLPLayer(in_size=d_model*pool_filter, hidden_arch=[1024], output_size=None)
        self.dropout = nn.Dropout(dropout)
        self.pool_filter = pool_filter
        self.n_quantiles = n_quantiles
        
        self.fc_out_state  = nn.Linear(1024, output_size*2)
        self.fc_out_power  = nn.Linear(1024, output_size*n_quantiles)
        nn.init.xavier_normal_(self.fc_out_state.weight)
        nn.init.xavier_normal_(self.fc_out_power.weight)
        self.fc_out_state.bias.data.fill_(0)
        self.fc_out_power.bias.data.fill_(0)
        
   
    def forward(self, x):
        B = x.size(0)
        x = x.permute(0,2,1)
        unet_out = self.dropout(self.unet(x))
        conv_out = self.conv_layer(unet_out)
        conv_out = self.dropout(F.adaptive_avg_pool1d(conv_out, self.pool_filter).reshape(x.size(0), -1))
        mlp_out  = self.dropout(self.mlp_layer(conv_out))
        states_logits   = self.fc_out_state(mlp_out).reshape(B, 2, -1)
        power_logits    = self.fc_out_power(mlp_out)
        if self.n_quantiles>1:
            power_logits = power_logits.reshape(B, self.n_quantiles, -1)
       
        return  states_logits,  power_logits    
        

