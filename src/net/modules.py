import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv1D, AttentionLayer, ConvBlock, MLPLayer, DeconvBlock

class CNN1DModel(nn.Module):
    def __init__(self, in_size=1, 
                 num_classes=5,
                 d_model=128,
                 dropout=0.01, 
                 seq_size=50,  
                 n_layers=5, 
                 n_quantiles=1, 
                 pool_filter=16,
                 mc=False,
                 pos_enconding=True):
        super().__init__()
        self.enc_net = ConvBlock(n_channels=in_size, 
                                 n_kernels=d_model, 
                                 n_layers=n_layers, 
                                 seq_size=seq_size, 
                                 dropout=dropout,
                                 mc=mc,
                                 pos_enconding=pos_enconding)
        self.final_conv_layer = nn.Sequential(
                                Conv1D(d_model, num_classes, kernel_size=1, dropout=dropout, mc=mc),
                                AttentionLayer(num_classes, num_classes, num_classes)
                                )
        self.mlp_layer = MLPLayer(in_size=seq_size*num_classes, hidden_arch=[512,1024], dropout=dropout, mc=mc)
        self.n_quantiles = n_quantiles
        self.fc_out_state  = nn.Linear(1024, num_classes*2)
        self.fc_out_power  = nn.Linear(1024, num_classes*n_quantiles)
        nn.init.xavier_normal_(self.fc_out_state.weight)
        nn.init.xavier_normal_(self.fc_out_power.weight)
        self.fc_out_state.bias.data.fill_(0)
        self.fc_out_power.bias.data.fill_(0)
        
   

    def forward(self, x):
        x = x.permute(0,2,1)
        B = x.size(0)
        conv_out = self.enc_net(x)
        conv_out = self.final_conv_layer(conv_out)
        mlp_out  = self.mlp_layer(conv_out.reshape(B, -1))
        states_logits   = self.fc_out_state(mlp_out).reshape(B, 2, -1)
        power_logits    = self.fc_out_power(mlp_out)
        if self.n_quantiles>1:
            power_logits = power_logits.reshape(B, self.n_quantiles, -1)
        return  states_logits,  power_logits    
        


class UNETNiLM(nn.Module):
       
    def __init__(
            self, num_classes: int = 5,
            num_layers: int = 5,
            features_start: int = 16,
            n_channels: int =1,
            seq_size=50,
            n_quantiles=1,
            dropout=0.1,
            mc = False,
            pos_enconding=True):
        super().__init__()
        self.num_layers = num_layers
        self.n_quantiles = n_quantiles
        layers = [ConvBlock(n_channels, features_start, 
                            n_layers=num_layers//2, 
                            seq_size=seq_size, 
                            dropout=dropout, 
                            mc=mc,
                            pos_enconding=pos_enconding)]
        feats = features_start
        for i in range(num_layers - 1):
            layers.append(Conv1D(feats, feats * 2, dropout=dropout, mc=mc))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(DeconvBlock(feats, feats // 2, dropout=dropout, mc=mc))
            feats //= 2

        final_conv = nn.Sequential(Conv1D(feats+n_channels, num_classes, kernel_size=1, dropout=dropout, mc=mc),
                             AttentionLayer(num_classes, num_classes, num_classes))
        
        layers.append(final_conv)
        self.layers = nn.ModuleList(layers)
        self.mlp_layer = MLPLayer(in_size=seq_size*num_classes, hidden_arch=[512,1024], dropout=dropout, mc=mc)
        self.fc_out_state  = nn.Linear(1024, num_classes*2)
        self.fc_out_power  = nn.Linear(1024, num_classes*n_quantiles)
        nn.init.xavier_normal_(self.fc_out_state.weight)
        nn.init.xavier_normal_(self.fc_out_power.weight)
        self.fc_out_state.bias.data.fill_(0)
        self.fc_out_power.bias.data.fill_(0)

    def forward(self, x):
        x = x.permute(0,2,1)
        B = x.size(0)
        xi = [self.layers[0](x)]
        #print(xi[-1].shape)
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
            #print(xi[-1].shape)
        
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])  
            #print(xi[-1].shape)
            
        xi[-1] = torch.cat((x, xi[-1]), 1)
        conv_out = self.layers[-1](xi[-1])
        #print(conv_out.shape)
        mlp_out  = self.mlp_layer(conv_out.flatten(1,2))
        states_logits   = self.fc_out_state(mlp_out).reshape(B, 2, -1)
        power_logits    = self.fc_out_power(mlp_out)
        if self.n_quantiles>1:
            power_logits = power_logits.reshape(B, self.n_quantiles, -1)
       
        return  states_logits,  power_logits    
        
      
    
