import torch
from einops import rearrange
from torch import nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim=50, heads = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim,dim*3*heads, bias = False)
        self.to_out = nn.Linear(dim*heads, dim)
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = torch.split(qkv, qkv.size(-1)//3, dim=-1)
        dots = torch.matmul(q.view(b, n, self.heads, -1), k.view(b, n, -1, self.heads)) *self.scale 
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, q.view(b, n, self.heads, -1))     # (bs, n_heads, q_length, dim_per_head)
        out = rearrange(out, 'b c h d -> b c (h d)')
        out =  self.to_out(out)
        return out
    
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ])
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)    
    
class NILMTransformer(nn.Module):
    def __init__(self, seq_len=50, patch_size=5, num_classes=12, 
                 dim=50, depth=4, heads=8, 
                 mlp_dim=128, channels = 1, n_quantiles=3):
        super().__init__()
        assert seq_len % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (seq_len // patch_size) 
        patch_dim = channels * patch_size 
        self.patch_size = patch_size
        self.n_quantiles = n_quantiles
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )
        self.mlp_classifier = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes*2))
        
        self.mlp_regress = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes*n_quantiles))
        
    def forward(self, x):
        p = self.patch_size
        B = x.size(0)
        x  = rearrange(x.unsqueeze(-1), 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = 1)
        x  = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        
        states_logits   = self.mlp_classifier(x).reshape(B, 2, -1)
        power_logits    = self.mlp_regress(x)
        if self.n_quantiles>1:
            power_logits = power_logits.reshape(B, self.n_quantiles, -1)
        return states_logits, power_logits    