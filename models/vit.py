import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, trunc_normal_


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = (dim // heads) ** -0.5
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
    
    def forward(self, x):
        # Fill this
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, expand_ratio=2):
        super().__init__()
        hidden_dim = dim * expand_ratio
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class VisionTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, expand_ratio=2, droppath=0.0):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_mixer = Attention(dim, heads)
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mixer = FeedForward(dim, expand_ratio)
        self.droppath = DropPath(droppath)
    
    def forward(self, x):
        x = x + self.droppath(self.token_mixer(self.token_norm(x)))
        x = x + self.droppath(self.channel_mixer(self.channel_norm(x)))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, num_patches):
        super().__init__()
        # Fill this

    def forward(self, x):
        # Fill this
        return x


class VisionTransformer(nn.Module):
    def __init__(self, 
                 blocks=12, 
                 dim=384, 
                 num_classes=10, 
                 image_size=32,
                 patch_size=4, 
                 num_heads=8, 
                 expand_ratio=2, 
                 droppath=0.0, 
                 dropout=0.0,
        ):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2 
        self.patch_embed = PatchEmbedding(3, dim, patch_size, num_patches)

        self.layers = nn.Sequential(*[
            VisionTransformerBlock(dim, num_heads, expand_ratio, droppath)  
            for _ in range(blocks)]
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layers(x)
        x = self.dropout(self.norm(x.mean([1])))
        x = self.head(x)
        return x
