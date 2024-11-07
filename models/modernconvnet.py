import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath


class FeedForward(nn.Module):
    def __init__(self, dim, expand_ratio):
        super().__init__()
        hidden_dim = dim * expand_ratio
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.w3 = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class ModernConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, expand_ratio=2, droppath=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # Fill this

    def forward(self, x):
        # Fill this
        x = x + self.droppath(h)
        return x


class ModernConvNet(nn.Module):
    def __init__(self, 
                 blocks, 
                 dims, 
                 kernel_size=3, 
                 expand_ratio=2, 
                 droppath=0.0, 
                 dropout=0.0, 
                 num_classes=10):
        
        super().__init__()

        self.downsamples = nn.ModuleList()
        for i in range(4):
            if i == 0:
                self.downsamples.append(Downsample(3, dims[0], 3, 1))
            else:
                self.downsamples.append(Downsample(dims[i-1], dims[i], 2, 2))
        
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(nn.Sequential(*[
                ModernConvBlock(dims[i], kernel_size, expand_ratio, droppath) 
                for _ in range(blocks[i])
            ]))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        # Fill this
        return x