import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim, 
            out_channels=out_dim, 
            kernel_size=kernel_size, 
            stride=stride,
        )
        self.norm = nn.BatchNorm2d(num_features=out_dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, droppath=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, 1, padding)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, 1, padding)
        self.norm2 = nn.BatchNorm2d(dim)
        self.droppath = DropPath(droppath)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x))) 
        h = F.relu(self.norm2(self.conv2(h)))
        x = x + self.droppath(h)
        return x


class ConvNet(nn.Module):
    def __init__(self, 
                 blocks, 
                 dims, 
                 kernel_size=3, 
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
                ConvBlock(dims[i], kernel_size, droppath) 
                for _ in range(blocks[i])
            ]))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsamples[i](x)
            x = self.layers[i](x)
        x = self.dropout(self.norm(x.mean([-1, -2])))
        x = self.head(x)
        return x