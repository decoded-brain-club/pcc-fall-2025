import torch
import torch.nn as nn
import torch.nn.functional as F
print('hello world')

class EMA1D(nn.Module):
    """Applies Efficient Multi Scale Attention (EMA) over 1D sequence data.

    Groups channels and applies intra-group attention mechanisms
    that include channel-wise convolution, group normalization, and dual-path
    cross-attention to enhance feature representations in temporal data.

    Args:
        channels (int): Number of input channels.
        factor (int): Number of groups to divide the channels into. Must divide `channels` evenly.
    """
    def __init__(self, channels, factor=32):
        super(EMA1D, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        assert channels % self.groups == 0, f"channels ({channels}) must be divisible by factor/groups ({self.groups})"

        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool1d(1)
        self.pool_c = nn.AdaptiveAvgPool1d(1) # Pool across sequence length
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, l = x.size()  # (b, c, l)
        group_x = x.reshape(b * self.groups, -1, l) # b*g, c//g, l
        
        # Apply channel-wise avg pool across length
        x_c = self.pool_c(group_x) # b*g, c//g, 1
        
        # Apply 1x1 conv to channel-pooled features
        c_att = self.conv1x1(x_c) # b*g, c//g, 1
        
        x1 = self.gn(group_x * c_att.sigmoid())
        x2 = self.conv3(group_x)
        
        # Cross-attention between x1 & x2
        # Path 1: x1 attends to x2
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # b*g, 1, c//g
        x12 = x2.reshape(b * self.groups, c // self.groups, -1) # b*g, c//g, l
        
        # Path 2: x2 attends to x1  
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)) # b*g, 1, c//g
        x22 = x1.reshape(b * self.groups, c // self.groups, -1) # b*g, c//g, l
        
        # Combine attention weights
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, l)
        
        return (group_x * weights.sigmoid()).reshape(b, c, l) # (b, c, l)