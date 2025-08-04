import torch.nn as nn
import torch.nn.functional as F
import ema_1d
print('hello world')

class CNN_EMA1D_Block(nn.Module):
    """A convolutional block that combines 1D convolution, ReLU activation,
    EMA-1D, and optional pooling or dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after convolution.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding added to both sides of the input.
        factor (int): Number of groups for the EMA1D attention mechanism.
        kernel_pool (int, optional): Kernel size for AvgPool1d (if pooling is used).
        stride_pool (int, optional): Stride for AvgPool1d (if pooling is used).
        pool_dropout (str, optional): If 'pool', applies average pooling;
                                      if 'dropout', applies dropout; if None, applies neither.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, factor, kernel_pool=None, stride_pool=None, pool_dropout=None):
        super(CNN_EMA1D_Block, self).__init__()
        
        # CNN_EMA1D Block
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU() # Activation function
        self.ema1d = ema_1d.EMA1D(channels=out_channels, factor=factor) # EMA-1d
        
        # Conditional to add either average pooling or dropout 
        self.pool_dropout = pool_dropout
        if pool_dropout == 'pool':
            assert kernel_pool is not None and stride_pool is not None, "Pooling selected but kernel_pool or stride_pool is None"
            
            self.pool = nn.AvgPool1d(kernel_size=kernel_pool, stride=stride_pool)
        elif pool_dropout == 'dropout':
            self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.ema1d(x)

        if self.pool_dropout == 'pool':
            x = self.pool(x)
        elif self.pool_dropout == 'dropout':
            x = self.dropout(x)

        return x # b, c, l
    
    
class CNN_EMA1D(nn.Module):
    """A stacked 1D convolutional network with EMA-1D attention applied at each layer.

    This module constructs a sequence of CNN_EMA1D_Block layers, where each block applies
    convolution, ReLU, EMA1D attention, and optionally average pooling or dropout.

    Args:
        channel_plan (List[int]): List defining the number of channels in each layer.
                                  Must have at least two elements: input and one output.
        kernel_size (int): Kernel size for all Conv1d layers.
        stride (int): Stride for all Conv1d layers.
        padding (int): Padding for all Conv1d layers.
        factor (int): Number of groups for the EMA1D attention mechanism.
        kernel_pool (int or None): Kernel size for AvgPool1d if pooling is used.
        stride_pool (int or None): Stride for AvgPool1d if pooling is used.
        pool_dropout (str or None): If 'pool', applies AvgPool1d; if 'dropout', applies dropout;
                                    if None, no additional operation is applied.
    """
    def __init__(self, channel_plan, kernel_size, stride, padding, factor, kernel_pool, stride_pool, pool_dropout):
        super(CNN_EMA1D, self).__init__()
        assert len(channel_plan) >= 2, "channel_plan must have at least input and one output"

        # Repeat layers 
        self.layers = nn.ModuleList()
        for i in range(len(channel_plan) - 1): # Loop over adjacent pairs in channel_plan
            in_channel = channel_plan[i] # Current input channel
            out_channel = channel_plan[i + 1] # Next output channel
            self.layers.append(
                CNN_EMA1D_Block(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    factor=factor,
                    kernel_pool=kernel_pool,
                    stride_pool=stride_pool,
                    pool_dropout=pool_dropout
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x # (b, c, l)
    
    
class Features_Block(nn.Module):
    """A feature extraction block combining stacked CNN_EMA1D layers followed by
    additional Conv1d, ReLU, EMA1D attention, and dropout layers.

    Args:
        channel_plan (List[int]): Channel sizes for the CNN_EMA1D block layers.
        kernel_size (int): Kernel size for all convolutional layers.
        stride (int): Stride for all convolutional layers.
        padding (int): Padding for all convolutional layers.
        factor (int): Number of groups for the EMA1D attention mechanism.
        kernel_pool (int or None): Kernel size for average pooling in CNN_EMA1D blocks.
        stride_pool (int or None): Stride for average pooling in CNN_EMA1D blocks.
        pool_dropout (str or None): Specifies pooling or dropout in CNN_EMA1D blocks.
        cnn_plan (List[int]): Channel sizes for the final convolutional layers after CNN_EMA1D.
    """
    def __init__(self, channel_plan, kernel_size, stride, padding, factor, kernel_pool, stride_pool, pool_dropout, cnn_plan):
        super(Features_Block, self).__init__()
        
        # Ensure that output_channel in channel_plan matches input_channel from cnn_plan
        assert channel_plan[-1] == cnn_plan[0], (
            f"Channel mismatch: Expected cnn_plan[0] ({cnn_plan[0]}) to match channel_plan[-1] ({channel_plan[-1]})"
        )
        assert len(cnn_plan) == 3, (f"cnn_plan must be list[int] length of 3")

        # CNN_EMA1D block (morphological feature extraction)
        self.cnn_ema1d = CNN_EMA1D(
            channel_plan,
            kernel_size,
            stride,
            padding,
            factor,
            kernel_pool,
            stride_pool,
            pool_dropout
        )
        
        # Sequential CNN + ReLU + EMA + Dropout block
        self.conv_1 = nn.Conv1d(in_channels=cnn_plan[0], out_channels=cnn_plan[1], kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu_1 = nn.ReLU() 
        self.conv_2 = nn.Conv1d(in_channels=cnn_plan[1], out_channels=cnn_plan[2], kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu_2 = nn.ReLU() 
        self.ema1d = ema_1d.EMA1D(channels=cnn_plan[2], factor=factor) # EMA-1d
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.cnn_ema1d(x)
        x = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x))
        x = self.ema1d(x)
        x = self.dropout(x)
        
        return x # (b, c, l)