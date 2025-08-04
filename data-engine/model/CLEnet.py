import torch
import torch.nn as nn
import torch.nn.functional as F
import features as feat
import ema_1d
import temporal as temp


class CLEnet(nn.Module):
    """CNN, LSTM, EMA-1D (CLEnet)

    This model is designed to process EEG data by extracting spatial features
    using convolutional blocks (`Features_Block`), capturing temporal dependencies
    using LSTM blocks (`Temporal`), and finally producing a denoised EEG output
    of the same shape as the original input using a fully connected layer.

    Args:
        channel_plan (list[int]): Configuration of channels for convolutional layers.
        kernel_sizes (list[int]): Two kernel sizes for dual-path convolutional branches.
        stride (int): Stride used in the convolutional layers.
        padding (int): Padding applied to the convolutional layers.
        factor (int): Expansion or bottleneck factor used in `Features_Block`.
        kernel_pool (int): Kernel size used in pooling layers.
        stride_pool (int): Stride used in pooling layers.
        pool_dropout (float): Dropout rate applied after pooling.
        cnn_plan (list[int]): Configuration for CNN layer channel dimensions.
        num_layers (int): Number of LSTM layers in each `Temporal` block.
        out_features (int): The number of output features (should match the length of the original EEG signal).

    """
    def __init__(self, channel_plan, kernel_sizes, stride, padding, factor, kernel_pool, stride_pool, pool_dropout, cnn_plan, num_layers, in_features, out_features):
        super(CLEnet, self).__init__()
        assert len(kernel_sizes) == 2, "kernel_sizes must be a list[int] of two elements"

        # Create two feature extractors with different kernel sizes
        self.features = nn.ModuleList([
            feat.Features_Block(
                channel_plan=channel_plan,
                kernel_size=k,
                stride=stride,
                padding=padding,
                factor=factor,
                kernel_pool=kernel_pool,
                stride_pool=stride_pool,
                pool_dropout=pool_dropout,
                cnn_plan=cnn_plan
            ) for k in kernel_sizes
        ])

        # Create two temporal extractors
        self.temporals = nn.ModuleList([
            temp.Temporal(in_channel=cnn_plan[-1], num_layers=num_layers)
            for _ in range(2)
        ])

        # in_features = (10 * 512) + (4 * 512) = 7168
        # out_features = length of original EEG signal
        self.fc = nn.Linear(in_features=in_features, out_features=out_features) 
        
    def forward(self, x):
        # Apply each feature and temporal extractor in parallel
        x_1 = self.temporals[0](self.features[0](x)) # (batch, length, hidden_size)
        x_2 = self.temporals[1](self.features[1](x)) # (b, l, h)
        
        # Transpose l & h to be able to concat
        x_1 = x_1.transpose(1, 2) # (b, h, l)
        x_2 = x_2.transpose(1, 2) # (b, h, l)
        
        # Concat along hidden_size and then flatten 
        x = torch.cat([x_1, x_2], dim=-1) # (b, l, h*2)
        x = x.reshape(x.size(0), -1) # (b, l * h*2)  

        x = self.fc(x)
        
        return x