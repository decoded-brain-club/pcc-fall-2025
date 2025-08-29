import torch
import torch.nn as nn

class CBR_Block(nn.Module):
    """Convolutional, Batch norm, Relu activation function block x2

    """
    def __init__(self, in_channel, out_channel):
        super(CBR_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Conv1d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)
    
    
class IC_UNet(nn.Module):
    """U-net model that downsamples, upsamples and concats to encode and decode EEG signals
    
    """
    def __init__(self, in_channel, out_channel):
        super(IC_UNet, self).__init__()
        
        # Encoder
        self.enc1 = CBR_Block(in_channel, out_channel)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = CBR_Block(out_channel, out_channel * 2)
        self.pool2 = nn.MaxPool1d(2)
        
        self.enc3 = CBR_Block(out_channel * 2, out_channel * 4)
        self.pool3 = nn.MaxPool1d(2)
        
        self.enc4 = CBR_Block(out_channel * 4, out_channel * 8)
        
        # Decoder
        self.up4 = nn.ConvTranspose1d(out_channel * 8, out_channel * 4, kernel_size=2, stride=2)
        self.dec4 = CBR_Block(out_channel * 8, out_channel * 4)
        
        self.up3 = nn.ConvTranspose1d(out_channel * 4, out_channel * 2, kernel_size=2, stride=2)
        self.dec3 = CBR_Block(out_channel * 4, out_channel * 2)
        
        self.up2 = nn.ConvTranspose1d(out_channel * 2, out_channel, kernel_size=2, stride=2)
        self.dec2 = CBR_Block(out_channel * 2, out_channel)
        
        # Final conv maps decoder
        self.final_conv = nn.Conv1d(out_channel, in_channel, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.enc1(x)
        p1 = self.pool1(d1)
        
        d2 = self.enc2(p1)
        p2 = self.pool2(d2)
        
        d3 = self.enc3(p2)
        p3 = self.pool3(d3)
        
        d4 = self.enc4(p3)
        
        # Decoder
        u4 = self.up4(d4)
        u4 = torch.cat([u4, d3], dim=1)
        u4 = self.dec4(u4)
        
        u3 = self.up3(u4)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.dec3(u3)
        
        u2 = self.up2(u3)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.dec2(u2)
                
        out = self.final_conv(u2)
        return out