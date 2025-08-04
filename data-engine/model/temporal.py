import torch.nn as nn
import torch.nn.functional as F
print('hello world')


class Temporal(nn.Module):
    """Temporal feature extractor module using fully connected layers followed by an LSTM.

    Args:
        in_channel (int): Number of input channels (should match the output channels of the preceding CNN).
        num_layers (int): Number of layers in the LSTM.
    """
    def __init__(self, in_channel, num_layers): # in_channel should be the same as cnn_plan[-1]
        super(Temporal, self).__init__()
        
        self.fc_1 = nn.Linear(in_features=in_channel, out_features=1024) 
        self.fc_2 = nn.Linear(in_features=1024, out_features=512)
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # (b, c, l) -> (b, l, c)
        x = self.fc_1(x)
        x = self.fc_2(x)

        x, _ = self.lstm(x)
        
        return x