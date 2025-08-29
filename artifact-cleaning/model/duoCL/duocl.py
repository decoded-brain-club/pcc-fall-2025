import torch
import torch.nn as nn

class duoCL(nn.Module):
    def __init__(self):
        super(duoCL, self).__init__()
        
        # === 1st Conv path with kernel = 3 ===
        self.conv_13 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.relu_11 = nn.ReLU()
        self.pool_11 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv_23 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.relu_12 = nn.ReLU()
        self.pool_12 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv_33 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=3, stride=1)
        self.relu_13 = nn.ReLU()
        self.pool_13 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv_43 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.relu_14 = nn.ReLU()
        self.pool_14 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv_53 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.dropout_1 = nn.Dropout(0.5)
        
        self.flatten_3 = nn.Flatten()
        
        self.fc_1 = nn.Linear(in_features=14336,out_features=1024)
        self.fc_2 = nn.Linear(in_features=1024,out_features=512)
        
        # === 2nd Conv path with kernel = 7 ===
        self.conv_17 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1)
        self.relu_21 = nn.ReLU()
        self.pool_21 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv_27 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1)
        self.relu_22 = nn.ReLU()
        self.pool_22 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv_37 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=7, stride=1)
        self.relu_23 = nn.ReLU()
        self.pool_23 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv_47 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=1)
        self.relu_24 = nn.ReLU()
        self.pool_24 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv_57 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=7, stride=1)
        self.dropout_2 = nn.Dropout(0.5)
        
        self.flatten_4 = nn.Flatten()

        self.fc_3 = nn.Linear(in_features=10240,out_features=1024)
        self.fc_4 = nn.Linear(in_features=1024,out_features=512)
        
        # === Separate LSTMs ===
        self.lstm_1 = nn.LSTM(input_size=512, hidden_size=512, num_layers=3, batch_first=False)
        self.lstm_2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=3, batch_first=False)
        
        # FC Layer after LSTM
        self.fc_5 = nn.Linear(in_features=1024, out_features=512)
        
    def forward(self, x):
        # --- Path 1 (kernel=3) ---
        x1 = self.pool_11(self.relu_11(self.conv_13(x)))
        x1 = self.pool_12(self.relu_12(self.conv_23(x1)))
        x1 = self.pool_13(self.relu_13(self.conv_33(x1)))
        x1 = self.pool_14(self.relu_14(self.conv_43(x1)))
        x1 = self.dropout_1(self.conv_53(x1))   # Shape: (B, 512, L)
        
        # Prepare for LSTM: transpose to (B, L1, 512)
        x1 = x1.permute(0, 2, 1)  # (B, L1, C)
        out1, _ = self.lstm_1(x1)  # (B, L1, 512)
        out1 = out1[:, -1, :]      # Take last time step (B, 512)

        # --- Path 2 (kernel=7) ---
        x2 = self.pool_21(self.relu_21(self.conv_17(x)))
        x2 = self.pool_22(self.relu_22(self.conv_27(x2)))
        x2 = self.pool_23(self.relu_23(self.conv_37(x2)))
        x2 = self.pool_24(self.relu_24(self.conv_47(x2)))
        x2 = self.dropout_2(self.conv_57(x2))   # Shape: (B, 512, L2)
        
        # Prepare for LSTM: transpose to (B, L2, 512)
        x2 = x2.permute(0, 2, 1)  # (B, L2, C)
        out2, _ = self.lstm_2(x2)  # (B, L2, 512)
        out2 = out2[:, -1, :]      # Take last time step (B, 512)

        # Concatenate outputs from both LSTMs
        out = torch.cat([out1, out2], dim=1)  # (B, 1024)

        # Final FC layer
        out = self.fc_5(out)  # (B, 512)
        out = out.unsqueeze(1) # (B, 1, 512)

        return out
