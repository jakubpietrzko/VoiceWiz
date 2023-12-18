import torch
from torch import nn

class F0Encoder(nn.Module):
    def __init__(self, in_channels):
        super(F0Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=10, stride=1, padding=5)
        self.norm1 = nn.InstanceNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=10, stride=1, padding=5)
        self.norm2 = nn.InstanceNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=10, stride=1, padding=5)
        self.norm3 = nn.InstanceNorm1d(64)

    def forward(self, x):
        # Przetwarzaj log F0 i maskę osobno
        log_f0 = x[:, :, 0, :]
        mask = x[:, :, 1, :]
        
        log_f0 = self.norm1(torch.relu(self.conv1(log_f0)))
        log_f0 = self.norm2(torch.relu(self.conv2(log_f0)))
        log_f0 = self.norm3(torch.relu(self.conv3(log_f0)))
        
        mask = self.norm1(torch.relu(self.conv1(mask)))
        mask = self.norm2(torch.relu(self.conv2(mask)))
        mask = self.norm3(torch.relu(self.conv3(mask)))
        
        # Połącz wyniki
        x = torch.cat((log_f0, mask), dim=1)
        
        return x
    
"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f0_encoder = F0Encoder(device)"""