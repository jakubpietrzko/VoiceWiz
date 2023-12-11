import torch
from torch import nn

class F0Encoder(nn.Module):
    def __init__(self):
        super(F0Encoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=10, stride=1, padding=5)
        self.norm1 = nn.InstanceNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=10, stride=1, padding=5)
        self.norm2 = nn.InstanceNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=10, stride=1, padding=5)
        self.norm3 = nn.InstanceNorm1d(64)

    def forward(self, x):
        x = self.norm1(torch.relu(self.conv1(x)))
        x = self.norm2(torch.relu(self.conv2(x)))
        x = self.norm3(torch.relu(self.conv3(x)))
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f0_encoder = F0Encoder(device)