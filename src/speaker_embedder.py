import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class SpeakerEmbedder(nn.Module):
    def __init__(self, in_channels, num_residual_layers):
        super(SpeakerEmbedder, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(in_channels) for _ in range(num_residual_layers)]
        )
        self.mean = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.log_var = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.res_blocks(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std