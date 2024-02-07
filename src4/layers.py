import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ConvSN2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvSN2D, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

    def forward(self, x):
        return self.conv(x)

class ConvSN2DTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvSN2DTranspose, self).__init__()
        self.conv_transpose = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))

    def forward(self, x):
        return self.conv_transpose(x)

class DenseSN(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseSN, self).__init__()
        self.dense = spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.dense(x)

class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=False):
        super(LeakyReLU, self).__init__(negative_slope, inplace)

class ReLU(nn.ReLU):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(inplace)

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d, self).__init__(num_features)

class UpSampling2D(nn.Upsample):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(UpSampling2D, self).__init__(size, scale_factor, mode, align_corners)

class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, *inputs):
        return torch.cat(inputs, dim=1)