import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
class ConvSN2D(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ConvSN2D, self).__init__(*args, **kwargs)
        self = spectral_norm(self)
class DenseSN(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(DenseSN, self).__init__(*args, **kwargs)
        self = spectral_norm(self)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        h, c = 80,1
        self.conv1 = ConvSN2D(c, 512, kernel_size=(h,3), stride=1)
        self.conv2 = ConvSN2D(512, 512, kernel_size=(1,9), stride=(1,2))
        self.conv3 = ConvSN2D(512, 512, kernel_size=(1,7), stride=(1,2))
        self.flatten = nn.Flatten()
        self.dense = DenseSN(9216, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = torch.nn.functional.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = torch.nn.functional.leaky_relu(self.conv3(x), negative_slope=0.2)
        xd2=x
        x = self.flatten(x)
        xd =x
        x = self.dense(x)
        #x = torch.sigmoid(x) 
        return x,xd,xd2Zg