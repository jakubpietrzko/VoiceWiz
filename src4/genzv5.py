import torch
from torch import nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
class ConvSN2D(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ConvSN2D, self).__init__(*args, **kwargs)
        self = spectral_norm(self)

class ConvSN2DTranspose(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super(ConvSN2DTranspose, self).__init__(*args, **kwargs)
        self = spectral_norm(self)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        h,  c = 80,1
        # downscaling
        self.padding = nn.ZeroPad2d((0, 1, 0, 0))
        self.conv1 = ConvSN2D(c, 512, kernel_size=(h,3), stride=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = ConvSN2D(512, 512, kernel_size=(1,9), stride=(1,2))
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = ConvSN2D(512, 512, kernel_size=(1,7), stride=(1,2))
        self.bn3 = nn.BatchNorm2d(512)
        # upscaling
        self.deconv1 = ConvSN2DTranspose(512, 512, kernel_size=(1,7), stride=(1,2))
        self.bn4 = nn.BatchNorm2d(512)
        self.deconv2 = ConvSN2DTranspose(512, 512, kernel_size=(1,9), stride=(1,2))
        self.deconv3 = ConvSN2DTranspose(512, 1, kernel_size=(h,1), stride=1)
        self.conv_emb = ConvSN2D(512, 512, kernel_size=(1,1), stride=1)
        self.bn_emb = nn.BatchNorm2d(512)


    def forward(self, x, emb):
        emb  = torch.nn.functional.leaky_relu(self.bn_emb(self.conv_emb(emb)), negative_slope=0.2)
        x = self.padding(x)
        x1 = torch.nn.functional.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x2 = torch.nn.functional.leaky_relu(self.bn2(self.conv2(x1)), negative_slope=0.2)
        x3 = torch.nn.functional.leaky_relu(self.bn3(self.conv3(x2)), negative_slope=0.2)
        #print(x3.shape,emb.shape)   
        emb= F.pad(emb, (0, 1))
        x3 = x3 + emb
        x4 = torch.nn.functional.leaky_relu(self.bn4(self.deconv1(x3)), negative_slope=0.2)
        x5 = torch.nn.functional.leaky_relu(self.deconv2(x4+x2), negative_slope=0.2)  # skip connection
        x6 = self.deconv3(x5+x1)  # skip connection
        #x6 = F.sigmoid(x6) #do h.pth bezniczego
        x6 = torch.tanh(x6)# najczestsze
        return x6