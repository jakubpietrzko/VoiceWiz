import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.batch_norm1(F.leaky_relu(self.conv1(x)))
        out = self.batch_norm2(F.leaky_relu(self.conv2(out)))
        residual = self.conv3(residual)
        out += residual
        out = F.relu(out)
        return out

class Generator(nn.Module):
    def __init__(self, speaker_embedding_dim):
        super(Generator, self).__init__()
        # Warstwa osadzająca mówcę
        self.fc_speaker1 = nn.Sequential(
            
            nn.Conv2d(speaker_embedding_dim, 128, kernel_size=5 , stride=2, padding=2),
            nn.BatchNorm2d(128),
            
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(64),
        )
        # Warstwy konwolucyjne 2D do przetwarzania melspektrogramu
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
        )
        self.conv_addChannel = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
        )
        # Warstwy dekonwolucyjne 2D do modyfikacji melspektrogramu
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1,0)),
            nn.BatchNorm2d(128),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
        )
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1,0)),
            
        )
        
        self.res_blocks = nn.ModuleList([ResidualBlock(128, 128, kernel_size=3, stride=1, padding=1) for i in range(3)])
        self.res_blocks1 = nn.ModuleList([ResidualBlock(128, 128, kernel_size=3, stride=1, padding=1) for i in range(3)])
        # Oryginalny vokoder
        
  
    def forward(self, x, speaker_embedding,ep,cnt):
        speaker_embedding = speaker_embedding.unsqueeze(1)
        speaker_embedding1 = self.fc_speaker1(speaker_embedding)
        xs= x.unsqueeze(1)
        x_std=xs.std()
        x_mean= xs.mean()
        eps = 1e-8
        if x_std < eps:
            x_std = eps
        xs = (xs - x_mean) / x_std
        # Przetwarzanie melspektrogramu
        x = F.leaky_relu(self.conv0(xs))
        x = F.leaky_relu(self.conv1(x))
        for res_block in self.res_blocks:
            x = res_block(x)       
        x= x + speaker_embedding1
        xd=x
        xd= F.leaky_relu(self.conv_addChannel(xd))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x=x+xd
        x = F.leaky_relu(self.conv4(x))
        for res_block in self.res_blocks1:
            x = res_block(x) 
        x = F.leaky_relu(self.deconv4(x))
        x= x+xd
        x = F.leaky_relu(self.deconv3(x))
        x = F.leaky_relu(self.deconv2(x))
        x= F.leaky_relu(self.deconv1(x))
        modified_mel = self.deconv0(x)
        #print(modified_mel.shape)
        modified_mel = modified_mel*x_std  + x_mean
        modified_mel = modified_mel.squeeze(1)
        return modified_mel
        if True or ep>5 or cnt ==10 or cnt%200==0:
        # Konwersja na audio przy użyciu oryginalnego vokodera
            modified_audio = self.vocoder.decode_batch(modified_mel)
            return modified_audio , modified_mel
        modified_audio = 0
        return modified_audio , modified_mel
