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
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(16),
        )
        # Warstwy konwolucyjne 2D do przetwarzania melspektrogramu
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(80),
        )
        # Warstwy dekonwolucyjne 2D do modyfikacji melspektrogramu
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(80, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1,1)),
            nn.BatchNorm2d(64),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1,0)),
            nn.BatchNorm2d(32),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1,0)),
            nn.BatchNorm2d(16),
        )
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1,0)),
            
        )
        
        self.res_blocks = nn.ModuleList([ResidualBlock(32, 32, kernel_size=3, stride=1, padding=1) for i in range(3)])
        #self.res_blocks1 = nn.ModuleList([ResidualBlock(160, 160, kernel_size=3, stride=1, padding=1) for i in range(2)])
        # Oryginalny vokoder
        
        self.fc1 = nn.Linear(16000, 8024)
        self.fc2 = nn.Linear(8024, 8024)
        self.fc3 = nn.Linear(8024, 8000)
    def forward(self, x, speaker_embedding,ep,cnt):
        #speaker_embedding = speaker_embedding.unsqueeze(1)
        #speaker_embedding1 = self.fc_speaker1(speaker_embedding)
        xs= x.unsqueeze(1)
        x_std=xs.std()
        x_mean= xs.mean()
        eps = 1e-8
        if x_std < eps:
            x_std = eps
        xs = (xs - x_mean) / x_std
        # Przetwarzanie melspektrogramu

        xs=xs
        x = F.leaky_relu(self.conv0(xs))
        x = F.leaky_relu(self.conv1(x))
    
        for res_block in self.res_blocks:
            x = res_block(x)       
        #x= x + speaker_embedding1
        x = F.leaky_relu(self.conv2(x))
   
        x = F.leaky_relu(self.conv3(x))
        #for res_block in self.res_blocks1:
            #x = res_block(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        print(speaker_embedding.shape)
        x= torch.cat((x, speaker_embedding),dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = x.view(-1, 80 ,5, 20)
        #print(x.shape)
        x = F.leaky_relu(self.deconv3(x))
        x = F.leaky_relu(self.deconv2(x))

        x= F.leaky_relu(self.deconv1(x))
        modified_mel = self.deconv0(x)
        #print(modified_mel.shape)
        modified_mel = modified_mel*x_std  + x_mean
        modified_mel = modified_mel.squeeze(1)
        return modified_mel