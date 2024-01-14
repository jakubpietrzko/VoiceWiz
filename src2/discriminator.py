import torch.nn as nn
import time
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5), stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(12800, 6400),
            nn.LeakyReLU(0.2),
            nn.Linear(6400, 3200),
            nn.LeakyReLU(0.2),
            nn.Linear(3200, 1048),
            nn.LeakyReLU(0.2),
            nn.Linear(1048, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
 
        x = x.unsqueeze(1)
        eps = 1e-8

        # Oblicz średnią i odchylenie standardowe dla całego batcha
        mean = x.mean([0, 2, 3], keepdim=True)
        std = x.std([0, 2, 3], keepdim=True)

        # Normalizuj dane
        x = (x - mean) / (std + eps)
        
        for layer in self.conv:
            x = layer(x)   
        
        return x