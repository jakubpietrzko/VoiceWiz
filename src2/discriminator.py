import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(5, 5), stride=2, padding=2),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 8, kernel_size=(5, 5), stride=2, padding=(2, 2)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(768, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x_std=x.std()
        eps = 1e-8
        if x_std < eps:
            x_std = eps
       
        x = (x - x.mean()) / x_std
        
        for layer in self.conv:
            x = layer(x)   
        return x