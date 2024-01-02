import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5), stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(1280, 1048),
            nn.LeakyReLU(0.2),
            nn.Linear(1048, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),  # zmienione z 512*4*4 na 512
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
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
