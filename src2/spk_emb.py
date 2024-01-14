import torch
from torch import nn

class SpeakerEmbedder(nn.Module):
    def __init__(self):
        super(SpeakerEmbedder, self).__init__()
        self.fc1 = nn.Linear(192, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc5 = nn.Linear(2048, 4096)
        self.fc6 = nn.Linear(4096, 8192) 
        self.fc7 = nn.Linear(8192, 25040)
    def forward(self, x):
        l_relu = nn.LeakyReLU()
        x= l_relu(self.fc1(x))
        x= l_relu(self.fc2(x))
        x= l_relu(self.fc3(x))
        x= l_relu(self.fc4(x))
        x= l_relu(self.fc5(x))
        x= l_relu(self.fc6(x))
        x= l_relu(self.fc7(x))
        x = x.view(-1, 80, 313)  
        return x