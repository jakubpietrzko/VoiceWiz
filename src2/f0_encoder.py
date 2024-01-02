import torch
from torch import nn

class F0Encoder(nn.Module):
    def __init__(self, in_channels):
        super(F0Encoder, self).__init__()
        num_channels_f0 = 129  # Ustawiamy na 130, aby na wyjściu były 2 kanały
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(2,5), stride=1, padding=(0,2))
        self.norm1 = nn.InstanceNorm2d(2)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=(1,5), stride=1, padding=(0,2))
        self.norm2 = nn.InstanceNorm2d(4)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=(1,5), stride=1, padding=(0,2))
        self.norm3 = nn.InstanceNorm2d(8)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        # Przetwarzaj log F0 i maskę jednocześnie
        x=x.unsqueeze(1)
        print(x.shape)
        x = self.norm1(self.leaky_relu(self.conv1(x)))
        x = self.norm2(self.leaky_relu(self.conv2(x)))
        x = self.norm3(self.leaky_relu(self.conv3(x)))
        print(x.shape)
        return x
"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f0_encoder = F0Encoder(device)"""
if __name__ == "__main__":
    # Tworzenie instancji modelu
    model = F0Encoder(in_channels=2)

    # Tworzenie losowego batcha danych wejściowych
    # Wymiary: [batch_size, in_channels, sequence_length]
    input_data = torch.load('..\\data\\fzeros\\common_voice_en_38024626.pt')
    print(input_data.shape)
    input_data = input_data.float()
   
    # Przekazanie danych przez model
    output_data = model(input_data)

    # Sprawdzenie wymiarów wyjścia
    print(output_data.shape)