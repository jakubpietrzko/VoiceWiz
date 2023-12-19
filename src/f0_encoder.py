import torch
from torch import nn

class F0Encoder(nn.Module):
    def __init__(self, in_channels):
        super(F0Encoder, self).__init__()
        num_channels_f0 = 129  # Ustawiamy na 130, aby na wyjściu były 2 kanały
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=10, stride=1, padding=5)
        self.norm1 = nn.InstanceNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=10, stride=1, padding=5)
        self.norm2 = nn.InstanceNorm1d(64)
        self.conv3 = nn.Conv1d(64, num_channels_f0, kernel_size=10, stride=1, padding=5)
        self.norm3 = nn.InstanceNorm1d(num_channels_f0)
        self.interpolation = nn.Upsample(301, mode='linear', align_corners=False)

    def forward(self, x):
        # Przetwarzaj log F0 i maskę jednocześnie
        x=x.squeeze(1)
        x = self.norm1(torch.relu(self.conv1(x)))
        x = self.norm2(torch.relu(self.conv2(x)))
        x = self.norm3(torch.relu(self.conv3(x)))
        
        # Interpolacja do docelowej długości sekwencji
        x = self.interpolation(x)
        
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