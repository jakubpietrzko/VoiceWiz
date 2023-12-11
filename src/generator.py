from torch import nn
import torch
class WaveNet(nn.Module):
    def __init__(self, layers=10, blocks=3):
        super(WaveNet, self).__init__()
        self.layers = layers
        self.blocks = blocks
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=2, dilation=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=2, dilation=2)
        self.conv3 = nn.Conv1d(128, 2, kernel_size=1)
        self.conv4 = nn.Conv1d(2, 1, kernel_size=1)
        self.conv_speaker = nn.Conv1d(128, 128, kernel_size=1) # Warstwa konwolucyjna, która przekształca osadzenie mówcy do wymiaru wejścia generatora  

    def forward(self, x, speaker_embedding):
        speaker_embedding = self.conv_speaker(speaker_embedding)
        for _ in range(self.blocks):
            for i in range(self.layers):
                x_tanh = self.tanh(self.conv1(x) + speaker_embedding)
                x_sigmoid = self.sigmoid(self.conv2(x) + speaker_embedding)
                x = x_tanh * x_sigmoid
                x = self.conv3(x)
        x = self.conv4(x)
        return x

class Generator(nn.Module):
    def __init__(self, asr_features, f0_features, speaker_features, generator_input_dim):
        super(Generator, self).__init__()
        # Warstwy generatora, które przekształcają wejście do surowego audio
        self.generator = WaveNet()
        # Warstwy konwolucyjne, które przekształcają cechy ASR, F0 i mówcy do wymiaru wejścia generatora
        self.conv_asr = nn.Conv3d(asr_features, generator_input_dim, kernel_size=3, padding=1)
        self.conv_f0 = nn.Conv2d(f0_features, generator_input_dim, kernel_size=3, padding=1)
        self.conv_speaker = nn.Conv2d(speaker_features, generator_input_dim, kernel_size=3, padding=1)

    def forward(self, asr_data, f0_data, speaker_data):
        # Dostosowanie wymiarów danych
        asr_data = self.conv_asr(asr_data)
        f0_data = self.conv_f0(f0_data.unsqueeze(1))
        speaker_data = self.conv_speaker(speaker_data.unsqueeze(1))
        # Sumowanie danych
        x = asr_data + f0_data + speaker_data
        return self.generator(x, speaker_data)  # przekazanie osadzenia mówcy do generatora