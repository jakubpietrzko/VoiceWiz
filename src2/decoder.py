import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNet(nn.Module):
    def __init__(self, num_channels, num_classes, num_blocks, num_layers):
        super(WaveNet, self).__init__()
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()

        for b in range(num_blocks):
            for l in range(num_layers):
                dilation = 2 ** l
                padding = (2 ** l) // 2
                conv = nn.Conv1d(num_channels, num_channels, kernel_size=2, dilation=dilation, padding=padding)
                self.conv_layers.append(conv)

        self.output_conv = nn.Conv1d(num_channels, num_classes, kernel_size=1)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = self.output_conv(x)
        return x

def spectrogram_to_audio(spectrogram, model):
    # Konwersja spektrogramu na tensor
    #if not isinstance(spectrogram, torch.Tensor):
     #   raise TypeError("Spektrogram musi być tensorem PyTorch.")

    # Generowanie dźwięku
    audio = model(spectrogram)
    return audio

# Parametry modelu
num_channels = 80  # liczba kanałów spektrogramu
num_classes = 256  # zakres wartości wyjściowych audio
num_blocks = 3     # liczba bloków
num_layers = 10    # liczba warstw w każdym bloku

# Inicjalizacja modelu
model = WaveNet(num_channels, num_classes, num_blocks, num_layers)

# Przykładowe dane
spectrogram = torch.load('..\\data\\mels\\common_voice_en_38024625.pt')
if not isinstance(spectrogram, torch.Tensor):
        raise TypeError("Spektrogram musi być tensorem PyTorch.")                        
spectrogram = spectrogram
# Konwersja spektrogramu na dźwięk
audio = spectrogram_to_audio(spectrogram, model)
import torchaudio
audio = audio.squeeze(0)
print(audio.shape)
torchaudio.save('speech1.wav', audio.detach().cpu(), sample_rate=22050)
