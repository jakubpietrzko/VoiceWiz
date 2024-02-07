from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import torch
from torch import nn
import torchaudio
class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.activation = nn.LeakyReLU(0.2)
        self.skip_connection = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = self.dilated_conv(x)
        residual = self.activation(residual)
        return self.skip_connection(x) + residual

class SpeakerEmbedder(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=5):
        super(SpeakerEmbedder, self).__init__()
        self.initial_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(out_channels, 2 ** i) for i in range(num_blocks)])
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.mean_layer = nn.Linear(out_channels, out_channels)
        self.log_var_layer = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.avg_pooling(x).squeeze(-1)
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mean + eps * std, log_var
        else:
            return mean, log_var
    
if __name__=="__main__": 
    # Parametry
    in_channels = 80  # Przykładowa liczba kanałów wejściowych (np. dla spektrogramów mel)
    out_channels = 80  # Przykładowa liczba kanałów wyjściowych
    # Utworzenie instancji modelu
    speaker_embedder = SpeakerEmbedder(in_channels, out_channels)
    # Testowanie modelu
    test_input, _ = torchaudio.load('..\\data\\parts6s_resampled\\common_voice_en_38024627.wav')  # Przykładowy tensor wejściowy
    spectrogram, _ = mel_spectogram(
    audio=test_input.squeeze(),
    sample_rate=16000,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    n_fft=1024,
    f_min=0.0,
    f_max=8000.0,
    power=1,
    normalized=False,
    min_max_energy_norm=True,
    norm="slaney",
    mel_scale="slaney",
    compression=True
)
    
    
    mean, log_var = speaker_embedder(spectrogram)
    
    print(mean)
    print(log_var)
    print(log_var.shape)
    print(mean.shape)