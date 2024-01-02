from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import torch
from torch import nn
import torchaudio
class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.activation = nn.LeakyReLU(0.2)
        self.skip_connection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = self.dilated_conv(x)
        residual = self.activation(residual)
        return self.skip_connection(x) + residual

class SpeakerEmbedder(nn.Module):
    def __init__(self, num_blocks=5):
        super(SpeakerEmbedder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1)
        self.decon3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.decon2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.initial_conv = nn.Conv2d(32, 32, kernel_size=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(32, 2 ** i) for i in range(num_blocks)])
        self.avg_pooling = nn.AvgPool2d(2)
        self.mean_layer = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.log_var_layer = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.conv2.parameters():
            param.requires_grad = False

        for param in self.conv3.parameters():
            param.requires_grad = False

    def forward(self, x,y,z):
        x= x.unsqueeze(1)
        x_std=x.std()
        eps = 1e-8
        if x_std < eps:
            x_std = eps
       
        x = (x - x.mean()) / x_std
        
        x = self.conv1(x)
        x, indx = self.pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        y = y.unsqueeze(1)
        y_std=y.std()
        if y_std < eps:
            y_std = eps
        y = (y - y.mean()) / y_std
        
        y = self.conv1(y)
        y, indy = self.pool(y)
        y = self.conv2(y)
        y = self.conv3(y)
        z = z.unsqueeze(1)
        z_std=z.std()
        if z_std < eps:
            z_std = eps
        z = (z - z.mean()) / z_std
        z = self.conv1(z)
        z, zind = self.pool(z)
        z = self.conv2(z)
        z = self.conv3(z)
        
        x= self.decon3(x)
       
        x= self.decon2(x)
        x= self.unpool(x,indx)
        y= self.decon3(y)
        y= self.decon2(y)
        y= self.unpool(y,indy)
        z= self.decon3(z)
        z= self.decon2(z)
        z = self.unpool(z,zind)
        x = torch.cat((x, y,z,x,y,z,x), dim=3)
        
        x = x[:, :, :, :376]
        
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.avg_pooling(x)
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