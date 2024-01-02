import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
from speechbrain.pretrained import HIFIGAN


class Generator(nn.Module):
    def __init__(self, speaker_embedding_dim):
        super(Generator, self).__init__()
            # Warstwa osadzająca mówcę
        # Warstwy osadzające mówcę 
        """self.fc_speaker1 = nn.Linear(speaker_embedding_dim,64)
        self.fc_speaker2 = nn.Sequential( 
        nn.Linear(speaker_embedding_dim,64), 
        nn.ReLU(),
        nn.Linear(64,32) ) 
        self.fc_speaker3 = nn.Sequential( 
        nn.Linear(speaker_embedding_dim,64), 
        nn.ReLU(), 
        nn.Linear(64,32), 
        nn.ReLU(), 
        nn.Linear(32,16) ) """
        self.fc_speaker1 = nn.Sequential(
            nn.Conv2d(speaker_embedding_dim, 16, kernel_size=5 , stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, kernel_size=5, stride=2, padding=2),
        )

        self.fc_speaker2 = nn.Sequential(
            nn.Conv2d(speaker_embedding_dim, 32, kernel_size=5 , stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
        )
        self.fc_speaker4 = nn.Sequential(
            nn.Conv2d(speaker_embedding_dim, 32, kernel_size=5 , stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
        )
        self.fc_speaker5 = nn.Sequential(
            nn.Conv2d(speaker_embedding_dim, 32, kernel_size=5 , stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
        )
        self.fc_speaker3 = nn.Sequential(
            nn.Conv2d(speaker_embedding_dim, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128,kernel_size=5, stride=1, padding=2)
        )
        self.conv0 = nn.Conv2d(1, 2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        # Warstwy konwolucyjne 2D do przetwarzania melspektrogramu
        self.conv1 = nn.Conv2d(2, 4, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) # zmieniamy wejście na 2
        
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.pool = nn.MaxPool2d(2, return_indices=True)
        
        self.conv4 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #spk2
        self.conv5 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.conv6 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #spk3
        # Warstwy dekonwolucyjne 2D do modyfikacji melspektrogramu
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #spk4
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.unpool = nn.MaxUnpool2d(2)
        #spk5
        self.deconv4 = nn.ConvTranspose2d(16, 8, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.deconv5 = nn.ConvTranspose2d(8, 4, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
        
        self.deconv6 = nn.ConvTranspose2d(4, 2, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
        self.deconv7 = nn.ConvTranspose2d(2, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        
        # Oryginalny vokoder
        self.vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="vocoder_16khz", run_opts={"device": "cuda"})
        self.vocoder.eval()
        
        
        for param in self.vocoder.parameters():
            param.requires_grad = False

    def forward(self, x, speaker_embedding):
        # Osadzanie mówcy
        #print(speaker_embedding.shape)
        """
        speaker_embedding1 = self.fc_speaker1(speaker_embedding).unsqueeze(-1).unsqueeze(-1)
        speaker_embedding2 = self.fc_speaker2(speaker_embedding).unsqueeze(-1).unsqueeze(-1)
        speaker_embedding3 = self.fc_speaker3(speaker_embedding).unsqueeze(-1).unsqueeze(-1)"""
        #print(speaker_embedding1.shape)
        #print(speaker_embedding2.shape)
        speaker_embedding1 = self.fc_speaker1(speaker_embedding)
        speaker_embedding2 = self.fc_speaker2(speaker_embedding)
        speaker_embedding3 = self.fc_speaker3(speaker_embedding) 
        speaker_embedding4 = self.fc_speaker4(speaker_embedding)
        speaker_embedding5 = self.fc_speaker5(speaker_embedding)
        xs= x.unsqueeze(1)
        x_std=xs.std()
        x_mean= xs.mean()
        eps = 1e-8
        if x_std < eps:
            x_std = eps
       
        xs = (xs - x_mean) / x_std
        
        #print(mel.shape)
        # Przetwarzanie melspektrogramu
        x = F.leaky_relu(self.conv0(xs))
        x = F.leaky_relu(self.conv1(x))
      
        x = F.leaky_relu(self.conv2(x))
        x, ind = self.pool(x) 

        x = x + speaker_embedding1
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = x + speaker_embedding2
        x = F.leaky_relu(self.conv5(x))
        
        x = F.leaky_relu(self.conv6(x))
        #print(x.shape)\
      
        # Modyfikacja melspektrogramu
        x = x + speaker_embedding3
        #print(x.shape)
        x = F.leaky_relu(self.deconv1(x))
        x = F.leaky_relu(self.deconv2(x))
 
        x = x + speaker_embedding4
        x = F.leaky_relu(self.deconv3(x))
        
        x = F.leaky_relu(self.deconv4(x))
        
        x = self.unpool(x,ind)
        
        x= F.leaky_relu(self.deconv5(x))
        
        x = F.leaky_relu(self.deconv6(x))
        
        modified_mel = torch.tanh(self.deconv7(x))
        modified_mel = (modified_mel  + x_mean)*x_std
        modified_mel = modified_mel.squeeze(1)
        
        # Konwersja na audio przy użyciu oryginalnego vokodera
        modified_audio = self.vocoder.decode_batch(modified_mel)
        modified_mel = modified_mel

        return modified_audio , modified_mel
if __name__ == "__main__":
    # Parametry
    asr_dim = 129  # Zaktualizuj zgodnie z wymiarem wyjścia ASR encoder
    f0_dim = 129  # Zaktualizuj zgodnie z wymiarem wyjścia F0 encoder
    speaker_dim = 1 # Zaktualizuj zgodnie z wymiarem wyjścia Speaker Embedder
    hidden_dim = 256  # Możesz dostosować zgodnie z potrzebami
    output_dim = 1   # Wymiary wyjściowe spektrogramu, dostosuj zgodnie z potrzebami

    # Utworzenie instancji modelu
    generator = Generator(asr_dim, f0_dim, speaker_dim, hidden_dim, output_dim)

    # Testowanie modelu
    test_input_asr = torch.randn(1, asr_dim)         # Przykładowy tensor wejściowy dla ASR
    test_input_f0 = torch.randn(1, f0_dim)           # Przykładowy tensor wejściowy dla F0
    test_input_speaker = torch.randn(1, 1, speaker_dim, 1) # Przykładowy tensor wejściowy dla Speaker Embedder
    print(test_input_asr.shape)
    output = generator(test_input_asr, test_input_f0, test_input_speaker)
    print(output.shape)  # Sprawdzenie wymiarów wyjścia