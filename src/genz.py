import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo.collections.tts.models import UnivNetModel
class Generator(nn.Module):
    def __init__(self, asr_dim, f0_dim, speaker_dim, output_dim):
        super(Generator, self).__init__()
        self.vocoder = UnivNetModel.from_pretrained(model_name="tts_en_libritts_univnet")
        for param in self.vocoder.parameters():
            param.requires_grad = False
        # Przygotowanie danych wejściowych
        self.fc_asr = nn.Linear(asr_dim, output_dim)
        self.fc_f0 = nn.Linear(f0_dim, output_dim)
        self.conv_speaker = nn.Conv2d(speaker_dim, output_dim, kernel_size=(3, 3), padding=(1, 1))  # Warstwa konwolucyjna dla speaker_features

        # Warstwy ukryte
        self.conv1 = nn.Conv2d(1, output_dim, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(output_dim)

        # Upsampling
        self.upconv1 = nn.ConvTranspose2d(output_dim, output_dim // 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(output_dim // 2)
        self.upconv2 = nn.ConvTranspose2d(output_dim // 2, output_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, asr_features, f0_features, speaker_features):
        # Przetwarzanie danych wejściowych
        asr_features = self.fc_asr(asr_features.permute(0,2,1))  # Przekształcenie na 2D
        f0_features = self.fc_f0(f0_features)  # Przekształcenie na 2D
        speaker_embedding = self.conv_speaker(speaker_features)  # Osadzanie speaker_features za pomocą warstwy konwolucyjnej

        # Łączenie cech
        combined_features = torch.cat((asr_features, f0_features), dim=2)  # Usunięcie speaker_features

        # Przechodzenie przez warstwy
        x = F.relu(self.bn1(self.conv1(combined_features)) + speaker_embedding)
        x = F.relu(self.bn2(self.conv2(x)) + speaker_embedding)
        x = F.relu(self.bn3(self.upconv1(x)) + speaker_embedding)
        x = torch.tanh(self.upconv2(x))
        print(x.shape)
        x = self.vocoder.convert_spectrogram_to_audio(spec=x)
        return x


if __name__ == "__main__":
    # Parametry
    asr_dim = 129  # Zaktualizuj zgodnie z wymiarem wyjścia ASR encoder
    f0_dim = 129  # Zaktualizuj zgodnie z wymiarem wyjścia F0 encoder
    speaker_dim = 1 # Zaktualizuj zgodnie z wymiarem wyjścia Speaker Embedder
    hidden_dim = 256  # Możesz dostosować zgodnie z potrzebami
    output_dim = 1   # Wymiary wyjściowe spektrogramu, dostosuj zgodnie z potrzebami

    # Utworzenie instancji modelu
    generator = VoiceConversionGenerator(asr_dim, f0_dim, speaker_dim, hidden_dim, output_dim)

    # Testowanie modelu
    test_input_asr = torch.randn(1, asr_dim)         # Przykładowy tensor wejściowy dla ASR
    test_input_f0 = torch.randn(1, f0_dim)           # Przykładowy tensor wejściowy dla F0
    test_input_speaker = torch.randn(1, 1, speaker_dim, 1) # Przykładowy tensor wejściowy dla Speaker Embedder
    output = generator(test_input_asr, test_input_f0, test_input_speaker)
    print(output.shape)  # Sprawdzenie wymiarów wyjścia