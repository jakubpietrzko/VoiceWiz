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
        self.conv_asr = nn.Conv1d(asr_dim, output_dim, kernel_size=3)  # Konwersja cech ASR
        self.conv_f0 = nn.Conv1d(f0_dim, output_dim, kernel_size=3)  # Konwersja cech F0
        self.fc_speaker = nn.Linear(speaker_dim, output_dim//2)
        # Warstwa liniowa dla osadzenia mówcy

        # Warstwy ukryte i upsampling
        self.conv1 = nn.Conv1d(output_dim, output_dim // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(output_dim // 2)
        self.conv2 = nn.Conv1d(output_dim // 2, output_dim // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(output_dim // 2)
        self.upconv1 = nn.ConvTranspose1d(output_dim // 2, output_dim // 4, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(output_dim // 4)
        self.upconv2 = nn.ConvTranspose1d(output_dim // 4, 80, kernel_size=4, stride=1, padding=1)


    def forward(self, asr_features, f0_features, speaker_features):
        # Przetwarzanie danych wejściowych
        asr_features = asr_features.permute(0,2,1)
        asr_features = self.conv_asr(asr_features)  # Przekształcenie na 2D
        f0_features = self.conv_f0(f0_features)
        
        # Łączenie cech z osadzeniem mówcy
        combined_features = torch.cat((asr_features, f0_features), dim=2)
             
        # Przechodzenie przez warstwy
        x = self.bn1(self.conv1(combined_features))
        input_dims = x.size()
        speaker_embedding = self.fc_speaker(speaker_features)
        # Rozszerz wymiary tensora speaker embeddera do wymiarów tensora wejściowego
        expanded_speaker_embedding = speaker_embedding.unsqueeze(2).expand(-1, -1, input_dims[2])
           
        x = x + expanded_speaker_embedding  # Integracja osadzenia mówcy
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        input_dims = x.size()
        speaker_embedding = self.fc_speaker(speaker_features)
        # Rozszerz wymiary tensora speaker embeddera do wymiarów tensora wejściowego
        expanded_speaker_embedding = speaker_embedding.unsqueeze(2).expand(-1, -1, input_dims[2])
        
        x = x + expanded_speaker_embedding  # Ponowna integracja osadzenia mówcy   
        x = F.relu(x)
        x = self.bn3(self.upconv1(x)) 
        x = torch.tanh(self.upconv2(x))     
        #print("przed melem")
        # Konwersja na melspektrogram i przekazanie do vocodera
        mel_spectrogram = x # Dopasowanie wymiarów do vocodera 
        self.vocoder = self.vocoder.eval()
        """print("przed vocoderem")
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0) 
        a = torch.cuda.memory_allocated(0)
        f = t-a  # free inside reserved
        print(f"Total memory przed mel: {t/1024**3}")
        print(f"Reserved memory: {r/1024**3}")
        print(f"Allocated memory: {a/1024**3}")
        print(f"Free inside reserved: {f/1024**3}")       """
        torch.cuda.empty_cache()    
        #print("rozmiar przed vocoderem",mel_spectrogram.shape)
        audio = self.vocoder.convert_spectrogram_to_audio(spec=mel_spectrogram)
        #print("koniec generatora")
        return audio

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
    output = generator(test_input_asr, test_input_f0, test_input_speaker)
    print(output.shape)  # Sprawdzenie wymiarów wyjścia