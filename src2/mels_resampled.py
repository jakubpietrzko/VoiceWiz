import os
import torch
import torchaudio
import concurrent.futures
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import os
import torchaudio

def create_melspectrogram(audio_folder, output_folder):
    # Przejrzyj wszystkie pliki w katalogu i podkatalogach
    for root, dirs, files in os.walk(audio_folder):
        for file in files:
            # Sprawdź, czy plik ma rozszerzenie .wav
            if file.endswith('.wav'):
                # Utwórz pełną ścieżkę do pliku audio
                audio_file = os.path.join(root, file)
                
                # Wczytaj plik audio
                signal, rate = torchaudio.load(audio_file)
                
                # Utwórz melspektrogram
                spectrogram, _ = mel_spectogram(
                    audio=signal.squeeze(),
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
                # Utwórz pełną ścieżkę do pliku wyjściowego
                output_file = os.path.join(output_folder, os.path.relpath(audio_file, audio_folder))
                output_file = os.path.splitext(output_file)[0] + ".pt"
                
                # Utwórz katalogi, jeśli nie istnieją
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Zapisz melspektrogram
                torch.save(spectrogram, output_file)

if __name__=="__main__":
    # Zdefiniuj folder źródłowy i docelowy
    source_folder = '..//data//wavs_16khz'
    target_folder = '..//data//wavs_16khz_mels'

    # Przejrzyj wszystkie pliki w folderze źródłowym
    create_melspectrogram(source_folder, target_folder)