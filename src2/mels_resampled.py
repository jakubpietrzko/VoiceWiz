import os
import torch
import torchaudio
import concurrent.futures
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
def create_melspectrogram(audio_folder, output_folder):

        for file in os.listdir(audio_folder):
            if file.endswith('.wav'):
                audio_file = os.path.join(audio_folder, file)
                signal, rate = torchaudio.load(os.path.join(audio_folder, file))
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
                output_file = os.path.join(output_folder, os.path.splitext(file)[0] + ".pt")
                torch.save(spectrogram, output_file)

if __name__=="__main__":
# Zdefiniuj folder źródłowy i docelowy
    source_folder = '..//data//parts6s_resampled'
    target_folder = '..//data//parts6s_mel'

    # Przejrzyj wszystkie pliki w folderze źródłowym
    create_melspectrogram(source_folder, target_folder)