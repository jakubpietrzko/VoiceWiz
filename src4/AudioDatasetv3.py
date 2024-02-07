import os
import random
import torch
import librosa
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, wav_directory, mel_directory, num_samples=1600):
        self.wav_directory = wav_directory
        self.mel_directory = mel_directory
        self.speaker_dir = 'p238'  # Stała dla mówcy, z którego p238 p239zawsze wybieramy waveform2118
        self.other_speaker_dir = 'p239'  # Stała dla mówcy, z którego zawsze wybieramy other_waveform
        self.data = [f for f in os.listdir(os.path.join(wav_directory, self.speaker_dir)) if f.endswith('.wav')]
        self.other_data = [f for f in os.listdir(os.path.join(wav_directory, self.other_speaker_dir)) if f.endswith('.wav')]
        self.data = random.sample(self.data, num_samples)
        self.other_data = random.sample(self.other_data, num_samples)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        wav_file = self.data[idx]
     
        mel_path = os.path.join(self.mel_directory, self.speaker_dir, wav_file.replace('.wav', '.pt'))
     
        mel_spectrogram = torch.load(mel_path)

        # Wybór pliku dźwiękowego od wybranego mówcy
        other_wav_file = self.other_data[idx]
      
  

        # Ładowanie mel-spektrogramu dla other_waveform
        other_mel_path = os.path.join(self.mel_directory, self.other_speaker_dir, other_wav_file.replace('.wav', '.pt'))
        other_mel_spectrogram = torch.load(other_mel_path)

        return mel_spectrogram,   other_mel_spectrogram