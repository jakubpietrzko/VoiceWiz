import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

class AudioDatasetRAW(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')][:5000]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.audio_files[idx])
        return waveform

class AudioDatasetMEL(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.spectrogram_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')][:5000]
    def __len__(self):
        return len(self.spectrogram_files)

    def __getitem__(self, idx):
        spectrogram = torch.load(self.spectrogram_files[idx])
        return spectrogram