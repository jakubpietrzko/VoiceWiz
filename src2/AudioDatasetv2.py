import os
import random
import torch
import librosa
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, wav_directory, mel_directory, num_samples=822):
        self.wav_directory = wav_directory
        self.mel_directory = mel_directory
        self.speaker_dirs = os.listdir(wav_directory)
        self.other_speaker_dir = 's5'  # Stała dla wybranego mówcy
        if self.other_speaker_dir in self.speaker_dirs:
            self.speaker_dirs.remove(self.other_speaker_dir)  # Usuń wybranego mówcę z listy
        self.data = []
        for speaker_dir in self.speaker_dirs:
            wav_files = [f for f in os.listdir(os.path.join(wav_directory, speaker_dir)) if f.endswith('.wav')]
            for wav_file in wav_files:
                self.data.append((speaker_dir, wav_file))
        self.data = random.sample(self.data, num_samples)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        speaker_dir, wav_file = self.data[idx]
        wav_path = os.path.join(self.wav_directory, speaker_dir, wav_file)
        mel_path = os.path.join(self.mel_directory, speaker_dir, wav_file.replace('.wav', '.pt'))

        waveform, _ = librosa.load(wav_path, sr=16000)
        waveform_length = waveform.shape[0]

        waveform = torch.from_numpy(waveform)
        waveform_length = torch.tensor([waveform_length])
        
        mel_spectrogram = torch.load(mel_path)

        # Losowy wybór pliku dźwiękowego od wybranego mówcy
        other_wav_file = random.choice([f for f in os.listdir(os.path.join(self.wav_directory, self.other_speaker_dir)) if f.endswith('.wav')])
        other_wav_path = os.path.join(self.wav_directory, self.other_speaker_dir, other_wav_file)
        other_waveform, _ = librosa.load(other_wav_path, sr=16000)
        other_waveform_length = other_waveform.shape[0]
        other_waveform = torch.from_numpy(other_waveform)
        
        other_waveform_length = torch.tensor([other_waveform_length])

        # Ładowanie mel-spektrogramu dla other_waveform
        other_mel_path = os.path.join(self.mel_directory, self.other_speaker_dir, other_wav_file.replace('.wav', '.pt'))
        other_mel_spectrogram = torch.load(other_mel_path)

        return waveform, mel_spectrogram, other_waveform, waveform_length, other_waveform_length, other_mel_spectrogram