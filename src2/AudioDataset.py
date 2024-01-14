import os
import random
import torch
import librosa
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, wav_directory, mel_directory, num_samples=5000):
        self.wav_directory = wav_directory
        self.mel_directory = mel_directory
        self.speaker_dirs = os.listdir(wav_directory)
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

        # Przekształć dane audio na tensor PyTorch i przenieś na urządzenie
        waveform = torch.from_numpy(waveform)

        # Zmień kształt tensora audio, aby miał dodatkowy wymiar
     
        # Przekształć długość sygnału audio na tensor PyTorch i przenieś na urządzenie
        waveform_length = torch.tensor([waveform_length])
        
        mel_spectrogram = torch.load(mel_path)
        #print(mel_spectrogram.shape)
        # Losowy wybór innego mówcy
        other_speaker_dir = random.choice([d for d in self.speaker_dirs if d != speaker_dir])
        other_wav_file = random.choice([f for f in os.listdir(os.path.join(self.wav_directory, other_speaker_dir)) if f.endswith('.wav')])
        other_wav_path = os.path.join(self.wav_directory, other_speaker_dir, other_wav_file)
        other_waveform, _ = librosa.load(other_wav_path, sr=16000)
        other_waveform_length = other_waveform.shape[0]
        other_waveform = torch.from_numpy(other_waveform)
        
        other_waveform_length = torch.tensor([other_waveform_length])

        #other_waveform = torch.from_numpy(other_waveform)
        return waveform, mel_spectrogram, other_waveform, waveform_length, other_waveform_length
from torch.utils.data import DataLoader

if __name__ == "__main__":
        
    # Utwórz instancję klasy CustomAudioDataset
    dataset = AudioDataset(wav_directory='..\\data\\wavs_22khz_5s', mel_directory='..\\data\\wavs_mels')

    # Utwórz DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Sprawdź, czy CUDA jest dostępne
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Iteruj przez dane
    for batch in dataloader:
        waveform_batch, mel_spectrogram_batch, other_waveform_batch, wave_len, oth_wav_len = batch

        # Przenieś dane na GPU
        waveform_batch = waveform_batch.to(device)
        mel_spectrogram_batch = mel_spectrogram_batch.to(device)
        other_waveform_batch = other_waveform_batch.to(device)
        wave_len = wave_len.to(device)
        oth_wav_len = oth_wav_len.to(device)
        # Tutaj możesz przetworzyć partię danych
        print(f'waveform_batch shape: {waveform_batch.shape}')
        print(f'mel_spectrogram_batch shape: {mel_spectrogram_batch.shape}')
        print(f'other_waveform_batch shape: {other_waveform_batch.shape}')
        