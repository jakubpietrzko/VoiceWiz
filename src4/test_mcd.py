import torch
import os
import random
import torch
import librosa
import matplotlib.pyplot as plt

from pathlib import Path
import torchaudio.transforms as T
import torchaudio
def mel_cepstral_distortion(mfcc1, mfcc2): # Assuming mfcc1 and mfcc2 are the MFCCs of two audio signals 
        diff = mfcc1 - mfcc2 
        mcd = torch.sqrt(torch.sum(diff ** 2, dim=-1)).mean() 
        return mcd   
def spectrogram_to_mfcc(spectrogram, sample_rate=16000, n_mfcc=13): 
        mfcc_transform = T.MFCC( sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False} ) 
        mfcc = mfcc_transform(spectrogram)
        return mfcc
def LRec(Msource, Mpred): 
        return torch.abs(Msource - Mpred).mean()
x = torch.load("..//data//wavs_16khz_mels//p225//p225_003_mic2_1.pt")
y = torch.load("..//data//wavs_16khz_mels//p225//p225_004_mic2.pt")
print(LRec(x,y))
"""waveform, _ = librosa.load("..//data//wavs_16khz//p225//p225_001_mic2.wav", sr=16000)
waveform_length = waveform.shape[0]

waveform = torch.from_numpy(waveform)
waveform_length = torch.tensor([waveform_length])
waveform2, _ = librosa.load("..//data//wavs_16khz//p225//p225_008_mic2_1.wav", sr=16000)
waveform_length2 = waveform2.shape[0]

waveform2 = torch.from_numpy(waveform2)
waveform_length2 = torch.tensor([waveform_length2])
         
mcc =  spectrogram_to_mfcc(waveform)
mcc1 =  spectrogram_to_mfcc(waveform2)
plt.figure(figsize=(10, 4))
plt.imshow(mcc, cmap='inferno', origin='lower')
plt.title('MFCC')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()"""
#print(mel_cepstral_distortion(mcc,mcc1))