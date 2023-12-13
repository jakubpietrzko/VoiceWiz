import torchaudio
import os
audio_folder = '..//data//parts6s'  # Ścieżka do folderu z plikami audio

for audio_file in os.listdir(audio_folder):
    if audio_file.endswith('.wav'):
        audio_file_path = os.path.join(audio_folder, audio_file)
        waveform, sr = torchaudio.load(audio_file_path)
        if sr != 32000:
            print(f"Plik {audio_file} ma częstotliwość próbkowania {sr}, a nie 32000.")