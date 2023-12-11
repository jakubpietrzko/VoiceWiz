import os
import librosa
import soundfile as sf
import numpy as np
from scipy import stats

class Prepare():
    def __init__(self, output_folder, audio_folder):
        self.output_folder = output_folder
        self.audio_folder = audio_folder
    def convert_mp3_wav(self):
        for filename in os.listdir(self.audio_folder):
            if filename.endswith('.mp3'):
                full_path = os.path.join(self.audio_folder, filename)
                data, sr = librosa.load(full_path, sr=None)
                output_file = os.path.join(self.output_folder, os.path.splitext(filename)[0] + ".wav")
                sf.write(output_file, data, sr)
    def get_duration(self):
        durations = []
        for audio_file in os.listdir(self.audio_folder):
            if audio_file.endswith('.wav'):
                audio_file = os.path.join(self.audio_folder, audio_file)
                y, sr = librosa.load(audio_file, sr=None)
                duration = librosa.get_duration(y, sr)  # obliczenie długości pliku audio w sekundach
                durations.append(duration)
        if durations:
            # obliczenie statystyk
            average_duration = np.mean(durations)
            longest_duration = np.max(durations)
            shortest_duration = np.min(durations)
            dominant_duration = stats.mode(durations)[0][0]

            print(f'Średnia długość: {average_duration} sekund')
            print(f'Najdłuższa długość: {longest_duration} sekund')
            print(f'Najkrótsza długość: {shortest_duration} sekund')
            print(f'Dominująca długość: {dominant_duration} sekund')
        else:
            print('Brak plików audio do analizy.')
        
audio_folder = '..\\data\\cv-corpus-15.0-delta-2023-09-08-en\\cv-corpus-15.0-delta-2023-09-08\\en\\clips\\'  
output_folder = '..\\data\\wavs\\'
x=Prepare(output_folder, audio_folder)
x.convert_mp3_wav()