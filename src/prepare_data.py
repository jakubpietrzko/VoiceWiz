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
                duration = len(y) / sr  # obliczenie długości pliku audio w sekundach
                durations.append(duration)
                
        if durations:
            # obliczenie statystyk
            average_duration = np.mean(durations)
            longest_duration = np.max(durations)
            shortest_duration = np.min(durations)
            print(f'Średnia długość: {average_duration} sekund')
            print(f'Najdłuższa długość: {longest_duration} sekund')
            print(f'Najkrótsza długość: {shortest_duration} sekund')
            # Oblicz liczby długości trwania spełniających określone warunki
            durations_equal_6 = [d for d in durations if d == 6.5]
            durations_less_than_6 = [d for d in durations if d < 6.5]
            durations_greater_than_6 = [d for d in durations if d > 6.5]

            # Wydrukuj wyniki
            print(f'Liczba długości trwania równych 6 sekund: {len(durations_equal_6)}')
            print(f'Liczba długości trwania krótszych niż 6 sekund: {len(durations_less_than_6)}')
            print(f'Liczba długości trwania dłuższych niż 6 sekund: {len(durations_greater_than_6)}')
            
        else:
            print('Brak plików audio do analizy.')
        
audio_folder = '..\\data\\cv-corpus-15.0-delta-2023-09-08-en\\cv-corpus-15.0-delta-2023-09-08\\en\\clips\\'  
output_folder = '..\\data\\wavs\\'
x=Prepare(output_folder, audio_folder)
#x.convert_mp3_wav()
x.audio_folder = output_folder
x.get_duration()
"""Średnia długość: 6.029381085011462 sekund
Najdłuższa długość: 153.936 sekund
Najkrótsza długość: 0.18 sekund
Liczba długości trwania równych 6 sekund: 0
Liczba długości trwania krótszych niż 6 sekund: 23604
Liczba długości trwania dłuższych niż 6 sekund: 16967"""