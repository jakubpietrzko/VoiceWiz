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
            durations_equal_6 = [d for d in durations if d == 6]
            durations_less_than_6 = [d for d in durations if d < 6]
            durations_greater_than_6 = [d for d in durations if d > 6]

            # Wydrukuj wyniki
            print(f'Liczba długości trwania równych 6 sekund: {len(durations_equal_6)}')
            print(f'Liczba długości trwania krótszych niż 6 sekund: {len(durations_less_than_6)}')
            print(f'Liczba długości trwania dłuższych niż 6 sekund: {len(durations_greater_than_6)}')
            
        else:
            print('Brak plików audio do analizy.')
    def process_audio_files(self):
        for audio_file in os.listdir(self.audio_folder):
            if audio_file.endswith('.wav'):
                audio_file_path = os.path.join(self.audio_folder, audio_file)
                y, sr = librosa.load(audio_file_path, sr=None)
                duration = len(y) / sr

                if duration < 6:
                    # Uzupełnij ciszą do 6 sekund
                    y_padded = np.pad(y, (0, int(6*sr - len(y))), 'constant')
                    output_file_path = os.path.join(self.output_folder, audio_file)
                    sf.write(output_file_path, y_padded, sr)
                elif duration > 6:
                    # Podziel na segmenty o długości 6 sekund
                    for i, start in enumerate(range(0, len(y), int(6*sr))):
                        segment = y[start:start+int(6*sr)]
                        output_file_name = f'{os.path.splitext(audio_file)[0]}_{i+1}.wav' if duration > 6 else audio_file
                        output_file_path = os.path.join(self.output_folder, output_file_name)
                        if len(segment) == int(6*sr):
                            # Zapisz 6-sekundowy segment
                            sf.write(output_file_path, segment, sr)
                        elif len(segment) / sr > 4:
                            # Uzupełnij ciszą do 6 sekund
                            segment_padded = np.pad(segment, (0, int(6*sr - len(segment))), 'constant')
                            sf.write(output_file_path, segment_padded, sr)         
                                          
audio_folder = '..\\data\\cv-corpus-15.0-delta-2023-09-08-en\\cv-corpus-15.0-delta-2023-09-08\\en\\clips\\'  
output_folder = '..\\data\\wavs\\'
output_folder1 = '..\\data\\parts6s\\'
x=Prepare(output_folder, audio_folder)
#x.convert_mp3_wav()
x.audio_folder = output_folder
x.output_folder = output_folder1
#x.get_duration()
"""Średnia długość: 6.029381085011462 sekund
Najdłuższa długość: 153.936 sekund
Najkrótsza długość: 0.18 sekund
Liczba długości trwania równych 6 sekund: 0
Liczba długości trwania krótszych niż 6 sekund: 23604
Liczba długości trwania dłuższych niż 6 sekund: 16967"""
x.process_audio_files()