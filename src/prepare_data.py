import os
import librosa
import soundfile as sf
import numpy as np
from scipy import stats
import torch
import torchaudio
import mels
import f0_utils
import shutil
import matplotlib.pyplot as plt
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
                print(f'Przetwarzanie pliku {audio_file}...')
                audio_file_path = os.path.join(self.audio_folder, audio_file)
                y, sr = librosa.load(audio_file_path, sr=None)
                duration = len(y) / sr

                if duration < 6 and duration > 4:
                    # Uzupełnij ciszą do 6 sekund
                    y_padded = np.pad(y, (0, int(6*sr - len(y))), 'constant')
                    output_file_path = os.path.join(self.output_folder, audio_file)
                    print(f'Zapisywanie pliku {output_file_path}...')
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
    def create_melspectrogram(self):
        for file in os.listdir(self.audio_folder):                 
            if file.endswith('.wav'):
                audio_file = os.path.join(self.audio_folder, file)
                y, sr = librosa.load(audio_file, sr=None)
                audio_params = mels.AudioFeaturesParams()
                audio = mels.load_and_preprocess_audio(audio_file, audio_params.sampling_rate)
                mel_spec = mels.mel_spectrogram(audio, audio_params)
                output_file = os.path.join(self.output_folder, os.path.splitext(file)[0] + ".pt")
                torch.save((mel_spec), output_file)

    #to przenosi po porstu polowe folderu do goal, mozna wybrac tych ktorych mamy wiecej probek potem przneisc tylko
    def split_goaset_source_set(self, new_folder):
        files = [file for file in os.listdir(self.audio_folder) if file.endswith('.pt')]
        half = len(files) // 2

        for file in files[:half]:
            src_file = os.path.join(self.audio_folder, file)
            dst_file = os.path.join(new_folder, file)
            shutil.move(src_file, dst_file)
                
    def create_f0(self):
        for file in os.listdir(self.audio_folder):                 
            if file.endswith('.wav'):
                audio_file = os.path.join(self.audio_folder, file)
                lf0_tensor = f0_utils.get_lf0_from_wav(audio_file)
                output_file = os.path.join(self.output_folder, os.path.splitext(file)[0] + ".pt")
                torch.save(lf0_tensor, output_file)    
                
    def prepare_dataset_split(audio_folder: str, split_ratio: float = 0.8):

        all_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')]

    # Mieszanie plików
        random.shuffle(all_files)

    # Podział na zbiory
        split_index = int(len(all_files) * split_ratio)
        train_files = all_files[:split_index]
        test_files = all_files[split_index:]

        return train_files, test_files

def display_melspectrogram_from_pt(file_path):
    mel_spectrogram = torch.load(file_path)  # Wczytaj melspektrogram i długość z pliku .pt
    mel_spectrogram = mel_spectrogram.squeeze(0)  # Usuń wymiary o rozmiarze 1
    print(mel_spectrogram.shape)
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.title('Melspectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()        
if __name__ == "__main__":
    audio_folder = '..\\data\\cv-corpus-15.0-delta-2023-09-08-en\\cv-corpus-15.0-delta-2023-09-08\\en\\clips\\'  
    output_folder = '..\\data\\wavs\\'
    output_folder1 = '..\\data\\parts6s\\'
    output_folder2 = '..\\data\\fzeros\\'
    output_folder3 = '..\\data\\mels\\'
    x=Prepare(output_folder3, output_folder1)
    #x.convert_mp3_wav()
    #x.audio_folder = output_folder
    #x.output_folder = output_folder1
    #x.get_duration()
    """Średnia długość: 6.029381085011462 sekund
    Najdłuższa długość: 153.936 sekund
    Najkrótsza długość: 0.18 sekund
    Liczba długości trwania równych 6 sekund: 0
    Liczba długości trwania krótszych niż 6 sekund: 23604
    Liczba długości trwania dłuższych niż 6 sekund: 16967"""
    #x.create_melspectrogram()
    file_path='..\\data\\mels\\common_voice_en_38024637.pt'
    display_melspectrogram_from_pt(file_path)