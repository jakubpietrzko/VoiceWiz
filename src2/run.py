import torch
import torchaudio
import os
import numpy as np  
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
def process_audio_files(audio_folder, output_folder):
    for root, dirs, files in os.walk(audio_folder):
        for audio_file in files:
            if audio_file.endswith('.wav'):
                print(f'Przetwarzanie pliku {audio_file}...')
                audio_file_path = os.path.join(root, audio_file)
                y, sr = librosa.load(audio_file_path, sr=None)
                duration = len(y) / sr

                # Tworzenie odpowiednich katalogów w folderze wyjściowym
                relative_path = os.path.relpath(root, audio_folder)
                output_folder_path = os.path.join(output_folder, relative_path)
                os.makedirs(output_folder_path, exist_ok=True)

                if duration < 5 and duration > 2:
                    # Uzupełnij ciszą do 6 sekund
                    y_padded = np.pad(y, (0, int(5*sr - len(y))), 'constant')
                    output_file_path = os.path.join(output_folder_path, audio_file)
                    print(f'Zapisywanie pliku {output_file_path}...')
                    sf.write(output_file_path, y_padded, sr)
                elif duration > 5:
                    # Podziel na segmenty o długości 6 sekund
                    for i, start in enumerate(range(0, len(y), int(5*sr))):
                        segment = y[start:start+int(5*sr)]
                        output_file_name = f'{os.path.splitext(audio_file)[0]}_{i+1}.wav' if duration > 5 else audio_file
                        output_file_path = os.path.join(output_folder_path, output_file_name)
                        if len(segment) == int(5*sr):
                            # Zapisz 5-sekundowy segment
                            sf.write(output_file_path, segment, sr)
                        elif len(segment) / sr > 2:
                            # Uzupełnij ciszą do 5 sekund
                            segment_padded = np.pad(segment, (0, int(5*sr - len(segment))), 'constant')
                            sf.write(output_file_path, segment_padded, sr)
def display_melspectrograms_from_pt(pt_file_path1, pt_file_path2):
    # Ładowanie melspektrogramów
    mel_spectrogram1 = torch.load(pt_file_path1)
    mel_spectrogram2 = torch.load(pt_file_path2)

    # Obliczanie różnicy między melspektrogramami
    #mel_difference = mel_spectrogram1 - mel_spectrogram2

        # Wyświetlanie melspektrogramów
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    im1 = axs[0].imshow(mel_spectrogram1.detach().squeeze().numpy(), aspect='auto', origin='lower')
    axs[0].set_title('Melspectrogram 1')
    fig.colorbar(im1, ax=axs[0], format='%+2.0f dB')

    im2 = axs[1].imshow(mel_spectrogram2.detach().squeeze().numpy(), aspect='auto', origin='lower')
    axs[1].set_title('Melspectrogram 2')
    fig.colorbar(im2, ax=axs[1], format='%+2.0f dB')

    #im3 = axs[2].imshow(mel_difference.detach().squeeze().numpy(), aspect='auto', origin='lower')
    #axs[2].set_title('Difference')
    #fig.colorbar(im3, ax=axs[2], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()


# Przykładowe użycie:
#display_melspectrograms_from_pt('..//data//results//gen_output_mel.pt','..//data//results//source.pt' )
display_melspectrograms_from_pt('..//data//wavs_mels//p225//p225_001_mic2.pt','..//data//wavs_mels//p226//p226_001_mic2.pt' )
#process_audio_files('..//data//wavs_22khz', '..//data//wavs_22khz_5s')