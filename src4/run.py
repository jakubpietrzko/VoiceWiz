import torch
import torchaudio
import os
import librosa
from speechbrain.pretrained import HIFIGAN
from nvidiapred import Nvidia_embedder
import numpy as np  
from model_v4 import VoiceConversionModel
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
def process_audio_files(audio_folder, output_folder):
    for root, dirs, files in os.walk(audio_folder):
        for audio_file in files:
            if audio_file.endswith('.wav'):
                print(f'Przetwarzanie pliku {audio_file}...')
                audio_file_path = os.path.join(root, audio_file)
                y, sr = librosa.load(audio_file_path, sr=16000)
                duration = len(y) / sr

                # Tworzenie odpowiednich katalogów w folderze wyjściowym
                relative_path = os.path.relpath(root, audio_folder)
                output_folder_path = os.path.join(output_folder, relative_path)
                os.makedirs(output_folder_path, exist_ok=True)

                if duration < 1.5 and duration > 1.3:
                    # Uzupełnij ciszą do 1.5 sekund
                    y_padded = np.pad(y, (0, int(1.5*sr - len(y))), 'constant')
                    output_file_path = os.path.join(output_folder_path, audio_file)
                    print(f'Zapisywanie pliku {output_file_path}...')
                    sf.write(output_file_path, y_padded, sr)
                elif duration > 1.5:
                    # Podziel na segmenty o długości 1.5 sekund
                    for i, start in enumerate(range(0, len(y), int(1.5*sr))):
                        segment = y[start:start+int(1.5*sr)]
                        output_file_name = f'{os.path.splitext(audio_file)[0]}_{i+1}.wav'
                        output_file_path = os.path.join(output_folder_path, output_file_name)
                        if len(segment) == int(1.5*sr):
                            # Zapisz 1.5-sekundowy segment
                            sf.write(output_file_path, segment, sr)
                        elif len(segment) / sr > 1.3:
                            # Uzupełnij ciszą do 1.5 sekund
                            segment_padded = np.pad(segment, (0, int(1.5*sr - len(segment))), 'constant')
                            sf.write(output_file_path, segment_padded, sr)
def display_melspectrograms_from_pt(pt_file_path1, pt_file_path2, pt_file_path3):
    # Ładowanie melspektrogramów
    mel_spectrogram1 = torch.load(pt_file_path1)
    mel_spectrogram2 = torch.load(pt_file_path2)
    mel_spectrogram3 = torch.load(pt_file_path3)
    # Obliczanie różnicy między melspektrogramami
    #mel_difference = mel_spectrogram1 - mel_spectrogram2

        # Wyświetlanie melspektrogramów
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axs[0].imshow(mel_spectrogram1.detach().squeeze().numpy(), aspect='auto', origin='lower')
    axs[0].set_title('Melspectrogram generated')
    fig.colorbar(im1, ax=axs[0], format='%+2.0f dB')

    im2 = axs[1].imshow(mel_spectrogram2.detach().squeeze().numpy(), aspect='auto', origin='lower')
    axs[1].set_title('Melspectrogram source')
    fig.colorbar(im2, ax=axs[1], format='%+2.0f dB')

    im3 = axs[2].imshow(mel_spectrogram3.detach().squeeze().numpy(), aspect='auto', origin='lower')
    axs[2].set_title('Melspectrogram goal')
    fig.colorbar(im3, ax=axs[2], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()
    
def run(mel_path,dest_path):
    device = torch.device('cuda')
    xd = VoiceConversionModel(device)
    xd =xd.to(device)
    state_dict = torch.load("..//best_model_one_one_nv4.pth")
    xd.load_state_dict(state_dict, strict=False)
    mel = torch.load(mel_path)
    gosl_mel = torch.load('..//data//mels//p239//p239_020_mic1_2.pt')
    mel= mel.unsqueeze(0).to(device)
    gosl_mel = gosl_mel.unsqueeze(0).to(device)
    new_voice=xd.generate(mel, gosl_mel)
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="vocoder_16khz" )
    vocoder.eval()
    for param in vocoder.parameters():
        param.requires_grad = False
    print(new_voice.shape)
    new_voice = vocoder.decode_batch(new_voice.squeeze(1))
    new_voice=new_voice.squeeze(1) 
    print(new_voice.shape)
    
    torchaudio.save(dest_path, new_voice[0].cpu().unsqueeze(0), 16000, format="WAV")
def save_wavs():
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="vocoder_16khz" )
    vocoder.eval()
    for param in vocoder.parameters():
        param.requires_grad = False
        
    generated=torch.load('..//data//results//gen_output_mel.pt')
    source = torch.load('..//data//results//source.pt')
    goal = torch.load('..//data//results//goal.pt')
    generated = vocoder.decode_batch(generated.squeeze(1))
    generated = generated.squeeze(1)
                #iarów do kształtu każdego tensora
    torchaudio.save('..//data//results//generated.wav', generated[0].cpu().unsqueeze(0), 16000, format="WAV")
    source = vocoder.decode_batch(source.squeeze(1))
    source = source.squeeze(1)
    torchaudio.save('..//data//results//source.wav', source[0].cpu().unsqueeze(0), 16000, format="WAV")   
    goal = vocoder.decode_batch(goal.squeeze(1))
    goal = goal.squeeze(1)
    torchaudio.save('..//data//results//goal.wav', goal[0].cpu().unsqueeze(0), 16000, format="WAV")
    
def test_pred(a, b):
    xd =Nvidia_embedder().from_pretrained("nvidia/speakerverification_en_titanet_large")
    a,sr = librosa.load(a, sr=16000)
    a = xd.get_embedding(a, sr)
    b,sr = librosa.load(b, sr=16000)
    b = xd.get_embedding(b, sr)
    
    
    ver = xd.verify_speakers(a,b)
    print(ver)
# Przykładowe użycie:
display_melspectrograms_from_pt('..//data//results//gen_output_mel.pt','..//data//results//source.pt','..//data//results//goal.pt')
#display_melspectrograms_from_pt('..//data//mels//p304//p304_002_mic1_2.pt','..//data//mels//s5//s5_002_mic2_2.pt',  '..//data//mels//p374//p374_002_mic1_2.pt')
#process_audio_files('..//data//wavs_16khz', '..//data//wavs_16khz_parts_1')
save_wavs()
run('..//data//mels//p238//p238_002_mic2_2.pt','..//data//results//p238_002_mic2_2.wav')
run('..//data//mels//p238//p238_003_mic2_2.pt','..//data//results//p238_003_mic2_2.wav')
run('..//data//mels//p238//p238_004_mic2_2.pt','..//data//results//p238_004_mic2_2.wav')
run('..//data//mels//p238//p238_005_mic2_2.pt','..//data//results//p238_005_mic2_2.wav')
run('..//data//mels//p238//p238_006_mic2_2.pt','..//data//results//p238_006_mic2_2.wav')
run('..//data//mels//p238//p238_007_mic2_2.pt','..//data//results//p238_007_mic2_2.wav')
run('..//data//mels//p238//p238_008_mic2_2.pt','..//data//results//p238_008_mic2_2.wav')
run('..//data//mels//p238//p238_009_mic2_2.pt','..//data//results//p238_009_mic2_2.wav')
run('..//data//mels//p238//p238_020_mic2_2.pt','..//data//results//p238_020_mic2_2.wav')
#test_pred('..//data//results//source.wav','..//data//results//goal.wav')