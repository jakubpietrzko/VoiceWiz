import nemo.collections.asr as nemo_asr
import torchaudio
import torch
import time
import os
class ASREncoder():
    def __init__(self, model_name="stt_en_squeezeformer_ctc_small_ls"):
        self.asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)
        #self.asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_pl_quartznet15x5")
        self.asr_model.encoder.register_forward_hook(self.capture_encoder_output)
        self.device = torch.device('cuda')

    def capture_encoder_output(self, module, input, output):
        self.encoder_output = output
    
    """def process_audio(self, audio_file):
        start_time = time.time()
        waveform, sample_rate = torchaudio.load(audio_file)
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
        mel_spectrogram = mel_transform(waveform)
        input_data = mel_spectrogram.mean(dim=1).to(self.device)
        length = torch.tensor([input_data.shape[1]]).to(self.device)
        end_time = time.time()
        print(f"melspektogram GPU: {end_time - start_time} sekund")
        start_time = time.time()
        output = self.asr_model(input_signal=input_data, input_signal_length=length)
        end_time = time.time()
        print(f"Czas obliczeń na GPU: {end_time - start_time} sekund")
        return self.encoder_output"""
    def process_audio(self, waveform, length):
        output = self.asr_model(input_signal=waveform, input_signal_length=length)
        return output
    
if __name__ == "__main__":
    asr_encoder = ASREncoder()
    asr_encoder.asr_model = asr_encoder.asr_model.to('cuda')
    asr_encoder.device = torch.device('cuda')
    device = torch.device('cuda')
    """# Przerzucanie modelu na GPU i mierzenie czasu
    if torch.cuda.is_available():
        asr_encoder.asr_model = asr_encoder.asr_model.to('cuda')

        output = asr_encoder.process_audio('gettysburg.wav')

        print(output)
    """
    # Przerzucanie modelu na CPU i mierzenie czasu
    """  asr_encoder.asr_model = asr_encoder.asr_model.to('cpu')
    asr_encoder.device = torch.device('cpu')"""
    
    audio_folder = '..\data\cv-corpus-15.0-delta-2023-09-08-en\cv-corpus-15.0-delta-2023-09-08\en\clips'  # ścieżka do folderu z plikami audio
    i=0
    waveforms = []
    sample_rates = []
    for audio_file in os.listdir(audio_folder):
        if i>=100:
            break
        i+=1
        audio_file = os.path.join(audio_folder, audio_file)
        waveform, sample_rate = torchaudio.load(audio_file)
        waveforms.append(waveform)
        sample_rates.append(sample_rate)

    waveforms = torch.stack(waveforms).to(device)  # przeniesienie wszystkich waveforms na raz na GPU

    mel_spectrograms = []
    lengths = []
    for waveform, sample_rate in zip(waveforms, sample_rates):
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate).to(device)
        mel_spectrogram = mel_transform(waveform)
        input_data = mel_spectrogram.mean(dim=1)
        length = torch.tensor([input_data.shape[1]], device=device)
        mel_spectrograms.append(input_data)
        lengths.append(length)
    
    start_time = time.time()
    outputs = []
   
    for mel_spectrogram, length in zip(mel_spectrograms, lengths):
        output = asr_encoder.process_audio(mel_spectrogram, length)
        outputs.append(output)
        i-=1
        print(i)
    
    end_time = time.time()
    print(f"Czas obliczeń na GPU: {end_time - start_time} sekund")