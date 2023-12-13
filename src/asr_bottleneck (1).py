import nemo.collections.asr as nemo_asr
import torchaudio
import torch
import time
import os

class ASREncoder():
    def __init__(self, model_name="stt_en_squeezeformer_ctc_small_ls"):
        self.asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)
        self.asr_model.encoder.register_forward_hook(self.capture_encoder_output)
        self.device = torch.device('cuda')

    def capture_encoder_output(self, module, input, output):
        self.encoder_output = output

    def process_audio(self, waveform, length):
        self.asr_model = self.asr_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            output = self.asr_model(input_signal=waveform, input_signal_length=length)  # Add an extra dimension for batch
        return output

if __name__ == "__main__":
    asr_encoder = ASREncoder()
    asr_encoder.asr_model = asr_encoder.asr_model.to('cuda')
    asr_encoder.device = torch.device('cuda')
    device = torch.device('cuda')

    audio_folder = '..\\data\\parts6s'  # ścieżka do folderu z plikami audio
    i=0
    waveforms = []
    sample_rates = []
    for audio_file in os.listdir(audio_folder):
        if i>=10:
            break
        i+=1
        audio_file = os.path.join(audio_folder, audio_file)
        waveform, sample_rate = torchaudio.load(audio_file)
        waveforms.append(waveform)
        sample_rates.append(sample_rate)

    waveforms = torch.stack(waveforms).to(device)  # przeniesienie wszystkich waveforms na raz na GPU
    
    """
    mel_spectrograms = []
    lengths = []
    for waveform, sample_rate in zip(waveforms, sample_rates):
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate).to(device)
        mel_spectrogram = mel_transform(waveform)
        input_data = mel_spectrogram.mean(dim=1)
        #input_data = mel_spectrogram
        length = torch.tensor([input_data.shape[1]], device=device)
        mel_spectrograms.append(input_data)
        lengths.append(length)"""
    mel_spectrograms = []
    lengths = []
    for waveform, sample_rate in zip(waveforms, sample_rates):
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate).to(device)
        mel_spectrogram = mel_transform(waveform)
        input_data = mel_spectrogram
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
    with open("output.txt", "w") as f:
        for output in outputs:
            f.write(f"{output}\n")
          