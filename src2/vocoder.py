import torch
# Load UnivNet
from nemo.collections.tts.models import UnivNetModel

available_models = UnivNetModel.list_available_models()
print(available_models)

model = UnivNetModel.from_pretrained(model_name="tts_en_libritts_univnet")

spectogram = torch.load("..\\data\\mels\\common_voice_en_38396397.pt").to(model.device)
print(spectogram.shape)
audio = model.convert_spectrogram_to_audio(spec=spectogram)
print(audio.shape)
import torchaudio

# Zapisz audio do pliku o nazwie speech.wav
torchaudio.save('speech1.wav', audio.detach().cpu(), sample_rate=22050)
waveform, sample_rate = torchaudio.load('speech1.wav')

# Resample the audio to 32000 Hz
resampler = torchaudio.transforms.Resample(orig_freq=22050, new_freq=32000)
waveform_resampled = resampler(waveform)

# Save the resampled audio to a file
torchaudio.save('speech1_resampled.wav', waveform_resampled, sample_rate=32000)
print(waveform_resampled.shape)
waveform, sample_rate = torchaudio.load('..//data//parts6s//common_voice_en_38024635.wav')
print(waveform.shape, sample_rate)