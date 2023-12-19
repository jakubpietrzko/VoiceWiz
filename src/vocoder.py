import torch
# Load UnivNet
from nemo.collections.tts.models import UnivNetModel

available_models = UnivNetModel.list_available_models()
print(available_models)

model = UnivNetModel.from_pretrained(model_name="tts_en_libritts_univnet")

spectogram = torch.load("..\\data\\mels\\common_voice_en_38025198.pt").to(model.device)
print(spectogram.shape)
audio = model.convert_spectrogram_to_audio(spec=spectogram)

import torchaudio

# Zapisz audio do pliku o nazwie speech.wav
torchaudio.save('speech1.wav', audio.detach().cpu(), sample_rate=22050)