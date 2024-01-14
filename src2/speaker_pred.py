import torch
import torchaudio
import nemo.collections.asr as nemo_asr
import librosa
from nvidiapred import nvidia_embedder
# Wczytaj model
speaker_model = nvidia_embedder.from_pretrained("nvidia/speakerverification_en_titanet_large")
speaker_model.eval()
for param in speaker_model.parameters():
    param.requires_grad = False
speaker_model = speaker_model.to('cuda')
path1 = '..//data//wavs_16khz//p236//p236_002_mic2.wav'
path2 = '..//data//wavs_16khz//p236//p236_003_mic1_1.wav'
audio1, sr = librosa.load(path1, sr=16000)
audio_length = audio1.shape[0]

# Przekształć dane audio na tensor PyTorch i przenieś na urządzenie
audio1 = torch.from_numpy(audio1).to('cuda')

# Zmień kształt tensora audio, aby miał dodatkowy wymiar
audio1 = audio1.unsqueeze(0)

# Przekształć długość sygnału audio na tensor PyTorch i przenieś na urządzenie
audio_length = torch.tensor([audio_length]).to('cuda')

audio2, sr = librosa.load(path2, sr=16000)  
audio_length2 = audio2.shape[0]
audio2 = torch.from_numpy(audio2).to('cuda')
audio2 = audio2.unsqueeze(0)
audio_length2 = torch.tensor([audio_length2]).to('cuda')
_, emb1 = speaker_model.forward(input_signal=audio1, input_signal_length=audio_length)

print(emb1)
_, emb2 = speaker_model.forward(input_signal=audio2, input_signal_length=audio_length2)
print(emb2)
result = speaker_model.verify_speakers(audio1, audio2,sr,audio_length, audio_length2)

print(result)

"""import torch
import torchaudio
import nemo.collections.asr as nemo_asr

# Wczytaj model
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

# Wczytaj audio za pomocą PyTorch
waveform1, sample_rate1 = torchaudio.load('..//data//wavs_22khz_5s//p226//p226_001_mic2.wav')
waveform2, sample_rate2 = torchaudio.load('..//data//wavs_22khz_5s//p225//p225_002_mic2.wav')
# Przenieś dane na GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
waveform1 = waveform1.to(device)
waveform2 = waveform2.to(device)
# Przetwórz audio tak, jak to robi model
audio1 = speaker_model.preprocessor(
    input_signal=waveform1,
    length=torch.tensor([waveform1.shape[1]]).to(waveform1.device)
)
audio2 = speaker_model.preprocessor(
    input_signal=waveform2,
    length=torch.tensor([waveform2.shape[1]]).to(waveform2.device)
)

# Przekaz audio do modelu
embeddings1 = speaker_model.encoder(audio_signal=audio1[0], length=audio1[1])
embeddings2 = speaker_model.encoder(audio_signal=audio2[0], length=audio2[1])
# Oblicz podobieństwo
similarity = torch.nn.functional.cosine_similarity(embeddings1[0], embeddings2[0])
print(similarity)
# instantiate pretrained model"""
"""from pyannote.audio import Model
model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")

from pyannote.audio import Inference
import torch
import numpy as np
inference = Inference(model, window="whole")
inference.to(torch.device("cuda"))
embedding1 = inference('..//data//wavs_22khz_5s//p227//p227_001_mic1.wav')
embedding2 = inference('..//data//wavs_22khz_5s//p227//p227_002_mic2.wav')
# `embeddingX` is (1 x D) numpy array extracted from the file as a whole.
embedding1 = embedding1[np.newaxis, :]
embedding2 = embedding2[np.newaxis, :]
from scipy.spatial.distance import cdist
distance = cdist(embedding1, embedding2, metric="cosine")[0,0]
print(distance)
# `distance` is a `float` describing how dissimilar speakers 1 and 2 are."""
"""import wespeaker

model = wespeaker.load_model('english')
similarity = model.compute_similarity('..//data//wavs_22khz_5s//p227//p227_005_mic2_1.wav', '..//data//wavs_22khz_5s//p227//p227_002_mic2.wav')
embedding = model.extract_embedding('..//data//wavs_22khz_5s//p227//p227_002_mic2.wav')
print(similarity)
print(embedding)"""