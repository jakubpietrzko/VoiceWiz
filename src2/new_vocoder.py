import torchaudio
from speechbrain.pretrained import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import time
# Load a pretrained HIFIGAN Vocoder
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="vocoder_16khz")

# Load an audio file (an example file can be found in this repository)
# Ensure that the audio signal is sampled at 16000 Hz; refer to the provided link for a 22050 Hz Vocoder.
#signal, rate = torchaudio.load('speechbrain/tts-hifigan-libritts-16kHz/example_16kHz.wav')
signal, rate = torchaudio.load('..//data//parts6s_resampled//common_voice_en_38024636_1.wav')
if rate != 16000:
    # Resample the waveform to 16kHz
    resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
    signal = resampler(signal)

# Ensure the audio is sigle channel
#signal = signal[0].squeeze()

#torchaudio.save('waveform.wav', signal.unsqueeze(0), 16000)

# Compute the mel spectrogram.
# IMPORTANT: Use these specific parameters to match the Vocoder's training settings for optimal results.
print(signal.shape)
start=time.time()
spectrogram, _ = mel_spectogram(
    audio=signal.squeeze(),
    sample_rate=16000,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    n_fft=1024,
    f_min=0.0,
    f_max=8000.0,
    power=1,
    normalized=False,
    min_max_energy_norm=True,
    norm="slaney",
    mel_scale="slaney",
    compression=True
)
stop = time.time()
print('time:', stop-start)
# Convert the spectrogram to waveform
print(spectrogram.shape)
start= time.time()
waveforms = hifi_gan.decode_batch(spectrogram)
stop = time.time()
print('time:', stop-start)
# Save the reconstructed audio as a waveform
print(waveforms.shape)

torchaudio.save('waveform_reconstructed.wav', waveforms.squeeze(1), 16000)

# If everything is set up correctly, the original and reconstructed audio should be nearly indistinguishable
