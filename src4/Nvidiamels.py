import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
import nemo.collections.asr as nemo_asr
# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts", run_opts={"device":"cuda"})
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir="tmpdir_vocoder", run_opts={"device":"cuda"})
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_squeezeformer_ctc_small_ls")
# Running the TTS

asr_model = asr_model.to('cuda')
text=asr_model.transcribe(paths2audio_files=['..//data//wavs_16khz//s5//s5_002_mic2.wav'], batch_size=1)
print(text)
mel_output, mel_length, alignment = tacotron2.encode_text(text[0])

# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_output)

# Save the waverform
torchaudio.save('s5002.wav', waveforms.squeeze(1).cpu(), 22050)