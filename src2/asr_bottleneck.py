import nemo.collections.asr as nemo_asr
import torchaudio
import torch
import time
import os
class ASREncoder():
    def __init__(self, model_name="stt_en_squeezeformer_ctc_small_ls"):
        self.asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)
        self.asr_model.encoder.register_forward_hook(self.capture_encoder_output)
        
         # Set the model to evaluation mode
    def capture_encoder_output(self, module, input, output):
        self.encoder_output = output

    def process_audio(self, waveform, length):
        self.asr_model = self.asr_model.eval() 
        with torch.no_grad():# Disable gradient calculation
            #waveform = waveform.squeeze(1)# Remove dimension with channel
            output = self.asr_model(input_signal=waveform, input_signal_length=length)
        return output[0]
def prepare_dataset(audio_folder,i):
        ys = []
        for audio_file in os.listdir(audio_folder):
            if audio_file.endswith('.wav'):
                if i>=10:
                    break
                i+=1
                audio_file_path = os.path.join(audio_folder, audio_file)
                y, _ = torchaudio.load(audio_file_path)
                ys.append(y)  # Przeniesienie tensora na GPU

        # Stacking list of tensors into a single tensor
        dataset = torch.stack(ys).to('cuda')
        return dataset
if __name__ == "__main__":
    asr_encoder = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_squeezeformer_ctc_small_ls")
    asr_encoder = asr_encoder.to('cuda')
    transcrition= asr_encoder.transcribe(paths2audio_files=['..//data//wavs_22khz_5s//p226//p226_001_mic2.wav'], batch_size=1)
    print(transcrition)
    help(nemo_asr.models.EncDecCTCModelBPE.transcribe)