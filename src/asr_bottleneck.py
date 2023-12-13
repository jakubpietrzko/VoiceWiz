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
            output = self.asr_model(input_signal=waveform, input_signal_length=length)
        return output
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
    asr_encoder = ASREncoder()
    asr_encoder.asr_model = asr_encoder.asr_model.to('cuda')

    audio_folder = '..//data//parts6s'
    waveforms = prepare_dataset(audio_folder= audio_folder,i=0)

    start_time = time.time()
    outputs = []
    """for waveform in waveforms:
        waveform = waveform  # Dodanie wymiaru batch
        length = torch.tensor([waveform.shape[1]], device='cuda')
        output = asr_encoder.process_audio(waveform, length)
        outputs.append(output)
    torch.save(outputs, "outputs.pt")"""
    for waveform in waveforms:
        waveform = waveform  # Dodanie wymiaru batch
        length = torch.tensor([waveform.shape[1]], device='cuda')
        output = asr_encoder.process_audio(waveform, length)
        outputs.append(output[0])  # Zapisz tylko tensor wyjściowy

    torch.save(outputs, "outputs.pt")
    end_time = time.time()
    
    print(f"Czas obliczeń na GPU: {end_time - start_time} sekund")
    # Wczytaj wyjście enkodera
    # Wczytaj wyjście enkodera
    """encoder_outputs = torch.load('outputs.pt')

    # Przepuść wyjście enkodera przez resztę modelu i uzyskaj transkrypcje
    transcriptions = []
    for encoder_output in encoder_outputs:
        # Transponuj osie 'time' i 'dimension'
        encoder_output = encoder_output.transpose(1, 2)
        log_probs, _ = asr_encoder.asr_model.decoder(encoder_output=encoder_output)
        predictions = asr_encoder.asr_model._wer.ctc_decoder_predictions_tensor(log_probs)
        transcriptions.append(predictions)

    # Wyświetl transkrypcje
    for transcription in transcriptions:
        print(transcription)"""
