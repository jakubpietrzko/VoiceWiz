import torch
from torch import nn
import torchaudio
from asr_bottleneck import ASREncoder
from f0_encoder import F0Encoder
from speaker_embedder import SpeakerEmbedder
from discriminator import Discriminator
from generator import Generator
from f0_utils import get_lf0_from_wav
import numpy as np
import os

class VoiceConversionModel(nn.Module):
    def __init__(self, device):
        super(VoiceConversionModel, self).__init__()
        self.device = device
        self.asr_encoder = ASREncoder().to(device)
        self.f0_encoder = F0Encoder().to(device)
        self.speaker_embedder = SpeakerEmbedder(in_features=128, num_residual_layers=5).to(device)
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)

            # Zamroź parametry ASR
        for param in self.asr_encoder.parameters():
            param.requires_grad = False
        # Zdefiniuj optymalizatory
        self.optimizer_f0 = torch.optim.Adam(self.f0_encoder.parameters(), lr=0.001)
        self.optimizer_speaker = torch.optim.Adam(self.speaker_embedder.parameters(), lr=0.001)
        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

    def async_data_preparation(self, data_queue, dataloader):
        for batch in dataloader:
            # Przetwarzanie danych na CPU
            processed_data = self.preprocess_data(batch)
            data_queue.put(processed_data)
    
    def forward(self, x, y):
        asr_features = self.asr_encoder(x)
        f0_features = self.f0_encoder(x)
        speaker_embedding = self.speaker_embedder(x)
        gen_output = self.generator(asr_features, f0_features, speaker_embedding)
        disc_output = self.discriminator(gen_output)
        return gen_output, disc_output
    
    def LRec(Msource, Mpred):
        return torch.abs(Msource - Mpred).sum()

    def LAdvP(Ds):
        return (Ds- 1)**2

    def LAdvD(D_fake, D_org):
        return (D_org- 1)**2 + (D_fake)**2

    def LFM(x, s):
        outputs_real = x
        outputs_fake = s
        loss = 0
        for out_real, out_fake in zip(outputs_real, outputs_fake):
            loss += torch.abs(out_real - out_fake).mean()
        return loss

    def LSpk(speaker_embedding):
        # Tworzymy dwa rozkłady normalne
        dist_pred = torch.distributions.Normal(speaker_embedding, torch.ones_like(speaker_embedding))
        dist_zero = torch.distributions.Normal(torch.zeros_like(speaker_embedding), torch.ones_like(speaker_embedding))

        # Obliczamy dywergencję KL
        return torch.distributions.kl_divergence(dist_pred, dist_zero).sum()
    def train_step(self, x, y):
        # Przejdź do przodu przez speaker_embedder, asr_encoder i f0_encoder
        speaker_embedding = self.speaker_embedder(y)
        asr_features = self.asr_encoder(x)
        f0_features = self.f0_encoder(x)

        # Przejdź do przodu przez generator
        gen_output = self.generator(asr_features, f0_features, speaker_embedding)
        # Utwórz transformację MelSpectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)

        # Przekształć surowy dźwięk na melspektrogram
        gen_output_mel = mel_transform(gen_output)
        # Przejdź do przodu przez discriminator z prawdziwą próbką
        disc_output_real = self.discriminator(y)
        disc_output_fake = self.discriminator(gen_output_mel)
        # oblicz loss_fm
        loss_fm = self.LFM(disc_output_real, disc_output_fake)

        disc_output_real = disc_output_real[-1]
        disc_output_fake = disc_output_fake[-1]
        # Oblicz stratę dyskryminatora dla prawdziwej i wygenerowanej próbki
        loss_disc = self.LAdvD(disc_output_fake,disc_output_real)


        # Wyzeruj gradienty dyskryminatora
        self.optimizer_disc.zero_grad()

        # Wykonaj backpropagation dla dyskryminatora
        loss_disc.backward()

        # Aktualizuj wagi dyskryminatora
        self.optimizer_disc.step()
        #obliczstrate rekonstrukcji
        loss_rec= self.LRec(x, gen_output_mel)
        # oblicz loss_adv_p
        loss_adv_p = self.LAdvP(disc_output_fake)
        # oblicz loss_spk
        loss_spk = self.LSpk(speaker_embedding)
        # Oblicz stratę generatora, f0_encoder i speaker_embedder
        loss_gen = 45 * loss_rec + loss_adv_p + loss_fm + 0.01 * loss_spk

        # Wyzeruj gradienty generatora, f0_encoder i speaker_embedder
        self.optimizer_gen.zero_grad()
        self.optimizer_f0.zero_grad()
        self.optimizer_speaker.zero_grad()

        # Wykonaj backpropagation dla generatora, f0_encoder i speaker_embedder
        loss_gen.backward()

        # Aktualizuj wagi generatora, f0_encoder i speaker_embedder
        self.optimizer_gen.step()
        self.optimizer_f0.step()
        self.optimizer_speaker.step()
    def prepare_dataset(self, audio_folder):
        ys = []
        srs = []
        for audio_file in os.listdir(audio_folder):
            if audio_file.endswith('.wav'):
                audio_file = os.path.join(audio_folder, audio_file)
                y, sr = librosa.load(audio_file, sr=None)
                ys.append(torch.tensor(y))
                srs.append(torch.tensor(sr))

        # Stacking lists of tensors into a single tensor
        dataset = torch.stack((ys, srs), dim=1)
        return dataset
    def z_train(self, train_loader, epochs):
        PATH_FOLDER = '..\\data\\parts6s\\'
        for audio_file in os.listdir(audio_folder):
            if i>=1000:
                break
            i+=1
            audio_file = os.path.join(audio_folder, audio_file)
            waveform, sample_rate = torchaudio.load(audio_file)
            waveforms.append(waveform)
            sample_rates.append(sample_rate)
    def train(self, dataloader, device, epochs):
        self.train()
        data_queue = queue.Queue(maxsize=10)  # Bufor na przetworzone dane

        # Wątek do asynchronicznego ładowania danych
        loader_thread = threading.Thread(target=self.async_data_preparation, args=(data_queue, dataloader))
        loader_thread.start()

        for epoch in range(epochs):
            while not data_queue.empty() or loader_thread.is_alive():
                try:
                    # Pobranie danych z bufora
                    batch = data_queue.get(timeout=30)
                    batch = batch.to(device)

                    # Trening na GPU
                    for i in range(0, len(batch), 32):
                        mini_batch = batch[i:i+32]
                        self.train_step(mini_batch, mini_batch)

                    print(f'Epoka: {epoch+1}, krok: {i+1}')

                except queue.Empty:
                    continue

        loader_thread.join()