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
from sklearn.model_selection import KFold
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
    def train_step(self, x, y, f0, asr_features):
        # Przejdź do przodu przez speaker_embedder, asr_encoder i f0_encoder
        
        
        speaker_embedding = self.speaker_embedder(y)
        asr_features = asr_features
        f0_features = self.f0_encoder(f0)

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
    def validate_step(self, x, y, f0, asr_features):
        # Przejdź do przodu przez speaker_embedder, asr_encoder i f0_encoder
        speaker_embedding = self.speaker_embedder(y)
        asr_features = asr_features 
        f0_features = self.f0_encoder(f0)

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

        #obliczstrate rekonstrukcji
        loss_rec= self.LRec(x, gen_output_mel)
        # oblicz loss_adv_p
        loss_adv_p = self.LAdvP(disc_output_fake)
        # oblicz loss_spk
        loss_spk = self.LSpk(speaker_embedding)
        # Oblicz stratę generatora, f0_encoder i speaker_embedder
        loss_gen = 45 * loss_rec + loss_adv_p + loss_fm + 0.01 * loss_spk

        return loss_gen, loss_disc

    def validate(self, device, val_x, val_y, val_f0, asr_features):
        self.eval()  # Przełącz model w tryb ewaluacji
        with torch.no_grad():  # Wyłącz obliczanie gradientów
            total_gen_loss = 0
            total_disc_loss = 0
            num_batches = 0
            for i in range(0, len(val_x), 1000):
                batch_x = val_x[i:i+1000].to(device)
                batch_y = val_y[i:i+1000].to(device)
                batch_f0 = val_f0[i:i+1000].to(device)

                for j in range(0, len(batch_x), 32):
                    mini_batch_x = batch_x[j:j+32]
                    mini_batch_y = batch_y[j:j+32]
                    mini_batch_f0 = batch_f0[j:j+32]

                    gen_loss, disc_loss = self.validate_step(mini_batch_x, mini_batch_y, mini_batch_f0, asr_features)
                    total_gen_loss += gen_loss.item()
                    total_disc_loss += disc_loss.item()
                    num_batches += 1

            avg_gen_loss = total_gen_loss / num_batches
            avg_disc_loss = total_disc_loss / num_batches

            print(f'Strata generatora na danych walidacyjnych: {avg_gen_loss}')
            print(f'Strata dyskryminatora na danych walidacyjnych: {avg_disc_loss}')
    #musimy zrobic to tak zeby inny glos bralo na asr i f0 a inny na spekaer embedder
    #na wejscie do asr
    def prepare_dataset_asr(self, audio_folder):
        ys = []
        for audio_file in os.listdir(audio_folder):
            if audio_file.endswith('.wav'):
                audio_file_path = os.path.join(audio_folder, audio_file)
                y, _ = torchaudio.load(audio_file_path)
                ys.append(y)  # Przeniesienie tensora na GPU

        # Stacking list of tensors into a single tensor
        dataset = torch.stack(ys).to('cuda') #trzeba bedzie sie pozbyc to cuda  przy treningu bo pozniej jest to zrobione
        return dataset
    #na wejscie do do speaker embeddera i moze dyskryminatora
    def prepare_data_mels(self, audio_folder):
        pass
    #na wejscie do f0
    def prepare_data_f0(self, audio_folder):
        pass
    #melspektogram glosu zrodlowego do funckji straty
    def prepare_dataset_mels_x(self, audio_folder):
        pass
    #jesli na RAM braknie miejsca bo zwiekszymy data set trzeba bedzie tez robic wsady do ramu ale to pozniej
    def train(self, device, epochs, n_splits=5):
        self.train()
        PATH_FOLDER = '..\\data\\parts6s\\'
        PATH_FOLDER_MELS = '..\\data\\mels\\'
        PATH_FOLDER_FZEROS = '..\\data\\fzeros\\'
        
        dataloader = self.prepare_dataset_asr(PATH_FOLDER)
        dataloader_mels = self.prepare_data_mels(PATH_FOLDER_MELS)
        dataloader_fzeros = self.prepare_data_f0(PATH_FOLDER_FZEROS)
        dataloader_mels_x = self.prepare_dataset_mels_x(PATH_FOLDER_MELS)
        # Utworzenie obiektu KFold
        kf = KFold(n_splits=n_splits)
        
        for epoch in range(epochs):
            # Walidacja krzyżowa
            for train_index, val_index in kf.split(dataloader):
                train_mels_x, val_mels_x = dataloader_mels_x[train_index], dataloader_mels_x[val_index]
                train_x, val_x = dataloader[train_index], dataloader[val_index]
                train_y, val_y = dataloader_mels[train_index], dataloader_mels[val_index]
                train_f0, val_f0 = dataloader_fzeros[train_index], dataloader_fzeros[val_index]
                
                # Trening na danych treningowych
                for i in range(0, len(train_x), 1000):
                    batch_x = train_x[i:i+1000].to(device)
                    batch_y = train_y[i:i+1000].to(device)
                    batch_f0 = train_f0[i:i+1000].to(device)

                    # Trening
                    for j in range(0, len(batch_x), 32):
                        mini_batch_mels_x = train_mels_x[j:j+32]
                        mini_batch_x = batch_x[j:j+32]
                        mini_batch_y = batch_y[j:j+32]
                        mini_batch_f0 = batch_f0[j:j+32]

                        length = torch.tensor([mini_batch_x.shape[1]], device=device)
                        asr_features = self.asr_encoder.process_audio(mini_batch_x, length)

                        self.train_step(mini_batch_mels_x, mini_batch_y, mini_batch_f0, asr_features)
                        print(f'Epoka: {epoch+1}, krok: {i+j+1}')
                
                # Walidacja na danych walidacyjnych
                length = torch.tensor([val_x.shape[1]], device=device)
                asr_features = self.asr_encoder.process_audio(val_x, length)
                self.validate(device, val_mels_x, val_y, val_f0, asr_features)