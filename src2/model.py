import torch
from torch import nn
import torchaudio
import warnings
from torch.utils.data import Dataset, DataLoader
import random
from AudioDataset import AudioDatasetMEL, AudioDatasetRAW
from asr_bottleneck import ASREncoder
from f0_encoder import F0Encoder
from speaker_embedderv2 import SpeakerEmbedder
from discriminator import Discriminator
from genz import Generator
from f0_utils import get_lf0_from_wav
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import os
from predictor import Predictor
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
class VoiceConversionModel(nn.Module):
    def __init__(self, device):
        super(VoiceConversionModel, self).__init__()
        self.device = device
        self.asr_encoder = ASREncoder()
        self.asr_encoder.asr_model = self.asr_encoder.asr_model.to(device)
        self.speaker_embedder = SpeakerEmbedder().to(device)
        self.generator = Generator(speaker_embedding_dim=32).to(device)
        self.discriminator = Discriminator().to(device)
        self.bce_loss = nn.BCELoss()
        self.predictor = Predictor().to(device)
        self.optimizer_pred = torch.optim.Adam(self.predictor.parameters(), lr=0.001)
            # Zamroź parametry ASR
        for param in self.asr_encoder.asr_model.parameters():
            param.requires_grad = False
        # zamrozenn parametry predyktora
        for param in self.predictor.parameters():
            param.requires_grad = False
        self.optimizer_speaker = torch.optim.Adam(self.speaker_embedder.parameters(), lr=0.001)
        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
        
    def forward(self, y):
        # Przejdź do przodu przez speaker_embedder, asr_encoder i f0_encoder
        x.eval()
        speaker_embedding = self.speaker_embedder(y)
        
        

        # Przejdź do przodu przez generator
        gen_output = self.generator(speaker_embedding)

        return gen_output
    
    def LRec(self, Msource, Mpred):
        return torch.abs(Msource - Mpred).mean()

    def LAdvP(self,Ds):
        real_labels = torch.ones_like(Ds)
        return self.bce_loss(Ds, real_labels)

    def LAdvD(self, D_fake, D_org):
        real_labels = torch.ones_like(D_org)
        fake_labels = torch.zeros_like(D_fake)
        
        real_loss = self.bce_loss(D_org, real_labels)
        fake_loss = self.bce_loss(D_fake, fake_labels)
        return (real_loss + fake_loss) / 2
    def LFM(self,x, s):
        outputs_real = x
        outputs_fake = s
        loss = 0
        for out_real, out_fake in zip(outputs_real, outputs_fake):
            loss += torch.abs(out_real - out_fake).mean()
        return loss
    def LPred(self, mel1, mel2,w):
        result=self.predictor.run_model(mel1, mel2).mean()
        if torch.isnan(result).any() == True or  torch.isinf(result).any() == True:
            print("result nan")
            return w*1.3
        return w / (1 + torch.exp(-7*(result-0.3)))
    def LAsr(self,mini_batch_x1,mini_batch_result):
        # Załóżmy, że asr_features1 i asr_features2 to batche tensorów wyjściowych z ASR dla oryginalnej i wygenerowanej mow
        mini_batch_x1 = mini_batch_x1.squeeze(1)
        mini_batch_result = mini_batch_result.squeeze(1)
        mini_batch_result = mini_batch_result.narrow(1, 0, mini_batch_x1.size(1))
        length1 = torch.full((mini_batch_x1.shape[0],), mini_batch_x1.shape[1], device=self.device)
        asr_features1 = self.asr_encoder.process_audio(mini_batch_x1, length1)
        length2 = torch.full((mini_batch_result.shape[0],), mini_batch_result.shape[1], device=self.device)
        asr_features2 = self.asr_encoder.process_audio(mini_batch_result, length2)
        # Oblicz różnicę między nimi
        differences = torch.norm(asr_features1 - asr_features2, dim=-1)

        # Oblicz średnią różnicę dla całego batcha
        mean_difference = torch.mean(differences)
        return mean_difference   
    def LSpk(self,speaker_embedding):
        # Tworzymy dwa rozkłady normalne
        dist_pred = torch.distributions.Normal(speaker_embedding, torch.ones_like(speaker_embedding))
        dist_zero = torch.distributions.Normal(torch.zeros_like(speaker_embedding), torch.ones_like(speaker_embedding))
        kl_div = torch.distributions.kl_divergence(dist_pred, dist_zero)
        return kl_div.mean(dim=(2, 3))
       # return torch.distributions.kl_divergence(dist_pred, dist_zero).mean()
    def train_step(self, x, goal,cnt, ep):
       
        warnings.filterwarnings("ignore")
        mel_spectrograms = []
        
        for audio in x:
            spectrogram, _ = mel_spectogram(
                audio=audio.squeeze(),
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
            mel_spectrograms.append(spectrogram)
       
        # Convert list of tensors to a single tensor
        y = torch.stack(mel_spectrograms)
        goal1 = self.predictor.select_random_segments(goal)
        goal2 = self. predictor.select_random_segments(goal)
        goal3= self.predictor.select_random_segments(goal)
        speaker_embedding, log_var = self.speaker_embedder(goal1,goal2,goal3)
        """t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0) 
        a = torch.cuda.memory_allocated(0)
        f = t-a  # free inside reserved
        print(f"Total memory przed gener: {t/1024**3}")
        print(f"Reserved memory: {r/1024**3}")
        print(f"Allocated memory: {a/1024**3}")
        print(f"Free inside reserved: {f/1024**3}")    """    
        #Przejdź do przodu przez generator
        gen_output, gen_output_mel = self.generator(y,speaker_embedding)
        #Utwórz transformację MelSpectrogram
        if cnt%200==0 or cnt == 10:
            # Zapisz tylko pierwszą próbkę z gen_output_mel
            torchaudio.save('..//data//results//generated.wav', gen_output[0].cpu().squeeze(1), 16000, format="WAV")
            torch.save(y[0].cpu(), '..//data//results//source.pt')
            # Zapisz tylko pierwszą próbkę z y
            torchaudio.save('..//data//results//source.wav', x[0].cpu().squeeze(1), 16000, format="WAV")
            torch.save(gen_output_mel[0].cpu(), '..//data//results//gen_output_mel.pt')
            goal_wav = self.generator.vocoder.decode_batch(goal[0])
            
            # Zapisz tylko pierwszą próbkę z goal.wav
            torchaudio.save('..//data//results//goal.wav', goal_wav.cpu().squeeze(1), 16000)
        # Przejdź do przodu przez discriminator z prawdziwą próbką
        #print("rozmiary na wej dyskr org", y.shape)
    
        disc_output_real = self.discriminator(y)
        #print("rozmiary na wej dyskr fake", gen_output_mel.shape)
        disc_output_fake = self.discriminator(gen_output_mel)

        #Oblicz stratę dyskryminatora dla prawdziwej i wygenerowanej próbki
        loss_disc = self.LAdvD(disc_output_fake,disc_output_real)
        w_pred = 0.2
        w_asr = 0.001
        w_rec = 1.5
        w_gen = 2
        if ep > 5:
            w_gen = 2
            w_pred = 10
            w_asr = 0.1
            w_rec = 0.5
        loss_pred = self.LPred(gen_output_mel, goal,w_pred)
        
        loss_asr = self.LAsr(x, gen_output)
        #obliczstrate rekonstrukcji
        loss_rec= self.LRec(y, gen_output_mel)
        #oblicz loss_adv_p
        loss_adv_p = self.LAdvP(disc_output_fake)
        #oblicz loss_spk
        #loss_spk = self.LSpk(log_var)
        #Oblicz stratę generatora, f0_encoder i speaker_embedder
        #wyprintuj wszystkie straty
        print("loss_rec", w_rec*loss_rec)
        print("loss_adv_p", w_gen*loss_adv_p)

        #print("01loss_spk", 0.1*loss_spk)
        print("10 loss_pred", loss_pred)
        print("loss_disc", loss_disc)
        print("01loss_asr", w_asr*loss_asr)
        loss_gen = w_rec*loss_rec + w_gen*loss_adv_p  + loss_pred +w_asr*loss_asr

        # Wyzeruj wszystkie gradienty
        self.optimizer_disc.zero_grad()
        self.optimizer_gen.zero_grad()
       
        self.optimizer_speaker.zero_grad()

        # Wykonaj backpropagation dla straty dyskryminatora
        
        loss_disc.mean().backward(retain_graph=True)

        

        # Wykonaj backpropagation dla straty generatora
        loss_gen.backward()
        # Aktualizuj wagi dyskryminatora
        self.optimizer_disc.step()
        # Aktualizuj wagi generatora, f0_encoder i speaker_embedder
        self.optimizer_gen.step()

        self.optimizer_speaker.step()
    
        return loss_gen.mean(), loss_disc.mean()

         
    def train_model(self, epochs=5, patience=3):
        self.train() 
        self.to(self.device)
        warnings.filterwarnings("ignore")
        PATH_FOLDER = '..\\data\\parts6s_resampled\\'
        PATH_FOLDER2 = '..\\data\\parts6s_mel\\'
        
        # Utwórz obiekty Dataset
        raw_dataset = AudioDatasetRAW(PATH_FOLDER)
        mel_dataset = AudioDatasetMEL(PATH_FOLDER2)
        batch_size_on = 12 
        # Utwórz obiekty DataLoader
        raw_dataloader = DataLoader(raw_dataset, batch_size=batch_size_on, shuffle=True)
        mel_dataloader = DataLoader(mel_dataset, batch_size=batch_size_on, shuffle=True)

        print("dane przygotowane")
        best_loss = float('inf')  # Inicjalizacja najlepszej straty jako nieskonczonosc
        no_improve_epochs = 0  # Licznik epok bez poprawy

        for epoch in range(epochs):
            running_loss = 0  # Inicjalizacja bieżącej straty
            cnt=0
            start = time.time()
            for (batch_x, batch_y) in zip(raw_dataloader, mel_dataloader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                cnt+=1
                # Trening na danych treningowych
                loss ,dys = self.train_step(batch_x, batch_y,cnt, epoch)
                stop = time.time()
                
                print()
                print(f'Krok {cnt*batch_size_on} z {len(raw_dataloader)*batch_size_on}, strata: {loss} czas od poczatku epoki: {stop-start}')
                running_loss += loss.item()
                print()

            # Obliczanie średniej straty dla epoki
            avg_loss = running_loss / len(mel_dataloader)
            print(f'Epoka: {epoch+1}, średnia strata: {avg_loss}')

            # Zapisz model, jeśli strata jest niższa
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.state_dict(), '..//best_model.pth')
                print(f'Zapisano model z epoki: {epoch+1}, srednia strata: {avg_loss}')
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # Jeśli strata nie poprawiła się przez 'patience' epok, zatrzymaj trening
            if no_improve_epochs >= patience:
                print('Early stopping')
                break
        
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    device = torch.device('cuda')
    x=VoiceConversionModel(device)
    print("trening start")
    x.train_model(epochs=50, patience=5)