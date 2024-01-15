import torch
from torch import nn
import torchaudio
import warnings
from torch.utils.data import Dataset, DataLoader
import random
from AudioDataset import AudioDataset
from spk_emb import SpeakerEmbedder
from discriminator import Discriminator
from genz import Generator
import nemo.collections.asr as nemo_asr
import os
import random
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import os
from speechbrain.pretrained import HIFIGAN
from predictor import Predictor
import time
import numpy as np
import pandas as pd
from nvidiapred import nvidia_embedder
class VoiceConversionModel(nn.Module):
    def __init__(self, device):
        super(VoiceConversionModel, self).__init__()
        self.device = device
        self.nvidiapred = nvidia_embedder.from_pretrained("nvidia/speakerverification_en_titanet_large")
        self.nvidiapred.eval()
        self.nvidiapred = self.nvidiapred.to(device)
        self.asr_encoder = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_squeezeformer_ctc_small_ls")
        self.asr_encoder = self.asr_encoder.to(device)
        self.asr_encoder = self.asr_encoder.eval()
        self.speaker_embedder = SpeakerEmbedder().to(device)
        self.generator = Generator(speaker_embedding_dim=1).to(device)
        self.discriminator = Discriminator().to(device)
        self.vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="vocoder_16khz", run_opts={"device": "cuda"})
    
        self.vocoder.eval()
        
        
        for param in self.vocoder.parameters():
            param.requires_grad = False
        self.bce_loss = nn.BCELoss()
            # Zamroź parametry ASR
        for param in self.asr_encoder.parameters():
            param.requires_grad = False
        for param in self.nvidiapred.parameters():
            param.requires_grad = False 
        self.optimizer_speaker = torch.optim.Adam(self.speaker_embedder.parameters(), lr=0.001)
        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)
        
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
    def LPred(self,embs1,embs2,w):
        #audio1 = audio1.squeeze(1)
        #print("audio1", audio1.shape)
        #print("audio2", audio2.shape)
        
        X = embs1 / torch.linalg.norm(embs1)
        Y = embs2 / torch.linalg.norm(embs2)
        # Score
        # Oblicz iloczyn skalarny dla każdej pary próbek
        dot_product = torch.einsum('ij,ij->i', X, Y)

        # Oblicz normy dla każdej próbki
        norms = torch.norm(X, dim=1) * torch.norm(Y, dim=1)

        # Oblicz podobieństwo kosinusowe
        similarity_score = dot_product / norms

        # Przeskaluj wynik do zakresu [0, 1]
        similarity_score = (similarity_score + 1) / 2
        #similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
        similarity_score = (similarity_score + 1) / 2
        # Decision
        similarity_score = similarity_score.mean()
        return w / (1 + torch.exp(-8*(similarity_score-0.7)))
        """if torch.isnan(result).any() == True or  torch.isinf(result).any() == True:
            print("result nan")
            return w*1.3
        return w / (1 + torch.exp(-5*(result-0.3)))"""
    def LAsr(self,mini_batch_x1,mini_batch_result, goal_len):
        mini_batch_x1 = mini_batch_x1.squeeze(1)
        mini_batch_result = mini_batch_result.squeeze(1)
        # Załóżmy, że asr_features1 i asr_features2 to batche tensorów wyjściowych z ASR dla oryginalnej i wygenerowanej mow
        #mini_batch_x1 = mini_batch_x1.squeeze(1)
        #mini_batch_result = mini_batch_result.squeeze(1)
        #mini_batch_result = mini_batch_result.narrow(1, 0, mini_batch_x1.size(1))
        #length1 = torch.full((mini_batch_x1.shape[0],), mini_batch_x1.shape[1], device=self.device)
        asr_features1,_,_ = self.asr_encoder(input_signal=mini_batch_x1, input_signal_length=goal_len)
        #length2 = torch.full((mini_batch_result.shape[0],), mini_batch_result.shape[1], device=self.device)
        asr_features2,_,_ = self.asr_encoder(input_signal=mini_batch_result, input_signal_length=goal_len)
        # Oblicz różnicę między nimi
        #print(asr_features1)
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
    def train_step(self, source,mel, goal,source_len, goal_len,cnt, ep):
        source_len = source_len.squeeze(1)
        goal_len = goal_len.squeeze(1)
        
        _, emb = self.nvidiapred.forward(input_signal=source, input_signal_length=source_len)
       
        speaker_embedding = self.speaker_embedder(emb)
        
        """t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0) 
        a = torch.cuda.memory_allocated(0)
        f = t-a  # free inside reserved
        print(f"Total memory przed gener: {t/1024**3}")
        print(f"Reserved memory: {r/1024**3}")
        print(f"Allocated memory: {a/1024**3}")
        print(f"Free inside reserved: {f/1024**3}")    """    
        #Przejdź do przodu przez generator
       
        gen_output_mel = self.generator(mel,speaker_embedding, ep,cnt)
        gen_output = self.vocoder.decode_batch(gen_output_mel)
        #Utwórz transformację MelSpectrogram
        gen_output = gen_output.narrow(2, 0, 80000)
        if  cnt%200==0 or cnt == 10 or cnt ==100:
            # Zapisz tylko pierwszą próbkę z gen_output_mel
            print("Nadpisano probki")
            print("Nadpisano probki")
            print("Nadpisano probki")
            print("Nadpisano probki")
       
                #iarów do kształtu każdego tensora
            torchaudio.save('..//data//results//generated.wav', gen_output[0].cpu().squeeze(1), 16000, format="WAV")
            torch.save(mel[0].cpu(), '..//data//results//source.pt')
            torchaudio.save('..//data//results//source.wav', source[0].cpu().unsqueeze(0), 16000, format="WAV")
            torch.save(gen_output_mel[0].cpu(), '..//data//results//gen_output_mel.pt')
            torchaudio.save('..//data//results//goal.wav', goal[0].cpu().unsqueeze(0), 16000)
        # Przejdź do przodu przez discriminator z prawdziwą próbką
        #print("rozmiary na wej dyskr org", y.shape)
        #print(gen_output_mel.shape)
        #print(mel.shape)
        #gen_output_mel = gen_output_mel.squeeze(1)
      
        disc_output_real = self.discriminator(mel)

        #print("rozmiary na wej dyskr fake", gen_output_mel.shape)
        #print("po d mel")
        
        disc_output_fake = self.discriminator(gen_output_mel.detach())
        
        #print()
        #disc_output_goal = self.discriminator(goal)
        real_labels = torch.ones_like(disc_output_real)
        fake_labels = torch.zeros_like(disc_output_fake)
        
        #Oblicz stratę dyskryminatora dla prawdziwej i wygenerowanej próbki
        
        loss_disc1 = self.bce_loss(disc_output_real, real_labels)
        loss_disc2=self.bce_loss(disc_output_fake, fake_labels)
        #loss_disc3=self.bce_loss(disc_output_goal, real_labels)
        loss_disc = loss_disc1 + loss_disc2 #+ loss_disc3
        #loss_disc = self.LAdvD(disc_output_fake,disc_output_real)
        w_pred = 10
        w_asr = 0
        w_rec = 45
        w_gen = 1
        if ep > 1:
            w_gen = 1
            w_pred = 20
            w_asr = 0
            w_rec = 45
            
        if ep>=5:
            w_pred = 20
            w_asr = 0.1
            loss_asr = self.LAsr(source, gen_output, goal_len)
        
        gen_output = gen_output.squeeze(1)
     
        with torch.no_grad():
            
            _, embs1 = self.nvidiapred.forward(input_signal=gen_output.detach() , input_signal_length=goal_len)
            
            embs1 = embs1.squeeze()
            
            _, embs2 = self.nvidiapred.forward(input_signal=goal, input_signal_length=goal_len)
            
            embs2 = embs2.squeeze() 
        
        loss_pred = self.LPred(embs1,embs2,w_pred)
        
        #obliczstrate rekonstrukcji
        loss_rec= self.LRec(mel, gen_output_mel)
        #oblicz loss_adv_p
        loss_adv_p = self.LAdvP(disc_output_fake.detach())
        #oblicz loss_spk
        #loss_spk = self.LSpk(log_var)
        #Oblicz stratę generatora, f0_encoder i speaker_embedder
        #wyprintuj wszystkie straty
        print("loss_rec", w_rec*loss_rec)
        print("loss_adv_p", w_gen*loss_adv_p)

        #print("01loss_spk", 0.1*loss_spk)
        print("10 loss_pred", loss_pred)
        print("loss_disc real", loss_disc1)
        print("loss_disc fake", loss_disc2)
        #print("loss_disc goal", loss_disc3)
        if ep >=5:
            print("01loss_asr", w_asr*loss_asr)
            loss_gen = w_rec*loss_rec + w_gen*loss_adv_p  + loss_pred +w_asr*loss_asr
        else:
            loss_gen = w_rec*loss_rec + w_gen*loss_adv_p  + loss_pred

        # Wyzeruj wszystkie gradienty
        
        # Na początku pętli treningowej
        self.optimizer_gen.zero_grad()
        self.optimizer_disc.zero_grad()
        self.optimizer_speaker.zero_grad()
        if   cnt%4==0:
            
            # Zamrożenie i trening dyskryminatora
            for param in self.generator.parameters():
                param.requires_grad = False
            for param in self.speaker_embedder.parameters():
                param.requires_grad = False
            loss_disc.mean().backward()
            self.optimizer_disc.step()
            print("discriminator")
            # Odmrożenie generatora i trening generatora
            for param in self.generator.parameters():
                param.requires_grad = True
            for param in self.speaker_embedder.parameters():
                param.requires_grad = True
        for param in self.discriminator.parameters():
            param.requires_grad = False
        
        loss_gen.backward()
        self.optimizer_gen.step()
        self.optimizer_speaker.step()
        
        for param in self.discriminator.parameters():
                param.requires_grad = True
        return loss_gen.mean()

    def evaluate_step(self, x, goal,cnt):
        self.eval()  # Ustawienie modelu w tryb ewaluacji
        with torch.no_grad():  # Wyłączenie obliczania gradientów
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

            y = torch.stack(mel_spectrograms)
            goal1 = self.predictor.select_random_segments(goal)
            goal2 = self.predictor.select_random_segments(goal)
            goal3 = self.predictor.select_random_segments(goal)
            speaker_embedding, log_var = self.speaker_embedder(goal1,goal2,goal3)

            gen_output, gen_output_mel = self.generator(y, speaker_embedding, 6)
            if cnt==0:
                # Zapisz tylko pierwszą próbkę z gen_output_mel
                print("Nadpisano probki")
                print("Nadpisano probki")
                print("Nadpisano probki")
                print("Nadpisano probki")
                
                torchaudio.save('..//data//results//generated.wav', gen_output[0].cpu().squeeze(1), 16000, format="WAV")
                torch.save(y[0].cpu(), '..//data//results//source.pt')
                # Zapisz tylko pierwszą próbkę z y
                torchaudio.save('..//data//results//source.wav', x[0].cpu().squeeze(1), 16000, format="WAV")
                torch.save(gen_output_mel[0].cpu(), '..//data//results//gen_output_mel.pt')
                goal_wav = self.generator.vocoder.decode_batch(goal[0])
                
                # Zapisz tylko pierwszą próbkę z goal.wav
                torchaudio.save('..//data//results//goal.wav', goal_wav.cpu().squeeze(1), 16000)
            disc_output_real = self.discriminator(y)
            disc_output_fake = self.discriminator(gen_output_mel)
            w_gen = 2
            w_pred = 10
            w_asr = 0.1
            w_rec = 25
            loss_asr = self.LAsr(x, gen_output)
            loss_disc = self.LAdvD(disc_output_fake, disc_output_real)
            loss_pred = self.LPred(gen_output_mel, goal, w_pred)
            loss_rec = self.LRec(y, gen_output_mel)
            loss_adv_p = self.LAdvP(disc_output_fake)

            loss_gen = w_rec*loss_rec + w_gen*loss_adv_p  + loss_pred +w_asr*loss_asr

        return loss_gen, loss_disc
    def train_model(self, epochs=5, patience=3, starting_epoch=1, batch_size=2):
        self.train() 
        self.to(self.device)
        warnings.filterwarnings("ignore")
        PATH_FOLDER = '..\\data\\wavs_16khz\\'
        PATH_FOLDER2 = '..\\data\\wavs_16khz_mels\\'
        
        # Utwórz obiekty Dataset
    
        best_loss = float('inf')  # Inicjalizacja najlepszej straty jako nieskonczonosc
        no_improve_epochs = 0  # Licznik epok bez poprawy

        for epoch in range(starting_epoch,epochs):
            running_loss = 0  # Inicjalizacja bieżącej straty
            cnt=0
            
            data = AudioDataset(PATH_FOLDER, PATH_FOLDER2)

            batch_size_on = batch_size
            # Utwórz obiekty DataLoader
            dataloader = DataLoader(data, batch_size=batch_size_on, shuffle=True)
            start = time.time()
            for batch in dataloader:
                source, mel, goal, source_len, goal_len = batch
                
                source=source.to(self.device)
                mel=mel.to(self.device)
                goal=goal.to(self.device)
                source_len=source_len.to(self.device)
                goal_len=goal_len.to(self.device)
                
                cnt+=1
                # Trening na danych treningowych
                loss  = self.train_step(source,mel,goal,source_len,goal_len,cnt, epoch)
                stop = time.time()
                
                print()
                print(f'Krok {cnt*batch_size_on} z {len(dataloader)*batch_size_on}, strata: {loss} czas od poczatku epoki: {stop-start}')
                running_loss += loss.item()
                print()

            # Obliczanie średniej straty dla epoki
            avg_loss = running_loss / len(dataloader)
            with open('log.txt', 'a') as f:
                f.write(f'Epoka: {epoch}, średnia strata: {avg_loss}\n')

                # Zapisz model, jeśli strata jest niższa
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(self.state_dict(), '..//best_modelep2.pth')
                    f.write(f'Zapisano model z epoki: {epoch+1}, srednia strata: {avg_loss}\n')
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

            # Jeśli strata nie poprawiła się przez 'patience' epok, zatrzymaj trening
            if no_improve_epochs >= patience:
                print('Early stopping')
                break
    def run_model(self, ):
        self.eval()
        self.to(self.device)
        PATH_FOLDER = '..\\data\\parts6s_resampled\\'
        PATH_FOLDER2 = '..\\data\\parts6s_mel\\'
        
        # Utwórz obiekty Dataset
        raw_dataset = AudioDataset(PATH_FOLDER)
        mel_dataset = AudioDataset(PATH_FOLDER2)
        batch_size_on = 10 
        # Utwórz obiekty DataLoader
        raw_dataloader = DataLoader(raw_dataset, batch_size=batch_size_on, shuffle=True)
        mel_dataloader = DataLoader(mel_dataset, batch_size=batch_size_on, shuffle=True)
        cnt = 0
        for raw, mel in zip(raw_dataloader, mel_dataloader):
            raw = raw.to(self.device)
            mel = mel.to(self.device)
            self.evaluate_step(raw, mel,cnt)
            cnt+=1
            
   
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    device = torch.device('cuda')
    x=VoiceConversionModel(device)
    x = x.to(device)
    # Wczytaj state_dict z pliku
    state_dict = torch.load("..//best_model.pth")

    # Usuń klucze związane z vocoderem
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vocoder')} 
    #print(state_dict.keys())
    # Wczytaj state_dict do modelu
    x.load_state_dict(state_dict, strict=False)
    #x.run_model()
    x.train_model(epochs=50, patience=5, starting_epoch=2, batch_size = 2)