import torch
import librosa
from torch import nn
import torchaudio
import torchaudio.transforms as T

from torch.utils.data import Dataset, DataLoader
import random
from AudioDatasetv3 import AudioDataset
from discriminatorv import Discriminator
from genzv5 import Generator
#import nemo.collections.asr as nemo_asr
import os

import random

import os
#from speechbrain.pretrained import HIFIGAN

import time
import numpy as np
import pandas as pd
class CustomScheduler:
    def __init__(self, optimizer, decrease_threshold=0.3 ,soft_increase_threshold=0.6, increase_threshold=1, decrease_factor=0.9, increase_factor=1.1, soft_increase_factor=1.05):
        self.optimizer = optimizer
        self.decrease_threshold = decrease_threshold
        self.increase_threshold = increase_threshold
        self.decrease_factor = decrease_factor
        self.increase_factor = increase_factor
        self.soft_increase_threshold = soft_increase_threshold
        self.soft_increase_factor = soft_increase_factor
    def step(self, loss):
        lr = self.optimizer.param_groups[0]['lr']
        if loss < self.decrease_threshold:
            lr *= self.decrease_factor
        elif loss > self.increase_threshold:
            lr *= self.increase_factor
        elif loss > self.soft_increase_threshold:
            lr *= self.soft_increase_factor
        self.optimizer.param_groups[0]['lr'] = lr
        
        
class VoiceConversionModel(nn.Module):
    def __init__(self, device):
        super(VoiceConversionModel, self).__init__()
        self.device = device
     
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        #self.vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="vocoder_16khz", run_opts={"device": "cuda"})
        
        #self.vocoder.eval()
        
        
        #for param in self.vocoder.parameters():
          #  param.requires_grad = False
        self.bce_loss = nn.BCELoss()
            # Zamroź parametry ASR
        """for param in self.asr_encoder.parameters():
            param.requires_grad = False"""
        
        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.00001)
        self.scheduler_disc = CustomScheduler(self.optimizer_disc)
        self.best=100
        
    def normalize_melspectrogram(self, melspectrogram):
        min_val = torch.min(melspectrogram)
        max_val = torch.max(melspectrogram)
        normalized_melspectrogram = (melspectrogram - min_val) / (max_val - min_val) # Normalizacja do 0-1
        normalized_melspectrogram = (normalized_melspectrogram * 2) - 1 # Przeskalowanie do -1 do 1
        return normalized_melspectrogram
    def normalize_melspectrogram_old(self,melspectrogram):
        min_val = torch.min(melspectrogram) 
        max_val = torch.max(melspectrogram) 
        normalized_melspectrogram = (melspectrogram - min_val) / (max_val - min_val) 
        return normalized_melspectrogram
    def denormalize_melspectrogram_old(self, normalized_melspectrogram, min_val, max_val):
        melspectrogram = normalized_melspectrogram * (max_val - min_val) + min_val
        return melspectrogram

    def denormalize_melspectrogram(self, normalized_melspectrogram, min_val, max_val):
        melspectrogram = (normalized_melspectrogram + 1) / 2 # Przeskalowanie z powrotem do 0-1
        melspectrogram = melspectrogram * (max_val - min_val) + min_val # Denormalizacja do oryginalnego zakresu
        return melspectrogram
    def generate(self, mel, goal):
    # Normalizacja mel-spektrogramu
        mel_min_val = torch.min(mel)
        mel_max_val = torch.max(mel)
        mel = self.normalize_melspectrogram(mel)#do h.pth old
        mel = mel.unsqueeze(1)

        # Przepuszczenie mel-spektrogramu przez generator
        gen_output_mel = self.generator(mel)

        # Denormalizacja wygenerowanego mel-spektrogramu
        generated = self.denormalize_melspectrogram(gen_output_mel, mel_min_val, mel_max_val)

        return generated
# Losses

    def mae(self, x, y):
        # Obcięcie
        if x.size(3) > y.size(3):
            x = x[:, :, :, :y.size(3)]
        elif y.size(3) > x.size(3):
            y = y[:, :, :, :x.size(3)]
        return torch.mean(torch.abs(x - y))

    def mse(self, x, y):
                # Obcięcie
        if x.size(3) > y.size(3):
            x = x[:, :, :, :y.size(3)]
        elif y.size(3) > x.size(3):
            y = y[:, :, :, :x.size(3)]
        return torch.mean((x - y) ** 2)


    def d_loss_f(self, fake):
        return torch.mean(torch.maximum(1 + fake, 0.1 * torch.ones_like(fake)))

    def d_loss_r(self, real):
        return torch.mean(torch.maximum(0.9 - real, torch.zeros_like(real)))

    def g_loss_f(self, fake):
        return torch.mean(-fake)
    def loss_dif(self, d_fake, d_real, d_src):
        d_real1, d_real2 = d_real.chunk(2, dim=0)
        diff_real = torch.abs(d_real1 - d_real2)
        d_src1, d_src2 = d_src.chunk(2, dim=0)
        diff_src = torch.abs(d_src1 - d_src2)
        d_fake1, d_fake2 = d_fake.chunk(2, dim=0)
        diff_fake = torch.abs(d_real1 - d_fake1)
        diff_fake1 = torch.abs(d_src1 - d_fake2)
        diff_final = torch.abs(diff_real - diff_fake)
        diff_final1 = torch.abs(diff_src - diff_fake1)
        return diff_final.sum(), diff_final1.sum()
        
        
    def train_step(self,mel, goal_mel,cnt):
        """t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0) 
        a = torch.cuda.memory_allocated(0)
        f = t-a  # free inside reserved
        print(f"Total memory przed gener: {t/1024**3}")
        print(f"Reserved memory: {r/1024**3}")
        print(f"Allocated memory: {a/1024**3}")
        print(f"Free inside reserved: {f/1024**3}")    """    
        #Przejdź do przodu przez generator

        mel_min_val = torch.min(mel)
        mel_max_val = torch.max(mel)
        goal_mel_min_val = torch.min(goal_mel)
        goal_mel_max_val = torch.max(goal_mel)
        mel = self.normalize_melspectrogram(mel)
        goal_mel = self.normalize_melspectrogram(goal_mel)
        mel = mel.unsqueeze(1)
        goal_mel = goal_mel.unsqueeze(1)
        
        disc_output_real, disc_vec_real, disc_embedding = self.discriminator(goal_mel)
        gen_output_mel= self.generator(mel, disc_embedding.detach())
        
        
        gen_output_mel_identity = self.generator(goal_mel, disc_embedding.detach())
        
        gen_output_mel_x = gen_output_mel.detach()
        if gen_output_mel_x.size(3) > goal_mel.size(3):
            gen_output_mel_x = gen_output_mel_x[:, :, :, :goal_mel.size(3)]
        elif goal_mel.size(3) > gen_output_mel_x.size(3):
            goal_mel = goal_mel[:, :, :, :gen_output_mel_x.size(3)]
        # Oblicz gradienty
       
        goal_mel = goal_mel.squeeze(1)
        gen_output_mel_x = gen_output_mel_x.squeeze(1)
        alpha = torch.rand(goal_mel.size(0), 1, 1, device=goal_mel.device)
        interpolates = alpha * goal_mel + ((1 - alpha) * gen_output_mel_x)
        
        interpolates= interpolates.unsqueeze(1)
        
        # Oblicz wyjście dyskryminatora dla interpolacji
        interpolates.requires_grad_(True)
        disc_interpolates,_ ,_= self.discriminator(interpolates)

        # Oblicz gradienty
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Oblicz karę za gradienty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        # Dodaj karę za gradienty do całkowitej straty
        lambda_gp = 20.0  
        gen_output_mel_x = gen_output_mel_x.unsqueeze(1)
        goal_mel = goal_mel.unsqueeze(1)
        # Oblicz wyjście dyskryminatora dla wygenerowanych próbek
        disc_output_fake, disc_vec_fake, _ = self.discriminator(gen_output_mel_x)
        
        # Utwórz etykiety dla prawdziwych i wygenerowanych próbek
        disc_output_src , disc_vec_src,_ = self.discriminator(mel)
        
       
        loss_gen = self.g_loss_f(disc_output_fake.detach())
        
        # Oblicz stratę dyskryminatora dla prawdziwej i wygenerowanej próbki
        loss_disc1 = self.d_loss_r(disc_output_real)
        loss_disc2 = self.d_loss_f(disc_output_fake)
        loss_rec = self.mae(gen_output_mel_identity, goal_mel)     
        loss_disc_src = self.d_loss_f(disc_output_src)
        # Oblicz całkowitą stratę dyskryminatora
        loss_disc = (2*loss_disc1 + loss_disc2 +loss_disc_src)/4.0
        loss_disc = loss_disc + lambda_gp * gradient_penalty
  
        loss_diff1, loss_diff2 = self.loss_dif(disc_vec_fake.detach(), disc_vec_real.detach(), disc_vec_src.detach())
        loss_diff = loss_diff1 - loss_diff2
        if   cnt%1536==0:
            # Zapisz tylko pierwszą próbkę z gen_output_mel
            print("Nadpisano probki")
            print("Nadpisano probki")
            print("Nadpisano probki")
            print("Nadpisano probki")
            #gen_output = self.vocoder.decode_batch(gen_output_mel.squeeze(1))
            #gen_output = gen_output.squeeze(1)
                #iarów do kształtu każdego tensora
            #torchaudio.save('..//data//results//generated.wav', gen_output[0].cpu().unsqueeze(0), 16000, format="WAV")
            generated=self.denormalize_melspectrogram(gen_output_mel, mel_min_val, mel_max_val)
            source = self.denormalize_melspectrogram(mel, mel_min_val, mel_max_val)
            goal = self.denormalize_melspectrogram(goal_mel, goal_mel_min_val, goal_mel_max_val)
            torch.save(source[0].cpu(), '..//data//results//source.pt')
            torch.save(generated[0].cpu(), '..//data//results//gen_output_mel.pt')
            torch.save(goal[0].cpu(), '..//data//results//goal.pt')
        
        #oblicz loss_spk
        #loss_spk = self.LSpk(log_var)
        #Oblicz stratę generatora, f0_encoder i speaker_embedder
        #wyprintuj wszystkie straty
      
        if cnt%1536==0: 
  
            print("loss_rec", loss_rec)
            print("loss_gen", loss_gen)
            print("loss_disc real", loss_disc1)
            print("loss_disc fake", loss_disc2)
            print('loss disc src', loss_disc_src)
            print("loss_disc", loss_disc)
            print("loss_diff", loss_diff1*0.0001)
            print("loss_diff1", loss_diff2*0.0001)
        #print("loss_disc goal", loss_disc3)
        loss_gen = 5*loss_gen + loss_diff*0.01+loss_rec*0.001

        # Wyzeruj wszystkie gradienty
        
        # Na początku pętli treningowej
        self.optimizer_gen.zero_grad()
        self.optimizer_disc.zero_grad()
     
        if   cnt%24==0:
            
            # Zamrożenie i trening dyskryminatora
            for param in self.generator.parameters():
                param.requires_grad = False
        
            loss_disc.mean().backward()
            self.optimizer_disc.step()
            
            # Odmrożenie generatora i trening generatora
            for param in self.generator.parameters():
                param.requires_grad = True
          
        for param in self.discriminator.parameters():
            param.requires_grad = False
        
        loss_gen.backward()
        self.optimizer_gen.step()
     
        
        for param in self.discriminator.parameters():
                param.requires_grad = True
        return loss_gen, loss_disc1, loss_disc2, loss_disc_src

 
    def train_model(self, epochs=5, patience=3, starting_epoch=1, batch_size=2):
        self.train() 
        self.to(self.device)
        PATH_FOLDER = '..\\data\\wavs_16khz_parts\\'
        PATH_FOLDER2 = '..\\data\\mels\\'
        
        # Utwórz obiekty Dataset
    
        best_loss = float('inf')  # Inicjalizacja najlepszej straty jako nieskonczonosc
        no_improve_epochs = 0  # Licznik epok bez poprawy
        data = AudioDataset(PATH_FOLDER, PATH_FOLDER2)
        batch_size_on = batch_size
        dataloader = DataLoader(data, batch_size=batch_size_on, shuffle=True)
        for epoch in range(starting_epoch,epochs):
            running_loss = 0  # Inicjalizacja bieżącej straty
            cnt=0 
            loss_sd=0
            # Utwórz obiekty DataLoader            
            start = time.time()
            for batch in dataloader:
                mel,  goal_mel = batch
                
                
                mel=mel.to(self.device)
               
                goal_mel=goal_mel.to(self.device)
                cnt+=batch_size
                # Trening na danych treningowych
                loss,loss_disc, loss_disc_fake, loss_disc_src  = self.train_step(mel, goal_mel,cnt)
                stop = time.time()
                if (cnt)%1536==0:
                    print()
                    print(f'Krok {cnt} z {len(dataloader)*batch_size_on}, strata: {loss} czas od poczatku epoki: {stop-start}')
                    print()
                running_loss += loss.item()
                loss_d = (loss_disc.item() + loss_disc_fake.item() + loss_disc_src.item())/3
                loss_sd+=loss_d
            
            loss_sd_avg = loss_sd / len(dataloader)    
            self.scheduler_disc.step(loss_sd_avg)   

            # Obliczanie średniej straty dla epoki
            #avg_loss_sd = loss_sd / 1704
            stop1 = time.time()
            avg_loss = running_loss / (len(dataloader))
            print(f'Epoka: {epoch}, srednia strata: {avg_loss}, czas: {stop1-start} strata dyskryminatora: {loss_sd_avg}')
            with open('log.txt', 'a') as f:
                f.write(f'Epoka: {epoch}, średnia strata: {avg_loss} strata dyskryminatora: {loss_sd_avg}  \n')

                # Zapisz model, jeśli strata jest niższa
                if avg_loss < best_loss:
                    if epoch >5:
                        best_loss = avg_loss
                    torch.save(self.state_dict(), '..//best_model_one_onev4.pth')
                    f.write(f'Zapisano model z epoki: {epoch}, srednia strata: {avg_loss}\n')
                    no_improve_epochs = 0
                elif epoch % 3 == 0:
                    no_improve_epochs += 1
                    torch.save(self.state_dict(), '..//best_model_one_one_nv4.pth')
                    f.write(f'Zapisano nmodel z epoki: {epoch}, srednia strata: {avg_loss}\n')
                else:
                    no_improve_epochs += 1
                    

            # Jeśli strata nie poprawiła się przez 'patience' epok, zatrzymaj trening
            if no_improve_epochs >= patience:
                print('Early stopping')
                #break
    
            
   
if __name__ == "__main__":
    device = torch.device('cuda')
    x=VoiceConversionModel(device)
    x = x.to(device)
    #state_dict_d = torch.load("..//e.pth")
    # Wczytaj state_dict z pliku
    #state_dict = torch.load("..//best_model_one_one_nv4.pth")
    #print(state_dict.keys())
    # Usuń klucze związane z dys
    #state_dict = {k: v for k, v in state_dict.items() if not k.startswith("discriminator")  } # 
    #print(state_dict.keys())
    #state_dict_d = {k: v for k, v in state_dict_d.items() if k.startswith("discriminator") }
    # Wczytaj state_dict do modelu
    #x.load_state_dict(state_dict, strict=False)
    #x.load_state_dict(state_dict_d, strict=False)"""
    #x.run_model()
    x.train_model(epochs=300, patience=10, starting_epoch=10, batch_size = 6)