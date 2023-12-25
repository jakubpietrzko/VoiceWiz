import torch
from torch import nn
import torchaudio
import random
from asr_bottleneck import ASREncoder
from f0_encoder import F0Encoder
from speaker_embedder import SpeakerEmbedder
from discriminator import Discriminator
from genz import Generator
from f0_utils import get_lf0_from_wav
import numpy as np
import os
import time
import pandas as pd
from sklearn.model_selection import KFold
class VoiceConversionModel(nn.Module):
    def __init__(self, device):
        super(VoiceConversionModel, self).__init__()
        self.device = device
        self.asr_encoder = ASREncoder()
        self.asr_encoder.asr_model = self.asr_encoder.asr_model.to(device)
        self.f0_encoder = F0Encoder(in_channels=2).to(device)
        self.speaker_embedder = SpeakerEmbedder(in_channels=80, out_channels=80).to(device)
        self.generator = Generator(asr_dim=129, f0_dim=129, speaker_dim=80, output_dim=80).to(device)
        self.discriminator = Discriminator().to(device)

            # Zamroź parametry ASR
        for param in self.asr_encoder.asr_model.parameters():
            param.requires_grad = False
        # Zdefiniuj optymalizatory
        self.optimizer_f0 = torch.optim.Adam(self.f0_encoder.parameters(), lr=0.001)
        self.optimizer_speaker = torch.optim.Adam(self.speaker_embedder.parameters(), lr=0.001)
        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
    def forward(self, y, f0, asr_features):
        # Przejdź do przodu przez speaker_embedder, asr_encoder i f0_encoder
        x.eval()
        speaker_embedding = self.speaker_embedder(y)
        asr_features = asr_features
        f0_features = self.f0_encoder(f0)

        # Przejdź do przodu przez generator
        gen_output = self.generator(asr_features, f0_features, speaker_embedding)

        return gen_output
    
    def LRec(self, Msource, Mpred):
        return torch.abs(Msource - Mpred).sum()

    def LAdvP(self,Ds):
        return (Ds- 1)**2

    def LAdvD(self,D_fake, D_org):
        return (D_org- 1)**2 + (D_fake)**2

    def LFM(self,x, s):
        outputs_real = x
        outputs_fake = s
        loss = 0
        for out_real, out_fake in zip(outputs_real, outputs_fake):
            loss += torch.abs(out_real - out_fake).mean()
        return loss

    def LSpk(self,speaker_embedding):
        # Tworzymy dwa rozkłady normalne
        dist_pred = torch.distributions.Normal(speaker_embedding, torch.ones_like(speaker_embedding))
        dist_zero = torch.distributions.Normal(torch.zeros_like(speaker_embedding), torch.ones_like(speaker_embedding))

        # Obliczamy dywergencję KL
        return torch.distributions.kl_divergence(dist_pred, dist_zero).sum()
    def train_step(self, x, y, f0, asr_features, goal):
        torch.autograd.set_detect_anomaly(True)
        # Przejdź do przodu przez speaker_embedder, asr_encoder i f0_encoder
        speaker_embedding, log_var = self.speaker_embedder(goal)
        asr_features = asr_features
        f0_features = self.f0_encoder(f0)
        """t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0) 
        a = torch.cuda.memory_allocated(0)
        f = t-a  # free inside reserved
        print(f"Total memory przed gener: {t/1024**3}")
        print(f"Reserved memory: {r/1024**3}")
        print(f"Allocated memory: {a/1024**3}")
        print(f"Free inside reserved: {f/1024**3}")    """    
        # Przejdź do przodu przez generator
        gen_output = self.generator(asr_features, f0_features, speaker_embedding)
        # Utwórz transformację MelSpectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=400, hop_length=256, n_mels=80).to(self.device)
        #print(gen_output.shape)
        # Przekształć surowy dźwięk na melspektrogram
        gen_output_mel_b = mel_transform(gen_output)
        # Przejdź do przodu przez discriminator z prawdziwą próbką
        #print("rozmiary na wej dyskr org", y.shape)
        gen_output_mel = gen_output_mel_b.clone()[:, :, :600]
        disc_output_real = self.discriminator(y)
        #print("rozmiary na wej dyskr fake", gen_output_mel.shape)
        disc_output_fake = self.discriminator(gen_output_mel)
        # oblicz loss_fm
        loss_fm = self.LFM(disc_output_real, disc_output_fake)

        disc_output_real = disc_output_real[-1]
        disc_output_fake = disc_output_fake[-1]
        # Oblicz stratę dyskryminatora dla prawdziwej i wygenerowanej próbki
        # Oblicz stratę dyskryminatora dla prawdziwej i wygenerowanej próbki
        # Oblicz stratę dyskryminatora dla prawdziwej i wygenerowanej próbki
        loss_disc = self.LAdvD(disc_output_fake,disc_output_real)

        #obliczstrate rekonstrukcji
        loss_rec= self.LRec(y, gen_output_mel)
        # oblicz loss_adv_p
        loss_adv_p = self.LAdvP(disc_output_fake)
        # oblicz loss_spk
        loss_spk = self.LSpk(log_var)
        # Oblicz stratę generatora, f0_encoder i speaker_embedder
        loss_gen = 45 * loss_rec + loss_adv_p + loss_fm + 0.01 * loss_spk

        # Wyzeruj wszystkie gradienty
        self.optimizer_disc.zero_grad()
        self.optimizer_gen.zero_grad()
        self.optimizer_f0.zero_grad()
        self.optimizer_speaker.zero_grad()

        # Wykonaj backpropagation dla straty dyskryminatora
        
        loss_disc.mean().backward(retain_graph=True)

        

        # Wykonaj backpropagation dla straty generatora
        loss_gen.mean().backward()
        # Aktualizuj wagi dyskryminatora
        self.optimizer_disc.step()
        # Aktualizuj wagi generatora, f0_encoder i speaker_embedder
        self.optimizer_gen.step()
        self.optimizer_f0.step()
        self.optimizer_speaker.step()

        return loss_gen.mean(), loss_disc.mean()
# Przykład użycia funkcji:
# train_files, test_files = prepare_dataset_split('ścieżka_do_folderu_z_dźwiękiem', 0.8)
# print(f"Liczba plików treningowych: {len(train_files)}, Liczba plików testowych: {len(test_files)}")

# Przykład użycia funkcji:   dataset = self.prepare_dataset(...) potem przerobic na tensory ys = torch.stack(dataset[0]) mels_x = torch.stack(dataset[1]) fzeros = torch.stack(dataset[2]) do speaker embbedera osobny prepare

    def prepare_dataset(self, audio_folder,mels_x_folder, fzeros_folder):
        data = [[],[],[]]
        ys = {}
        mels ={}
        fzeros={}
        #poki fzero jest male zeby liczylo szybciej
        names= set()
        for filename in os.listdir(fzeros_folder):  
            if filename.endswith('.pt'):
                audio_file_path = os.path.join(fzeros_folder, filename)
                name = filename[:-3]
                y = torch.load(audio_file_path).float()
                print(y.shape)
                fzeros[name] = y
                names.add(name)
        cnt=0
        for filename in os.listdir(audio_folder):
            if filename.endswith('.wav') and filename[:-4] in names:
                if cnt>=320:
                    break
                cnt+=1
                audio_file_path = os.path.join(audio_folder, filename)
                name = filename[:-4]
                y, _ = torchaudio.load(audio_file_path)
                ys[name] = y
        cnt=0
        for filename in os.listdir(mels_x_folder):
            if filename.endswith('.pt') and filename[:-3] in names:
                if cnt>=320:
                    break
                cnt+=1
                audio_file_path = os.path.join(mels_x_folder, filename)
                name = filename[:-3]
                y = torch.load(audio_file_path)
                mels[name] = y
        """for filename in os.listdir(fzeros_folder):  
            if filename.endswith('.pt'):
                audio_file_path = os.path.join(fzeros_folder, filename)
                name = filename[:-3]
                y = torch.load(audio_file_path)
                fzeros[name] = y"""
        
        for name in ys.keys():
            if name in mels and name in fzeros:
                data[0].append(ys[name])
                data[1].append(mels[name])
                data[2].append(fzeros[name])
        return data
    
    def prepare_dataset_goal(self, mels_goal):
        data = []
        cnt =0
        for filename in os.listdir(mels_goal):
            if cnt>=320:
                break
            cnt+=1
            if filename.endswith('.pt'):
                audio_file_path = os.path.join(mels_goal, filename)
                name = filename[:-3]
                y = torch.load(audio_file_path)
                data.append(y)
                
        random.shuffle(data)  #przy any to one mozna usunac
        
        data = torch.stack(data)
        
        return data.float()
                 
    #jesli na RAM braknie miejsca bo zwiekszymy data set trzeba bedzie tez robic wsady do ramu ale to pozniej
    #mozna dodac np early stopping
    def train_model(self, epochs=5):
        self.train() 
        
        PATH_FOLDER = '..\\data\\parts6s\\'
        PATH_FOLDER_MELS = '..\\data\\mels\\'
        PATH_FOLDER_FZEROS = '..\\data\\fzeros\\'
        """PATH_FOLDER = '..\\data\\x\\'
        PATH_FOLDER_MELS = '..\\data\\y\\'
        PATH_FOLDER_FZEROS = '..\\data\\f\\'"""
        #zbyt malo danych
        #PATH_FOLDER_GOAL = '..\\data\\mels_goal\\' #nie istnieje jeszcze przyda sie zwlaszcza jak bedziemy chcieli zrobic any to one lub po prostu rozdizleic na dwa foldery
        dataset = self.prepare_dataset(audio_folder=PATH_FOLDER,mels_x_folder=PATH_FOLDER_MELS, fzeros_folder=PATH_FOLDER_FZEROS)
        dataset_goals = self.prepare_dataset_goal(PATH_FOLDER_MELS) # jesli rozdzielimy to warto pozmieniac
        dataloader_fzeros = torch.stack(dataset[2]).squeeze(1).float()
        dataloader= torch.stack(dataset[0]).squeeze(1).float()
        dataloader_mels = torch.stack(dataset[1]).squeeze(1).float()
        dataloader_goals = dataset_goals.squeeze(1)
        print("dane przygotowane")
        best_loss = float('inf')  # Inicjalizacja najlepszej straty jako nieskonczonosc
        for epoch in range(epochs):
            running_loss = 0  # Inicjalizacja bieżącej straty
            size_on_gpu = 128
            for i in range(0, len(dataloader), size_on_gpu):
                # Przekształcenie ramki w postaci indeksowanej nazwami plików tensory na tensor pytorch
                print(i)
                batch_x = dataloader[i:i+size_on_gpu]
                batch_y = dataloader_mels[i:i+size_on_gpu]
                batch_f0 = dataloader_fzeros[i:i+size_on_gpu]
                batch_goal = dataloader_goals[i:i+size_on_gpu]
                
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_f0 = batch_f0.to(self.device)
                batch_goal =batch_goal.to(self.device)
                # Trening na danych treningowych
                size_of_mini_batch = 2
                x=time.time()
                for j in range(0, len(batch_x), size_of_mini_batch):
                    mini_batch_x = batch_x[j:j+size_of_mini_batch]
                    mini_batch_y = batch_y[j:j+size_of_mini_batch]
                    mini_batch_f0 = batch_f0[j:j+size_of_mini_batch]
                    mini_batch_goal = batch_goal[j:j+size_of_mini_batch]
                    length = torch.full((mini_batch_x.shape[0],), mini_batch_x.shape[1], device=self.device)  # Utwórz tensor z długościami
                    asr_features = self.asr_encoder.process_audio(mini_batch_x, length)
                    print(asr_features.shape)
                    loss ,dys = self.train_step(mini_batch_x, mini_batch_y, mini_batch_f0, asr_features,mini_batch_goal)
                    running_loss += loss
                    print(f'Epoka: {epoch+1}, krok: {i+j+1}, strata: {loss, dys }')
                    a = time.time()
                    print(a-x)
                    torch.cuda.empty_cache()
            # Obliczanie średniej straty dla epoki
            avg_loss = running_loss / len(dataloader)
            print(f'Epoka: {epoch+1}, średnia strata: {avg_loss}')

            # Zapisywanie modelu, jeśli średnia strata jest mniejsza od najlepszej dotychczasowej straty
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.state_dict(), '..//best_model.pth')
                print(f'Zapisano model z epoki: {epoch+1}, strata: {avg_loss}')
                    
    
        
if __name__ == "__main__":
    device = torch.device('cuda')
    x=VoiceConversionModel(device)
    print("trening start")
    x.train_model(epochs=2)