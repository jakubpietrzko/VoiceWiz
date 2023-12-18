import torch
from torch import nn
import torchaudio
import random
from asr_bottleneck import ASREncoder
from f0_encoder import F0Encoder
from speaker_embedder import SpeakerEmbedder
from discriminator import Discriminator
from generator import Generator
from f0_utils import get_lf0_from_wav
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold
class VoiceConversionModel(nn.Module):
    def __init__(self, device):
        super(VoiceConversionModel, self).__init__()
        self.device = device
        self.asr_encoder = ASREncoder()
        self.asr_encoder.asr_model = self.asr_encoder.asr_model.to(device)
        self.f0_encoder = F0Encoder(in_channels=1).to(device)
        self.speaker_embedder = SpeakerEmbedder(in_channels=1, num_residual_layers =5).to(device)
        self.generator = Generator(asr_features=1, f0_features=1, speaker_features=1, generator_input_dim=1).to(device)
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
        speaker_embedding = self.speaker_embedder(y)
        asr_features = asr_features
        f0_features = self.f0_encoder(f0)

        # Przejdź do przodu przez generator
        gen_output = self.generator(asr_features, f0_features, speaker_embedding)

        return gen_output
    
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
        return data
                   
    #musimy zrobic to tak zeby inny glos bralo na asr i f0 a inny na spekaer embedder
    #na wejscie do asr
    def prepare(self, audio_folder):
        names=[]
        data=[]
        for filename in os.listdir(audio_folder):
            if filename[:-4] in self.names_1 or filename[:-3] in self.names_1:#self.names na czas niepelnego zbioru f0
                if filename.endswith('.wav'):
                    audio_file_path = os.path.join(audio_folder, filename)
                    name = filename[:-4]
                    y, _= torchaudio.load(audio_file_path)
                elif filename.endswith('.pt'):
                    audio_file_path = os.path.join(audio_folder, filename)
                    name = filename[:-3]
                    y = torch.load(audio_file_path)
                data.append(y)
                names.append(name)
        df = pd.DataFrame({'name': names,'data': data})

        # Ustaw nazwę pliku jako indeks
        df.set_index('name', inplace=True)

        return df
    
    def preparefzero(self, audio_folder):
        names=[]
        data=[]
        cnt=0
        self.names_1=set()
        for filename in os.listdir(audio_folder):
            if cnt>10:
                break
            cnt+=1
            if filename.endswith('.wav'):
                audio_file_path = os.path.join(audio_folder, filename)
                name = filename[:-4]
                y, _= torchaudio.load(audio_file_path)
            elif filename.endswith('.pt'):
                audio_file_path = os.path.join(audio_folder, filename)
                name = filename[:-3]
                self.names_1.add(name)
                y = torch.load(audio_file_path)
            data.append(y)
            names.append(name)
        df = pd.DataFrame({'name': names,'data': data})
        
        # Ustaw nazwę pliku jako indeks
        df.set_index('name', inplace=True)

        return df
        
    def prepare_dataset_asr(self, audio_folder, max_source_voices=15000, reverse = False):
        ys = []
        names = set()
        cnt=0
        xd=sorted(os.listdir(audio_folder))
        if reverse:
            for audio_file in reversed(xd):
                if cnt >= max_source_voices:
                    break
                cnt+=1
                if audio_file.endswith('.wav'):
                    audio_file_path = os.path.join(audio_folder, audio_file)
                    audio_file = audio_file[:-4]
                    y, _ = torchaudio.load(audio_file_path)
                    names.add(audio_file)
                    ys.append(y)
        else:
            for audio_file in xd:
                if cnt >= max_source_voices:
                    break
                cnt+=1
                if audio_file.endswith('.wav'):
                    audio_file_path = os.path.join(audio_folder, audio_file)
                    audio_file = audio_file[:-4]
                    y, _ = torchaudio.load(audio_file_path)
                    names.add(audio_file)
                    ys.append(y)
        ys = torch.stack(ys)
        return ys, names
    #na wejscie do do speaker embeddera i moze dyskryminatora (muszą byc z innych audio niz do pozostlaych)

    def prepare_data_mels(self, audio_folder, names, max_source_voices=15000, reverse=False):
        cnt=0
        ys = []
        xd=sorted(os.listdir(audio_folder))
        if reverse:
            print(cnt)
            for audio_file in reversed(xd):
                if cnt>=max_source_voices:
                    break
                cnt+=1
                audio_file_without_ext = audio_file[:-4]
                if audio_file.endswith('.pt') and audio_file_without_ext not in names:
                    audio_file_path = os.path.join(audio_folder, audio_file)
                    y = torch.load(audio_file_path)
                    ys.append(y)
        
        else:
            for audio_file in xd:
                if cnt>=max_source_voices:
                    break
                cnt+=1
                audio_file_without_ext = audio_file[:-4]
                if audio_file.endswith('.pt') and audio_file_without_ext not in names:
                    audio_file_path = os.path.join(audio_folder, audio_file)
                    y = torch.load(audio_file_path)
                    ys.append(y)
        ys = torch.stack(ys)
        return ys
            
    #na wejscie do f0
    def prepare_data_f0(self, audio_folder,names):
        ys = []
        for audio_file in os.listdir(audio_folder):
            audio_file_without_ext = audio_file[:-4]
            if audio_file.endswith('.pt') and audio_file_without_ext in names:
                audio_file_path = os.path.join(audio_folder, audio_file)
                y = torch.load(audio_file_path)
                ys.append(y)
        ys = torch.stack(ys)
        return ys
    #melspektogram glosu zrodlowego do funckji straty
    def prepare_dataset_mels_x(self, audio_folder,names):
        ys = []
        for audio_file in os.listdir(audio_folder):
            audio_file_without_ext = audio_file[:-4]
            if audio_file.endswith('.pt') and audio_file_without_ext in names:
                audio_file_path = os.path.join(audio_folder, audio_file)
                y= torch.load(audio_file_path)
                ys.append(y)
        ys = torch.stack(ys)
        return ys
    #jesli na RAM braknie miejsca bo zwiekszymy data set trzeba bedzie tez robic wsady do ramu ale to pozniej
    def train_model(self, epochs=5):
        self.train() 
        
        PATH_FOLDER = '..\\data\\parts6s\\'
        PATH_FOLDER_MELS = '..\\data\\mels\\'
        PATH_FOLDER_FZEROS = '..\\data\\fzeros\\' #zbyt malo danych
        #PATH_FOLDER_GOAL = '..\\data\\mels_goal\\' #nie istnieje jeszcze przyda sie zwlaszcza jak bedziemy chcieli zrobic any to one lub po prostu rozdizleic na dwa foldery
        dataset = self.prepare_dataset(audio_folder=PATH_FOLDER,mels_x_folder=PATH_FOLDER_MELS, fzeros_folder=PATH_FOLDER_FZEROS)
        dataset_goals = self.prepare_dataset_goal(PATH_FOLDER_MELS) # jesli rozdzielimy to warto pozmieniac
        dataloader_fzeros = torch.stack(dataset[2])
        dataloader= torch.stack(dataset[0])
        dataloader_mels = torch.stack(dataset[1])

        print("dane przygotowane")
        best_loss = float('inf')  # Inicjalizacja najlepszej straty jako nieskończoność

        for epoch in range(epochs):
            running_loss = 0  # Inicjalizacja bieżącej straty
            for i in range(0, len(dataloader), 1024):
                # Przekształcenie ramki w postaci indeksowanej nazwami plików tensory na tensor pytorch
                batch_x = dataloader[i:i+1024].to(self.device)
                batch_y = dataloader_mels[i:i+1024].to(self.device)
                batch_f0 = dataloader_fzeros[i:i+1024].to(self.device)

                # Trening na danych treningowych
                for j in range(0, len(batch_x), 32):
                    mini_batch_x = batch_x[j:j+32]
                    mini_batch_y = batch_y[j:j+32]
                    mini_batch_f0 = batch_f0[j:j+32]

                    mini_batch_x = mini_batch_x.squeeze(1)  # Usuń wymiar 1
                    length = torch.full((mini_batch_x.shape[0],), mini_batch_x.shape[1], device=self.device)  # Utwórz tensor z długościami
                    asr_features = self.asr_encoder.process_audio(mini_batch_x, length)

                    loss = self.train_step(mini_batch_x, mini_batch_y, mini_batch_f0, asr_features)
                    running_loss += loss.item()
                    print(f'Epoka: {epoch+1}, krok: {i+j+1}, strata: {loss.item()}')

            # Obliczanie średniej straty dla epoki
            avg_loss = running_loss / len(dataloader)
            print(f'Epoka: {epoch+1}, średnia strata: {avg_loss}')

            # Zapisywanie modelu, jeśli średnia strata jest mniejsza od najlepszej dotychczasowej straty
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.state_dict(), 'best_model.pth')
                print(f'Zapisano model z epoki: {epoch+1}, strata: {avg_loss}')
                    
    
        
if __name__ == "__main__":
    device = torch.device('cuda')
    x=VoiceConversionModel(device)
    print("trening start")
    x.train_model(epochs=2)