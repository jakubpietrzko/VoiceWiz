import torch
from torch import nn
import os
import random
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import torch
import time
import torchaudio
from predictor_dataloader import SpeakerDataset
from torchvision import transforms
import torch.nn.functional as F
class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),  # Dodajemy normalizację wsadową
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),  # Dodajemy normalizację wsadową
                nn.LeakyReLU(0.2),
            )
        #encoder like U-Net
        self.encoder = nn.Sequential(
            conv_block(1, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2, 2),
            conv_block(64, 128),
            nn.MaxPool2d(2, 2),
            conv_block(128, 256),
            nn.MaxPool2d(2, 2),
            conv_block(256, 256),
            nn.MaxPool2d(2, 2),
            conv_block(256, 256),
            nn.MaxPool2d(2, 2),
         
        )
        
        self.fc0 = nn.Linear(13312, 7000)
        self.fc1 = nn.Linear(7000, 3048) 
        self.fc2 = nn.Linear(3048, 1024)
        self.fc3= nn.Linear(1024, 256)
        self.fc4= nn.Linear(256, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        self.loss = nn.BCELoss()
        self.sample_rate = 22050
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #self.load_state_dict(torch.load('best_predictor.pth'))
    def forward(self, mel1, mel2):
       
        mel1 = mel1.unsqueeze(1)
        mel2 = mel2.unsqueeze(1)
        m1std=mel1.std()
        m2std=mel2.std()
        eps = 1e-8
        if m1std < eps:
            m1std = eps
        if m2std < eps:
            m2std = eps
        mel1 = (mel1 - mel1.mean()) / m1std
        mel2 = (mel2 - mel2.mean()) / m2std
        #mel1 = F.softmax(mel1, dim=2)
        #mel2 = F.softmax(mel2, dim=2)
        #print(mel1)
        #print(mel2)
        """x1 = self.pool(F.leaky_relu(self.conv1(mel1)))
        x1 = self.pool(F.leaky_relu(self.conv2(x1)))
        x1 = self.pool(F.leaky_relu(self.conv3(x1)))
        x1 = x1.view(x1.size(0), -1)  # Spłaszczenie

        x2 = self.pool(F.leaky_relu(self.conv1(mel2)))
        x2 = self.pool(F.leaky_relu(self.conv2(x2)))
        x2 = self.pool(F.leaky_relu(self.conv3(x2)))
        x2 = x2.view(x2.size(0), -1)  # Spłaszczenie"""
        x1 = self.encoder(mel1)
        x1 = x1.view(x1.size(0), -1)  # Spłaszczenie
        x2 = self.encoder(mel2)
        x2 = x2.view(x2.size(0), -1)  # Spłaszczenie
        x = torch.cat((x1, x2), dim=1)
        x= F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x= F.leaky_relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        #print("x", x)
        return x
    def train_on_batch(self, batch):
        # Dzielenie batcha na dane i etykiety
        batch_x,batch_z ,batch_y = batch
        batch_x = batch_x.to(self.device)
        batch_z = batch_z.to(self.device)
        batch_y = batch_y.to(self.device)
        #batch_x = self.select_random_segments(batch_x)
        #batch_z = self.select_random_segments(batch_z)
        # Przejście i propagacja wsteczna
        self.optimizer.zero_grad()
        y_pred = self(batch_x, batch_z)
        if not torch.isnan(y_pred).any():
            batch_y = batch_y.float() 
            loss = self.loss(y_pred.squeeze(), batch_y)
            loss.backward()
            self.optimizer.step()

            return loss.item()
        print("syf")
        return 0

    def validation_on_batch(self, batch):
        # Dzielenie batcha na dane i etykiety
        batch_x, batch_y = batch
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        # Przejście
        y_pred = self(batch_x)
        if not torch.isnan(y_pred).any():
            loss = self.loss(y_pred.squeeze(), batch_y)

            return loss.item()
        print("syf")
        return 0
    def run_model(self, mel1, mel2):

        self.eval()

        mel1=mel1.to(self.device)
        mel2=mel2.to(self.device)
        mel1=mel1.unsqueeze(0)
        mel2=mel2.unsqueeze(0)
        with torch.no_grad():
            print(mel1)
            mel1segments = self.select_random_segments(mel1)
            mel2segments = self.select_random_segments(mel2)    
            #print(mel1segments.shape, mel2segments.shape)
            result = self(mel1segments, mel2segments)
            #print(result)
            return result
    """def validation_on_batch(self, batch_y):
        # Dzielenie batcha na dwie części
        half_batch_size = len(batch_y) // 2
        batch_y1 = batch_y[:half_batch_size]
        batch_y2 = batch_y[half_batch_size:]

        # Wybieranie losowych 1,5-sekundowych fragmentów z każdej próbki
        batch_y1_segments = self.select_random_segments(batch_y1)
        batch_y1_an_segments = self.select_random_segments(batch_y1)
        batch_y2_segments = self.select_random_segments(batch_y2)

        # Pierwsze przejście
        y_pred1 = self(batch_y1_segments, batch_y1_an_segments)
        y_pred2 = self(batch_y1_segments, batch_y2_segments)
        if not torch.isnan(y_pred1).any() and not torch.isnan(y_pred2).any():
            loss1 = self.loss(y_pred1.squeeze(), torch.zeros(half_batch_size, device=self.device))
             # Drugie przejście
            loss2 =  self.loss(y_pred2.squeeze(), torch.ones(half_batch_size, device=self.device))
        #print("walidacja",loss1.item() + loss2.item())
            return (loss1.item() + loss2.item()) / 2
        print("syf")
        return 0"""
    def select_random_segments(self, samples, segment_length = 256):
        #print(samples.shape)
        segments = []
        for sample in samples:
            
            start = torch.randint(0, sample.shape[1] - segment_length, (1,)).item()
            segment = sample[:, start:start+segment_length]
            segments.append(segment)
        return torch.stack(segments)
        
    
    """def prepare_dataset_goal(self, mels_goal):
        dataset = AudioDataset(mels_goal)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

        data = []
        for batch in data_loader:
            data.append(batch)

        data = torch.cat(data)

        return data.float().cuda()
    def train_model(self, epochs, patience):
        self.train()
        torch.cuda.empty_cache()
        self.to(self.device)
        
        dataloader = self.prepare_dataset_goal('..\\data\\parts6s_resampled\\')
        dataloader = dataloader.squeeze(1)
        print(dataloader.shape)
        # Podziel dane na zestaw treningowy i walidacyjny
        train_size = int(0.8 * len(dataloader))
        print(train_size)
        val_size = len(dataloader) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataloader, [train_size, val_size])
        print(len(train_dataset), len(val_dataset))
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            running_loss = 0
            size_on_gpu = 64
            sumtime_in_epoch=0
            start_time = time.time()
            steps =0
            # Trening
            for i in range(0, len(train_dataset), size_on_gpu):
                batch_y = train_dataset[i:i+size_on_gpu]
                batch_y = batch_y.to(self.device) 
                size_of_mini_batch = 16
                for j in range(0, len(batch_y), size_of_mini_batch):
                    mini_batch_y = batch_y[j:j+size_of_mini_batch]
                    
                    if len(mini_batch_y) == size_of_mini_batch:  # Pomiń, jeśli batch jest niepełny
                        running_loss += self.train_on_batch(mini_batch_y)
                        steps+=1
                        
                print(steps)
                start_time_empty = time.time()
                torch.cuda.empty_cache()
                end_time_empty = time.time()

                elapsed_empty_time = end_time_empty - start_time_empty
                sumtime_in_epoch += elapsed_empty_time
            print(f"Czas wykonania empty w epoce: {sumtime_in_epoch} sekund")
            #steps-=5
            print(f"Training Loss: {running_loss/steps:.4f}")
            val_steps = 0
            # Walidacja
            val_loss = 0
            self.eval()  # Przełącz model w tryb oceny
            with torch.no_grad():
                for i in range(0, len(val_dataset), size_on_gpu):
                    batch_y = val_dataset[i:i+size_on_gpu]
                    batch_y = batch_y.to(self.device) 
                    for j in range(0, len(batch_y), size_of_mini_batch):
                        mini_batch_y = batch_y[j:j+size_of_mini_batch]
                        if len(mini_batch_y) == size_of_mini_batch:  # Pomiń, jeśli batch jest niepełny
                            val_loss += self.validation_on_batch(mini_batch_y)
                            val_steps += 1
            #val_steps-=2
            print(f"Validation Loss: {val_loss/val_steps:.4f}")
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"Czas wykonania: {elapsed_time} sekund")
            # Sprawdź, czy potrzebne jest Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print("best model")
                torch.save(self.state_dict(), 'best_predictor.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break

            self.train()  # Przełącz model z powrotem w tryb treningu  """    
    def train_model(self, epochs, patience):
        self.train()
        torch.cuda.empty_cache()
        self.to(self.device)
        
        dataset = SpeakerDataset('..\\data\\wavs_mels')
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        batchsize=4
        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, num_workers=4)
        
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Trening
            cnt=0
            start = time.time()
            for batch in train_dataloader:
                cnt+=1
                loss = self.train_on_batch(batch)
                stop = time.time()
                print(f'Ep {epoch} step: {cnt* batchsize}, Loss: {loss}, time: {stop-start}')
            
            # Walidacja
            total_val_loss = 0
            self.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    val_loss = self.validation_on_batch(batch)
                    total_val_loss += val_loss
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f'Validation Loss: {avg_val_loss}')
            
            # Sprawdzanie, czy strata walidacji jest najlepsza
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Zapisywanie najlepszego modelu
                torch.save(self.state_dict(), 'best_predictor v2.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                # Wczesne zatrzymanie, jeśli strata walidacji nie poprawiła się przez określoną liczbę epok
                if patience_counter == patience:
                    print('Early stopping')
                    break     
    def evaluate_model(self, test_dataset_path):
        self.eval()

        # Przygotuj dane testowe
        test_dataloader = self.prepare_dataset_goal(test_dataset_path)
        test_dataloader = test_dataloader.squeeze(1)

        # Inicjalizuj zmienne do obliczania metryk
        total_loss = 0
        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        num_batches = 0

        # Ustal rozmiar mini-batcha
        size_of_mini_batch = 12

        with torch.no_grad():
            for i in range(0, len(test_dataloader), size_of_mini_batch):
                batch_y = test_dataloader[i:i+size_of_mini_batch]
                batch_y = batch_y.to(self.device)

                if len(batch_y) == size_of_mini_batch:  # Pomiń, jeśli batch jest niepełny
                    # Oblicz strate
                    loss = self.validation_on_batch(batch_y)
                    total_loss += loss
                    half_batch_size = len(batch_y) // 2
                    batch_y1 = batch_y[:half_batch_size]
                    batch_y2 = batch_y[half_batch_size:]

                    # Wybieranie losowych 1,5-sekundowych fragmentów z każdej próbki
                    batch_y1_segments = self.select_random_segments(batch_y1)
                    batch_y1_an_segments = self.select_random_segments(batch_y1)
                    batch_y2_segments = self.select_random_segments(batch_y2)

                    # Pierwsze przejście
                    y_pred1 = self(batch_y1_segments, batch_y1_an_segments)
                    y_pred2 = self(batch_y1_segments, batch_y2_segments)
                    # Oblicz predykcje
                    
                    y_pred1 = torch.where(torch.isnan(y_pred1), torch.zeros_like(y_pred1), y_pred1)
                    y_pred1 = torch.where(torch.isinf(y_pred1), torch.ones_like(y_pred1), y_pred1)

                    # Sprawdź, czy y_pred2 zawiera jakiekolwiek wartości NaN lub Inf
                    y_pred2 = torch.where(torch.isnan(y_pred2), torch.zeros_like(y_pred2), y_pred2)
                    y_pred2 = torch.where(torch.isinf(y_pred2), torch.ones_like(y_pred2), y_pred2)

# Oblicz pr
                    y_pred1 = torch.round(y_pred1)
                    y_pred2 = torch.round(y_pred2)
                                        # Tworzenie etykiet docelowych
                    y_true1 = torch.zeros(half_batch_size, device=self.device)
                    y_true2 = torch.ones(half_batch_size, device=self.device)

                    # Obliczanie metryk
                    accuracy1 = accuracy_score(y_true1.cpu(), y_pred1.cpu())
                    accuracy2 = accuracy_score(y_true2.cpu(), y_pred2.cpu())
                    precision1 = precision_score(y_true1.cpu(), y_pred1.cpu(), average='macro',zero_division=1)
                    precision2 = precision_score(y_true2.cpu(), y_pred2.cpu(), average='macro',zero_division=1)
                    recall1 = recall_score(y_true1.cpu(), y_pred1.cpu(), average='macro',zero_division=1)
                    recall2 = recall_score(y_true2.cpu(), y_pred2.cpu(), average='macro',zero_division=1)

                    total_accuracy += (accuracy1 + accuracy2) / 2
                    total_precision += (precision1 + precision2) / 2
                    total_recall += (recall1 + recall2) / 2

                    num_batches += 1

        # Oblicz średnie metryki
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_precision = total_precision / num_batches
        avg_recall = total_recall / num_batches

        print(f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")
        
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Ustal urządzenie
    model = Predictor()  # Tworzenie modelu
    model = model.to(device)
    #mel1 = torch.load('..\\data\\wavs_mels\\p226\\p226_179_mic1.pt')
    #mel2 = torch.load('..\\data\\wavs_mels\\p232\\p232_006_mic2_1.pt')
    #x=model.run_model(mel1, mel2)
    #model.evaluate_model('..\\data\\parts6s_resampled\\')
    model.train_model(epochs=100, patience=5)
    ##print(x)
    #Loss: 0.1767, Accuracy: 0.9325, Precision: 0.7929, Recall: 0.9663