import torch
import asr_bottleneck
import os
import pandas as pd
# Wczytaj tensor z pliku
f = '..\\data\\parts6s\\'
def prepare(audio_folder):
        names=[]
        data=[]
        for filename in os.listdir(audio_folder):
            if filename != 'common_voice_en_38024627.wav':
                continue
            if filename.endswith('.wav'):
                audio_file_path = os.path.join(audio_folder, filename)
                name = filename[:-4]
                y = torch.load(audio_file_path)
            data.append(y)
            names.append(name)
        df = pd.DataFrame({'name': names,'data': data})

        # Ustaw nazwę pliku jako indeks
        df.set_index('name', inplace=True)

        return df
df=prepare(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xd=df.loc['common_voice_en_38024627']
oop = xd[0].to(device) if isinstance(xd[0], torch.Tensor) else torch.tensor(xd[0].values).to(device)
asr_encoder=asr_bottleneck.ASREncoder()

length = torch.tensor(oop, device=device)
asr_features = asr_encoder.process_audio(oop, length)
# Wyświetl wymiary tensora
print(asr_features.shape)