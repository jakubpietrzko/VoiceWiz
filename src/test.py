import pandas as pd
from collections import Counter
import os
def speaker_matching(df):
    same = pd.DataFrame(columns=['path1', 'path2'])
    different = pd.DataFrame(columns=['path1', 'path2'])
    next_diff_cnt=0
    cnt=0
    for i, row_i in df.iterrows():
        for j, row_j in df.iloc[i+1:].iterrows():  
            if row_i['client_id'] == row_j['client_id']:
                path1, _ = os.path.splitext(row_i['path'])
                path2, _ = os.path.splitext(row_j['path'])
                new_row = {'path1': path1, 'path2': path2}
                same = pd.concat([same, pd.DataFrame(new_row, index=[0])], ignore_index=True)
                next_diff_cnt += 1
                cnt+=1
                print(cnt)
            elif row_i['client_id'] != row_j['client_id'] and next_diff_cnt > 0:
                path1, _ = os.path.splitext(row_i['path'])
                path2, _ = os.path.splitext(row_j['path'])
                new_row = {'path1': path1, 'path2': path2}
                different = pd.concat([different, pd.DataFrame(new_row, index=[0])], ignore_index=True)
                next_diff_cnt -= 1
                cnt+=1
                print(cnt)
        
        if cnt>1000:
            print()
            print(next_diff_cnt)
            break
    return same, different
def sum_values(dict_id):
    return sum(dict_id.values())
def dict_id(df):
    dict_id = {}
    for i, row in df.iterrows():
        if row['client_id'] not in dict_id:
            dict_id[row['client_id']] = 1
        else:
            dict_id[row['client_id']] += 1    
    return dict_id
def mix_dataframes(df1, df2):
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    return df
if __name__ == "__main__":  
    # Czytanie pliku tsv
    df = pd.read_csv('..\\data\\cv-corpus-15.0-delta-2023-09-08-en\\cv-corpus-15.0-delta-2023-09-08\\en\\validated.tsv', sep='\t')
    df = df.iloc[:, [0, 1]]  # Wybierz tylko kolumny 1 i 2
    df.columns = ['client_id', 'path']  # Przypisz nowe nazwy kolumn
    d=dict_id(df)
    total = sum_values(d)

    print(d.values())
    print(len(d))
    print(total)
    """same, diff=speaker_matching(df)
    print(same.head(10))
    print()
    print(diff.head(10))"""
    
    
"""# Wczytanie pliku clip_durations.tsv
df_durations = pd.read_csv('..\\data\\cv-corpus-15.0-delta-2023-09-08-en\\cv-corpus-15.0-delta-2023-09-08\\en\\clip_durations.tsv', sep='\t')
df_durations.columns = ['path', 'duration']
df = pd.merge(df, df_durations, on='path')
# Funkcja do znalezienia 100 najczęściej występujących client_id
def find_top_client_ids(df):
    counter = Counter(df['client_id'])
    return counter.most_common(10)
def find_top_durations(df):
    return df.groupby('client_id')['duration'].sum().nlargest(100)

# Użycie funk
# Funkcja do znalezienia wszystkich ścieżek dla danego zestawu client_id
def find_paths(df, client_ids):
    return df[df['client_id'].isin(client_ids)]['path']

"""


"""# Użycie funkcji
top_durations = find_top_durations(df)

print(top_durations)"""
"""import torch

# Wczytaj tensor
f0_tensor = torch.load('..\\data\\mels\\common_voice_en_38024627.pt')

# Sprawdź długość sekwencji
sequence_length = f0_tensor.shape# Zakładamy, że długość sekwencji jest drugim wymiarem

print(f"Długość sekwencji: {sequence_length}")"""
