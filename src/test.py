import pandas as pd
from collections import Counter

# Czytanie pliku tsv
df = pd.read_csv('..\\data\\cv-corpus-15.0-delta-2023-09-08-en\\cv-corpus-15.0-delta-2023-09-08\\en\\invalidated.tsv', sep='\t')
df = df.iloc[:, [0, 1]]  # Wybierz tylko kolumny 1 i 2
df.columns = ['client_id', 'path']  # Przypisz nowe nazwy kolumn
# Wczytanie pliku clip_durations.tsv
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

# Użycie funkcji
top_durations = find_top_durations(df)

print(top_durations)
