import os
import soundfile as sf

# Ścieżka do katalogu z plikami FLAC
flac_dir = '..\\data\\wav48_silence_trimmed'

# Ścieżka do katalogu, gdzie mają być zapisane pliki WAV
wav_dir = '..\\data\\wavs'

# Przejrzyj wszystkie pliki w katalogu i podkatalogach
for root, dirs, files in os.walk(flac_dir):
    for file in files:
        # Sprawdź, czy plik ma rozszerzenie .flac
        if file.endswith('.flac'):
            # Utwórz pełną ścieżkę do pliku FLAC
            flac_path = os.path.join(root, file)
            
            # Utwórz pełną ścieżkę do pliku WAV
            wav_path = os.path.join(wav_dir, os.path.relpath(flac_path, flac_dir))
            wav_path = os.path.splitext(wav_path)[0] + '.wav'
            
            # Utwórz katalogi, jeśli nie istnieją
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            
            # Wczytaj plik FLAC
            data, samplerate = sf.read(flac_path)
            
            # Zapisz dane jako WAV
            sf.write(wav_path, data, samplerate)