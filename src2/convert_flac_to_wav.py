import soundfile as sf

# Ścieżka do pliku FLAC
flac_path = 'ścieżka/do/pliku.flac'

# Ścieżka do pliku WAV
wav_path = 'ścieżka/do/pliku.wav'

# Wczytaj plik FLAC
data, samplerate = sf.read(flac_path)

# Zapisz dane jako WAV
sf.write(wav_path, data, samplerate)
