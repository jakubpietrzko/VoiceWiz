import os
from concurrent.futures import ThreadPoolExecutor
import torchaudio

def process_file(file_path, output_directory):
    # Load the audio data
    waveform, rate = torchaudio.load(file_path)

    # Resample to 16kHz if necessary
    if rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
        waveform = resampler(waveform)

    # Ensure the audio is single channel
    waveform = waveform[0].squeeze()

    # Save the resampled audio
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_directory, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, waveform.unsqueeze(0), 16000)

# Directory containing the audio files
directory = '..//data//wavs_22khz_5s'
new_directory = '..//data//wavs_16khz'

# Get a list of all audio files in the directory and its subdirectories
audio_files = []
for dirpath, dirnames, filenames in os.walk(directory):
    for filename in [f for f in filenames if f.endswith(".wav")]:
        audio_files.append(os.path.join(dirpath, filename))

# Process all files in multiple threads
with ThreadPoolExecutor(max_workers=8) as executor:
    output_directories = [os.path.join(new_directory, os.path.dirname(os.path.relpath(file_path, directory))) for file_path in audio_files]
    executor.map(process_file, audio_files, output_directories)