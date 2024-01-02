import os
import torchaudio
from concurrent.futures import ThreadPoolExecutor

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
    torchaudio.save(output_path, waveform.unsqueeze(0), 16000)

# Directory containing the audio files
directory = '..//data//parts6s'
new_directory = '..//data//parts6s_resampled'

# Create the output directory if it doesn't exist
os.makedirs(new_directory, exist_ok=True)

# Get a list of all audio files in the directory
audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

# Process all files in multiple threads
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_file, audio_files, [new_directory]*len(audio_files))