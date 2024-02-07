import os
import random
from torch.utils.data import Dataset
import torch

class SpeakerDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.speakers = os.listdir(root_dir)
        self.speaker_to_files = {speaker: os.listdir(os.path.join(root_dir, speaker)) for speaker in self.speakers}

    def __len__(self):
        return 10000#sum(len(files) for files in self.speaker_to_files.values())

    def __getitem__(self, idx):
        same_speaker = random.choice([True, False])
        speaker1 = random.choice(self.speakers)
        file1 = random.choice(self.speaker_to_files[speaker1])
        file1_path = os.path.join(self.root_dir, speaker1, file1)
        melspec1 = torch.load(file1_path)

        if same_speaker:
            # Wybierz drugi plik od tego samego mówcy
            file2 = random.choice(self.speaker_to_files[speaker1])
        else:
            # Wybierz plik od innego mówcy
            speaker2 = random.choice([speaker for speaker in self.speakers if speaker != speaker1])
            file2 = random.choice(self.speaker_to_files[speaker2])

        file2_path = os.path.join(self.root_dir, speaker2 if not same_speaker else speaker1, file2)
        melspec2 = torch.load(file2_path)

        return melspec1, melspec2, int(not same_speaker)