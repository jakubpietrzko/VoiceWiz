import torch
import matplotlib.pyplot as plt

def display_melspectrogram_from_pt(pt_file_path):
    # Ładowanie melspektrogramu
    mel_spectrogram = torch.load(pt_file_path)

    # Wyświetlanie melspektrogramu
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram.detach().squeeze().numpy(), aspect='auto', origin='lower')
    plt.title('Melspectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


# Przykładowe użycie:
display_melspectrogram_from_pt('..//data//results//gen_output_mel.pt')

