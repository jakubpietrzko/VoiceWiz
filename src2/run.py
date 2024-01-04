import torch
import matplotlib.pyplot as plt

def display_melspectrograms_from_pt(pt_file_path1, pt_file_path2):
    # Ładowanie melspektrogramów
    mel_spectrogram1 = torch.load(pt_file_path1)
    mel_spectrogram2 = torch.load(pt_file_path2)

    # Obliczanie różnicy między melspektrogramami
    mel_difference = mel_spectrogram1 - mel_spectrogram2

        # Wyświetlanie melspektrogramów
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axs[0].imshow(mel_spectrogram1.detach().squeeze().numpy(), aspect='auto', origin='lower')
    axs[0].set_title('Melspectrogram 1')
    fig.colorbar(im1, ax=axs[0], format='%+2.0f dB')

    im2 = axs[1].imshow(mel_spectrogram2.detach().squeeze().numpy(), aspect='auto', origin='lower')
    axs[1].set_title('Melspectrogram 2')
    fig.colorbar(im2, ax=axs[1], format='%+2.0f dB')

    im3 = axs[2].imshow(mel_difference.detach().squeeze().numpy(), aspect='auto', origin='lower')
    axs[2].set_title('Difference')
    fig.colorbar(im3, ax=axs[2], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()


# Przykładowe użycie:
display_melspectrograms_from_pt('..//data//results//gen_output_mel.pt','..//data//results//source.pt' )
