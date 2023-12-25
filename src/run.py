from model import VoiceConversionModel
import torch
if __name__ == "__main__":
    device = torch.device('cuda')
    x=VoiceConversionModel(device)
    x.eval()
    PATH_FOLDER = '..\\data\\parts6s\\'
    PATH_FOLDER_MELS = '..\\data\\mels\\'
    PATH_FOLDER_FZEROS = '..\\data\\fzeros\\'
    x.load_state_dict(torch.load('model.pth', map_location=device))

    x(y,f0,asr)


