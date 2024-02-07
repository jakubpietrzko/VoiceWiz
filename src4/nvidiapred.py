import nemo.collections.asr as nemo_asr
import torch
import time
import librosa
import numpy as np
class Nvidia_embedder(nemo_asr.models.EncDecSpeakerLabelModel): 
    def __init__(self, cfg, trainer=None):
        super(Nvidia_embedder, self).__init__(cfg=cfg, trainer=trainer)
    @torch.no_grad()
    def infer_file(self, audio, sr):#def infer_file(self, path2audio_file):
        """
        Args:
            path2audio_file: path to an audio wav file

        Returns:
            emb: speaker embeddings (Audio representations)
            logits: logits corresponding of final layer
        """
        
        #audio, sr = librosa.load(path2audio_file, sr=None)   
        """target_sr = self._cfg.train_ds.get('sample_rate', 16000)
        print(target_sr)
        if sr != target_sr:
            audio = librosa.core.resample(audio, orig_sr=sr, target_sr=target_sr)"""
        """        audio_length = audio.shape[0]
        device = self.device
        audio = np.array([audio])
        audio_signal, audio_signal_len = (
            torch.tensor(audio, device=device),
            torch.tensor([audio_length], device=device),
        )
        mode = self.training
        self.freeze()
        """
        
        logits, emb = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        #print(logits)
        """self.train(mode=mode)
        if mode is True:
            self.unfreeze()"""
        del audio_signal, audio_signal_len
        return emb, logits


    def get_embedding(self, audio , sr):#def get_embedding(self, path2audio_file):
        """
        Returns the speaker embeddings for a provided audio file.

        Args:
            path2audio_file: path to an audio wav file

        Returns:
            emb: speaker embeddings (Audio representations)
        """

        emb, _ = self.infer_file(audio= audio, sr =sr)#emb, _ = self.infer_file(path2audio_file=path2audio_file)

        return emb

    @torch.no_grad()
    def verify_speakers(self, embs1, embs2):
        """
        Verify if two audio files are from the same speaker or not.

        Args:
            path2audio_file1: path to audio wav file of speaker 1
            path2audio_file2: path to audio wav file of speaker 2
            threshold: cosine similarity score used as a threshold to distinguish two embeddings (default = 0.7)

        Returns:
            True if both audio files are from same speaker, False otherwise
        """
        
        X = embs1 / torch.linalg.norm(embs1)
        Y = embs2 / torch.linalg.norm(embs2)
        # Score
        # Oblicz iloczyn skalarny dla każdej pary próbek
        dot_product = torch.einsum('ij,ij->i', X, Y)

        # Oblicz normy dla każdej próbki
        norms = torch.norm(X, dim=1) * torch.norm(Y, dim=1)

        # Oblicz podobieństwo kosinusowe
        similarity_score = dot_product / norms

        # Przeskaluj wynik do zakresu [0, 1]
        similarity_score = (similarity_score + 1) / 2
        #similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
        similarity_score = (similarity_score + 1) / 2
        # Decision
    
        return similarity_score
        """if similarity_score >= threshold:
            logging.info(" two audio files are from same speaker")
            return True
        else:
            logging.info(" two audio files are from different speakers")
            return False"""