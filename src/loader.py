import numpy as np
import os
import librosa
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

class Loader():
    
    def __init__(self, speech_path: str,save_path: str, verbose: bool, audio_size: int, limit: int) -> None:
        self.speech_path = speech_path
        self.verbose = verbose
        self.audio_size = audio_size
        self.save_path = save_path
        self.limit = limit
        
    def _list_files_actor(self, i: int) -> list:  
        base_path = f"{self.speech_path}/Actor_{i:02d}"
        return [f"{base_path}/{e}" for e in sorted(os.listdir(base_path))][:self.limit]
    
    def _extract_label(self, path: str) -> int:
        filename = path.split("/")[-1].replace(".mp4", "")
        parts = [int(e) for e in filename[:filename.find(".")].split("-")]
        emotion = parts[2] - 1
        vocal_channel = parts[1] - 1
        gender = parts[-1] % 2
        return np.array([emotion, vocal_channel, gender])
    
    def _load_audio(self, path: str) -> np.array:
        y, sr = librosa.load(path)
        assert len(y) <= self.audio_size
        y = librosa.util.fix_length(y, self.audio_size)
        return y
    
    def load(self, overwrite: bool):
        
        p1 = f"{self.save_path}/raw_load.npy"
        p2 = f"{self.save_path}/feature_load.npy"
    
        if not overwrite and os.path.exists(p1) and os.path.exists(p2):
            return np.load(open(p1, "rb")), np.load(open(p2, "rb"))
        
        if self.verbose: print("Loading audio from actors:")
        gen = tqdm(range(1, 25)) if self.verbose else range(1, 25)
        data, labels = [], []
        for actor in gen:
            l = self._list_files_actor(actor)
            data += list(map(self._load_audio, l))
            labels += list(map(self._extract_label, l))
        
        data, labels = np.array(data), np.array(labels)
        np.save(open(p1, "wb"), data)
        np.save(open(p2, "wb"), labels)
        
        return data, labels
    
    
class Augmenter():
    
    def __init__(self, loader, augmenter) -> None:
        self.loader = loader
        self.aug = augmenter
        
    def augment(self):
        data, labels = self.loader.load(False)
        # sample rate like this is dirty
        augmented_samples = self.aug(data, 22050)
        return augmented_samples, labels