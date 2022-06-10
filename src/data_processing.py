from ssl import PROTOCOL_TLSv1_1
import pandas as pd
import numpy as np
import librosa as lb
import os
from tqdm import tqdm

class FeatureExtractor: 
    
    SAMPLE_FIXED_LEN = 140526
    
    def __init__(self, 
                 speech_path: str, 
                 song_path: str, 
                 save_path: str, 
                 verbose: bool, 
                 file_per_actor_limit: int) -> None:
        
        self.speech_path          = speech_path
        self.song_path            = song_path   
        self.verbose              = verbose
        self.file_per_actor_limit = file_per_actor_limit
        self.save_path            = save_path
    
    def _extract_features(self, path: str) -> np.array:       
        y, sr = lb.load(path, sr = None)
        y = lb.util.fix_length(y, FeatureExtractor.SAMPLE_FIXED_LEN)
        
        mfcc = lb.feature.mfcc(y, sr, n_mfcc=13)
        x, y = mfcc.shape
        return mfcc.reshape(x*y)
    
    def _extract_label(self, path: str) -> int:
        filename = path.split("/")[-1]
        parts = [int(e) for e in filename[:filename.find(".")].split("-")]
        # all labels starts from one
        emotion     = parts[2]-1
        intensity   = parts[3]-1
        statement   = parts[4]-1
        return np.array([emotion, intensity, statement])
    
    def _list_files_actor(self, i: int, mode: str) -> list:
        
        base_path = ""
        if mode == "speech":
            base_path = self.speech_path
        elif mode == "song":
            base_path = self.song_path
        else: 
            raise Exception("Unsupported mode for listing actor files")
        
        base_path += f"/Actor_{i:02d}"
        
        return [f"{base_path}/{e}" for e in sorted(os.listdir(base_path))][:self.file_per_actor_limit]
        
    def get_training_data(self, overwrite: bool) -> tuple:
        p1 = f"{self.save_path}/speech_feature_array.npy"
        p2 = f"{self.save_path}/song_feature_array.npy"
        p3 = f"{self.save_path}/speech_label_array.npy"
        p4 = f"{self.save_path}/song_label_array.npy"
        
        if not overwrite and os.path.exists(p1):
            if self.verbose:
                print("Data found on disk")
            f1 = np.load(open(p1, "rb"))
            f2 = np.load(open(p2, "rb"))
            f3 = np.load(open(p3, "rb"))
            f4 = np.load(open(p4, "rb"))
            return f1, f2, f3, f4
            
        
        speech_feature_array, song_feature_array = [], []
        speech_label_array, song_label_array = [], []
        if self.verbose: 
            print("Extracting features from audio files: ")
            
        gen = range(1, 25)
        if self.verbose: gen = tqdm(gen)
        for actor_id in gen:
            
            l1 = self._list_files_actor(actor_id, "speech")
            speech_feature_array += list(map(self._extract_features, l1))
            speech_label_array += list(map(self._extract_label, l1))
            
            l2 = self._list_files_actor(actor_id, "song")
            song_feature_array += list(map(self._extract_features, l2))
            song_label_array += list(map(self._extract_label, l2))
            
        speech_feature_array = np.array(speech_feature_array)
        song_feature_array = np.array(song_feature_array)
        speech_label_array = np.array(speech_label_array)
        song_label_array = np.array(song_label_array)
        
        np.save(open(p1, "wb"), speech_feature_array)
        np.save(open(p2, "wb"), song_feature_array)
        np.save(open(p3, "wb"), speech_label_array)
        np.save(open(p4, "wb"), song_label_array)
        
        return speech_feature_array, song_feature_array, speech_label_array, song_label_array