import numpy as np
import librosa as lb
import os
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

class FeatureExtractor: 
    
    def __init__(self, 
                 raw_data: np.array, 
                 labels: np.array, 
                 save_path: str, 
                 file_name: str,
                 augmenter: callable, 
                 verbose: bool) -> None:
        
        self.raw_data = raw_data
        self.labels = labels
        self.save_path = save_path
        self.file_name = file_name
        self.augmenter = augmenter
        self.verbose = verbose
        self.sr = 48000
    
    def _extract_features(self, array: np.array) -> np.array:

        y = self.augmenter(array)
        mfccs = lb.feature.mfcc(y=y, sr=self.sr, n_mfcc=40)
        return mfccs
      
    def get_training_data(self, overwrite: bool) -> tuple:
        p =  f"{self.save_path}/{self.file_name}.npy"
        
        if not overwrite and os.path.exists(p):
            if self.verbose:
                print(f"Filename: {self.file_name} found on disk\n")
            return np.load(open(p, "rb")), self.labels
                    
        # apply the feature extraction to all data
                
        if self.verbose: 
            print(f"Extracting: \n\t- filename: {self.file_name}")
        
        res = []
        with Pool(processes=4) as pool:
            res = pool.map(self._extract_features, tqdm(self.raw_data))
        data = np.array(res)
        np.save(open(p, "wb"), data)
        
        return data, self.labels
    
def noise(x):
    noise_amp = 0.05*np.random.uniform()*np.amax(x)   
    x = x.astype('float64') + noise_amp * np.random.normal(size=x.shape[0])
    return x

def stretch(x, rate=0.8):
    data = lb.effects.time_stretch(x, rate)
    return lb.util.fix_length(data, 116247) 
    # return data

def speedpitch(x):
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.4  / length_change 
    tmp = np.interp(np.arange(0,len(x),speed_fac),np.arange(0,len(x)),x)
    minlen = min(x.shape[0], tmp.shape[0])
    x *= 0
    x[0:minlen] = tmp[0:minlen]
    return x

def identity(x):
    return x