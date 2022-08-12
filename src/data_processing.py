import numpy as np
import librosa
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
                 verbose: bool, 
                 only_mfcc: bool) -> None:
        
        self.raw_data = raw_data
        self.labels = labels
        self.save_path = save_path
        self.file_name = file_name
        self.verbose = verbose
        self.sr = 22050
        self.only_mfcc = only_mfcc
    
    def _extract_features(self, array: np.array) -> np.array:
        if self.only_mfcc:           
             features = np.mean(librosa.feature.mfcc(y=array, sr=self.sr, n_mfcc=40).T, axis=0)
        else:
            
            stft = np.abs(librosa.stft(array))
            pitches, magnitudes = librosa.piptrack(y=array, sr=self.sr, S=stft, fmin=70, fmax=400)
            
            pitch = []
            for i in range(magnitudes.shape[1]):
                index = magnitudes[:, 1].argmax()
                pitch.append(pitches[index, i])

            pitch_tuning_offset = librosa.pitch_tuning(pitches)
            pitchmean = np.mean(pitch)
            pitchstd = np.std(pitch)
            pitchmax = np.max(pitch)
            pitchmin = np.min(pitch)

            cent = librosa.feature.spectral_centroid(y=array, sr=self.sr)
            cent = cent / np.sum(cent)
            meancent = np.mean(cent)
            stdcent = np.std(cent)
            maxcent = np.max(cent)

            flatness = np.mean(librosa.feature.spectral_flatness(y=array))

            mfccs = np.mean(librosa.feature.mfcc(y=array, sr=self.sr, n_mfcc=50).T, axis=0)
            mfccsstd = np.std(librosa.feature.mfcc(y=array, sr=self.sr, n_mfcc=50).T, axis=0)
            mfccmax = np.max(librosa.feature.mfcc(y=array, sr=self.sr, n_mfcc=50).T, axis=0)

            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=self.sr).T, axis=0)

            mel = np.mean(librosa.feature.melspectrogram(y=array, sr=self.sr).T, axis=0)

            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=self.sr).T, axis=0)

            zerocr = np.mean(librosa.feature.zero_crossing_rate(array))

            S, phase = librosa.magphase(stft)
            meanMagnitude = np.mean(S)
            stdMagnitude = np.std(S)
            maxMagnitude = np.max(S)

            rmse = librosa.feature.rms(S=S)[0]
            meanrms = np.mean(rmse)
            stdrms = np.std(rmse)
            maxrms = np.max(rmse)

            ext_features = np.array([
                flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
                maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
                pitch_tuning_offset, meanrms, maxrms, stdrms
            ])

            features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))
        return features
      
    def get_training_data(self, overwrite: bool) -> tuple:
        p =  f"{self.save_path}/{self.file_name}.npy"
        
        if not overwrite and os.path.exists(p):
            if self.verbose:
                print(f"Filename: {self.file_name} found on disk\n")
            return np.load(open(p, "rb")), self.labels
                
        if self.verbose: 
            print(f"Extracting: \n\t- filename: {self.file_name}")
        
        res = []
        with Pool(processes=4) as pool:
            res = pool.map(self._extract_features, tqdm(self.raw_data))
        data = np.array(res)
        np.save(open(p, "wb"), data)
        
        return data, self.labels