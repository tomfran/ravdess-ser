import numpy as np
import librosa as lb
import os
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift


class FeatureExtractor: 
    
    def __init__(self, 
                 speech_path: str, 
                 song_path: str, 
                 save_path: str, 
                 verbose: bool, 
                 file_per_actor_limit: int, 
                 audio_fixed_size: int) -> None:
        
        self.speech_path          = speech_path
        self.song_path            = song_path   
        self.verbose              = verbose
        self.file_per_actor_limit = file_per_actor_limit
        self.save_path            = save_path
        self.audio_fixed_size     = audio_fixed_size
        self.augmenter = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ])
    
    def _extract_features(self, args: tuple) -> np.array:
        path, augment = args
        y, sr = lb.load(path, sr = None)
        y = lb.util.fix_length(y, self.audio_fixed_size)
        if augment:
            y = self.augmenter(y, sr)
            
        stft = np.abs(lb.stft(y))
        pitches, magnitudes = lb.piptrack(y=y, sr=sr, S=stft, fmin=70, fmax=400)
        pitch = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, 1].argmax()
            pitch.append(pitches[index, i])

        pitch_tuning_offset = lb.pitch_tuning(pitches)
        pitchmean = np.mean(pitch)
        pitchstd = np.std(pitch)
        pitchmax = np.max(pitch)
        pitchmin = np.min(pitch)

        cent = lb.feature.spectral_centroid(y=y, sr=sr)
        cent = cent / np.sum(cent)
        meancent = np.mean(cent)
        stdcent = np.std(cent)
        maxcent = np.max(cent)

        flatness = np.mean(lb.feature.spectral_flatness(y=y))

        mfccs = np.mean(lb.feature.mfcc(y=y, sr=sr, n_mfcc=50).T, axis=0)
        mfccsstd = np.std(lb.feature.mfcc(y=y, sr=sr, n_mfcc=50).T, axis=0)
        mfccmax = np.max(lb.feature.mfcc(y=y, sr=sr, n_mfcc=50).T, axis=0)

        chroma = np.mean(lb.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

        mel = np.mean(lb.feature.melspectrogram(y=y, sr=sr).T, axis=0)

        contrast = np.mean(lb.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)

        zerocr = np.mean(lb.feature.zero_crossing_rate(y))

        S, phase = lb.magphase(stft)
        meanMagnitude = np.mean(S)
        stdMagnitude = np.std(S)
        maxMagnitude = np.max(S)

        rmse = lb.feature.rms(S=S)[0]
        meanrms = np.mean(rmse)
        stdrms = np.std(rmse)
        maxrms = np.max(rmse)

        ext_features = np.array([
            flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
            maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
            pitch_tuning_offset, meanrms, maxrms, stdrms
        ])

        ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))
        return ext_features
     
    def _extract_label(self, path: str) -> int:
        filename = path.split("/")[-1].replace(".mp4", "")
        parts = [int(e) for e in filename[:filename.find(".")].split("-")]
        emotion = parts[2] - 1
        vocal_channel = parts[1] - 1
        gender = parts[-1] % 2
        return np.array([emotion, vocal_channel, gender])
    
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
        p5 = f"{self.save_path}/speech_augmented_feature_array.npy"
        p6 = f"{self.save_path}/song_augmented_feature_array.npy"
        
        if not overwrite and os.path.exists(p1):
            if self.verbose:
                print("Data found on disk")
            f1 = np.load(open(p1, "rb"))
            f2 = np.load(open(p2, "rb"))
            f3 = np.load(open(p3, "rb"))
            f4 = np.load(open(p4, "rb"))
            f5 = np.load(open(p5, "rb"))
            f6 = np.load(open(p6, "rb"))
            return f1, f2, f3, f4, f5, f6
            
        speech_feature_array, song_feature_array = [], []
        speech_augmented_feature_array, song_augmented_feature_array = [], []
        speech_label_array, song_label_array = [], []
        
        if self.verbose: 
            print("Extracting features from audio files.")
            
        gen = range(1, 25)
        if self.verbose: gen = tqdm(gen)
        for actor_id in gen:
            
            l1 = self._list_files_actor(actor_id, "speech")
            speech_feature_array += list(map(self._extract_features, zip(l1, [False]*len(l1))))
            speech_augmented_feature_array += list(map(self._extract_features, zip(l1, [True]*len(l1))))
            speech_label_array += list(map(self._extract_label, l1))
            
            l2 = self._list_files_actor(actor_id, "song")
            song_feature_array += list(map(self._extract_features, zip(l2, [False]*len(l2))))
            song_augmented_feature_array += list(map(self._extract_features, zip(l2, [True]*len(l2))))
            song_label_array += list(map(self._extract_label, l2))
            
        speech_feature_array = np.array(speech_feature_array)
        song_feature_array = np.array(song_feature_array)
        speech_augmented_feature_array = np.array(speech_augmented_feature_array)
        song_augmented_feature_array = np.array(song_augmented_feature_array)
        speech_label_array = np.array(speech_label_array)
        song_label_array = np.array(song_label_array)
        
        if self.verbose: 
            print("Saving to disk.")
        
        np.save(open(p1, "wb"), speech_feature_array)
        np.save(open(p2, "wb"), song_feature_array)
        np.save(open(p3, "wb"), speech_label_array)
        np.save(open(p4, "wb"), song_label_array)
        np.save(open(p5, "wb"), speech_augmented_feature_array)
        np.save(open(p6, "wb"), song_augmented_feature_array)
        
        
        return speech_feature_array, song_feature_array, speech_label_array, song_label_array, speech_augmented_feature_array, song_augmented_feature_array