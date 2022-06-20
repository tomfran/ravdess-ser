import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale


class Dataset():
    
    def __init__(self, data: tuple) -> None:
        self.X_speech   = data[0]
        self.X_song     = data[1]
        self.y_speech   = data[2]
        self.y_song     = data[3]
        
    def _get_splits(self, data: str, train_perc: float, val_perc: float) -> list:
        if data == "speech":
            X, y = self.X_speech, self.y_speech
        elif data == "song":
            X, y = self.X_song, self.y_song
        elif data == "merge":
            X, y = np.concatenate((self.X_song, self.X_speech), axis=0), np.concatenate((self.y_song, self.y_speech), axis=0)
        else: 
            raise Exception(f"Data must be speech or song, not {data}")
            
        X_train, X_test, y_train, y_test =  train_test_split(X, y, train_size=train_perc+val_perc, 
                                                             stratify=y, random_state=0)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_perc/(train_perc+val_perc), 
                                                          stratify=y_train, random_state=0)
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    
    def _train_scaler(self, train_data: np.array) -> StandardScaler:
        s = StandardScaler()
        s.fit(train_data)
        return s
    
    def get_training_data(self, data: str, train_perc: float, val_perc: float) -> list: 
        X_train, X_val, X_test, y_train, y_val, y_test = self._get_splits(data, train_perc, val_perc)
        scaler = self._train_scaler(X_train)
        return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test), y_train, y_val, y_test