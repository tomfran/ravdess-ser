import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale


class Dataset():
    
    def __init__(self, data: tuple) -> None:
        self.X_speech   = data[0]
        self.X_song     = data[1]
        self.y_speech   = data[2]
        self.y_song     = data[3]
        
    def _get_splits(self, data: str, train_perc: float) -> list:
        if data == "speech":        
            return train_test_split(self.X_speech, self.y_speech, 
                                    train_size=train_perc, stratify=self.y_speech, 
                                    random_state=0)
        elif data == "song":
            return train_test_split(self.X_song, self.y_song, 
                                    train_size=train_perc, stratify=self.y_song, 
                                    random_state=0)
        raise Exception(f"Data must be speech or song, not {data}")
    
    def _train_scaler(self, train_data: np.array) -> StandardScaler:
        s = StandardScaler()
        s.fit(train_data)
        return s
    
    def get_training_data(self, data: str, train_perc: float) -> list: 
        X_train, X_test, y_train, y_test = self._get_splits(data, train_perc)
        scaler = self._train_scaler(X_train)
        return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test