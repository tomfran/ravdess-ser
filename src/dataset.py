import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale


class Dataset():
    
    def __init__(self, data: tuple, augment: bool) -> None:
        self.X_speech     = data[0]
        self.X_song       = data[1]
        self.y_speech     = data[2]
        self.y_song       = data[3]
        self.X_aug_speech = data[4]
        self.X_aug_song   = data[5]
        self.augment      = augment
        
    def _get_splits(self, data: str, train_perc: float, val_perc: float) -> list:
        if data == "speech":
            X, y, XX = self.X_speech, self.y_speech, self.X_aug_speech
        elif data == "song":
            X, y, XX = self.X_song, self.y_song, self.X_aug_song
        elif data == "merge":
            X, y, XX = np.concatenate((self.X_song, self.X_speech), axis=0), np.concatenate((self.y_song, self.y_speech), axis=0), np.concatenate((self.X_aug_song, self.X_aug_speech), axis=0)
        else: 
            raise Exception(f"Data must be speech or song, not {data}")
               
        x_indices = np.arange(len(X))
        
        X_train_ind, X_test_ind, y_train, y_test = train_test_split(x_indices, y, train_size=train_perc+val_perc, 
                                                            stratify=y, random_state=0)
        # extract the test from the train data
        X_test = X[X_test_ind]
        # add augmented data to X and y
        if self.augment:    
            X = np.concatenate((X[X_train_ind], XX[X_train_ind]), axis = 0)
            y = np.concatenate((y[X_train_ind], y[X_train_ind]), axis = 0)
        else:
            X = X[X_train_ind]
            y = y[X_train_ind]
            
        # split for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_perc/(train_perc+val_perc), 
                                                          stratify=y, random_state=0)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    
    def _train_scaler(self, train_data: np.array) -> StandardScaler:
        s = StandardScaler()
        s.fit(train_data)
        return s
    
    def get_training_data(self, data: str, train_perc: float, val_perc: float) -> list: 
        X_train, X_val, X_test, y_train, y_val, y_test = self._get_splits(data, train_perc, val_perc)
        scaler = self._train_scaler(X_train)
        return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test), y_train, y_val, y_test