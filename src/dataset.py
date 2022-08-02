import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale


class Dataset():
    
    def __init__(self, original_data: tuple, augmented_data: list) -> None:
        self.X     = original_data[0]
        self.y     = original_data[1]
        self.augmented_data = augmented_data
        
    def _get_splits(self, train_perc: float, val_perc: float, flatten: bool) -> list:      
    
        X, y = self.X, self.y
        
        
        if flatten: 
            a, b, c = X.shape
            X = X.reshape(a, b*c)

        x_indices = np.arange(len(X))
        
        X_train_ind, X_test_ind, y_train, y_test = train_test_split(x_indices, y, train_size=train_perc+val_perc, 
                                                            stratify=y, random_state=0)
        # extract the test from the original data
        X_test = X[X_test_ind]

        if self.augmented_data:
            
            if flatten: 
                def f(x):
                    a, b, c = x.shape
                    return x.reshape(a, b*c)
                self.augmented_data = [(f(e[0]), e[1]) for e in self.augmented_data]
            
            augmented_X = tuple([X[X_train_ind]] + [e[0][X_train_ind] for e in self.augmented_data])
            augmented_y = tuple([y[X_train_ind]] + [e[1][X_train_ind] for e in self.augmented_data])
            X = np.concatenate(augmented_X, axis = 0)
            y = np.concatenate(augmented_y, axis = 0)
        else:
            X = X[X_train_ind]
            y = y[X_train_ind]
            
        # split for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_perc/(train_perc+val_perc), 
                                                          stratify=y, random_state=0)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    
    def _train_scaler(self, train_data: np.array) -> StandardScaler:
        s = StandardScaler()
        
        x = train_data.shape
        if len(x) == 2:    
            s.fit(train_data)
        else:
            reshaped_data = train_data.reshape((x[0], x[1]*x[2]))
            s.fit(reshaped_data)
        return s
    
    def _scale_data(self, s, data):
        x = data.shape
        if len(x) == 2:    
            return s.transform(data)
        else:
            reshaped_data = data.reshape((x[0], x[1]*x[2]))
            scaled_data = s.transform(reshaped_data)
            return scaled_data.reshape((x[0], x[1], x[2]))
    
    def get_training_data(self, label: str, train_perc: float, val_perc: float, flatten=False) -> list: 
        
        
        X_train, X_val, X_test, y_train, y_val, y_test = self._get_splits(train_perc, val_perc, flatten)
        s = self._train_scaler(X_train)
        
        
        
        label_mapping = {"emotion" : 0, "vocal_channel" : 1, "gender" : 2}
        if label == "all":    
            return (self._scale_data(s, X_train), self._scale_data(s, X_val), self._scale_data(s, X_test), 
                    y_train, y_val, y_test)
        else:
            i = label_mapping[label]
            return (self._scale_data(s, X_train), self._scale_data(s, X_val), self._scale_data(s, X_test), 
                    y_train[ :, i], y_val[ :, i], y_test[ :, i])