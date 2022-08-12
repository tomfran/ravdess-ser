import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, scale


class Dataset():
    
    def __init__(self, original_data: tuple, augmented_data: tuple) -> None:
        self.X     = original_data[0]
        self.y     = original_data[1]
        self.augmented_data = augmented_data
        
    def _get_splits(self, train_perc: float) -> list:      
    
        X, y = self.X, self.y
        
        x_indices = np.arange(len(X))
        
        X_train_ind, X_test_ind, _, y_test = train_test_split(x_indices, y, train_size=train_perc, 
                                                            stratify=y, random_state=0)
        # extract the test from the original data
        X_test = X[X_test_ind]

        if self.augmented_data:
            augmented_X = (X[X_train_ind], self.augmented_data[0][X_train_ind])
            augmented_y = (y[X_train_ind], self.augmented_data[1][X_train_ind])
            X = np.concatenate(augmented_X, axis = 0)
            y = np.concatenate(augmented_y, axis = 0)
        else:
            X = X[X_train_ind]
            y = y[X_train_ind]
        
        return X, X_test, y, y_test
        
    
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
    
    def get_training_data(self, label: str, train_perc: float) -> list: 
        
        
        X_train, X_test, y_train,y_test = self._get_splits(train_perc)
        s = self._train_scaler(X_train)

        label_mapping = {"emotion" : 0, "vocal_channel" : 1, "gender" : 2}
        if label == "all":    
            return (self._scale_data(s, X_train), self._scale_data(s, X_test), 
                    y_train, y_test)
        else:
            i = label_mapping[label]
            return (self._scale_data(s, X_train), self._scale_data(s, X_test), 
                    y_train[ :, i], y_test[ :, i])
            
    def get_cross_val_generator(self, num_splits: int):
        # as we are not calling get_training data, the label never get column sliced, 
        # this will do, as cross validation is performed only on emotion
        self.y = self.y[:, 0]
        if self.augmented_data: 
            return KFoldGenerator(num_splits, self.X, self.y, 
                                  (self.augmented_data[0], self.augmented_data[1][:, 0]))
        return KFoldGenerator(num_splits, self.X, self.y, None)
    
    
class KFoldGenerator:
    
    def __init__(self, num_splits, X, y, augmented_data) -> None:
        self.num_splits = num_splits 
        self.X = X 
        self.y = y 
        self.augmented_data = augmented_data
        
    def __iter__(self):
        self.split_gen = StratifiedKFold(n_splits=self.num_splits).split(self.X, self.y)
        return self
    
    def __next__(self):
        # this raises stop iteration when completed
        train_index, test_index = next(self.split_gen)
        
        if self.augmented_data:
            augmented_X = (self.X[train_index], self.augmented_data[0][train_index])
            augmented_y = (self.y[train_index], self.augmented_data[1][train_index])
            train_X = np.concatenate(augmented_X, axis = 0)
            train_y = np.concatenate(augmented_y, axis = 0)
        else:
            train_X = self.X[train_index]
            train_y = self.y[train_index]
            
        test_X = self.X[test_index]
        test_y = self.y[test_index]
        return (train_X, train_y), (test_X, test_y)
        