import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation, Flatten, MaxPool1D, Conv1D, LSTM
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import L2, L1L2
from tensorflow.keras.optimizers import RMSprop
from .utility import plot_history

from collections import Counter
import numpy as np

def build_train_simple(train, test, model):
    model.fit(train[0], train[1])
    test_score = model.score(test[0], test[1])
    return test_score

def class_weight(y: np.array):
    seq = y.copy().reshape(y.shape[0])
    t, L = len(np.unique(seq)), len(seq)
    return {k : L/(t*v) for k, v in Counter(seq).items()}

def build_nn(shape: tuple):
    inp = Input(shape=shape)
    x = Dense(128, activation="relu", kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5))(inp)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu", kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5))(inp)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu", kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5))(x)
    x = Dropout(0.3)(x)
    out = Dense(8, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=RMSprop(lr=0.00005, decay=1e-6), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model


def build_cnn_slow(shape: tuple):
    inp = Input(shape=shape)
    x = Conv1D(32, 8, padding="same")(inp)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    x = MaxPool1D()(x)

    x = Conv1D(64, 6, padding="same")(inp)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    x = MaxPool1D()(x)
    
    x = Conv1D(128, 4, padding="same")(inp)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    x = MaxPool1D()(x)
        
    x = Flatten()(x)
    x = Dense(128)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    out = Dense(8, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=RMSprop(lr=0.00001, decay=1e-6), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model

def build_cnn(shape: tuple):
    inp = Input(shape=shape)
    x = Conv1D(32, 8, padding="same")(inp)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    x = MaxPool1D()(x)

    x = Conv1D(64, 6, padding="same")(inp)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    x = MaxPool1D()(x)
    
    x = Conv1D(128, 4, padding="same")(inp)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    x = MaxPool1D()(x)
        
    x = Flatten()(x)
    x = Dense(128)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    out = Dense(8, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=RMSprop(lr=0.00001, decay=1e-6), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])
    return model

def build_lstm(shape: tuple):
	inp = Input(shape=shape)
	x = LSTM(128, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5))(inp)
	x = Dropout(0.3)(x)
	x = Dense(128, activation="relu", kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5))(x)
	x = Dropout(0.3)(x)
	out = Dense(8, activation="softmax")(x)
	model = Model(inputs=inp, outputs=out)
	model.compile(optimizer=RMSprop(lr=0.0001, decay=1e-6), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])
	return model


def train_nn(train_data, test_data, build_fun, epochs, verbose, plot):
    m = build_fun(train_data[0][0].shape)
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="data/models/tmp.hdf5",
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    w = class_weight(train_data[1])
    
    history = m.fit(train_data[0], train_data[1], validation_split=0.2, 
                          epochs=epochs, verbose=verbose, callbacks=[model_checkpoint_callback], class_weight=w)      
        
    m.load_weights("data/models/tmp.hdf5")
    l, a = m.evaluate(test_data[0], test_data[1], verbose=0)
    if plot:
        plot_history(history)
    return l, a