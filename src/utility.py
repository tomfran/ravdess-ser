import os
import librosa as lb
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def show_duration_distribution(speech_path : str, song_path : str, limit_per_actor: int) -> None:
    
    p1 = "data/processed/speech_duration.npy"
    p2 = "data/processed/song_duration.npy"

    if os.path.exists(p1):
        speech_len = np.load(open(p1, "rb"))
        song_len = np.load(open(p2, "rb"))
    else:
        max_len = 0
        def get_file_paths(s):
            return sorted([f"{s}/{e}" for e in os.listdir(s)][:limit_per_actor])

        def get_audio_duration(a):
            y, sr = lb.load(a)
            nonlocal max_len
            max_len = max(len(y), max_len)
            return lb.get_duration(y, sr)

        song_len, speech_len = [], []

        for i in range(1, 25):
            els1 = get_file_paths(f"{speech_path}/Actor_{i:02d}")
            speech_len += list(map(get_audio_duration, els1))
            els2 = get_file_paths(f"{song_path}/Actor_{i:02d}")
            song_len += list(map(get_audio_duration, els2))
        song_len = np.array(song_len)
        speech_len = np.array(speech_len)
        np.save(open(p1, "wb"), speech_len)
        np.save(open(p2, "wb"), song_len)
        
    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(15,5))
    fig.suptitle('Samples length distribution')
    ax[0].set_title('Speech files')
    ax[0].set_xlabel("Seconds")
    sns.histplot(ax = ax[0], data=speech_len)
    ax[1].set_title('Song files')
    ax[1].set_xlabel("Seconds")
    sns.histplot(ax = ax[1], data=song_len)

    plt.show()
    return max(max(song_len), max(speech_len))

def plot_history(history):
    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(15,5))
    sns.lineplot(data=history.history['accuracy'], ax=ax[0])
    sns.lineplot(data=history.history['val_accuracy'], ax=ax[0])
    ax[0].set_title('Accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'val'], loc='upper left')

    sns.lineplot(data=history.history['loss'], ax=ax[1])
    sns.lineplot(data=history.history['val_loss'], ax=ax[1])
    ax[1].set_title('Loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'val'], loc='upper left')
    plt.show()
    
def plot_classes(y):
    c = Counter(y[:, 0].reshape(y.shape[0]))
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.barplot(ax=ax, x=list(c.keys()), y=list(c.values()))
    ax.set_xlabel("Classes")
    ax.set_ylabel("Frequency")
    ax.set_title("Class distribution")
    plt.show()
    
def plot_clusters(d):
    pca = PCA(n_components=2)
    X_train, _, _, y_train, _, _ = d.get_training_data(data="song", label="all", train_perc=0.6, val_perc=0.2)
    reduced_data = pca.fit_transform(X_train)
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    ax[0][0].set_title("Songs labeled by emotion")
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], ax=ax[0][0], hue=y_train[:, 0])
    ax[0][1].set_title("Songs labeled by gender")
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], ax=ax[0][1], hue=y_train[:, 2])
    X_train, _, _, y_train, _, _ = d.get_training_data(data="speech", label="all", train_perc=0.6, val_perc=0.2)
    reduced_data = pca.fit_transform(X_train)
    ax[1][0].set_title("Speech labeled by emotion")
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], ax=ax[1][0], hue=y_train[:, 0])
    ax[1][1].set_title("Speech labeled by gender")
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], ax=ax[1][1], hue=y_train[:, 2])
    X_train, _, _, y_train, _, _ = d.get_training_data(data="merge", label="all", train_perc=0.6, val_perc=0.2)
    reduced_data = pca.fit_transform(X_train)
    ax[2][0].set_title("Speech and songs labeled by emotion")
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], ax=ax[2][0], hue=y_train[:, 0])
    ax[2][1].set_title("Speech and songs labeled by gender")
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], ax=ax[2][1], hue=y_train[:, 2])
    plt.show()