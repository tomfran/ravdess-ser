import os
import librosa as lb
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from tqdm import tqdm

def show_duration_distribution(speech_path : str, save_path:str, limit_per_actor: int) -> None:
    
    p1 = f"{save_path}/speech_duration.npy"

    if os.path.exists(p1):
        speech_len = np.load(open(p1, "rb"))
    else:
        max_len = 0
        def get_file_paths(s):
            return sorted([f"{s}/{e}" for e in os.listdir(s)][:limit_per_actor])

        def get_audio_duration(a):
            y, sr = lb.load(a, sr=None)
            nonlocal max_len
            max_len = max(len(y), max_len)
            return lb.get_duration(y, sr)

        speech_len = []

        for i in tqdm(range(1, 25)):
            els1 = get_file_paths(f"{speech_path}/Actor_{i:02d}")
            speech_len += list(map(get_audio_duration, els1))
        print(max_len)
        speech_len = np.array([max_len] + speech_len)
        print(speech_len)
        np.save(open(p1, "wb"), speech_len)
        
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10,5))
    fig.suptitle('Samples length distribution')
    ax.set_title('Speech files')
    ax.set_xlabel("Seconds")
    sns.histplot(ax = ax, data=speech_len[1:])
    plt.show()
    return speech_len[0]

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
    
def plot_encoder_loss(history):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10,5))
    sns.lineplot(data=history.history['loss'], ax=ax)
    sns.lineplot(data=history.history['val_loss'], ax=ax)
    ax.set_title('Auto encoder reconstruction loss')
    ax.set_ylabel('mean absolute erorr')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'val'], loc='upper left')
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
    X_train, _, y_train, _ = d.get_training_data(label="all", train_perc=0.98)
    reduced_data = pca.fit_transform(X_train)
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].set_title("Speech labeled by emotion")
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], ax=ax[0], hue=y_train[:, 0])
    ax[1].set_title("K means clusters")
    clusters = KMeans(n_clusters=3).fit_predict(reduced_data)
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], ax=ax[1], hue=clusters)
    plt.show()