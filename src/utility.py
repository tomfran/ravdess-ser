import os
import librosa as lb
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def show_duration_distribution(speech_path : str, song_path : str, limit_per_actor: int) -> None:

    max_len = 0

    def get_file_paths(s):
        return sorted([f"{s}/{e}" for e in os.listdir(s)][:limit_per_actor])

    def get_audio_duration(a):
        y, sr = lb.load(a)
        nonlocal max_len
        max_len = max(len(y), max_len)
        return lb.get_duration(y, sr)

    song_len, speech_len = [], []

    for i in tqdm(range(1, 25)):
        els1 = get_file_paths(f"{speech_path}/Actor_{i:02d}")
        speech_len += list(map(get_audio_duration, els1))
        els2 = get_file_paths(f"{song_path}/Actor_{i:02d}")
        song_len += list(map(get_audio_duration, els2))
    
    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(15,5))
    fig.suptitle('Samples length distribution')
    ax[0].set_title('Speech files')
    ax[0].set_xlabel("Seconds")
    sns.histplot(ax = ax[0], data=speech_len)
    ax[1].set_title('Song files')
    ax[1].set_xlabel("Seconds")
    sns.histplot(ax = ax[1], data=song_len)

    plt.show()
    return max_len