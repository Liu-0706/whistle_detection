import os
import numpy as np
import librosa
import tensorflow as tf

DATA_PATH = "data"
SAMPLE_RATE = 16000
DURATION = 1  # 秒
NUM_MFCC = 10

def extract_mfcc(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
    else:
        audio = audio[:SAMPLE_RATE]
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
    return mfcc.T  # (frames, 10)

def prepare_dataset():
    X, y = [], []
    for label, folder in enumerate(['noise', 'whistle']):
        folder_path = os.path.join(DATA_PATH, folder)
        for file in os.listdir(folder_path):
            mfcc = extract_mfcc(os.path.join(folder_path, file))
            X.append(mfcc)
            y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = prepare_dataset()
    np.savez('dataset.npz', X=X, y=y)
