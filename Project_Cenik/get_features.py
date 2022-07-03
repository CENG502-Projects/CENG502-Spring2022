import glob
import os
import pickle
import torch
from python_speech_features import logfbank, fbank, sigproc
import numpy as np
import librosa
from tqdm import tqdm

# This part of the code is adapted from the Head Fusion Net [3] to extract the features


def process_data(path, t=2, overlap=1, RATE=40000):
    path = path.rstrip('/')
    wav_files = glob.glob(path + '/*.wav')
    meta_dict = {}

    LABEL_DICT1 = {
        '01': 'neutral',
        '02': 'frustration',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        # '06': 'fearful',
        '07': 'excitement'
        # '08': 'surprised'
    }

    print("constructing meta dictionary for {}...".format(path))
    for i, wav_file in enumerate(tqdm(wav_files)):
        label = str(os.path.basename(wav_file).split('-')[2])
        if (label not in LABEL_DICT1):
            continue
        label = LABEL_DICT1[label]
        wav_data, _ = librosa.load(wav_file, sr=RATE)
        X1 = []
        y1 = []
        index = 0
        if (t * RATE >= len(wav_data)):
            continue
        while (index + t * RATE < len(wav_data)):
            X1.append(wav_data[index:(index + t * RATE)])
            y1.append(label)
            index += (t - overlap) * RATE

        X1 = np.array(X1)
        meta_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }

    print("building X, y...")
    X = []
    y = []
    for k in meta_dict:
        X.append(meta_dict[k]['X'])
        y += meta_dict[k]['y']
    X = np.row_stack(X)
    y = np.array(y)
    assert len(X) == len(y), "X length and y length must match! X shape: {}, y length: {}".format(X.shape, y.shape)
    return X, y


def process_features(X, u=255):
    X = torch.from_numpy(X)
    max = X.max()
    X = X / max
    X = X.float()
    X = torch.sign(X) * (torch.log(1 + u * torch.abs(X)) / torch.log(torch.Tensor([1 + u])))
    X = X.numpy()
    return X


class FeatureExtractor(object):
    def __init__(self, rate):
        self.rate = rate

    def get_features(self, features_to_use, X):
        X_features = None
        accepted_features_to_use = ("logfbank", 'mfcc', 'fbank', 'melspectrogram', 'spectrogram', 'pase')
        if features_to_use not in accepted_features_to_use:
            raise NotImplementedError("{} not in {}!".format(features_to_use, accepted_features_to_use))
        if features_to_use in ('logfbank'):
            X_features = self.get_logfbank(X)
        if features_to_use in ('mfcc'):
            X_features = self.get_mfcc(X,26)
        if features_to_use in ('fbank'):
            X_features = self.get_fbank(X)
        if features_to_use in ('melspectrogram'):
            X_features = self.get_melspectrogram(X)
        if features_to_use in ('spectrogram'):
            X_features = self.get_spectrogram(X)
        if features_to_use in ('pase'):
            X_features = self.get_Pase(X)
        return X_features

    def get_logfbank(self, X):
        def _get_logfbank(x):
            out = logfbank(signal=x, samplerate=self.rate, winlen=0.040, winstep=0.010, nfft=1024, highfreq=4000,
                           nfilt=40)
            return out

        X_features = np.apply_along_axis(_get_logfbank, 1, X)
        return X_features

    def get_mfcc(self, X, n_mfcc=13):
        def _get_mfcc(x):
            mfcc_data = librosa.feature.mfcc(x, sr=self.rate, n_mfcc=n_mfcc)
            return mfcc_data

        X_features = np.apply_along_axis(_get_mfcc, 1, X)
        return X_features

    def get_fbank(self, X):
        def _get_fbank(x):
            out, _ = fbank(signal=x, samplerate=self.rate, winlen=0.040, winstep=0.010, nfft=1024)
            return out

        X_features = np.apply_along_axis(_get_fbank, 1, X)
        return X_features

    def get_melspectrogram(self, X):
        def _get_melspectrogram(x):
            mel = librosa.feature.melspectrogram(y=x, sr=self.rate, n_fft=800, hop_length=400)[np.newaxis, :]
            delta = librosa.feature.delta(mel)
            delta_delta = librosa.feature.delta(delta)
            out = np.concatenate((mel, delta, delta_delta))
            return out

        X_features = np.apply_along_axis(_get_melspectrogram, 1, X)
        return X_features

    def get_spectrogram(self, X):
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.logpowspec(frames, NFFT=3198)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        X_features = np.apply_along_axis(_get_spectrogram, 1, X)
        return X_features


    def get_Pase(self,X):
        return X

if __name__ == '__main__':
    X, y = process_data('C:\\Users\\hjjh\\PycharmProjects\\pythonProject\\CENG502-PROJECT\\IEMOCAP\\Test/IEMOCAP')
    n = len(X)
    train_indices = list(np.random.choice(range(n), int(n * 0.9), replace=False))
    valid_indices = list(set(range(n)) - set(train_indices))
    train_X = X[train_indices]
    train_y = y[train_indices]
    valid_X = X[valid_indices]
    valid_y = y[valid_indices]
    features = {'train_X': train_X, 'train_y': train_y,
                'val_X': valid_X, 'val_y': valid_y, }
    with open('meta_dicts.pkl', 'wb') as f:
        pickle.dump(features, f)

