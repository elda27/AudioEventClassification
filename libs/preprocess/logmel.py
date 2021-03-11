import librosa
import numpy as np


def logmel_spectrum(
    audio, sr, hop_length=256, n_mels=128,
    fmin=0.0, fmax=None
):
    mag = librosa.feature.melspectrogram(
        audio, sr=sr, hop_length=hop_length, n_mels=n_mels,
        fmin=fmin, fmax=fmax,
    )
    logmag = np.log(mag + 1e8)
    return logmag
