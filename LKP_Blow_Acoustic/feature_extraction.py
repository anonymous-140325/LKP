"""
feature_extraction.py

Extracts a 43-dimensional acoustic feature vector from a WAV file.

Feature layout:
  [0:20]  MFCC means        (20 coefficients)
  [20:40] MFCC stds         (20 coefficients)
  [40]    Zero Crossing Rate (1 value)
  [41]    RMS Energy         (1 value)
  [42]    Spectral Centroid  (1 value)
"""

import librosa
import numpy as np


def extract_features(audio_path):

    y, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = np.concatenate([
        np.mean(mfcc, axis=1),                                        # [0:20]  MFCC means
        np.std(mfcc, axis=1),                                         # [20:40] MFCC stds
        [np.mean(librosa.feature.zero_crossing_rate(y))],             # [40]    ZCR
        [np.mean(librosa.feature.rms(y=y))],                          # [41]    RMS
        [np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))]      # [42]    Spectral Centroid
    ])

    print(f"Features extracted from {audio_path}")
    print(f"Feature shape: {features.shape}")

    return features
