import os
import numpy as np
import librosa
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

# CQCC feature extraction function
def compute_cqcc(y, sr, n_cqcc=15, hop_length=512):
    cqt_spec = np.abs(librosa.cqt(y=y, sr=sr, n_bins=84, bins_per_octave=12, hop_length=hop_length))
    log_cqt = np.log(cqt_spec + 1e-6)
    N_bins, _ = log_cqt.shape
    n = np.arange(N_bins)
    k = np.arange(n_cqcc)[:, None]
    dct_basis_cqcc = np.cos(np.pi * k * (2*n + 1) / (2 * N_bins))
    cqcc = dct_basis_cqcc.dot(log_cqt)
    return cqcc

# Preprocessing function
def extract_features_from_path(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_combined = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
    mfcc_mean = np.mean(mfcc_combined, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_delta_std = np.std(mfcc_delta, axis=1)
    mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
    mfcc_delta2_std = np.std(mfcc_delta2, axis=1)
    mfcc_liftered = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30, lifter=22)
    mfcc_liftered_mean = mfcc_liftered.mean(axis=1)

    # CQCC
    cqcc = compute_cqcc(y, sr, n_cqcc=15)
    cqcc_delta = librosa.feature.delta(cqcc, order=1)
    cqcc_delta2 = librosa.feature.delta(cqcc, order=2)
    cqcc_mean = np.mean(cqcc, axis=1)
    cqcc_std = np.std(cqcc, axis=1)
    cqcc_delta_mean = np.mean(cqcc_delta, axis=1)
    cqcc_delta_std = np.std(cqcc_delta, axis=1)
    cqcc_delta2_mean = np.mean(cqcc_delta2, axis=1)
    cqcc_delta2_std = np.std(cqcc_delta2, axis=1)

    # Feature vector concatenation
    feature_vector = np.concatenate([
        mfcc_mean, mfcc_std,
        mfcc_delta_mean, mfcc_delta_std,
        mfcc_delta2_mean, mfcc_delta2_std,
        mfcc_liftered_mean,
        cqcc_mean, cqcc_std,
        cqcc_delta_mean, cqcc_delta_std,
        cqcc_delta2_mean, cqcc_delta2_std
    ])

    return feature_vector

# Pipeline creation function
def make_pipe(n_kpca_components=15, lda_components=1):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('kpca',   KernelPCA(n_components=n_kpca_components, kernel='rbf')),
        ('lda',    LDA(n_components=lda_components, solver='svd')),
        ('svc',    SVC(kernel='rbf', probability=True, class_weight={'Fake': 9, 'Real': 1}))
    ])
    return pipe