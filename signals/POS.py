import math

import numpy as np
from scipy import signal
from .utils import getRGB, detrend


def POS(frames, Fs):
    WinSec = 1.6
    rgb_data = getRGB(frames)
    N = rgb_data.shape[0]
    BVP = np.zeros((1, N))
    WinL = math.ceil(WinSec * Fs)

    for n in range(WinL, N):  # window_step = 1
        m = n - WinL
        Cn = np.true_divide(rgb_data[m:n, :], np.mean(rgb_data[m:n, :], axis=0))
        S = np.array([[0, 1, -1], [-2, 1, 1]]).dot(Cn.T)
        h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]  # h: WinL,
        mean_h = np.mean(h)
        for temp in range(len(h)):
            h[temp] = h[temp] - mean_h
        BVP[0, m:n] = BVP[0, m:n] + h  # overlap, accumulate

    BVP = detrend(BVP.T, 100).reshape(-1)
    b, a = signal.butter(1, [0.75 / Fs * 2, 3 / Fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP
