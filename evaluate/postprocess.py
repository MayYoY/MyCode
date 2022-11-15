import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags


def mag2db(mag):
    return 20. * np.log10(mag)


def calculate_SNR(pxx_pred, f_pred, currHR, signal):
    """
    信噪比
    :param pxx_pred: predict PSD
    :param f_pred: predict frequency
    :param currHR: ground truth
    :param signal: signal type
    """
    currHR = currHR / 60
    f = f_pred
    pxx = pxx_pred
    gtmask1 = (f >= currHR - 0.1) & (f <= currHR + 0.1)
    gtmask2 = (f >= currHR * 2 - 0.1) & (f <= currHR * 2 + 0.1)
    # np.take(): 按下标取值
    sPower = np.sum(np.take(pxx, np.where(gtmask1 | gtmask2)))
    if signal == 'pulse':
        fmask2 = (f >= 0.75) & (f <= 4)
    else:
        fmask2 = (f >= 0.08) & (f <= 0.5)
    allPower = np.sum(np.take(pxx, np.where(fmask2)))
    SNR_temp = mag2db(sPower / (allPower - sPower))
    return SNR_temp
