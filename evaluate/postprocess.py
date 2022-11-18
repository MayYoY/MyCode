import numpy as np
import scipy
import scipy.io
from scipy.signal import butter, periodogram
from scipy.sparse import spdiags


def mag2db(mag):
    return 20. * np.log10(mag)


def detrend(signal, lambda_value):
    """
    :param signal: T, or B x T
    :param lambda_value:
    :return:
    """
    T = signal.shape[-1]
    # observation matrix
    H = np.identity(T)  # T x T
    ones = np.ones(T)  # T,
    minus_twos = -2 * np.ones(T)  # T,
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (T - 2), T).toarray()
    designal = (H - np.linalg.inv(H + (lambda_value ** 2) * D.T.dot(D))).dot(signal.T).T
    return designal


def calculate_SNR(psd, freq, gtHR, target):
    """
    信噪比
    :param psd: predict PSD
    :param freq: predict frequency
    :param gtHR: ground truth
    :param target: signal type
    """
    gtHR = gtHR / 60
    gtmask1 = (freq >= gtHR - 0.1) & (freq <= gtHR + 0.1)
    gtmask2 = (freq >= gtHR * 2 - 0.1) & (freq <= gtHR * 2 + 0.1)
    sPower = psd[np.where(gtmask1 | gtmask2)].sum()
    if target == 'pulse':
        mask = (freq >= 0.75) & (freq <= 4)
    else:
        mask = (freq >= 0.08) & (freq <= 0.5)
    allPower = psd[np.where(mask)].sum()
    ret = mag2db(sPower / (allPower - sPower))
    return ret


"""def calculate_HR(pxx_pred, frange_pred, fmask_pred, pxx_label, frange_label, fmask_label):
    pred_HR = np.take(frange_pred, np.argmax(
        np.take(pxx_pred, fmask_pred), 0))[0] * 60
    ground_truth_HR = np.take(frange_label, np.argmax(
        np.take(pxx_label, fmask_label), 0))[0] * 60
    return pred_HR, ground_truth_HR"""


def calculate_physiology(signal: np.ndarray, target="pulse", fs=30, diff=True):
    """
    根据预测信号计算 HR or FR
    get filter -> detrend -> get psd and freq -> get mask -> get HR
    :param signal: T, or B x T
    :param target: pulse or respiration
    :param fs:
    :param diff: 是否为差分信号
    :return:
    """
    # TODO: respiration 是否需要 cumsum
    # get filter and detrend
    if target == "pulse":
        # regular heart beats are 0.75 * 60 and 2.5 * 60
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        if diff:
            signal = signal.cumsum(axis=-1)
        signal = detrend(signal, 100)
    else:
        # regular respiration is 0.08 * 60 and 0.5 * 60
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
        if diff:
            signal = signal.cumsum()
    # bandpass
    signal = scipy.signal.filtfilt(b, a, np.double(signal))
    # get psd
    freq, psd = periodogram(signal, fs=fs, nfft=4 * signal.shape[-1], detrend=False)
    # get mask
    if target == 'pulse':
        mask = np.argwhere((freq >= 0.75) & (freq <= 2.5))
    else:
        mask = np.argwhere((freq >= 0.08) & (freq <= 0.5))
    # get peak
    freq = freq[mask]
    if len(signal.shape) == 1:
        # phys = np.take(freq, np.argmax(np.take(psd, mask))) * 60
        idx = psd[mask.reshape(-1)].argmax(-1)
    else:
        idx = psd[:, mask.reshape(-1)].argmax(-1)
    phys = freq[idx] * 60
    return phys
