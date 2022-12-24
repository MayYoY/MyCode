import numpy as np
import scipy
import scipy.io
from scipy.signal import butter, periodogram, find_peaks, firwin, lombscargle
from scipy.sparse import spdiags
from scipy import interpolate


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def bandpass_hamming(Fs, taps_num=128, cutoff=None):
    """
    bandpass filtered (128-point Hamming window, 0.7–4 Hz)
    :return:
    """
    if cutoff is None:
        cutoff = [0.7, 4]
    return firwin(numtaps=taps_num, cutoff=cutoff, window="hamming",
                  fs=Fs, pass_zero="bandpass")


def average_filter(wave, win_size=5):
    """five-point moving average filter"""
    """temp = np.zeros(len(wave) + win_size - 1)
    temp[win_size // 2: win_size // 2 + len(wave)] = wave
    return np.convolve(temp, np.ones(win_size) / win_size, mode="valid")"""
    return np.convolve(wave, np.ones(win_size) / win_size, mode="same")


def mag2db(mag):
    return 20. * np.log10(mag)


def detrend(signal, lambda_value=100):
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


# TODO: respiration 是否需要 cumsum; 短序列心率计算不准确
def fft_physiology(signal: np.ndarray, target="pulse", Fs=30, diff=True, detrend_flag=True):
    """
    利用 fft 计算 HR or FR
    get filter -> detrend -> get psd and freq -> get mask -> get HR
    :param signal: T, or B x T
    :param target: pulse or respiration
    :param Fs:
    :param diff: 是否为差分信号
    :param detrend_flag: 是否需要 detrend
    :return:
    """
    if diff:
        signal = signal.cumsum(axis=-1)
    if detrend_flag:
        signal = detrend(signal, 100)
    # get filter and detrend
    if target == "pulse":
        # regular heart beats are 0.75 * 60 and 2.5 * 60
        [b, a] = butter(1, [0.75 / Fs * 2, 2.5 / Fs * 2], btype='bandpass')
    else:
        # regular respiration is 0.08 * 60 and 0.5 * 60
        [b, a] = butter(1, [0.08 / Fs * 2, 0.5 / Fs * 2], btype='bandpass')
    # bandpass
    signal = scipy.signal.filtfilt(b, a, np.double(signal))
    # get psd
    N = next_power_of_2(signal.shape[-1])
    freq, psd = periodogram(signal, fs=Fs, nfft=N, detrend=False)
    # get mask
    if target == "pulse":
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
    return phys.reshape(-1)


def peak_physiology(signal: np.ndarray, target="pulse", Fs=30, diff=True, detrend_flag=True):
    """
    利用 ibi 计算 HR or FR
    get filter -> detrend -> get psd and freq -> get mask -> get HR
    :param signal: T, or B x T
    :param target: pulse or respiration
    :param Fs:
    :param diff: 是否为差分信号
    :param detrend_flag: 是否需要 detrend
    :return:
    """
    if diff:
        signal = signal.cumsum(axis=-1)
    if detrend_flag:
        signal = detrend(signal, 100)
    if target == 'pulse':
        [b, a] = butter(1, [0.75 / Fs * 2, 2.5 / Fs * 2],
                        btype='bandpass')  # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / Fs * 2, 0.5 / Fs * 2], btype='bandpass')
    # bandpass
    signal = scipy.signal.filtfilt(b, a, np.double(signal))
    T = signal.shape[-1]
    signal = signal.reshape(-1, T)
    phys = []
    for s in signal:
        peaks, _ = find_peaks(s)
        phys.append(60 * Fs / np.diff(peaks).mean(axis=-1))

    return np.asarray(phys)


def cal_hrv(wave, Fs, bandpass=True, interpolation=True):
    """
     detrend -> normalize -> filters -> interpolate -> calculate PSD -> get hrv
    :param wave:
    :param Fs:
    :param bandpass: 
    :param interpolation: 发现插值的效果更好, 更接近 heartpy
    :return:
    """
    temp = wave.copy()
    ret = [0.] * 4  # LF, HF, LF / HF, RR
    # detrend
    temp = detrend(temp, lambda_value=10)
    # normalize
    temp = (temp - temp.mean()) / temp.std()
    # five-point moving average filter and bandpass filtered
    temp = average_filter(temp)
    if bandpass:
        b = bandpass_hamming(Fs)
        temp = scipy.signal.filtfilt(b, 1, temp)
    if interpolation:
        # interpolated with a cubic spline function at a sampling frequency of 256 Hz.
        fun = interpolate.CubicSpline(range(len(temp)), temp)
        x = np.linspace(0, len(temp) - 1, num=round(256 / Fs * len(temp)))
        y = fun(x)
        x = np.linspace(0, (len(y) - 1) / 256, num=len(y))  # 采样时间, 假设为均匀采样
    else:
        y = temp.copy()
        x = np.linspace(0, (len(y) - 1) / Fs, num=len(y))  # 采样时间, 假设为均匀采样
    #  HRV was performed by PSD estimation using the Lomb periodogram
    freq = np.arange(1, 501) / 1000  # 需要计算的频率点
    psd = lombscargle(x, y, freq, normalize=True)
    # LF, HF:the area under the PSD curve corresponding
    # to 0.04–0.15 and 0.15–0.4 Hz, respectively
    mask1 = np.argwhere((0.15 >= freq) & (freq >= 0.04))  # LF
    mask2 = np.argwhere((0.4 >= freq) & (freq >= 0.15))  # HF
    ret[0] = psd[mask1].sum()
    ret[1] = psd[mask2].sum()
    ret[2] = ret[0] / ret[1]
    # calculate RR from the center frequency of the HF peak f_{HFpeak} in the HRV PSD
    hf_freq = freq[mask2]
    hf_psd = psd[mask2]
    ret[3] = 60 * hf_freq[hf_psd.argmax()][0]
    # print(f"LF: {ret[0]}; HF: {ret[1]}; LF/HF: {ret[2]:.3f}; RR: {ret[3]:.3f}")
    return ret
