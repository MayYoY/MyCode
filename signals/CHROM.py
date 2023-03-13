import numpy as np
import math
from scipy import signal
from .utils import getRGB


def CHROM(frames, Fs):
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6

    rgb_data = getRGB(frames)  # 计算平均 RGB 信号
    FN = rgb_data.shape[0]  # T
    NyquistF = 0.5 * Fs
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], btype='bandpass')  # 巴特沃斯滤波

    WinL = math.ceil(WinSec * Fs)
    if WinL % 2:
        WinL = WinL + 1
    NWin = math.floor((FN - WinL // 2) / (WinL // 2))
    WinS = 0
    WinM = WinS + WinL // 2
    WinE = WinS + WinL
    total_len = (WinL // 2) * (NWin + 1)
    BVP = np.zeros(total_len)

    for i in range(NWin):
        RGBBase = np.mean(rgb_data[WinS: WinE, :], axis=0)  # 1 x 3
        RGBNorm = np.zeros((WinE - WinS, 3))  # len x 3
        for temp in range(WinS, WinE):
            RGBNorm[temp - WinS] = np.true_divide(rgb_data[temp], RGBBase) - 1
        Xs = np.squeeze(3 * RGBNorm[:, 0] - 2 * RGBNorm[:, 1])
        Ys = np.squeeze(1.5 * RGBNorm[:, 0] + RGBNorm[:, 1] - 1.5 * RGBNorm[:, 2])
        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys)

        Alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf - Alpha * Yf
        SWin = np.multiply(SWin, signal.hanning(WinL))

        if i == -1:
            BVP = SWin
        else:
            # temp = SWin[:int(WinL // 2)]
            BVP[WinS: WinM] = BVP[WinS: WinM] + SWin[: WinL // 2]  # overlap, accumulate
            BVP[WinM: WinE] = SWin[WinL // 2:]
        WinS = WinM
        WinM = WinS + WinL // 2
        WinE = WinS + WinL
    return BVP
