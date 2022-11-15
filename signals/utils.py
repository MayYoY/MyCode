import numpy as np
from scipy import sparse, signal


def getRGB(frames):
    """
    取平均, 得到 RGB 时序信号
    :param frames: T x H x W x C
    :return:
    """
    RGB = []
    for frame in frames:
        acc = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(acc / (frame.shape[0] * frame.shape[1]))
    return np.asarray(RGB)


def detrend(input_signal, lambda_value):
    """
    去趋势化
    scipy.signal.detrend(input)
    :param input_signal:
    :param lambda_value:
    :return:
    """
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)  # T x T
    ones = np.ones(signal_length)  # T,
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])  # 1, -2, 1
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                       (signal_length - 2),
                       signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))),
                             input_signal)
    return filtered_signal
