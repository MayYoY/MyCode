import torch
import numpy as np

from . import postprocess


class Accumulate:
    def __init__(self, n):
        self.n = n
        self.cnt = [0] * n
        self.acc = [0] * n

    def update(self, val: list, n):
        if not isinstance(n, list):
            n = [n] * self.n
        if not isinstance(val, list):
            val = [val] * self.n
        self.cnt = [a + b for a, b in zip(self.cnt, n)]
        self.acc = [a + b for a, b in zip(self.acc, val)]

    def reset(self):
        self.cnt = [0] * self.n
        self.acc = [0] * self.n


def cal_metric(preds: torch.Tensor, labels: torch.Tensor, config, post="fft", methods=None):
    """
    :param preds:
    :param labels:
    :param config:
    :param post: 后处理计算 phys 的方法 fft or peak
    :param methods: 评估指标
    :return:
    """
    if methods is None:
        methods = ["MAE", "RMSE", "MAPE", "R"]
    if post == "fft":
        pred_phys = postprocess.fft_physiology(preds.cpu().numpy(), target=config.target,
                                               Fs=config.Fs, diff=config.diff,
                                               detrend_flag=config.detrend_flag).reshape(-1)
        label_phys = postprocess.fft_physiology(labels.cpu().numpy(), target=config.target,
                                                Fs=config.Fs, diff=config.diff,
                                                detrend_flag=config.detrend_flag).reshape(-1)
    else:
        pred_phys = postprocess.peak_physiology(preds.cpu().numpy(), target=config.target,
                                                Fs=config.Fs, diff=config.diff,
                                                detrend_flag=config.detrend_flag).reshape(-1)
        label_phys = postprocess.peak_physiology(labels.cpu().numpy(), target=config.target,
                                                 Fs=config.Fs, diff=config.diff,
                                                 detrend_flag=config.detrend_flag).reshape(-1)
    ret = [] * len(methods)
    for m in methods:
        if m == "MAE":
            ret.append(np.abs(pred_phys - label_phys).mean())
        elif m == "RMSE":
            ret.append(np.sqrt((np.square(pred_phys - label_phys)).mean()))
        elif m == "MAPE":
            ret.append((np.abs((pred_phys - label_phys) / label_phys)).mean() * 100)
        elif m == "R":
            temp = np.corrcoef(pred_phys, label_phys)[0, 1]
            if np.isnan(temp).any() or np.isinf(temp).any():
                ret.append(-1 * np.ones(1))
            else:
                ret.append(temp[0, 1])
    return ret
