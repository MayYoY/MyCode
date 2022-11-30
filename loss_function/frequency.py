import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn


# TODO: 短序列心率计算不准确
class HRCELoss(nn.Module):
    def __init__(self, T=300, delta=3, reduction="mean", use_snr=False):
        """
        :param T: 序列长度
        :param delta: 信号带宽, 带宽外的认为是噪声, 验证阶段的 delta 为 60 * 0.1
        :param reduction:
        :param use_snr:
        """
        super(HRCELoss, self).__init__()
        self.T = T
        self.delta = delta
        self.low_bound = 40
        self.high_bound = 150
        # for DFT
        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype=torch.float) / 60.
        self.two_pi_n = Variable(2 * math.pi * torch.arange(0, self.T, dtype=torch.float))
        self.hanning = Variable(torch.from_numpy(np.hanning(self.T)).type(torch.FloatTensor),
                                requires_grad=True).view(1, -1)  # 1 x N
        # criterion
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)

        self.use_snr = use_snr

    def forward(self, wave, labels, fps):
        """
        DFT: F(**k**) = \sum_{n = 0}^{N - 1} f(n) * \exp{-j2 \pi n **k** / N}
        :param wave: predict ecg  B x N
        :param labels: heart rate B,
        :param fps: B,
        :return:
        """
        # 多 GPU 训练下, 确保同一 device
        self.bpm_range = self.bpm_range.to(wave.device)
        self.two_pi_n = self.two_pi_n.to(wave.device)
        self.hanning = self.hanning.to(wave.device)

        # DFT
        B = wave.shape[0]
        # DFT 中的 k = N x fk / fs, x N 与 (-j2 \pi n **k** / N) 抵消
        k = self.bpm_range[None, :] / fps[:, None]
        k = k.view(B, -1, 1)  # B x range x 1
        # 汉宁窗
        preds = wave * self.hanning  # B x N
        preds = preds.view(B, 1, -1)  # B x 1 x N
        # 2 \pi n
        temp = self.two_pi_n.repeat(B, 1)
        temp = temp.view(B, 1, -1)  # B x 1 x N
        # B x range
        complex_absolute = torch.sum(preds * torch.sin(k * temp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(k * temp), dim=-1) ** 2
        # 归一化
        norm_t = (torch.ones(B, device=wave.device) / torch.sum(complex_absolute, dim=1))
        norm_t = norm_t.view(-1, 1)  # B x 1
        complex_absolute = complex_absolute * norm_t  # B x range
        # 平移区间 [40, 150] -> [0, 110]
        labels -= self.low_bound
        labels = labels.type(torch.long).view(B)

        """# 预测心率
        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound"""

        if self.use_snr:
            # CE loss
            loss = self.cross_entropy(complex_absolute, labels)
            # truncate
            left = labels - self.delta  # B,
            left[left.le(0)] = 0
            right = labels + self.delta
            right[right.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1
            # SNR
            loss_snr = 0.0
            for i in range(0, B):
                loss_snr += 1 - torch.sum(complex_absolute[i, left[i]:right[i]])
            if self.reduction == "mean":
                loss_snr = loss_snr / B
            loss += loss_snr
        else:
            loss = self.cross_entropy(complex_absolute, labels)

        return loss  # , whole_max_idx
