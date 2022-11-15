import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


# TODO: 修改为自己的风格
class HRCELoss(nn.Module):
    def __init__(self, clip_length=300, delta=3, use_snr=False):
        super(HRCELoss, self).__init__()

        self.clip_length = clip_length
        self.time_length = 300
        self.delta = delta
        # self.delta_distribution = [0.4, 0.25, 0.05]
        self.low_bound = 40
        self.high_bound = 150

        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype=torch.float).cuda()
        self.bpm_range = self.bpm_range / 60.0

        two_pi_n = Variable(2 * math.pi * torch.arange(0, self.time_length, dtype=torch.float))
        hanning = Variable(torch.from_numpy(np.hanning(self.time_length)).type(torch.FloatTensor),
                           requires_grad=True).view(1, -1)  # 1 x N

        self.two_pi_n = two_pi_n.cuda()
        self.hanning = hanning.cuda()

        self.cross_entropy = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        self.l1 = nn.L1Loss()

        self.use_snr = use_snr

    def forward(self, wave, gt, fps):  # all variable operation
        """
        DFT: F(**k**) = \sum_{n = 0}^{N - 1} f(n) * \exp{-j2 \pi n **k** / N}
        :param wave: predict ecg  B x N
        :param gt: bvp B,
        :param fps:
        :return:
        """
        # 将 label 换算为 HR
        # gt x (fps x 60 / N) = gt / (N / (fps x 60))
        hr = torch.mul(gt, fps)  # 哈达玛积
        hr = hr * 60 / self.clip_length
        hr[hr.ge(self.high_bound)] = self.high_bound - 1  # truncate
        hr[hr.le(self.low_bound)] = self.low_bound

        # DFT
        batch_size = wave.shape[0]  # N
        k = self.bpm_range / fps  # DFT 中的 k = N x fk / fs, x N 与 (-j2 \pi n **k** / N) 抵消
        # 汉宁窗
        preds = wave * self.hanning  # B x N
        preds = preds.view(batch_size, 1, -1)  # B x 1 x N
        # 求 k
        k = k.view(batch_size, -1, 1)  # B x range x 1
        # 2 \pi n
        tmp = self.two_pi_n.repeat(batch_size, 1)
        tmp = tmp.view(batch_size, 1, -1)  # B x 1 x N
        # B x range
        complex_absolute = torch.sum(preds * torch.sin(k * tmp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(k * tmp), dim=-1) ** 2

        # 平移区间 [40, 150] -> [0, 110]
        target = hr - self.low_bound
        target = target.type(torch.long).view(batch_size)

        """# 预测心率
        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound"""

        if self.use_snr:
            norm_t = (torch.ones(batch_size).cuda() / torch.sum(complex_absolute, dim=1))
            norm_t = norm_t.view(-1, 1)  # B x 1
            complex_absolute = complex_absolute * norm_t  # B x range

            loss = self.cross_entropy(complex_absolute, target)

            idx_l = target - self.delta
            idx_l[idx_l.le(0)] = 0
            # truncate
            idx_r = target + self.delta
            idx_r[idx_r.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1

            loss_snr = 0.0
            for i in range(0, batch_size):
                loss_snr += 1 - torch.sum(complex_absolute[i, idx_l[i]:idx_r[i]])

            loss_snr = loss_snr / batch_size

            loss += loss_snr
        else:
            loss = self.cross_entropy(complex_absolute, target)

        return loss
