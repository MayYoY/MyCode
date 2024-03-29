import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn


class SNRLoss_dB_Signals(torch.nn.Module):
    def __init__(self, N=1024, pulse_band=None):
        super(SNRLoss_dB_Signals, self).__init__()
        if pulse_band is None:
            pulse_band = [45 / 60., 180 / 60.]
        self.N = N
        self.pulse_band = torch.tensor(pulse_band, dtype=torch.float32)

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, Fs=30):
        device = preds.device
        self.pulse_band = self.pulse_band.to(device)
        T = preds.shape[-1]  # T
        win_size = int(self.N / 256)  # 4

        freq = torch.linspace(0, Fs / 2, int(self.N / 2) + 1, dtype=torch.float32).to(device)

        low_idx = torch.argmin(torch.abs(freq - self.pulse_band[0]))
        up_idx = torch.argmin(torch.abs(freq - self.pulse_band[1]))

        preds = preds.view(-1, T)
        labels = labels.view(-1, T)

        # Generate GT heart indices from GT signals
        Y = torch.fft.rfft(labels, n=self.N, dim=1, norm='forward')  # 单边傅里叶变换, B x (N / 2 + 1)
        Y2 = torch.abs(Y) ** 2  # gt
        hr_idx = torch.argmax(Y2[:, low_idx:up_idx], dim=1) + low_idx

        P = torch.fft.rfft(preds, n=self.N, dim=1, norm='forward')
        P2 = torch.abs(P) ** 2  # predict

        loss = torch.empty((P.shape[0],), dtype=torch.float32)
        for i, hr in enumerate(hr_idx):
            signal_amp = torch.sum(P2[i, hr - win_size: hr + win_size])  # 信号项
            # 噪声项
            noise_amp = torch.sum(P2[i, low_idx: hr - win_size]) + torch.sum(P2[i, hr + win_size: up_idx])
            loss[i] = -10 * torch.log10(signal_amp / (noise_amp + 1e-7))  # 转换为分贝
        loss.to(device)
        return torch.mean(loss)


# TODO: 短序列心率计算不准确
class FreqCELoss(nn.Module):
    def __init__(self, T=300, delta=3, reduction="mean", use_snr=False):
        """
        :param T: 序列长度
        :param delta: 信号带宽, 带宽外的认为是噪声, 验证阶段的 delta 为 60 * 0.1
        :param reduction:
        :param use_snr:
        """
        super(FreqCELoss, self).__init__()
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
        gts = labels.clone()
        gts -= self.low_bound
        gts[gts.le(0)] = 0
        gts[gts.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1
        gts = gts.type(torch.long).view(B)

        """# 预测心率
        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound"""

        if self.use_snr:
            # CE loss
            loss = self.cross_entropy(complex_absolute, gts)
            # truncate
            left = gts - self.delta  # B,
            left[left.le(0)] = 0
            right = gts + self.delta
            right[right.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1
            # SNR
            loss_snr = 0.0
            for i in range(0, B):
                loss_snr += 1 - torch.sum(complex_absolute[i, left[i]:right[i]])
            if self.reduction == "mean":
                loss_snr = loss_snr / B
            loss += loss_snr
        else:
            loss = self.cross_entropy(complex_absolute, gts)

        return loss  # , whole_max_idx


class PeriodicLoss(torch.nn.Module):
    """ refer to https://arxiv.org/abs/2303.07944 """
    def __init__(self, N=1024, delta=6, pulse_band=None, weights=None):
        super(PeriodicLoss, self).__init__()
        if pulse_band is None:
            pulse_band = [40 / 60., 180 / 60.]
        if weights is None:
            weights = [1., 1., 1.]
        self.N = N
        self.delta = delta / 60  # bpm -> Hz
        self.pulse_band = torch.tensor(pulse_band, dtype=torch.float32)
        self.weights = weights

    def forward(self, preds: torch.Tensor, Fs: float):
        device = preds.device
        self.pulse_band = self.pulse_band.to(device)
        band_loss = torch.empty((preds.shape[0],), dtype=torch.float32, device=device)
        sparse_loss = torch.empty((preds.shape[0],), dtype=torch.float32, device=device)
        batch_psd = torch.zeros(int(self.N / 2) + 1, dtype=torch.float32, device=device)

        freq = torch.linspace(0, Fs / 2, int(self.N / 2) + 1, dtype=torch.float32).to(device)
        left = torch.argmin(torch.abs(freq - self.pulse_band[0]))
        right = torch.argmin(torch.abs(freq - self.pulse_band[1]))
        for i in range(len(preds)):

            temp = preds[i]
            psd = torch.fft.rfft(temp, n=self.N, norm="backward")
            psd = torch.abs(psd) ** 2
            batch_psd = batch_psd + psd

            band_loss[i] = (torch.sum(psd[: left]) + torch.sum(psd[right:])) / (1e-8 + torch.sum(psd))

            peak = torch.argmax(psd[left: right]) + left
            delta = max(1, round(self.delta / (Fs / 2 / len(freq))))
            sparse_loss[i] = (torch.sum(psd[left: peak - delta]) +
                              torch.sum(psd[peak + delta: right])) / torch.sum(psd[left: right] + 1e-8)

        batch_psd = batch_psd / (1e-8 + batch_psd.sum())
        Q = torch.zeros_like(batch_psd, device=device)
        Q[0] = batch_psd[0]
        for i in range(1, Q.shape[0]):
            Q[i] = batch_psd[i] + Q[i - 1]
        Q = Q.clamp(min=0, max=1)
        P = torch.distributions.Uniform(low=0, high=Fs / 2).cdf(freq).to(device)
        variance_loss = ((P - Q) ** 2).mean()

        band_loss = band_loss.mean()
        sparse_loss = sparse_loss.mean()
        return self.weights[0] * band_loss + self.weights[1] * sparse_loss + self.weights[2] * variance_loss


class PeriodicLoss_diffFs(torch.nn.Module):
    """
    refer to https://arxiv.org/abs/2303.07944
    Fs 波动, 无法计算 variance loss
    """
    def __init__(self, N=1024, delta=6, pulse_band=None, weights=None):
        super(PeriodicLoss_diffFs, self).__init__()
        if pulse_band is None:
            pulse_band = [40 / 60., 180 / 60.]
        if weights is None:
            weights = [1., 1., 1.]
        self.N = N
        self.delta = delta / 60  # bpm -> Hz
        self.pulse_band = torch.tensor(pulse_band, dtype=torch.float32)
        self.weights = weights

    def forward(self, preds: torch.Tensor, Fs: torch.Tensor):
        device = preds.device
        self.pulse_band = self.pulse_band.to(device)
        band_loss = torch.empty((preds.shape[0],), dtype=torch.float32, device=device)
        sparse_loss = torch.empty((preds.shape[0],), dtype=torch.float32, device=device)
        for i in range(len(preds)):
            # 需要逐样本计算, 因为 Fs 波动,
            freq = torch.linspace(0, Fs[i] / 2, int(self.N / 2) + 1, dtype=torch.float32).to(device)
            left = torch.argmin(torch.abs(freq - self.pulse_band[0]))
            right = torch.argmin(torch.abs(freq - self.pulse_band[1]))

            temp = preds[i]
            psd = torch.fft.rfft(temp, n=self.N, norm="backward")
            psd = torch.abs(psd) ** 2

            band_loss[i] = (torch.sum(psd[: left]) + torch.sum(psd[right:])) / (1e-8 + torch.sum(psd))

            peak = torch.argmax(psd[left: right]) + left
            delta = max(1, round(self.delta / float(Fs[i] / 2 / len(freq))))
            sparse_loss[i] = (torch.sum(psd[left: peak - delta]) +
                              torch.sum(psd[peak + delta: right])) / torch.sum(psd[left: right] + 1e-8)

        band_loss = band_loss.mean()
        sparse_loss = sparse_loss.mean()
        return self.weights[0] * band_loss + self.weights[1] * sparse_loss


if __name__ == '__main__':
    loss_fun = PeriodicLoss_diffFs()
    preds = torch.randn(4, 160)
    Fs = torch.tensor([30, 25, 30, 20])
    loss = loss_fun(preds, Fs)
    print(loss)
