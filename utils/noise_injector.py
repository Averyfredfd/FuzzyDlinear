import torch
import numpy as np

class NoiseInjector:
    """
    时序数据噪声注入器
    仅对输入特征注入噪声，标签保持干净

    spike_amp: 脉冲幅度，单位是"各通道标准差的倍数"(k)。设置后飞点 = x ± k*std_c，
               用来表达简历里的"不同倍率强脉冲噪声"；保持 None 时退化为旧公式
               randn * (intensity * 5 + 0.5)，兼容早期实验。
    """
    def __init__(self, noise_type='gaussian', intensity=0.1, spike_ratio=0.03, spike_amp=None):
        self.noise_type = noise_type      # 'gaussian', 'spike', 'mixed'
        self.intensity = intensity        # 高斯噪声标准差
        self.spike_ratio = spike_ratio    # 脉冲噪声比例
        self.spike_amp = spike_amp        # 脉冲幅度（标准差倍数），None=旧行为

    def _spike_values(self, x):
        """生成与 x 同形状的脉冲增量。"""
        if self.spike_amp is None:
            # 旧行为：独立高斯采样，幅度随 intensity 放大
            return torch.randn_like(x) * (self.intensity * 5 + 0.5)
        # 新行为：幅度 = k * 通道标准差，方向随机，模拟正负向跳变
        sigma = x.std(dim=1, keepdim=True).clamp_min(1e-6)     # [B, 1, C]
        sign = torch.where(torch.rand_like(x) < 0.5, -1.0, 1.0)
        return sign * self.spike_amp * sigma

    def __call__(self, x):
        """
        x: [Batch, Seq_len, Channel] tensor
        return: noisy_x (same shape)
        """
        if self.noise_type == 'none':
            return x
        if self.intensity == 0 and self.spike_amp is None:
            return x

        noisy_x = x.clone()

        if self.noise_type in ('gaussian', 'mixed'):
            gaussian_noise = torch.randn_like(noisy_x) * self.intensity
            noisy_x = noisy_x + gaussian_noise

        if self.noise_type in ('spike', 'mixed'):
            mask = torch.rand_like(noisy_x) < self.spike_ratio
            # 叠加脉冲增量（而不是直接替换），保证 k 就是"偏离正常值 k 个标准差"
            noisy_x = torch.where(mask, noisy_x + self._spike_values(noisy_x), noisy_x)

        return noisy_x