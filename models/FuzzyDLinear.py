import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ---------------------------------------------------------
# [Inject 1]: Fuzzy Confidence Filter (模糊置信度提纯器)
# 作用：抑制 Seasonal 高频分量中的电网跳变噪声
# ---------------------------------------------------------
class FuzzyConfidenceFilter(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.membership_fc = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            nn.GELU(),
            nn.Linear(seq_len // 2, seq_len),
            nn.Sigmoid()  # 输出 0~1 的置信度权重
        )

    def forward(self, seasonal_part):
        # seasonal_part shape: [Batch, Channel, Seq_len]
        # nn.Linear 自动作用于最后一维 Seq_len
        fuzzy_weight = self.membership_fc(seasonal_part)
        return seasonal_part * fuzzy_weight


# ---------------------------------------------------------
# [Inject 1b]: Fuzzy Soft-Threshold (基于隶属度的模糊软截断)
# 作用：在输入端识别遥测飞点，用隶属度做"软替换"而不是硬裁剪
#   1) 局部中位数 m_t 作为该时刻对脉冲不敏感的"可信期望值"
#   2) 局部 MAD 作为尺度，归一化偏差 d_t = |x_t - m_t| / MAD_t
#   3) 隶属度 mu_t = 1 / (1 + (d_t / c)^(2p))：正常点 mu≈1，飞点 mu→0
#   4) 输出 mu_t * x_t + (1 - mu_t) * m_t，权重连续，不产生台阶
# ---------------------------------------------------------
class FuzzySoftThreshold(nn.Module):
    def __init__(self, window=5, center=3.0, sharpness=2.0, eps=1e-3):
        super().__init__()
        self.window = window
        self.center = center          # 隶属度降到 0.5 时的偏差（单位：MAD）
        self.sharpness = sharpness    # 隶属度曲线陡峭程度
        self.eps = eps                # 防止 MAD=0 时除零

    def forward(self, x):
        # x: [Batch, Seq_len, Channel]
        if self.window <= 1:
            return x
        half = self.window // 2
        z = x.permute(0, 2, 1)                                  # [B, C, L]
        zp = F.pad(z, (half, half), mode='reflect')
        win = zp.unfold(-1, self.window, 1)                     # [B, C, L, w]
        med = win.median(dim=-1).values                         # [B, C, L]
        mad = (win - med.unsqueeze(-1)).abs().median(dim=-1).values
        dev = (z - med).abs() / (mad + self.eps)                # 归一化偏差
        mu = 1.0 / (1.0 + torch.clamp(dev / self.center, max=1e6) ** (2 * self.sharpness))
        out = mu * z + (1 - mu) * med
        return out.permute(0, 2, 1)


# ---------------------------------------------------------
# [Inject 2]: Topology-aware Global Mixer (图空间拓扑混淆器)
# 作用：聚合不同 Channel(变电站/节点) 之间的空间潮流关系
# ---------------------------------------------------------
class SpatialGNNMixer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 初始化为单位矩阵，默认一开始“只看自己，不看邻居”
        self.spatial_adj = nn.Parameter(torch.eye(channels))

        # 加入一个可学习的门控标量，让模型自动控制空间特征的流入量
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        # x shape: [Batch, Channel, Pred_len]

        # 不用 softmax，防止过度平滑，允许负相关的拓扑连接
        # 使用 sigmoid 将邻接矩阵约束在 0~1 之间
        adj = torch.sigmoid(self.spatial_adj)

        # 提取空间特征
        spatial_features = torch.einsum('b j l, c j -> b c l', x, adj)

        # 【核心抢救】：残差连接！自身特征(x) 为主，空间特征(spatial_features) 加权辅助
        out = x + self.alpha * spatial_features
        return out


class Model(nn.Module):
    """
    Fuzzy-Spatial DLinear (针对南网业务定制改进版)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # [Modified]: 修复 kernel_size 硬编码，优先读取配置
        kernel_size = getattr(configs, 'moving_avg', 25)
        self.decompsition = series_decomp(kernel_size)

        self.individual = configs.individual
        self.channels = configs.enc_in

        # [Inject 3]: 实例化你个人的研究模块
        # fuzzy_mode: 'mlp' = 学出来的置信度掩码；'membership' = 隶属度软截断（输入级）
        self.fuzzy_mode = getattr(configs, 'fuzzy_mode', 'mlp')
        if self.fuzzy_mode == 'membership':
            self.fuzzy_input = FuzzySoftThreshold(window=getattr(configs, 'fuzzy_window', 5),
                                                  center=getattr(configs, 'fuzzy_center', 3.0),
                                                  sharpness=getattr(configs, 'fuzzy_sharpness', 2.0))
            self.fuzzy_filter = None
        else:
            self.fuzzy_input = None
            self.fuzzy_filter = FuzzyConfidenceFilter(self.seq_len)
        self.spatial_mixer = SpatialGNNMixer(self.channels)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()  # TSlib 遗留物，保持原样防报错
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Decoder.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Decoder = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # 0. 隶属度软截断：在分解前先把飞点按置信度软替换掉
        if self.fuzzy_input is not None:
            x = self.fuzzy_input(x)

        seasonal_init, trend_init = self.decompsition(x)

        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # 1. 模糊提纯 (仅对高频残差)
        if self.fuzzy_filter is not None:
            seasonal_init = self.fuzzy_filter(seasonal_init)

        # 2. 独立/共享线性映射
        #    individual=True  : 每个通道一套权重，避免异构特征互相干扰
        #    individual=False : 所有通道共享一套权重，参数量最小（轻量化档位）
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # nn.Linear 作用在最后一维(Seq_len)上，[B, C, L] -> [B, C, P]
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # 时间维度上的融合: [Batch, Channel, Pred_len]
        x_temporal = seasonal_output + trend_output

        # 3. 空间维度的残差混淆 (安全地融合邻居信息)
        x_final = self.spatial_mixer(x_temporal)

        return x_final.permute(0, 2, 1)