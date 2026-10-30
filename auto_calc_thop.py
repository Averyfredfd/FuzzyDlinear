import torch
import argparse
from thop import profile, clever_format

# 导入你写好的模型类（请确保路径正确）
from models.FuzzyDLinear import Model, SpatialGNNMixer


def count_spatial_gnn(m, x, y):
    """
    自定义 THOP Hook：让 thop 能够自动识别并统计底层 nn.Parameter (图网络参数)
    """
    # 1. 自动计算参数量：spatial_adj + alpha
    m.total_params[0] = m.spatial_adj.numel() + m.alpha.numel()

    # 2. 自动计算 MACs 乘加操作
    # 输入 x 是一个 tuple, x[0] 的形状为 [Batch, Channel, Pred_len]
    B, C, P = x[0].shape

    # sigmoid 运算量 (C*C) + einsum 矩阵运算量 (B * C * C * P)
    m.total_ops[0] = (C * C) + (B * C * C * P)


def main():
    # 1. 完全模拟你的 run.py 训练命令超参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=7)
    # 开启通道独立
    parser.add_argument('--individual', action='store_true', default=True)
    parser.add_argument('--moving_avg', type=int, default=25)

    configs = parser.parse_args([])

    # 2. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(configs).to(device).eval()

    # 3. 构造模拟输入流 (Batch=1, Seq=96, Channel=7)
    dummy_input = torch.randn(1, configs.seq_len, configs.enc_in, device=device)

    # 4. 注册自定义算子并使用 THOP 自动探测
    # 将 SpatialGNNMixer 绑定到自动统计规则上
    custom_ops = {SpatialGNNMixer: count_spatial_gnn}

    print("正在使用 THOP 自动遍历前向传播计算图...")
    macs, params = profile(model, inputs=(dummy_input,), custom_ops=custom_ops, verbose=False)

    # 5. 自动格式化（M = 兆，K = 千）
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")
    flops_formatted = clever_format([macs * 2], "%.3f")

    print("=" * 65)
    print(f"【训练配置】: seq_len=96, pred_len=96, enc_in=7, individual=True")
    print(f"【THOP 自动统计有效 Params】: {params_formatted} (精确值: {int(params)} 个)")
    print(f"【THOP 自动统计计算量 MACs】: {macs_formatted}")
    print(f"【THOP 自动推算算力 FLOPs】 : ≈ {flops_formatted} (单次推理)")
    print("=" * 65)


if __name__ == "__main__":
    main()