# calc_complexity.py
import torch
from argparse import Namespace
from models.FuzzyDLinear import Model
import thop

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def manual_params(seq_len, pred_len, channels):
    """
    仅针对当前 Model(individual=True) 的参数量公式。
    注意：Linear_Decoder 在 forward 中未使用，但会被 model.parameters() 统计，
    所以这里也把它算进去。
    """
    L, P, C = seq_len, pred_len, channels
    H = L // 2

    # FuzzyConfidenceFilter:
    # Linear(L -> H): L*H + H
    # Linear(H -> L): H*L + L
    fuzzy = 2 * L * H + H + L

    # SpatialGNNMixer:
    # spatial_adj: C*C
    # alpha: 1
    spatial = C * C + 1

    # individual=True:
    # Linear_Seasonal: C * (P*L + P)
    # Linear_Trend:    C * (P*L + P)
    # Linear_Decoder:  C * (P*L + P)  (未使用但存在参数)
    linear = 3 * C * (P * L + P)

    return fuzzy + spatial + linear


def manual_flops(B, L, P, C, K):
    """
    手动统计当前 forward 的矩阵乘 MACs 和近似 FLOPs。
    B: batch size
    L: seq_len
    P: pred_len
    C: channels / enc_in
    K: moving_avg kernel_size，要求为奇数
    """
    assert K % 2 == 1, "moving_avg 的 kernel_size 需要为奇数，默认 25"
    H = L // 2

    # 矩阵乘/Linear/einsum 的 MACs
    # FuzzyConfidenceFilter 两个 Linear
    macs_fuzzy = B * C * L * H + B * C * H * L

    # Linear_Seasonal + Linear_Trend
    macs_linear = 2 * B * C * L * P

    # SpatialGNNMixer 中的 einsum: 'b j l, c j -> b c l'
    macs_spatial = B * C * C * P

    matmul_macs = macs_fuzzy + macs_linear + macs_spatial
    matmul_flops = 2 * matmul_macs   # 一次乘加按 2 FLOPs 计

    # 额外逐元素/池化算子近似 FLOPs
    extra_flops = (
        B * C * L * K       # AvgPool1d 近似
        + B * C * L         # x - moving_mean
        + B * C * H * 8     # GELU 粗略估计
        + B * C * L * 4     # Sigmoid(fuzzy_weight)
        + B * C * L         # seasonal_part * fuzzy_weight
        + B * C * P         # seasonal_output + trend_output
        + C * C * 4         # Sigmoid(spatial_adj)
        + 2 * B * C * P     # alpha * spatial_features + x
    )

    total_flops_approx = matmul_flops + extra_flops
    return matmul_macs, matmul_flops, total_flops_approx


def main():
    # 与 run.py 默认值保持一致，可按需修改
    B = 1
    L = 96       # seq_len
    P = 96       # pred_len
    C = 7        # enc_in
    K = 25       # moving_avg

    # 注意：当前 forward 中 self.Linear_Seasonal[i] 强制按通道索引，
    # 因此必须 individual=True，否则 individual=False 分支会报错。
    individual = True

    configs = Namespace(
        seq_len=L,
        pred_len=P,
        enc_in=C,
        individual=individual,
        moving_avg=K,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(configs).to(device).eval()

    total_params, trainable_params = count_params(model)
    manual_p = manual_params(L, P, C)

    print("=" * 70)
    print(f"[Config] B={B}, seq_len={L}, pred_len={P}, channels={C}, moving_avg={K}")
    print(f"[Params] total     = {total_params / 1e6:.6f} M")
    print(f"[Params] trainable = {trainable_params / 1e6:.6f} M")
    print(f"[Params] manual    = {manual_p / 1e6:.6f} M")
    print("=" * 70)

    x = torch.randn(B, L, C, device=device)

    with torch.no_grad():
        y = model(x)
    print(f"[Output] shape = {tuple(y.shape)}")

    # 1) PyTorch 自带 FlopCounterMode，需要 PyTorch >= 2.0
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with torch.no_grad():
            with FlopCounterMode(display=False) as f:
                model(x)

        auto_flops = f.get_total_flops()
        print(f"[FlopCounterMode] FLOPs = {auto_flops / 1e9:.6f} GFLOPs (B={B})")
    except Exception as e:
        print(f"[FlopCounterMode] 不可用或统计失败: {repr(e)}")

    # 2) 手动公式
    macs, mm_flops, total_flops = manual_flops(B, L, P, C, K)
    print(f"[Manual] MatMul MACs      = {macs / 1e9:.6f} GMACs")
    print(f"[Manual] MatMul FLOPs     = {mm_flops / 1e9:.6f} GFLOPs")
    print(f"[Manual] total approx     = {total_flops / 1e9:.6f} GFLOPs (B={B})")

    # 3) THOP 可选，注意 THOP 对 einsum 可能统计不全
    try:
        from thop import profile

        macs_thop, params_thop = profile(model, inputs=(x,), verbose=False)
        print(f"[THOP] MACs   = {macs_thop / 1e9:.6f} GMACs")
        print(f"[THOP] Params = {params_thop / 1e6:.6f} M")
    except Exception as e:
        print(f"[THOP] 未安装或统计失败: {repr(e)}")

    print("=" * 70)


if __name__ == "__main__":
    main()