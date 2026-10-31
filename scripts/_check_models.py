import sys
sys.path.insert(0, '.')
from types import SimpleNamespace

from scripts.noise_robustness import build_model, DATASET_DEFAULTS

for key, d in DATASET_DEFAULTS.items():
    cfg = SimpleNamespace(seq_len=d['seq_len'], pred_len=d['pred_len'], individual=True,
                          enc_in=d['enc_in'], moving_avg=25)
    for m in ['DLinear', 'FuzzyDLinear:membership']:
        mdl = build_model(m, cfg)
        n = sum(q.numel() for q in mdl.parameters())
        print(f'{key:12s} enc_in={d["enc_in"]:4d} {m:<26s} params={n / 1e6:.3f}M')