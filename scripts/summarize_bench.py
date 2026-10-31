"""Aggregate per-dataset benchmark jsons into comparison tables.

The benchmark writes one jsonl per dataset (results/ETTh1.jsonl, ...).
This script scans a directory, merges all records, and prints comparison
tables for the clean and noisy protocols.

Usage:
    python scripts/summarize_bench.py --input_dir results --out results/multi_summary.md
"""
import argparse
import glob
import json
import os

import numpy as np


def load_dir(d):
    paths = sorted(glob.glob(os.path.join(d, '*.jsonl')))
    paths = [p for p in paths if os.path.basename(p) not in ('_smoke.jsonl', '_smoke2.jsonl')]
    recs = []
    for p in paths:
        try:
            recs += [json.loads(l) for l in open(p, encoding='utf-8')]
        except Exception:
            pass
    return recs


def mean_by(recs, dataset, protocol, model, amp):
    rows = [r for r in recs if r['dataset'] == dataset and r['protocol'] == protocol
            and r['model'] == model and abs(r['amp'] - amp) < 1e-9]
    if not rows:
        return None
    return np.mean([r['mse'] for r in rows]), np.mean([r['mae'] for r in rows])


def fmt_pair(base, ours, key):
    if base is None or ours is None:
        return '--'
    b = base[key]
    o = ours[key]
    d = (b - o) / b * 100 if b != 0 else 0.0
    return f"{o:.4f} ({d:+.1f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', default='./results')
    ap.add_argument('--out', default='./results/multi_summary.md')
    args = ap.parse_args()

    recs = load_dir(args.input_dir)
    if not recs:
        raise SystemExit(f'no records found in {args.input_dir}')
    datasets = sorted({r['dataset'] for r in recs})
    protocols = sorted({r['protocol'] for r in recs})
    amps = [0.0, 5.0, 10.0]

    lines = []
    lines.append('# 多数据集噪声鲁棒性对比\n')
    lines.append('口径：标准化尺度 MSE/MAE；同一噪声种子配对比较；'
                 '输入 336 步 / 预测 96 步（ILI 为 36/24）。DLinear 与 FuzzyDLinear 同参数量（0.68M）。\n')

    for proto in protocols:
        lines.append(f'\n## 协议 {proto}\n')
        lines.append('| 数据集 | 噪声 | DLinear MSE | Fuzzy MSE (Δ) | DLinear MAE | Fuzzy MAE (Δ) |')
        lines.append('|---|---|---|---|---|---|')
        for ds in datasets:
            for amp in amps:
                base = mean_by(recs, ds, proto, 'DLinear', amp)
                ours = mean_by(recs, ds, proto, 'FuzzyDLinear:membership', amp)
                if base is None or ours is None:
                    continue
                labels = {0.0: '0σ', 5.0: '5σ', 10.0: '10σ'}
                lines.append(
                    f"| {ds} | {labels[amp]} | {base[0]:.4f} | {fmt_pair(base, ours, 0)} | "
                    f"{base[1]:.4f} | {fmt_pair(base, ours, 1)} |"
                )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('wrote', args.out)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()