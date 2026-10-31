"""Generate the two figures used in docs/项目文档.md from results/bench.jsonl.

Usage:
    python scripts/make_figures.py --input ./results/bench.jsonl --out ./docs/figs

fig1_noise_curve.png : MSE / MAE vs impulse amplitude (clean protocol)
fig2_protocol_bars.png : clean vs noisy training protocol, grouped bars at 0/5/10 sigma

Labels are intentionally in English: CJK fonts are often missing on servers and
would render as tofu boxes.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

DISPLAY = {
    'DLinear': 'DLinear (baseline)',
    'FuzzyDLinear': 'Fuzzy-DLinear (learned gate)',
    'FuzzyDLinear:membership': 'Fuzzy-DLinear (membership)',
}
STYLE = {
    'DLinear': dict(color='#666666', marker='o', ls='-'),
    'FuzzyDLinear': dict(color='#d08b3a', marker='s', ls='--'),
    'FuzzyDLinear:membership': dict(color='#2f6fb3', marker='^', ls='-'),
}


def load(path):
    recs = [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]
    agg = defaultdict(lambda: defaultdict(list))
    for r in recs:
        agg[(r['protocol'], r['model'])][float(r['amp'])].append((r['mse'], r['mae']))
    out = {}
    for key, by_amp in agg.items():
        amps = sorted(by_amp)
        mse = [float(np.mean([v[0] for v in by_amp[a]])) for a in amps]
        mae = [float(np.mean([v[1] for v in by_amp[a]])) for a in amps]
        out[key] = (amps, mse, mae)
    return out


def fig_curve(data, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model in [m for m in DISPLAY if ('clean', m) in data]:
        amps, mse, mae = data[('clean', model)]
        st = STYLE.get(model, {})
        axes[0].plot(amps, mse, label=DISPLAY[model], lw=2, ms=6, **st)
        axes[1].plot(amps, mae, label=DISPLAY[model], lw=2, ms=6, **st)

    for ax, title, ylab in ((axes[0], 'MSE', 'MSE (standardized)'), (axes[1], 'MAE', 'MAE (standardized)')):
        ax.set_xlabel('Impulse amplitude $k$ ($k\\sigma$, 3% of points)')
        ax.set_ylabel(ylab)
        ax.set_title(f'{title} vs impulse strength (clean-trained, noisy test)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    p = os.path.join(out_dir, 'fig1_noise_curve.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def fig_bars(data, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = [m for m in DISPLAY if m != 'FuzzyDLinear']
    amps_of_interest = [0.0, 5.0, 10.0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, protocol in zip(axes, ('clean', 'noisy')):
        x = np.arange(len(amps_of_interest))
        width = 0.36
        for i, model in enumerate(models):
            if (protocol, model) not in data:
                continue
            amps, mse, _ = data[(protocol, model)]
            lookup = dict(zip(amps, mse))
            vals = [lookup.get(a, np.nan) for a in amps_of_interest]
            ax.bar(x + (i - 0.5) * width, vals, width, label=DISPLAY[model], color=STYLE[model]['color'], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{a:g}$\\sigma$' for a in amps_of_interest])
        ax.set_title(f'protocol = {protocol}')
        ax.set_ylabel('MSE (standardized)')
        ax.grid(axis='y', alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.suptitle('Robustness under impulse noise: clean-trained vs noise-augmented', fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, 'fig2_protocol_bars.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def print_table(data):
    print(f"{'protocol':<7} {'model':<26} {'amp':>5} {'MSE':>8} {'MAE':>8}")
    for (protocol, model) in sorted(data):
        amps, mse, mae = data[(protocol, model)]
        for a, m1, m2 in zip(amps, mse, mae):
            print(f"{protocol:<7} {model:<26} {a:>5g} {m1:8.4f} {m2:8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='./results/bench.jsonl')
    ap.add_argument('--out', default='./docs/figs')
    cfg = ap.parse_args()

    data = load(cfg.input)
    os.makedirs(cfg.out, exist_ok=True)
    print_table(data)
    try:
        p1 = fig_curve(data, cfg.out)
        p2 = fig_bars(data, cfg.out)
        print(f"\nwrote {p1}\nwrote {p2}")
    except ImportError:
        print("\nmatplotlib not installed; run: pip install matplotlib")


if __name__ == '__main__':
    main()
