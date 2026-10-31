"""Generate the figures for the printable one-pager (docs/项目速览).

Numbers come from results/multi_summary.md after the follow-up 4090 run:
ETTh1 / ETTm1 / ETTm2, clean-training protocol, seed 0, official per-dataset
hyper-parameters. ETTh2 was excluded because the official lr=5e-2 made the
fuzzy input filter train unstably on clean data.

Run with a python that has matplotlib:
    python scripts/make_report_figures.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = 'docs/figs'
OURS_C = '#1f5fa9'

# (dataset, 5σ MSE improve %, 10σ MSE improve %, 5σ MAE %, 10σ MAE %)
DATA = [
    ('ETTh1', 6.7, 22.7, 5.7, 16.9),
    ('ETTm1', 20.9, 50.7, 13.3, 30.3),
    ('ETTm2', 44.9, 75.3, 22.2, 42.9),
]


def multi():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.05), dpi=240)
    names = [d[0] for d in DATA]
    for ax, idx, title, avg5, avg10 in (
        (axes[0], 1, 'MSE 改善（越高越好）', 24.2, 49.6),
        (axes[1], 3, 'MAE 改善（越高越好）', 13.7, 30.0),
    ):
        v5 = [d[idx] for d in DATA]
        v10 = [d[idx + 1] for d in DATA]
        x = range(len(names))
        w = 0.34
        b1 = ax.bar([i - w / 2 for i in x], v5, w, label='5σ 脉冲', color='#7fa8d0')
        b2 = ax.bar([i + w / 2 for i in x], v10, w, label='10σ 脉冲', color=OURS_C)
        ax.bar_label(b1, fmt='%+.1f%%', fontsize=6.8, padding=2)
        ax.bar_label(b2, fmt='%+.1f%%', fontsize=6.8, padding=2)
        ax.axhline(0, color='#888888', lw=0.8)
        ax.set_title(f'{title}  平均 5σ/10σ = {avg5:.1f}% / {avg10:.1f}%', fontsize=9, pad=6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_ylim(0, 88)
        ax.tick_params(labelsize=8)
        ax.grid(axis='y', alpha=0.25, lw=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7.5, frameon=False, loc='upper left')

    axes[0].set_ylabel('MSE 降幅（%）', fontsize=8.5)
    axes[1].set_ylabel('MAE 降幅（%）', fontsize=8.5)
    fig.tight_layout(pad=0.4)
    p = os.path.join(OUT, 'fig_multi.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', p)


def structure():
    fig, ax = plt.subplots(figsize=(7.4, 1.5), dpi=240)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 26)
    ax.axis('off')

    boxes = [
        ('输入\n336×7', False),
        ('模糊软截断\n零参数', True),
        ('序列分解\nMA 窗口 25', False),
        ('逐通道线性\n336→96', False),
        ('空间混合\n7×7 + 门控', False),
        ('输出\n96×7', False),
    ]
    w, h, gap = 13.2, 9.5, 3.6
    y = 9
    xs = []
    x = 1.0
    for label, hi in boxes:
        face = '#dce9f7' if hi else '#f4f4f4'
        edge = OURS_C if hi else '#bfbfbf'
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.35,rounding_size=1.2',
                                    facecolor=face, edgecolor=edge, lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=7.6,
                color='#222222', linespacing=1.45)
        xs.append((x, x + w))
        x += w + gap

    for i in range(len(xs) - 1):
        ax.add_patch(FancyArrowPatch((xs[i][1] + 0.3, y + h / 2), (xs[i + 1][0] - 0.3, y + h / 2),
                                     arrowstyle='-|>', mutation_scale=9, lw=1.1, color='#7f7f7f'))

    ax.text(1.0 + w / 2, y + h + 2.6, '现场遥测：3% 点位出现 ±kσ 飞点', ha='center', fontsize=7.2,
            color='#b45309')
    ax.text(1.0 + w + gap + w / 2, y - 3.6, '飞点在分解前被软替换', ha='center', fontsize=7.2,
            color=OURS_C)
    ax.text(100, y - 3.6, '仅 +50 参数', ha='right', fontsize=7.2, color='#7f7f7f')

    fig.tight_layout(pad=0.2)
    p = os.path.join(OUT, 'fig_structure.png')
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    multi()
    structure()