"""Spike-noise robustness benchmark: DLinear (baseline) vs FuzzyDLinear.

Protocols
  clean : train on clean windows, evaluate under spike noise (inference-time robustness)
  noisy : train with spike-noise injection, evaluate under spike noise (noise-aware training)

Noise model
  additive impulses: at each position with prob `ratio`, x <- x + k * z, z ~ N(0,1)
  k is the amplitude multiplier; data are standardized, so k is roughly "k sigma".

Device
  --device auto (default): CUDA when available, otherwise CPU. Force with --device cpu / cuda.
  Evaluation noise is generated from a fixed generator per amplitude on the same device,
  so every model sees exactly the same corrupted inputs (paired comparison).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_loader import Dataset_ETT_hour
from models import DLinear, FuzzyDLinear

MODEL_ZOO = {'DLinear': DLinear, 'FuzzyDLinear': FuzzyDLinear}


def resolve_device(spec):
    if spec == 'auto':
        spec = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(spec)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spike(x, amp, ratio, gen):
    """Additive impulse noise, applied to inputs only."""
    if amp == 0:
        return x
    mask = torch.rand(x.shape, generator=gen, device=x.device) < ratio
    noise = torch.randn(x.shape, generator=gen, device=x.device) * amp
    return torch.where(mask, x + noise, x)


def build_loaders(cfg, device):
    kw = dict(root_path=cfg.root_path, data_path=cfg.data_path, size=[cfg.seq_len, cfg.label_len, cfg.pred_len],
              features=cfg.features, target=cfg.target, scale=True, timeenc=1, freq=cfg.freq)
    tr = Dataset_ETT_hour(flag='train', **kw)
    va = Dataset_ETT_hour(flag='val', **kw)
    te = Dataset_ETT_hour(flag='test', **kw)
    pin = device.type == 'cuda'

    def mk(ds, sh):
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=sh, num_workers=cfg.num_workers,
                          drop_last=True, pin_memory=pin)
    return (tr, mk(tr, True)), (va, mk(va, False)), (te, mk(te, False))


def build_model(name, cfg):
    base, _, variant = name.partition(':')  # e.g. "FuzzyDLinear:membership"
    mc = SimpleNamespace(seq_len=cfg.seq_len, pred_len=cfg.pred_len, individual=cfg.individual,
                         enc_in=cfg.enc_in, moving_avg=cfg.moving_avg,
                         fuzzy_mode=variant or 'mlp')
    return MODEL_ZOO[base].Model(mc)


def train(model, loader, cfg, train_amp, gen, device):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.MSELoss()
    model.train()
    for ep in range(cfg.epochs):
        t0 = time.time()
        tot = 0.0
        for bx, by, _, _ in loader:
            opt.zero_grad()
            bx = bx.float().to(device, non_blocking=True)
            by = by.float().to(device, non_blocking=True)
            if train_amp > 0:
                bx = spike(bx, train_amp, cfg.ratio, gen)
            out = model(bx)[:, -cfg.pred_len:, :]
            loss = lossf(out, by[:, -cfg.pred_len:, :])
            loss.backward()
            opt.step()
            tot += loss.item()
        for g in opt.param_groups:  # type-1 schedule
            g['lr'] *= 0.5
        print(f"    epoch {ep + 1}/{cfg.epochs} loss={tot / len(loader):.5f} ({time.time() - t0:.1f}s)", flush=True)


@torch.no_grad()
def evaluate(model, loader, cfg, amp, device):
    model.eval()
    # identical noise for every model: same seed, same draw order
    gen = torch.Generator(device=device.type).manual_seed(90210 + int(amp))
    se, ae, n = 0.0, 0.0, 0
    for bx, by, _, _ in loader:
        bx = bx.float().to(device, non_blocking=True)
        bx = spike(bx, amp, cfg.ratio, gen)
        out = model(bx)[:, -cfg.pred_len:, :]
        tgt = by.float().to(device, non_blocking=True)[:, -cfg.pred_len:, :]
        se += ((out - tgt) ** 2).sum().item()
        ae += (out - tgt).abs().sum().item()
        n += tgt.numel()
    return se / n, ae / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', default='DLinear,FuzzyDLinear')
    p.add_argument('--protocols', default='clean,noisy')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seq_len', type=int, default=336)
    p.add_argument('--label_len', type=int, default=168)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--enc_in', type=int, default=7)
    p.add_argument('--moving_avg', type=int, default=25)
    p.add_argument('--individual', type=str, default='true', help='true/false, DLinear 逐通道权重')
    p.add_argument('--ratio', type=float, default=0.03)
    p.add_argument('--amps', default='0,1,3,5,10')
    p.add_argument('--train_amp', type=float, default=3.0)
    p.add_argument('--data', default='ETTh1')
    p.add_argument('--root_path', default='./dataset/ETT-small/')
    p.add_argument('--data_path', default='ETTh1.csv')
    p.add_argument('--features', default='M')
    p.add_argument('--target', default='OT')
    p.add_argument('--freq', default='h')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--out', default='./results/noise_bench.jsonl')
    cfg = p.parse_args()
    cfg.amps = [float(a) for a in cfg.amps.split(',')]
    cfg.individual = str(cfg.individual).lower() in ('1', 'true', 'yes')
    cfg.models = [m.strip() for m in cfg.models.split(',') if m.strip()]
    cfg.protocols = [x.strip() for x in cfg.protocols.split(',') if x.strip()]
    cfg.seeds = [int(s) for s in cfg.seeds.split(',')]

    device = resolve_device(cfg.device)
    print(f"device={device} (torch {torch.__version__}, cuda_available={torch.cuda.is_available()})", flush=True)
    if device.type == 'cuda':
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    (tr_ds, tr_ld), (va_ds, va_ld), (te_ds, te_ld) = build_loaders(cfg, device)
    print(f"split sizes: train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)}", flush=True)

    with open(cfg.out, 'w', encoding='utf-8') as f:
        for protocol in cfg.protocols:
            for name in cfg.models:
                for seed in cfg.seeds:
                    set_seed(seed)
                    model = build_model(name, cfg).to(device)
                    nparam = sum(q.numel() for q in model.parameters())
                    t0 = time.time()
                    gen = torch.Generator(device=device.type).manual_seed(1000 + seed)
                    train_amp = cfg.train_amp if protocol == 'noisy' else 0.0
                    print(f"[{protocol}] {name} seed={seed} params={nparam / 1e6:.3f}M train_amp={train_amp}", flush=True)
                    train(model, tr_ld, cfg, train_amp, gen, device)
                    for amp in cfg.amps:
                        mse, mae = evaluate(model, te_ld, cfg, amp, device)
                        rec = dict(protocol=protocol, model=name, seed=seed, amp=amp,
                                   mse=round(mse, 6), mae=round(mae, 6), params=nparam, device=str(device))
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        print(f"    amp={amp:>4}  MSE={mse:.4f}  MAE={mae:.4f}", flush=True)
                    print(f"  total {time.time() - t0:.1f}s", flush=True)

    summarize(cfg)


def summarize(cfg):
    recs = [json.loads(l) for l in open(cfg.out, encoding='utf-8')]
    print("\n==== paired summary (mean over seeds) ====")
    amps = sorted({r['amp'] for r in recs})
    for protocol in cfg.protocols:
        print(f"\nprotocol={protocol}")
        print(f"{'amp':>5} | {'model':<24} | {'MSE':>8} | {'MAE':>8} | {'dMSE%':>7} | {'dMAE%':>7}")
        for amp in amps:
            base = {}
            for name in cfg.models:
                rows = [r for r in recs if r['protocol'] == protocol and r['model'] == name and r['amp'] == amp]
                if rows:
                    base[name] = (np.mean([x['mse'] for x in rows]), np.mean([x['mae'] for x in rows]))
            ref = base.get('DLinear')
            for name, (mse, mae) in base.items():
                if ref is None or name == 'DLinear':
                    dmse = dmae = 0.0
                else:
                    dmse = (ref[0] - mse) / ref[0] * 100
                    dmae = (ref[1] - mae) / ref[1] * 100
                print(f"{amp:>5} | {name:<24} | {mse:8.4f} | {mae:8.4f} | {dmse:+7.1f} | {dmae:+7.1f}")


if __name__ == '__main__':
    main()
