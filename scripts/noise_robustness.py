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
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute
from models import DLinear, FuzzyDLinear

MODEL_ZOO = {'DLinear': DLinear, 'FuzzyDLinear': FuzzyDLinear}


class PlainDataset:
    """Standardized CSV loader without the repo's business-level Hampel cleaning.

    Split follows the repo's custom datasets: 70% train / 10% val / 20% test.
    x window is [start, start+seq_len); y window is [start+seq_len, start+seq_len+pred_len).
    """

    def __init__(self, root_path, data_path, flag='train', seq_len=336, pred_len=96,
                 target=None):
        df_raw = pd.read_csv(os.path.join(root_path, data_path), encoding='utf-8')
        if target is None:
            target = df_raw.columns[-1]
        if target in df_raw.columns:
            cols = [c for c in df_raw.columns if c != 'date']
            values = df_raw[cols].values.astype(np.float32)
        else:
            values = df_raw.drop(columns=['date']).values.astype(np.float32)
        n = len(values)
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_val = n - num_train - num_test
        borders = {'train': (0, num_train), 'val': (num_train, num_train + num_val),
                   'test': (num_train + num_val, n)}
        s, e = borders[flag]
        _scaler = _StandardScaler()
        _scaler.fit(values[:num_train])
        self.scaled = _scaler.transform(values[s:e]).astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.scaled) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x = self.scaled[index:index + self.seq_len]
        y = self.scaled[index + self.seq_len:index + self.seq_len + self.pred_len]
        return x, y, np.zeros((x.shape[0], 1), dtype=np.float32), np.zeros((y.shape[0], 1), dtype=np.float32)


DATASET_DEFAULTS = {
    'ETTh1': dict(loader='ett_hour', root='./dataset/ETT-small/', file='ETTh1.csv', enc_in=7, seq_len=336, pred_len=96, freq='h', lr=1e-4, batch_size=8),
    'ETTh2': dict(loader='ett_hour', root='./dataset/ETT-small/', file='ETTh2.csv', enc_in=7, seq_len=336, pred_len=96, freq='h', lr=5e-2, batch_size=32),
    'ETTm1': dict(loader='ett_min', root='./dataset/ETT-small/', file='ETTm1.csv', enc_in=7, seq_len=336, pred_len=96, freq='t', lr=1e-4, batch_size=8),
    'ETTm2': dict(loader='ett_min', root='./dataset/ETT-small/', file='ETTm2.csv', enc_in=7, seq_len=336, pred_len=96, freq='t', lr=1e-3, batch_size=32),
    'exchange': dict(loader='custom', root='./dataset/exchange_rate/', file='exchange_rate.csv', enc_in=8, seq_len=336, pred_len=96, freq='h', lr=5e-4, batch_size=8),
    'weather': dict(loader='custom', root='./dataset/weather/', file='weather.csv', enc_in=21, seq_len=336, pred_len=96, freq='h', lr=1e-4, batch_size=16),
    'electricity': dict(loader='custom', root='./dataset/electricity/', file='electricity.csv', enc_in=321, seq_len=336, pred_len=96, freq='h', lr=1e-3, batch_size=16),
    'traffic': dict(loader='custom', root='./dataset/traffic/', file='traffic.csv', enc_in=862, seq_len=336, pred_len=96, freq='h', lr=5e-2, batch_size=16),
    'illness': dict(loader='custom', root='./dataset/illness/', file='national_illness.csv', enc_in=7, seq_len=36, pred_len=24, freq='w', lr=5e-2, batch_size=32),
}


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
    pin = device.type == 'cuda'

    if cfg.loader == 'ett_hour':
        kw = dict(root_path=cfg.root_path, data_path=cfg.data_path, size=[cfg.seq_len, cfg.label_len, cfg.pred_len],
                  features=cfg.features, target=cfg.target, scale=True, timeenc=1, freq=cfg.freq)
        tr = Dataset_ETT_hour(flag='train', **kw)
        va = Dataset_ETT_hour(flag='val', **kw)
        te = Dataset_ETT_hour(flag='test', **kw)
    elif cfg.loader == 'ett_min':
        kw = dict(root_path=cfg.root_path, data_path=cfg.data_path, size=[cfg.seq_len, cfg.label_len, cfg.pred_len],
                  features=cfg.features, target=cfg.target, scale=True, timeenc=1, freq=cfg.freq)
        tr = Dataset_ETT_minute(flag='train', **kw)
        va = Dataset_ETT_minute(flag='val', **kw)
        te = Dataset_ETT_minute(flag='test', **kw)
    else:  # 'custom'
        tr = PlainDataset(cfg.root_path, cfg.data_path, 'train', cfg.seq_len, cfg.pred_len)
        va = PlainDataset(cfg.root_path, cfg.data_path, 'val', cfg.seq_len, cfg.pred_len)
        te = PlainDataset(cfg.root_path, cfg.data_path, 'test', cfg.seq_len, cfg.pred_len)

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

    def ds_choice(s):
        if s in DATASET_DEFAULTS:
            return s
        raise argparse.ArgumentTypeError(f'unknown dataset: {s}')

    p.add_argument('--models', default='DLinear,FuzzyDLinear')
    p.add_argument('--protocols', default='clean,noisy')
    p.add_argument('--seeds', default='0')
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
    p.add_argument('--datasets', default='', help='comma list of dataset keys, e.g. ETTh1,ETTh2,exchange; empty = use --data+--loader')
    p.add_argument('--loader', default='ett_hour', choices=['ett_hour', 'ett_min', 'custom'])
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

    if cfg.datasets:
        cfg.datasets = [d.strip() for d in cfg.datasets.split(',') if d.strip()]
        total = 0
        for k in cfg.datasets:
            d = DATASET_DEFAULTS[k]
            run_one_datasets(cfg, k, d, device=resolve_device(cfg.device))
        return
    else:
        cfg.loader = getattr(cfg, 'loader', 'ett_hour')
        run_one_datasets(cfg, cfg.data, DATASET_DEFAULTS.get(cfg.data, dict(
            loader=cfg.loader, root=cfg.root_path, file=cfg.data_path,
            enc_in=cfg.enc_in, seq_len=cfg.seq_len, pred_len=cfg.pred_len, freq=cfg.freq)), device=None)


def run_one_datasets(cfg, name, d, device=None):
    # d: dict with loader/root/file/enc_in/seq_len/pred_len/freq
    import copy
    c = copy.copy(cfg)
    for k, v in d.items():
        setattr(c, k, v)
    c.loader = d['loader']
    c.root_path = d['root']
    c.data_path = d['file']
    c.enc_in = d['enc_in']
    c.seq_len = d['seq_len']
    c.pred_len = d['pred_len']
    c.freq = d['freq']
    if 'lr' in d:
        c.lr = d['lr']
    if 'batch_size' in d:
        c.batch_size = d['batch_size']
    c.out = os.path.join(os.path.dirname(cfg.out), f'{name}.jsonl')
    if device is None:
        device = resolve_device(c.device)
    else:
        c.device = str(device)

    (tr_ds, tr_ld), (va_ds, va_ld), (te_ds, te_ld) = build_loaders(c, device)
    print(f"\n==== dataset={name} loader={c.loader} enc_in={c.enc_in} "
          f"seq={c.seq_len} pred={c.pred_len} freq={c.freq} ====", flush=True)
    print(f"split sizes: train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)}", flush=True)

    os.makedirs(os.path.dirname(c.out), exist_ok=True)
    with open(c.out, 'w', encoding='utf-8') as f:
        for protocol in c.protocols:
            for mname in c.models:
                for seed in c.seeds:
                    set_seed(seed)
                    model = build_model(mname, c).to(device)
                    nparam = sum(q.numel() for q in model.parameters())
                    t0 = time.time()
                    gen = torch.Generator(device=device.type).manual_seed(1000 + seed)
                    train_amp = c.train_amp if protocol == 'noisy' else 0.0
                    print(f"[{protocol}] {mname} seed={seed} params={nparam / 1e6:.3f}M train_amp={train_amp}", flush=True)
                    train(model, tr_ld, c, train_amp, gen, device)
                    for amp in c.amps:
                        mse, mae = evaluate(model, te_ld, c, amp, device)
                        rec = dict(dataset=name, protocol=protocol, model=mname, seed=seed, amp=amp,
                                   mse=round(mse, 6), mae=round(mae, 6), params=nparam, device=str(device))
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        print(f"    amp={amp:>4}  MSE={mse:.4f}  MAE={mae:.4f}", flush=True)
                    print(f"  total {time.time() - t0:.1f}s", flush=True)

    summarize(c.out)


def summarize(path):
    recs = [json.loads(l) for l in open(path, encoding='utf-8')]
    print("\n==== paired summary (mean over seeds) ====")
    amps = sorted({r['amp'] for r in recs})
    protocols = sorted({r['protocol'] for r in recs})
    models = sorted({r['model'] for r in recs})
    for protocol in protocols:
        print(f"\nprotocol={protocol}")
        print(f"{'amp':>5} | {'model':<24} | {'MSE':>8} | {'MAE':>8} | {'dMSE%':>7} | {'dMAE%':>7}")
        for amp in amps:
            base = {}
            for name in models:
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
