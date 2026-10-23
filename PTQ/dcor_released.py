#!/usr/bin/env python3
"""Distance correlation between metadata and the constrained best-mode label
(and the PTQ accuracy change) on the released pools.  python dcor_released.py"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import numpy as np, pandas as pd
import predictor as P
M = P.MODES


def dcor(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.ndim == 1: x = x[:, None]
    if y.ndim == 1: y = y[:, None]
    def cen(z):
        d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
        return d - d.mean(0) - d.mean(1)[:, None] + d.mean()
    A, B = cen(x), cen(y)
    v = (A * B).mean(); vx = (A * A).mean(); vy = (B * B).mean()
    return np.sqrt(max(v, 0) / np.sqrt(vx * vy))


def onehot(s):
    return pd.get_dummies(s).values.astype(float)


for c in ('cifar10', 'imagenet100'):
    d = pd.read_csv(f'results/v83/dataset_full_{c}.csv')
    d = d.sample(n=min(len(d), 2500), random_state=0)   # O(n^2) memory
    y = onehot(d.label)
    lat32 = np.log(d[[f'lat_{m}' for m in M]].min(axis=1))   # proxy: log best latency
    feats = {'log model size': np.log(d.size_fp32_kib), 'model family': onehot(d.family),
             'device identity': onehot(d.device), 'log best latency': lat32,
             'input resolution': d.in_px, 'FP32 accuracy': d.acc_fp32}
    out = {}
    for k, v in feats.items():
        v = np.asarray(v, float)
        ok = np.isfinite(v).all(1) if v.ndim == 2 else np.isfinite(v)
        out[k] = round(dcor(v[ok], y[ok]), 2) if np.nanstd(v) > 0 else None
    print(c, out)
    # accuracy change vs mode / device (long format)
    L = []
    for m in M[1:]:
        t = d[['device', f'accdrop_{m}']].dropna().rename(columns={f'accdrop_{m}': 'drop'}); t['mode'] = m; L.append(t)
    L = pd.concat(L).sample(n=2500, random_state=0)
    print('   acc change ~ mode', round(dcor(onehot(L['mode']), L['drop']), 3),
          ' ~ device', round(dcor(onehot(L.device), L['drop']), 3))
