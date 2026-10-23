#!/usr/bin/env python3
"""A1 + A3 (v85): accelerator profiles.
For each device and backend (CPU/GPU/NPU), median latency of each PTQ mode
relative to FP32 on the SAME backend (paired per architecture, both corpora
pooled, all valid runs), with IQR and n; FP32 backend ratios by latency class;
backend availability and static-INT8 counts on the Qualcomm NNAPI path.
    python accel_profile.py > results/v85/accel_profile.txt   (+ accel_profile.json)
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json, numpy as np, pandas as pd
NAME = {'24040RN64Y': 'Redmi 13', 'Pixel 4': 'Pixel 4', 'SM-T970': 'Tab S7+'}
MODES = ['fp32', 'fp16', 'mixed', 'dynamic', 'int8']
BE = ['cpu', 'gpu', 'npu']
r = pd.read_csv('results/v83/runs_long.csv')
r = r[r.valid].copy()
for b in BE:
    r.loc[~(r[f'{b}_ms'] > 0.1), f'{b}_ms'] = np.nan   # same 0.1 ms floor as the corpus
out = {'profile': {}, 'availability': {}, 'fp32_ratio': {}}
print('== A1 median(mode/FP32) per backend, paired by (corpus,model); [Q1,Q3]; n')
for dv in NAME:
    s = r[r.device == dv]
    for b in BE:
        w = s.pivot_table(index=['corpus', 'model'], columns='mode', values=f'{b}_ms')
        for m in MODES[1:]:
            x = (w[m] / w['fp32']).dropna()
            if len(x) < 5:
                print(f'{NAME[dv]:9s} {b.upper():3s} {m:8s} n={len(x)}'); continue
            q = x.quantile([.25, .5, .75]).values
            out['profile'][f'{NAME[dv]}|{b}|{m}'] = dict(med=q[1], q1=q[0], q3=q[2], n=int(len(x)))
            print(f'{NAME[dv]:9s} {b.upper():3s} {m:8s} {q[1]:5.2f} [{q[0]:4.2f},{q[2]:4.2f}] n={len(x)}')
print('\n== A3 availability: share of valid runs with a measured backend latency (>0.1 ms)')
for dv in NAME:
    s = r[r.device == dv]
    for m in MODES:
        t = s[s['mode'] == m]
        row = {b: float(t[f'{b}_ms'].notna().mean()) for b in BE}
        row['n'] = int(len(t))
        out['availability'][f'{NAME[dv]}|{m}'] = row
        print(f'{NAME[dv]:9s} {m:8s} n={len(t):5d} ' + ' '.join(f'{b}={100*row[b]:5.1f}%' for b in BE))
print('\n== FP32 backend ratios by model latency class (class = min over backends of FP32 latency on this device, ms)')
for dv in NAME:
    s = r[(r.device == dv) & (r['mode'] == 'fp32')].copy()
    s['best'] = s[[f'{b}_ms' for b in BE]].min(axis=1)
    for lo, hi, nm in [(0, 2, '<2'), (2, 10, '2-10'), (10, 1e9, '>=10')]:
        t = s[(s.best >= lo) & (s.best < hi)]
        d = {}
        for a, b in [('gpu', 'cpu'), ('npu', 'cpu'), ('npu', 'gpu')]:
            x = (t[f'{a}_ms'] / t[f'{b}_ms']).dropna()
            d[f'{a}/{b}'] = (float(x.median()) if len(x) else None, int(len(x)))
        out['fp32_ratio'][f'{NAME[dv]}|{nm}'] = d
        print(f'{NAME[dv]:9s} {nm:5s} n={len(t):5d} ' + '  '.join(f'{k}={v[0]:.2f}(n={v[1]})' for k, v in d.items() if v[0] is not None))
json.dump(out, open('results/v85/accel_profile.json', 'w'), indent=1)
