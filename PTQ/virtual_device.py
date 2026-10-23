#!/usr/bin/env python3
"""A2 (v85): within-device control. Recompute the constrained optimum when a
backend is removed from the candidate set (same accuracy constraint, same pool).
Outcome: share of pairs whose optimum is an integer mode {mixed,dynamic,int8}
vs a float mode {fp32,fp16}; the Redmi-minus-Qualcomm gap with an
architecture-clustered bootstrap. Scenarios: all backends (real), no GPU,
no NPU, CPU only, GPU only.
    python virtual_device.py > results/v85/virtual_device.txt
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json, numpy as np, pandas as pd
import v83_analysis as V
import guideline as G
MODES = V.MODES; NAME = V.NAME; DEVS = V.DEVS
INT = {'mixed', 'dynamic', 'int8'}
runs = pd.read_csv('results/v83/runs_long.csv'); runs = runs[runs.valid]
SC = {'all backends (real)': ['cpu', 'gpu', 'npu'], 'no GPU': ['cpu', 'npu'], 'no NPU': ['cpu', 'gpu'],
      'CPU only': ['cpu'], 'GPU only': ['gpu']}
rng = np.random.RandomState(1)
res = {}
for c in ('cifar10', 'imagenet100'):
    d = G.load(c).reset_index(drop=True)
    rc = runs[runs.corpus == c]
    base = {}
    for b in ('cpu', 'gpu', 'npu'):
        t = rc.copy(); t.loc[~(t[f'{b}_ms'] > 0.1), f'{b}_ms'] = np.nan
        base[b] = t.set_index(['model', 'device', 'mode'])[f'{b}_ms']
    idx = pd.MultiIndex.from_frame(d[['model', 'device']].assign(k=0)[['model', 'device']])
    res[c] = {}
    for sc, bes in SC.items():
        lab, bi, bf = [], [], []
        for r in d.itertuples():
            best, bl = None, np.inf
            li, lf = np.inf, np.inf
            for m in MODES:
                if m not in r.feas: continue
                v = np.nanmin([base[b].get((r.model, r.device, m), np.nan) for b in bes] + [np.inf])
                if v < bl: best, bl = m, v
                if m in INT: li = min(li, v)
                else: lf = min(lf, v)
            lab.append(best); bi.append(li); bf.append(lf)
        d['lab'] = lab
        d['isint'] = d.lab.isin(INT)
        # decisive integer win: best integer mode >5% faster than best float mode (ties are noise, excluded)
        d['dec_int'] = [1.0 if i < 0.95 * f else 0.0 for i, f in zip(bi, bf)]
        d['dec_flt'] = [1.0 if f < 0.95 * i else 0.0 for i, f in zip(bi, bf)]
        d.loc[d.lab.isna(), ['dec_int', 'dec_flt']] = np.nan
        out = {}
        for dv in DEVS:
            s = d[(d.device == dv) & d.lab.notna()]
            out[NAME[dv]] = dict(int_share=100 * float(s.isint.mean()), dec_int=100 * float(s.dec_int.mean()), dec_flt=100 * float(s.dec_flt.mean()), n=int(len(s)),
                                 modes={m: 100 * float((s.lab == m).mean()) for m in MODES})
        # Redmi minus mean(Qualcomm) gap, clustered bootstrap over architectures
        w = d.pivot_table(index='model', columns='device', values='isint', aggfunc='mean')
        w = w.dropna()
        g = (w['24040RN64Y'] - (w['Pixel 4'] + w['SM-T970']) / 2).values
        bs = [g[rng.randint(0, len(g), len(g))].mean() for _ in range(2000)]
        out['gap'] = dict(mean=100 * float(g.mean()), lo=100 * float(np.percentile(bs, 2.5)), hi=100 * float(np.percentile(bs, 97.5)), n_arch=int(len(g)))
        wd = d.pivot_table(index='model', columns='device', values='dec_int', aggfunc='mean').dropna()
        gd = (wd['24040RN64Y'] - (wd['Pixel 4'] + wd['SM-T970']) / 2).values
        bd = [gd[rng.randint(0, len(gd), len(gd))].mean() for _ in range(2000)]
        out['gap_dec'] = dict(mean=100 * float(gd.mean()), lo=100 * float(np.percentile(bd, 2.5)), hi=100 * float(np.percentile(bd, 97.5)))
        # Tab vs Pixel gap (same family)
        g2 = (w['SM-T970'] - w['Pixel 4']).values
        bs2 = [g2[rng.randint(0, len(g2), len(g2))].mean() for _ in range(2000)]
        out['gap_TabPixel'] = dict(mean=100 * float(g2.mean()), lo=100 * float(np.percentile(bs2, 2.5)), hi=100 * float(np.percentile(bs2, 97.5)))
        # exact winner identical on all three devices
        ww = d.pivot_table(index='model', columns='device', values='lab', aggfunc='first').dropna()
        out['same_winner_all3'] = 100 * float((ww.nunique(axis=1) == 1).mean())
        res[c][sc] = out
    print(f'== {c}')
    for sc, o in res[c].items():
        print(f'  {sc:20s} integer-mode optimum share: ' + '  '.join(f'{k}={o[k]["int_share"]:5.1f}%' for k in map(NAME.get, DEVS)) +
              f'  | Redmi-Qualcomm gap {o["gap"]["mean"]:+5.1f} [{o["gap"]["lo"]:+5.1f},{o["gap"]["hi"]:+5.1f}] pp'
              f'  | decisive-int share ' + '/'.join(f'{o[k]["dec_int"]:.0f}' for k in map(NAME.get, DEVS)) + f' (gap {o["gap_dec"]["mean"]:+.1f} [{o["gap_dec"]["lo"]:+.1f},{o["gap_dec"]["hi"]:+.1f}])' + f'  | Tab-Pixel {o["gap_TabPixel"]["mean"]:+5.1f} [{o["gap_TabPixel"]["lo"]:+5.1f},{o["gap_TabPixel"]["hi"]:+5.1f}]'
              f'  | same winner on all 3: {o["same_winner_all3"]:4.1f}%')
json.dump(res, open('results/v85/virtual_device.json', 'w'), indent=1)
