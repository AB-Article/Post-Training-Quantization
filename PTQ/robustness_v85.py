#!/usr/bin/env python3
"""F1-F7 (v85): robustness of the headline shipping results.
Scenarios (each recomputes feasibility/labels where needed and re-runs the policy comparison):
 base | F1 acc_FP32>=50% | F2 mixed+static INT8 merged | F3 1 pp absolute budget | F4 min-latency labels |
 F6 by architecture family | F7 share of pairs where a quantized mode is >1.5x slower than FP32 | F5 timestamp drift
Reported per scenario: best single artifact (K=1) regret, always dynamic, vendor-split K=2 regret, share of the
K=1 regret removed by the split, and the Redmi/Pixel cost of tuning on the Tab S7+.
    python robustness_v85.py > results/v85/robustness_v85.txt
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
import v83_analysis as V, guideline as G, one_or_many as O, k_artifacts as K
M = V.MODES; NAME = V.NAME; DEVS = V.DEVS
EX = pd.read_csv('results/v85/runs_extra.csv'); EX = EX[EX.valid]

def relabel(d):
    lab = []
    for r in d.itertuples():
        c = {m: getattr(r, f'lat_{m}') for m in M if m in r.feas and pd.notna(getattr(r, f'lat_{m}'))}
        lab.append(min(c, key=c.get) if c else np.nan)
    d = d.copy(); d['label'] = lab
    return d[d.label.notna()].reset_index(drop=True)

def prep(d):
    d = d.copy().reset_index(drop=True)
    d['best_ms'] = d[[f'lat_{m}' for m in M]].min(axis=1) / 1e6
    d['model_ms'] = d.groupby('model').best_ms.transform('max')
    return d

def scen(d):
    d = prep(d)
    A, B_, C_ = DEVS
    o = {}
    for nm, p in (('K1', O.single_artifact_oracle(d)), ('dyn', np.array(['dynamic'] * len(d), dtype=object)),
                  ('K2', K.policy(d, [[A], [B_, C_]]))):
        rat, ms = O.regret_of(d, p)
        o[nm] = O.summarize(rat, ms)['geo']
        o[nm + '_light'] = O.summarize(rat, ms, (d.model_ms < 2).values)['geo']
    for ref in ('SM-T970',):
        p, mk = O.flagship_tuned(d, ref); rat, ms = O.regret_of(d, p)
        for dv in (A, B_):
            o[f'tab->{NAME[dv]}'] = O.summarize(rat, ms, mk & (d.device == dv).values)['geo']
    o['removed'] = 100 * (1 - o['K2'] / o['K1']); o['n'] = int(len(d)); o['arch'] = int(d.model.nunique())
    return o

def show(c, name, o):
    print(f'  {name:42s} n={o["n"]:5d} arch={o["arch"]:4d} | K=1 {o["K1"]:5.1f}% (light {o["K1_light"]:5.1f}%) | always-dyn {o["dyn"]:6.1f}% (light {o["dyn_light"]:6.1f}%) | '
          f'K=2 vendor {o["K2"]:4.1f}% (light {o["K2_light"]:4.1f}%) | removed {o["removed"]:3.0f}% | tuned-on-Tab -> Redmi {o["tab->Redmi 13"]:5.1f}% Pixel {o["tab->Pixel 4"]:5.1f}%')

res = {}
for c in ('cifar10', 'imagenet100'):
    print(f'== {c}')
    d0 = G.load(c).reset_index(drop=True)
    res[c] = {}
    def run(name, d):
        o = scen(d); res[c][name] = o; show(c, name, o)
    run('base', d0)
    keep = d0.groupby('model').acc_fp32.first(); ok = set(keep[keep >= 0.5].index)
    run('F1 FP32 accuracy >= 50%', d0[d0.model.isin(ok)])
    d = d0.copy()
    d['lat_int8'] = d[['lat_int8', 'lat_mixed']].min(axis=1)
    d['lat_mixed'] = np.nan
    d['feas'] = [[m for m in f if m != 'mixed'] if ('int8' in f or 'mixed' in f) else f for f in d.feas]
    d['feas'] = [(f + ['int8'] if ('int8' not in f and 'mixed' in g) else f) for f, g in zip(d.feas, d0.feas)]
    run('F2 mixed + static INT8 merged', relabel(d))
    d = d0.copy()
    d['feas'] = [['fp32'] + [m for m in M[1:] if pd.notna(r[f'acc_{m}']) and r[f'acc_{m}'] >= r['acc_fp32'] - 0.01] for _, r in d.iterrows()]
    run('F3 1 pp absolute accuracy budget', relabel(d))
    # F4: min-latency labels (per-backend min over the 20 iterations, then best backend)
    e = EX[EX.corpus == c].copy()
    for b in ('cpu', 'gpu', 'npu'):
        e.loc[~(e[f'{b}_min_ms'] > 0.1), f'{b}_min_ms'] = np.nan
    e['mn'] = e[['cpu_min_ms', 'gpu_min_ms', 'npu_min_ms']].min(axis=1) * 1e6
    piv = e.pivot_table(index=['model', 'device'], columns='mode', values='mn')
    d = d0.copy()
    for m in M:
        key = list(zip(d.model, d.device))
        d[f'lat_{m}'] = [piv[m].get(k, np.nan) if m in piv else np.nan for k in key]
    run('F4 min-latency (best of 20) labels', relabel(d))
    # F6 by family
    for fam, nm in (('unq', 'F6 family unq'), ('rag', 'F6 family rag'), ('named', 'F6 named')):
        sub = d0[d0.family == fam]
        if sub.model.nunique() >= 15: run(nm, sub)
    # F7 slow-down share
    print('  F7 share of pairs where a mode is >1.5x slower than FP32 on the same device (and where the best mode is faster than FP32):')
    out7 = {}
    for m in M[1:]:
        x = (d0[f'lat_{m}'] / d0['lat_fp32']).dropna()
        out7[m] = dict(slower15=100 * float((x > 1.5).mean()), slower1=100 * float((x > 1.0).mean()), n=int(len(x)))
        print(f'     {m:8s} >1.5x slower {out7[m]["slower15"]:5.1f}%   slower at all {out7[m]["slower1"]:5.1f}%   (n={len(x)})')
    res[c]['F7'] = out7
    # F5 drift
    print('  F5 drift (Spearman rho of run timestamp vs the weight-detrended log GPU/CPU latency ratio, FP32 runs of the corpus; and span in days)')
    out5 = {}
    for dv in DEVS:
        t = e[(e.device == dv) & (e['mode'] == 'fp32')].dropna(subset=['gpu_ms', 'cpu_ms', 'ts'])
        t = t[(t.gpu_ms > 0.1) & (t.cpu_ms > 0.1)]
        y0 = np.log(t.gpu_ms / t.cpu_ms)
        xx = np.log(t.cpu_ms)
        y = y0 - np.polyval(np.polyfit(xx, y0, 3), xx)   # remove the dependence on model weight (run order is not random w.r.t. size)
        rho, p = spearmanr(t.ts, y)
        span = (t.ts.max() - t.ts.min()) / 86400
        # slope in % per 10 days, robust: median of first vs last third
        n3 = len(t) // 3; o = np.argsort(t.ts.values)
        first, last = np.median(y.values[o[:n3]]), np.median(y.values[o[-n3:]])
        out5[NAME[dv]] = dict(rho=float(rho), p=float(p), span_days=float(span), shift_pct=float(100 * (np.exp(last - first) - 1)), n=int(len(t)))
        print(f'     {NAME[dv]:9s} rho={rho:+.3f} (p={p:.3f}) span={span:5.1f} d  last-third vs first-third median detrended GPU/CPU ratio shift {100*(np.exp(last-first)-1):+.1f}%  n={len(t)}')
    res[c]['F5'] = out5
json.dump(res, open('results/v85/robustness_v85.json', 'w'), indent=1, default=float)
