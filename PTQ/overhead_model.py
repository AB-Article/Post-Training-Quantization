#!/usr/bin/env python3
"""B1 + B2 (v85): overhead + throughput model.
Per device and path (backend, mode): L_path = a + s * x, x = FP32 CPU latency of the
same model on the same device (ms, a hardware-agnostic compute proxy), Theil-Sen fit.
a = dispatch/fixed overhead (ms), s = relative throughput cost. Predicted crossover
between two paths: L* = (a1 - a2) / (s2 - s1), checked against the empirical 50%
crossing of P(path 1 faster) in x (logistic fit) and against the 2-10 ms bands.
Also B2: absolute regret of wrong modes by quartile of best feasible latency.
    python overhead_model.py > results/v85/overhead_model.txt
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
NAME = {'24040RN64Y': 'Redmi 13', 'Pixel 4': 'Pixel 4', 'SM-T970': 'Tab S7+'}
rng = np.random.RandomState(0)
r = pd.read_csv('results/v83/runs_long.csv'); r = r[r.valid].copy()
for b in ('cpu', 'gpu', 'npu'):
    r.loc[~(r[f'{b}_ms'] > 0.1), f'{b}_ms'] = np.nan

def theil_sen(x, y, n=300000):
    i = rng.randint(0, len(x), n); j = rng.randint(0, len(x), n)
    ok = np.abs(x[i] - x[j]) > 1e-6
    sl = (y[i][ok] - y[j][ok]) / (x[i][ok] - x[j][ok])
    s = np.median(sl); a = np.median(y - s * x)
    return a, s

PATHS = {'Redmi 13': [('cpu', 'fp32'), ('cpu', 'dynamic'), ('gpu', 'fp32'), ('npu', 'fp32'), ('npu', 'int8'), ('npu', 'dynamic')],
         'Pixel 4': [('cpu', 'fp32'), ('cpu', 'dynamic'), ('cpu', 'int8'), ('gpu', 'fp32'), ('npu', 'fp16'), ('npu', 'fp32')],
         'Tab S7+': [('cpu', 'fp32'), ('cpu', 'dynamic'), ('cpu', 'int8'), ('gpu', 'fp32'), ('npu', 'fp16'), ('npu', 'fp32')]}
CROSS = {'Redmi 13': [(('npu', 'int8'), ('gpu', 'fp32')), (('npu', 'dynamic'), ('gpu', 'fp32'))],
         'Pixel 4': [(('cpu', 'dynamic'), ('gpu', 'fp32')), (('npu', 'fp16'), ('gpu', 'fp32'))],
         'Tab S7+': [(('cpu', 'dynamic'), ('gpu', 'fp32')), (('npu', 'fp16'), ('gpu', 'fp32'))]}
out = {'fit': {}, 'cross': {}}
print('== B1 L_path = a + s*x   (x = FP32 CPU latency, ms; both corpora; Theil-Sen)')
for dv, nm in NAME.items():
    s = r[r.device == dv]
    x = s[s['mode'] == 'fp32'].set_index(['corpus', 'model']).cpu_ms.rename('x')
    cache = {}
    for b, m in PATHS[nm]:
        t = s[s['mode'] == m].set_index(['corpus', 'model'])[f'{b}_ms'].rename('y')
        j = pd.concat([x, t], axis=1).dropna()
        j = j[j.x < 400]
        a, sl = theil_sen(j.x.values, j.y.values)
        pred = a + sl * j.x.values
        r2 = 1 - ((j.y - pred) ** 2).sum() / ((j.y - j.y.mean()) ** 2).sum()
        mre = float(np.median(np.abs(pred - j.y.values) / j.y.values))
        out['fit'][f'{nm}|{b}|{m}'] = dict(a=float(a), s=float(sl), r2=float(r2), medrelerr=mre, n=int(len(j)))
        cache[(b, m)] = (j, a, sl)
        print(f'{nm:9s} {b.upper():3s} {m:8s} a={a:6.2f} ms  s={sl:5.2f}  R2={r2:5.2f}  med.rel.err={100*mre:4.0f}%  n={len(j)}')
    for p1, p2 in CROSS[nm]:
        (j1, a1, s1), (j2, a2, s2) = cache[p1], cache[p2]
        L = (a1 - a2) / (s2 - s1) if s2 != s1 else np.nan
        jj = pd.concat([j1.y.rename('y1'), j2.y.rename('y2'), j1.x], axis=1).dropna()
        z = (jj.y1 < jj.y2).astype(int)
        if 0 < z.mean() < 1:
            lr = LogisticRegression(C=100).fit(np.log(jj[['x']].values), z)
            emp = float(np.exp(-lr.intercept_[0] / lr.coef_[0][0]))
            share_lo = float(z[jj.x < 2].mean()); share_hi = float(z[jj.x > 20].mean())
        else:
            emp = np.nan; share_lo = share_hi = float(z.mean())
        out['cross'][f'{nm}|{p1}|{p2}'] = dict(L_pred=float(L), L_emp=emp, p1_faster_small=share_lo, p1_faster_large=share_hi, n=int(len(jj)))
        print(f'   crossover {p1[0].upper()}-{p1[1]} vs {p2[0].upper()}-{p2[1]}: predicted L*={L:7.2f} ms; empirical 50% crossing x={emp:7.2f} ms; '
              f'P(path1 faster): x<2 ms {share_lo:.2f}, x>20 ms {share_hi:.2f}  (n={len(jj)})')

# B2 absolute regret by quartile of best feasible latency
import guideline as G, one_or_many as O
print('\n== B2 absolute regret (ms) of always-dynamic and best single artifact by quartile of model latency class (best feasible latency on slowest device)')
for c in ('cifar10', 'imagenet100'):
    d = G.load(c).reset_index(drop=True)
    d['best_ms'] = d[[f'lat_{m}' for m in O.M]].min(axis=1) / 1e6
    d['model_ms'] = d.groupby('model').best_ms.transform('max')
    q = d.drop_duplicates('model').model_ms.quantile([.25, .5, .75]).values
    edges = [0, *q, 1e9]
    pols = {'always dynamic': np.array(['dynamic'] * len(d), dtype=object), 'single artifact': O.single_artifact_oracle(d)}
    out.setdefault('abs', {})[c] = {}
    print(f'-- {c}: quartile edges of model latency {np.round(q,2)} ms')
    for k, p in pols.items():
        rat, ms = O.regret_of(d, p)
        row = []
        for i in range(4):
            mk = ((d.model_ms >= edges[i]) & (d.model_ms < edges[i + 1])).values
            row.append(dict(q=i + 1, lat_med=float(d.best_ms[mk].median()), abs_mean=float(ms[mk].mean()), abs_med=float(np.median(ms[mk])),
                            rel_geo=float(100 * (np.exp(np.log(rat[mk]).mean()) - 1)), n=int(mk.sum())))
        out['abs'][c][k] = row
        for e in row:
            print(f'   {k:15s} Q{e["q"]} (median best latency {e["lat_med"]:6.2f} ms): mean abs regret {e["abs_mean"]:5.2f} ms, median {e["abs_med"]:5.2f} ms, relative {e["rel_geo"]:6.1f}%  n={e["n"]}')
json.dump(out, open('results/v85/overhead_model.json', 'w'), indent=1)
