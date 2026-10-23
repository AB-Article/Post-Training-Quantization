#!/usr/bin/env python3
"""C1 (v85): how many artifacts to ship? K = 1, 2 (fixed vendor split, any split), 3 (per device).
Per model, each artifact is a feasible mode, each device is assigned to one artifact.
Group choice minimizes the mean log latency ratio over the devices of the group
(oracle, in-sample like the single-artifact oracle of Table III).
Reports geometric-mean regret (%), total-time regret (sum ms regret / sum best ms),
by model latency class, clustered bootstrap CIs over architectures, and the share of
the one-artifact regret removed by the second artifact.
    python k_artifacts.py > results/v85/k_artifacts.txt
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import itertools, json, numpy as np, pandas as pd
import v83_analysis as V, guideline as G, one_or_many as O
M = V.MODES; DEVS = V.DEVS; NAME = V.NAME
BANDS = O.BANDS
rng = np.random.RandomState(2)

def modemat(g):
    """per device row -> dict mode -> log ratio (fallback applied)"""
    R = {}
    for r in g.itertuples():
        c = {k: getattr(r, f'lat_{k}') for k in M if pd.notna(getattr(r, f'lat_{k}'))}
        if not c: continue
        b = min(c.values()); fb = next(x for x in V.FALLBACK if x in c)
        R[r.device] = {m: np.log(c.get(m, c[fb]) / b) for m in M if m in g.feas.iloc[0]}
    return R

def best_group(R, devs):
    best, bv = None, np.inf
    for m in M:
        if all(m in R[dv] for dv in devs):
            v = np.mean([R[dv][m] for dv in devs])
            if v < bv: best, bv = m, v
    return best

def policy(d, groups):
    """groups: list of lists of devices (devices absent from a model are skipped)"""
    pick = np.empty(len(d), dtype=object)
    for mo, g in d.groupby('model'):
        R = modemat(g)
        asg = {}
        for grp in groups:
            dd = [x for x in grp if x in R]
            if not dd: continue
            m = best_group(R, dd) or 'fp32'
            for x in dd: asg[x] = m
        for i, dv in zip(g.index.values, g.device.values):
            pick[i] = asg.get(dv, 'fp32')
    return pick

def free_two(d):
    pick = np.empty(len(d), dtype=object)
    parts = [[['24040RN64Y'], ['Pixel 4', 'SM-T970']], [['Pixel 4'], ['24040RN64Y', 'SM-T970']], [['SM-T970'], ['24040RN64Y', 'Pixel 4']]]
    for mo, g in d.groupby('model'):
        R = modemat(g)
        bestp, bv = None, np.inf
        for p in parts:
            tot = 0; asg = {}
            for grp in p:
                dd = [x for x in grp if x in R]
                if not dd: continue
                m = best_group(R, dd) or 'fp32'
                tot += sum(R[x][m] for x in dd) if m in R[dd[0]] else 0
                for x in dd: asg[x] = m
            if tot < bv: bv, bestp = tot, asg
        for i, dv in zip(g.index.values, g.device.values):
            pick[i] = bestp.get(dv, 'fp32')
    return pick

def boot(d, rat, mask, B=1000):
    mod = d.model.values[mask]; lr = np.log(rat[mask])
    u, inv = np.unique(mod, return_inverse=True)
    s = np.bincount(inv, lr); n = np.bincount(inv)
    out = []
    for _ in range(B):
        k = rng.randint(0, len(u), len(u))
        out.append(100 * (np.exp(s[k].sum() / n[k].sum()) - 1))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def main():
    res = {}
    for c in ('cifar10', 'imagenet100'):
        d = G.load(c).reset_index(drop=True)
        d['best_ms'] = d[[f'lat_{m}' for m in M]].min(axis=1) / 1e6
        d['model_ms'] = d.groupby('model').best_ms.transform('max')
        A, B_, C_ = '24040RN64Y', 'Pixel 4', 'SM-T970'
        pols = {'K=1 single artifact': policy(d, [DEVS]),
                'K=2 vendor split (Redmi | Pixel+Tab)': policy(d, [[A], [B_, C_]]),
                'K=2 (Pixel | Redmi+Tab)': policy(d, [[B_], [A, C_]]),
                'K=2 (Tab | Redmi+Pixel)': policy(d, [[C_], [A, B_]]),
                'K=2 best split per model': free_two(d),
                'K=3 per device': policy(d, [[A], [B_], [C_]])}
        res[c] = {}
        print(f'== {c}')
        base = None
        for k, p in pols.items():
            rat, ms = O.regret_of(d, p)
            row = {'all': O.summarize(rat, ms)}
            tot = {}
            for b, lo, hi in BANDS:
                mk = ((d.model_ms >= lo) & (d.model_ms < hi)).values
                row['model ' + b] = O.summarize(rat, ms, mk)
                row['model ' + b]['ci'] = boot(d, rat, mk, 500)
                row['model ' + b]['tot'] = 100 * float(ms[mk].sum() / d.best_ms[mk].sum())
            mk10 = (d.model_ms < 10).values
            row['model <10 ms'] = O.summarize(rat, ms, mk10); row['model <10 ms']['tot'] = 100 * float(ms[mk10].sum() / d.best_ms[mk10].sum())
            row['all']['ci'] = boot(d, rat, np.ones(len(d), bool), 500)
            row['all']['tot'] = 100 * float(ms.sum() / d.best_ms.sum())
            res[c][k] = row
            print(f'  {k:38s} all: geo {row["all"]["geo"]:5.2f}% [{row["all"]["ci"][0]:.2f},{row["all"]["ci"][1]:.2f}] total-time {row["all"]["tot"]:5.2f}% | ' +
                  ' | '.join(f'{b}: {row["model "+b]["geo"]:5.1f}% (tot {row["model "+b]["tot"]:4.1f}%)' for b, _, _ in BANDS))
        k1 = res[c]['K=1 single artifact']; k2 = res[c]['K=2 vendor split (Redmi | Pixel+Tab)']
        rec = {b: 100 * (1 - k2['model ' + b]['geo'] / k1['model ' + b]['geo']) for b, _, _ in BANDS}
        rec['model <10 ms'] = 100 * (1 - k2['model <10 ms']['geo'] / k1['model <10 ms']['geo'])
        rec['all'] = 100 * (1 - k2['all']['geo'] / k1['all']['geo'])
        res[c]['recovered_by_second'] = rec
        print('  share of K=1 regret removed by the vendor split: ' + ', '.join(f'{k} {v:.0f}%' for k, v in rec.items()))
    json.dump(res, open('results/v85/k_artifacts.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
