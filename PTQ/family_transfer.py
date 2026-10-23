#!/usr/bin/env python3
"""D1 (v85): does a selector trained on one device transfer to another?
Train on device X (architectures of the training folds), test on device Y
(architectures of the held-out fold; same grouped 5-fold split for every pair), offline features
only, feasibility-masked. X = Y is the in-device reference. Also the 'majority mode of X'
baseline (feasibility-aware). Within family: Pixel<->Tab; across: Redmi<->Qualcomm.
    python family_transfer.py > results/v85/family_transfer.txt
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json, numpy as np, pandas as pd
from grouped_cv import GroupKFold   # deterministic drop-in; see grouped_cv.py (sklearn >= 1.6 changed GroupKFold's fold assignment)
import predictor as P, v83_analysis as V, guideline as G
NAME = V.NAME; DEVS = V.DEVS; NA, CA = P.NUM_ARCH, P.CAT_ARCH
FAM = {'24040RN64Y': 'MediaTek', 'Pixel 4': 'Qualcomm', 'SM-T970': 'Qualcomm'}
res = {}
for c in ('cifar10', 'imagenet100'):
    d = G.load(c).reset_index(drop=True)
    res[c] = {}
    print(f'== {c}')
    picks = {(x, y): np.empty(len(d), dtype=object) for x in DEVS for y in DEVS}
    maj = {(x, y): np.empty(len(d), dtype=object) for x in DEVS for y in DEVS}
    for tr, te in GroupKFold(5).split(d, d.label, groups=d.model):
        trd, ted = d.iloc[tr], d.iloc[te]
        for x in DEVS:
            t = trd[trd.device == x]
            m = P.model(NA, CA, 'hgb').fit(t[NA + CA], t.label)
            order = V.win_order(t)
            for y in DEVS:
                mk = (ted.device == y).values
                if mk.sum() == 0: continue
                sub = ted[mk]
                picks[(x, y)][te[mk]] = V.mask_pred(m, sub, NA + CA)
                maj[(x, y)][te[mk]] = [next((q for q in order if q in f), 'fp32') for f in sub.feas]
    for x in DEVS:
        for y in DEVS:
            mk = (d.device == y).values
            sub = d[mk].reset_index(drop=True)
            for kind, store in (('GBT', picks), ('majority', maj)):
                h, r, ms = V.evaluate(sub, store[(x, y)][mk])
                s = V.summary(h, r, ms)
                lo, hi = V.cluster_boot(sub.model.values, [h, r], lambda h, r: np.array([100 * h.mean(), 100 * (np.exp(np.log(r).mean()) - 1)]), reps=500)
                s['ci_geo'] = [lo[1], hi[1]]
                res[c][f'{kind}|{NAME[x]}->{NAME[y]}'] = s
                rel = 'same device' if x == y else ('same family' if FAM[x] == FAM[y] else 'cross family')
                s['rel'] = rel
                print(f'  {kind:8s} train {NAME[x]:9s} -> test {NAME[y]:9s} ({rel:12s}) exact {s["exact"]:5.1f}  geo {s["geo"]:5.1f} [{lo[1]:4.1f},{hi[1]:4.1f}]  p90 {s["p90"]:5.1f}')
json.dump(res, open('results/v85/family_transfer.json', 'w'), indent=1, default=float)
