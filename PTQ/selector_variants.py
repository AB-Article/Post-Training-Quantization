#!/usr/bin/env python3
"""C2 + E1 + E2 (v85).
 C2 per-family selector: device identity replaced by accelerator family (vendor)
 E1 predict-then-select baseline: per-mode log-latency regressors, argmin over
    feasible modes (same features, same architecture-grouped 5-fold CV)
 E2 feasibility guarantee: accuracy-budget violations of unmasked vs masked picks
    python selector_variants.py > results/v85/selector_variants.txt
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from grouped_cv import GroupKFold   # deterministic drop-in; see grouped_cv.py (sklearn >= 1.6 changed GroupKFold's fold assignment)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import predictor as P, v83_analysis as V, guideline as G
M = V.MODES; NAME = V.NAME; DEVS = V.DEVS
VENDOR = {'24040RN64Y': 'MediaTek', 'Pixel 4': 'Qualcomm', 'SM-T970': 'Qualcomm'}
ND, CD = P.NUM_ARCH + P.NUM_DEV, P.CAT_ARCH + P.CAT_DEV
NA, CA = P.NUM_ARCH, P.CAT_ARCH

def reg(num, cat):
    pre = ColumnTransformer([('n', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), num),
                             ('c', OneHotEncoder(handle_unknown='ignore'), cat)])
    return Pipeline([('p', pre), ('m', HistGradientBoostingRegressor(max_iter=300, learning_rate=.08, max_leaf_nodes=15,
                                                                      l2_regularization=1., min_samples_leaf=10, random_state=0))])

def viol(d, pick):
    """share of picks violating the accuracy budget (pick not in feasible set)"""
    return 100 * float(np.mean([p not in f for p, f in zip(pick, d.feas)]))

def report(name, d, pick, store):
    h, r, ms = V.evaluate(d, pick); s = V.summary(h, r, ms)
    lo, hi = V.cluster_boot(d.model.values, [h, r], lambda h, r: np.array([100 * h.mean(), 100 * (np.exp(np.log(r).mean()) - 1)]), reps=500)
    s['ci_exact'] = [lo[0], hi[0]]; s['ci_geo'] = [lo[1], hi[1]]; s['viol'] = viol(d, pick)
    store[name] = s
    print(f'  {name:46s} exact {s["exact"]:5.1f} [{lo[0]:4.1f},{hi[0]:4.1f}]  geo {s["geo"]:5.1f} [{lo[1]:4.1f},{hi[1]:4.1f}]  p90 {s["p90"]:5.1f}  ms {s["ms_mean"]:.2f}  budget violations {s["viol"]:.1f}%')
    return h, r

def cv_class(d, num, cat, masked):
    pick = np.empty(len(d), dtype=object)
    for tr, te in GroupKFold(5).split(d, d.label, groups=d.model):
        m = P.model(num, cat, 'hgb').fit(d.iloc[tr][num + cat], d.iloc[tr].label)
        pick[te] = V.mask_pred(m, d.iloc[te], num + cat) if masked else m.predict(d.iloc[te][num + cat])
    return pick

def cv_reg(d, num, cat, masked):
    pick = np.empty(len(d), dtype=object)
    for tr, te in GroupKFold(5).split(d, d.label, groups=d.model):
        pred = {}
        for m in M:
            y = d.iloc[tr][f'lat_{m}']; ok = y.notna().values
            if ok.sum() < 50: continue
            mod = reg(num, cat).fit(d.iloc[tr][num + cat][ok], np.log(y[ok]))
            pred[m] = mod.predict(d.iloc[te][num + cat])
        for j, i in enumerate(te):
            fs = d.feas.iloc[i] if masked else M
            cand = [m for m in fs if m in pred]
            pick[i] = min(cand, key=lambda m: pred[m][j]) if cand else 'fp32'
    return pick

res = {}
for c in ('cifar10', 'imagenet100'):
    d = G.load(c).reset_index(drop=True)
    d['vendor'] = d.device.map(VENDOR)
    print(f'== {c}: {len(d)} pairs')
    st = {}
    report('direct GBT, per-device (masked)', d, cv_class(d, ND, CD, True), st)
    report('direct GBT, per-device (unmasked)', d, cv_class(d, ND, CD, False), st)
    report('direct GBT, per-family (masked)', d, cv_class(d, NA, CA + ['vendor'], True), st)
    report('direct GBT, per-family + cores/RAM (masked)', d, cv_class(d, ND, CA + ['vendor'], True), st)
    report('direct GBT, offline only (masked)', d, cv_class(d, NA, CA, True), st)
    report('latency regressors, per-device (masked)', d, cv_reg(d, ND, CD, True), st)
    report('latency regressors, per-device (unmasked)', d, cv_reg(d, ND, CD, False), st)
    report('latency regressors, per-family (masked)', d, cv_reg(d, NA, CA + ['vendor'], True), st)
    res[c] = st
json.dump(res, open('results/v85/selector_variants.json', 'w'), indent=1, default=float)
