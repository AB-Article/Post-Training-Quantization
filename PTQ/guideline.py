#!/usr/bin/env python3
"""Deployment look-up table and interpretable shallow tree (v83 tasks 1 and 3).

Look-up rule: for each (device, FP32 model-size band) cell, rank the PTQ modes
by how often they are the constrained winner in the training folds and ship the
first mode that satisfies the accuracy constraint (known offline). Model size
is used instead of latency because it is known before benchmarking (Spearman
rho = 0.8 with the best latency on both corpora).

Shallow tree: depth-limited decision tree on the same offline + device
features, feasibility-masked, architecture-grouped five-fold CV.

    python guideline.py > results/v83/guideline.txt   # also writes guideline.json, guideline_rows.tex
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json
import numpy as np, pandas as pd
from grouped_cv import GroupKFold   # deterministic drop-in; see grouped_cv.py (sklearn >= 1.6 changed GroupKFold's fold assignment)
from sklearn.tree import DecisionTreeClassifier, export_text

import predictor as P
import v83_analysis as V

MODES = P.MODES
SHORT = {'fp32': 'FP32', 'fp16': 'FP16', 'mixed': 'Mixed', 'dynamic': 'Dyn.', 'int8': 'INT8'}
BANDS = [(0, 1, '$<1$\\,MB'), (1, 10, '$1$--$10$\\,MB'), (10, 1e12, '$\\geq10$\\,MB')]
DEVS = V.DEVS
NAME = V.NAME


def band(mb):
    for i, (lo, hi, _) in enumerate(BANDS):
        if lo <= mb < hi:
            return i


def load(c):
    d = pd.read_csv(f'results/v83/dataset_full_{c}.csv')
    rng = np.random.RandomState(0)
    order = {g: i for i, g in enumerate(rng.permutation(pd.unique(d.model.values)))}
    d = d.iloc[np.argsort([order[g] for g in d.model.values], kind='stable')].reset_index(drop=True)
    d['feas'] = [V._feasible_acc(r) for _, r in d.iterrows()]
    d['band'] = (d.size_fp32_kib / 1024).map(band)
    return d


def cell_orders(tr):
    out = {}
    for (dv, b), g in tr.groupby(['device', 'band']):
        out[(dv, b)] = V.win_order(g)
    glob = V.win_order(tr)
    return out, glob


def lookup_pick(te, orders, glob):
    return np.array([next((m for m in orders.get((r.device, r.band), glob) if m in r.feas), 'fp32')
                     for r in te.itertuples()], dtype=object)


def summarize(d, pick):
    h, r, ms = V.evaluate(d, pick)
    return V.summary(h, r, ms)


def run(c):
    d = load(c)
    res = {'corpus': c, 'n': len(d)}
    # cross-validated look-up rule
    pick = np.empty(len(d), dtype=object)
    for tr, te in GroupKFold(5).split(d, d.label, groups=d.model):
        o, g = cell_orders(d.iloc[tr])
        pick[te] = lookup_pick(d.iloc[te], o, g)
    res['lookup_cv'] = summarize(d, pick)
    # full-data table for publication
    o, g = cell_orders(d)
    cells = {}
    for dv in DEVS:
        for b in range(3):
            sub = d[(d.device == dv) & (d.band == b)]
            share = (100 * sub.label.value_counts(normalize=True)).round(0)
            cells[f'{NAME[dv]}|{b}'] = dict(order=o.get((dv, b), g)[:3], n=len(sub),
                                            top_share=float(share.iloc[0]) if len(share) else None)
    res['cells'] = cells
    # shallow trees, feasibility-masked
    num = P.NUM_ARCH + P.NUM_DEV
    X = pd.get_dummies(d[num + ['device']], columns=['device']).fillna(-1)
    X = X.rename(columns={f'device_{k}': f'device={v}' for k, v in NAME.items()})
    res['tree'] = {}
    for depth in (2, 3, 4, 6):
        pick = np.empty(len(d), dtype=object)
        for tr, te in GroupKFold(5).split(d, d.label, groups=d.model):
            t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=30, random_state=0).fit(X.iloc[tr], d.label.iloc[tr])
            pr = t.predict_proba(X.iloc[te]); cls = list(t.classes_)
            for j, i in enumerate(te):
                ordr = np.argsort(-pr[j])
                pick[i] = next((cls[k] for k in ordr if cls[k] in d.feas.iloc[i]), cls[ordr[0]])
        res['tree'][depth] = summarize(d, pick)
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0).fit(X, d.label)
    res['tree_rules_depth3'] = export_text(t, feature_names=list(X.columns), decimals=3)
    return res


def main():
    out = {c: run(c) for c in ('cifar10', 'imagenet100')}
    for c, r in out.items():
        print(f'== {c}: {r["n"]} pairs')
        s = r['lookup_cv']; print(f'  look-up (CV): exact {s["exact"]:.1f} geo {s["geo"]:.1f} p90 {s["p90"]:.1f} ms {s["ms_mean"]:.2f}')
        for k, s in r['tree'].items():
            print(f'  tree depth {k} (CV, masked): exact {s["exact"]:.1f} geo {s["geo"]:.1f} p90 {s["p90"]:.1f} ms {s["ms_mean"]:.2f}')
        for k, v in r['cells'].items():
            print('   ', k, v)
        print(r['tree_rules_depth3'])
    json.dump(out, open('results/v83/guideline.json', 'w'), indent=1, default=str)
    # LaTeX table: device x size band, top-2 modes per corpus
    L = []
    for dv in DEVS:
        row = [NAME[dv]]
        for c in ('cifar10', 'imagenet100'):
            for b in range(3):
                v = out[c]['cells'][f'{NAME[dv]}|{b}']
                row.append(' $>$ '.join(SHORT[m] for m in v['order'][:2]))
        L.append(' & '.join(row) + r' \\')
    open('results/v83/guideline_rows.tex', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
