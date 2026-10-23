#!/usr/bin/env python3
"""E3 (v85): named-model table. For the named (human-designed) architectures of the released
constrained pools: input size, FP32 accuracy, FP32 size, constrained best mode and latency per device,
regret of the best single artifact, and how the depth-3 rules / look-up rule (architecture-grouped CV) do on them.
    python named_models.py > results/v85/named_models.txt
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json, numpy as np, pandas as pd
from grouped_cv import GroupKFold   # deterministic drop-in; see grouped_cv.py (sklearn >= 1.6 changed GroupKFold's fold assignment)
from sklearn.tree import DecisionTreeClassifier
import predictor as P, v83_analysis as V, guideline as G, one_or_many as O
NAME = V.NAME; DEVS = V.DEVS; M = V.MODES
SH = {'fp32': 'FP32', 'fp16': 'FP16', 'mixed': 'Mix', 'dynamic': 'Dyn', 'int8': 'I8'}
res = {}
for c in ('cifar10', 'imagenet100'):
    d = G.load(c).reset_index(drop=True)
    d['best_ms'] = d[[f'lat_{m}' for m in M]].min(axis=1) / 1e6
    single = O.single_artifact_oracle(d)
    rat1, _ = O.regret_of(d, single)
    d['pick1'] = single; d['r1'] = rat1
    # CV rule picks
    num = P.NUM_ARCH + P.NUM_DEV
    X = pd.get_dummies(d[num + ['device']], columns=['device']).fillna(-1)
    tree = np.empty(len(d), dtype=object); look = np.empty(len(d), dtype=object)
    for tr, te in GroupKFold(5).split(d, d.label, groups=d.model):
        t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0).fit(X.iloc[tr], d.label.iloc[tr])
        pr = t.predict_proba(X.iloc[te]); cls = list(t.classes_)
        for j, i in enumerate(te):
            o = np.argsort(-pr[j]); tree[i] = next((cls[q] for q in o if cls[q] in d.feas.iloc[i]), cls[o[0]])
        o_, g_ = G.cell_orders(d.iloc[tr]); look[te] = G.lookup_pick(d.iloc[te], o_, g_)
    nm = d[d.family == 'named']
    print(f'== {c}: {nm.model.nunique()} named architectures, {len(nm)} pairs')
    rows = []
    for mo, g in nm.groupby('model'):
        r0 = g.iloc[0]
        row = dict(model=mo, px=f'{int(r0.in_h)}', acc=100 * r0.acc_fp32 if r0.acc_fp32 <= 1 else r0.acc_fp32, mb=r0.size_fp32_kib / 1024,
                   single=r0.pick1, r1=100 * (np.exp(np.log(g.r1).mean()) - 1))
        for dv in DEVS:
            s = g[g.device == dv]
            row[NAME[dv]] = (f'{SH[s.label.iloc[0]]} {s.best_ms.iloc[0]:.1f}' if len(s) else '--')
        rows.append(row)
    T = pd.DataFrame(rows).sort_values('mb')
    print(T.to_string(index=False, float_format=lambda v: f'{v:.1f}'))
    summ = {}
    for nmr, pk in (('depth-3 rules', tree), ('look-up rule', look)):
        h, r, ms = V.evaluate(d, pk)
        mk = (d.family == 'named').values
        s = V.summary(h[mk], r[mk], ms[mk]); sa = V.summary(h, r, ms)
        summ[nmr] = dict(named=s, all=sa)
        print(f'  {nmr:14s} on named: exact {s["exact"]:.1f} geo {s["geo"]:.1f} p90 {s["p90"]:.1f} | on all pairs: exact {sa["exact"]:.1f} geo {sa["geo"]:.1f}')
    # single artifact for named models
    print(f'  best single artifact on named models: geo regret {100*(np.exp(np.log(nm.r1).mean())-1):.1f}% ; on generated: {100*(np.exp(np.log(d[d.family!="named"].r1).mean())-1):.1f}%')
    # device dependence in named subset
    w = nm.pivot_table(index='model', columns='device', values='label', aggfunc='first').dropna()
    print(f'  named models labeled on all 3 devices: {len(w)}; winner differs on some device for {100*(w.nunique(axis=1)>1).mean():.0f}%')
    print('  winner share (named):', (100 * nm.label.value_counts(normalize=True)).round(0).to_dict())
    res[c] = dict(table=T.to_dict('records'), rules=summ)
json.dump(res, open('results/v85/named_models.json', 'w'), indent=1, default=float)
