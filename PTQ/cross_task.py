#!/usr/bin/env python3
"""D2 (v85): cross-task reuse. For the architectures present in both corpora
(1,167), compare the constrained optimum on CIFAR-10 and ImageNet-100 per device;
regret of reusing the CIFAR-10 choice on ImageNet-100 (when feasible there); and the
depth-3 CIFAR-10 tree applied to ImageNet-100 (device features identical).
    python cross_task.py > results/v85/cross_task.txt
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json, numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
import predictor as P, v83_analysis as V, guideline as G, one_or_many as O
NAME = V.NAME; DEVS = V.DEVS; M = V.MODES
c10 = G.load('cifar10').reset_index(drop=True); im = G.load('imagenet100').reset_index(drop=True)
k = c10.merge(im, on=['model', 'device'], suffixes=('_c', '_i'))
print(f'shared (model, device) pairs: {len(k)}; shared architectures: {k.model.nunique()}')
agree = (k.label_c == k.label_i)
same_f = [set(a) == set(b) for a, b in zip(k.feas_c, k.feas_i)]
print(f'exact agreement of the constrained winner: {100*agree.mean():.1f}%  (same feasible set: {100*np.mean(same_f):.1f}% of pairs; agreement within them {100*agree[same_f].mean():.1f}%)')
for dv in DEVS:
    s = k[k.device == dv]; print(f'   {NAME[dv]:9s} agreement {100*(s.label_c==s.label_i).mean():.1f}% (n={len(s)})')
# regret of reusing the CIFAR-10 choice on ImageNet-100
imi = im.set_index(['model', 'device'])
pick, keep = [], []
for r in k.itertuples():
    p = r.label_c
    keep.append(True)
    pick.append(p if p in r.feas_i else 'fp32')
sub = im.merge(k[['model', 'device']], on=['model', 'device']).reset_index(drop=True)
sub = sub.set_index(['model', 'device']).loc[list(zip(k.model, k.device))].reset_index()
h, r, ms = V.evaluate(sub, np.array(pick, dtype=object))
s = V.summary(h, r, ms)
print(f'reuse CIFAR-10 choice on ImageNet-100 (fallback FP32 if infeasible): exact {s["exact"]:.1f}  geo regret {s["geo"]:.1f}  p90 {s["p90"]:.1f}  ms {s["ms_mean"]:.2f}')
# baselines on the same rows
for nm, p in [('always dynamic', ['dynamic'] * len(sub)), ('per-device majority (ImageNet)', None)]:
    if p is None:
        mj = sub.groupby('device').label.agg(lambda x: x.value_counts().index[0]); order = {dv: V.win_order(sub[sub.device == dv]) for dv in DEVS}
        p = [next((q for q in order[r.device] if q in r.feas), 'fp32') for r in sub.itertuples()]
    h2, r2, m2 = V.evaluate(sub, np.array(p, dtype=object)); s2 = V.summary(h2, r2, m2)
    print(f'   baseline {nm:32s} exact {s2["exact"]:.1f}  geo {s2["geo"]:.1f}  p90 {s2["p90"]:.1f}')
# CIFAR-10 depth-3 tree on ImageNet-100
num = P.NUM_ARCH + P.NUM_DEV
def X(d):
    x = pd.get_dummies(d[num + ['device']], columns=['device']).fillna(-1)
    return x.rename(columns={f'device_{kk}': f'device={v}' for kk, v in NAME.items()})
Xc, Xi = X(c10), X(im)
Xi = Xi.reindex(columns=Xc.columns, fill_value=0)
t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0).fit(Xc, c10.label)
pr = t.predict_proba(Xi); cls = list(t.classes_)
pk = []
for j, f in enumerate(im.feas):
    o = np.argsort(-pr[j]); pk.append(next((cls[q] for q in o if cls[q] in f), cls[o[0]]))
h, r, ms = V.evaluate(im, np.array(pk, dtype=object)); s3 = V.summary(h, r, ms)
tm = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0).fit(Xi, im.label)
pr2 = tm.predict_proba(Xi); cls2 = list(tm.classes_)
pk2 = [next((cls2[q] for q in np.argsort(-pr2[j]) if cls2[q] in f), cls2[np.argmax(pr2[j])]) for j, f in enumerate(im.feas)]
h2, r2, m2 = V.evaluate(im, np.array(pk2, dtype=object)); s4 = V.summary(h2, r2, m2)
print(f'depth-3 tree trained on CIFAR-10, applied to all ImageNet-100 pairs: exact {s3["exact"]:.1f}  geo {s3["geo"]:.1f}  p90 {s3["p90"]:.1f}')
print(f'   reference: depth-3 tree trained on ImageNet-100 itself (in-sample): exact {s4["exact"]:.1f}  geo {s4["geo"]:.1f}  p90 {s4["p90"]:.1f}')
json.dump(dict(agree=float(agree.mean()), reuse=s, tree_c2i=s3, tree_i2i=s4, n=int(len(k))), open('results/v85/cross_task.json', 'w'), indent=1, default=float)
