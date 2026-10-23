#!/usr/bin/env python3
"""Small facts quoted in the text that are not produced by the other scripts.
    python misc_checks.py NND_ROOT   ->  results/v83/misc_checks.txt"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json, sys
import numpy as np, pandas as pd
import predictor as P

root = sys.argv[1] if len(sys.argv) > 1 else '/path/to/nn-dataset'
out = []
# identical mixed / static INT8 converted accuracy
for c, D in (('CIFAR-10', f'{root}/_work/stats'), ('ImageNet-100', 'im100_acc')):
    a = json.load(open(f'{D}/all_models_accuracy_mixed.json')); b = json.load(open(f'{D}/all_models_accuracy_int8.json'))
    k = set(a) & set(b); same = sum(a[x]['accuracy'] == b[x]['accuracy'] for x in k)
    out.append(f'{c}: mixed == static INT8 accuracy for {same}/{len(k)} models ({100*same/len(k):.1f}%)')
# input resolution, valid CIFAR runs
r = pd.read_csv('results/v83/runs_long.csv'); v = r[(r.corpus == 'cifar10') & r.valid]
out.append(f'CIFAR-10 valid runs: max input side {int(v.in_h.max())} px, median {int(v.in_h.median())}, '
           f'32x32 share {100*np.mean(v.in_h == 32):.1f}%')
# size as an offline proxy for latency
for c in ('cifar10', 'imagenet100'):
    d = pd.read_csv(f'results/v83/dataset_full_{c}.csv')
    best = d[[f'lat_{m}' for m in P.MODES]].min(axis=1)
    out.append(f'{c}: Spearman(size_fp32, best latency) = {d.size_fp32_kib.corr(best, method="spearman"):.2f}; '
               f'label share ' + str((100 * d.label.value_counts(normalize=True)).round(1).to_dict()))
    fam = d.model.map(P.family).value_counts(normalize=True)
    out.append(f'   family share of pairs: ' + str((100 * fam).round(1).to_dict()))
# per-family exact match of the masked selector (CIFAR-10)
p = pd.read_csv('results/v83/picks_A.csv'); p['fam'] = p.model.map(P.family)
g = p.groupby('fam')['learned +dev (GBT, masked)|hit'].agg(['size', 'mean'])
out.append('CIFAR-10 masked GBT exact match by family: ' + str({k: (int(s), round(100 * m, 1)) for k, (s, m) in g.iterrows()}))
txt = '\n'.join(out); print(txt); open('results/v83/misc_checks.txt', 'w').write(txt + '\n')
