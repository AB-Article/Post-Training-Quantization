#!/usr/bin/env python3
"""robustness.py statistics (MC relabeling, margins, Welch, eps bands, policy
regret) for ImageNet-100, using the per-mode accuracy files from
huggingface.co/NN-Dataset/tflite (<mode>/img-classification_imagenet100_acc/all_models.json).
    python robustness_im100.py --data-root NND --acc-dir im100_acc"""
import argparse, collections, glob, json, os
import numpy as np
import robustness as R

ap = argparse.ArgumentParser(); ap.add_argument('--data-root', required=True); ap.add_argument('--acc-dir', required=True)
a = ap.parse_args()
acc = {m: json.load(open(f'{a.acc_dir}/all_models_accuracy_{m}.json')) for m in R.MODES}
runs = collections.defaultdict(dict)
for m in R.MODES:
    for f in glob.glob(f'{a.data_root}/ab/nn/stat/run/tflite/imagenet100/{m}/*/android_*.json'):
        r = json.load(open(f))
        if not (r.get('valid') and r.get('device_type') in R.DEVICES): continue
        d = r.get('duration')
        if not d or d < R.FLOOR: continue
        k = R.unit_key(r.get('unit'))
        runs[(r['model_name'], r['device_type'])][m] = (float(d), float(r.get(f'{k}_std_dev') or 0.0))
A, B = R.pools(acc, runs)
for nm, P in (('A unconstrained', A), ('B constrained', B)):
    print(f'=== POOL {nm}: {len(P)} pairs ===')
    d = R.dep_rate(P); lo, hi = R.cluster_bootstrap(P, R.dep_rate)
    print(f'  device-dependence {100*d:.1f}% [{100*lo:.1f}, {100*hi:.1f}]')
    mg = R.margins(P)
    print(f'  margin median {100*np.median(mg):.1f}%, {100*np.mean(mg<0.05):.1f}% below 5%')
    sg, ns = R.welch_sig(P); print(f'  Welch significant {100*sg:.1f}% of {ns}')
    for eps in (0.02, 0.05, 0.10):
        r, nn = R.eps_device_dependence(P, eps); print(f'  eps={100*eps:.0f}% {100*r:.1f}% (n={nn})')
    dep, stab, nf = R.mc_perturb(P, reps=500)
    print(f'  MC {100*dep.mean():.1f}% [{100*np.percentile(dep,2.5):.1f}, {100*np.percentile(dep,97.5):.1f}], stability {100*stab.mean():.1f}%')
