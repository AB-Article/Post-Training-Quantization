#!/usr/bin/env python3
"""v85: per-run extras not in runs_long.csv: per-backend MIN durations and the run timestamp.
    python load_runs_extra.py --data-root NND --out results/v85/runs_extra.csv
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import argparse, glob, json, os
import pandas as pd
MODES = ['fp32', 'fp16', 'mixed', 'dynamic', 'int8']
DEVICES = ['24040RN64Y', 'Pixel 4', 'SM-T970']
ap = argparse.ArgumentParser(); ap.add_argument('--data-root', required=True); ap.add_argument('--out', default='results/v85/runs_extra.csv')
a = ap.parse_args()
rows = []
for corpus, sub in (('cifar10', ''), ('imagenet100', 'imagenet100')):
    R = os.path.join(a.data_root, 'ab', 'nn', 'stat', 'run', 'tflite', sub)
    for m in MODES:
        for f in glob.glob(f'{R}/{m}/*/android_*.json'):
            r = json.load(open(f))
            if r.get('device_type') not in DEVICES: continue
            row = {'corpus': corpus, 'model': r.get('model_name'), 'device': r['device_type'], 'mode': m, 'valid': bool(r.get('valid')),
                   'ts': (r.get('device_analytics') or {}).get('timestamp')}
            for b in ('cpu', 'gpu', 'npu'):
                row[f'{b}_min_ms'] = (r.get(f'{b}_min_duration') or 0) / 1e6 or None
                row[f'{b}_ms'] = (r.get(f'{b}_duration') or 0) / 1e6 or None
            rows.append(row)
pd.DataFrame(rows).to_csv(a.out, index=False)
print(len(rows))
