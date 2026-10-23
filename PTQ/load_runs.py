#!/usr/bin/env python3
"""Flatten the released TFLite run records into one long table.

One row per (corpus, model, device, mode) with the fastest-backend duration
(`duration`, the quantity used throughout the paper), the per-backend means
and standard deviations, input shape and sizes.

    python load_runs.py --data-root /path/to/nn-dataset --out results/v83/runs_long.csv
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import argparse, glob, json, os
import pandas as pd

MODES = ['fp32', 'fp16', 'mixed', 'dynamic', 'int8']
DEVICES = ['24040RN64Y', 'Pixel 4', 'SM-T970']


def load(root):
    rows = []
    for corpus, sub in (('cifar10', ''), ('imagenet100', 'imagenet100')):
        R = os.path.join(root, 'ab', 'nn', 'stat', 'run', 'tflite', sub)
        for m in MODES:
            for f in glob.glob(f'{R}/{m}/*/android_*.json'):
                r = json.load(open(f))
                if r.get('device_type') not in DEVICES:
                    continue
                ci = r.get('device_analytics', {}).get('cpu_info', {})
                row = {'corpus': corpus, 'model': r.get('model_name'), 'device': r['device_type'],
                       'mode': m, 'valid': bool(r.get('valid')), 'unit': r.get('unit'),
                       'duration': r.get('duration'),
                       'in_h': r.get('in_dim_1'), 'in_w': r.get('in_dim_2'), 'in_c': r.get('in_dim_3'),
                       'size_fp32_kib': r.get('original_model_size_kib'),
                       'size_kib': r.get('quantized_model_size_kib'),
                       'total_ram_kb': r.get('total_ram_kb'), 'cores': ci.get('cpu_cores'),
                       'soc': ci.get('arm_architecture', {}).get('hardware'),
                       'os': r.get('os_version')}
                for b in ('cpu', 'gpu', 'npu'):
                    row[f'{b}_ms'] = (r.get(f'{b}_duration') or 0) / 1e6 or None
                    row[f'{b}_sd_ms'] = (r.get(f'{b}_std_dev') or 0) / 1e6 or None
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', default='results/v83/runs_long.csv')
    a = ap.parse_args()
    df = load(a.data_root)
    df.to_csv(a.out, index=False)
    print(df.groupby(['corpus', 'valid']).size())
