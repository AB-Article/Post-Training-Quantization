#!/usr/bin/env python3
"""Evaluate the hand-written rules of thumb (main-text Table VI) as a policy.
    python rules_eval.py"""
import numpy as np, pandas as pd
import v83_analysis as V
import guideline as G

CFG = {'cifar10': dict(size_col='size_dynamic_kib', thr=1180, int8_mode='mixed', loss=0.004),
       'imagenet100': dict(size_col='size_fp16_kib', thr=4095, int8_mode='int8', loss=0.003)}
for c, k in CFG.items():
    d = G.load(c)
    pick = []
    for r in d.itertuples():
        small = getattr(r, k['size_col']) <= k['thr'] if pd.notna(getattr(r, k['size_col'])) else False
        if r.device != '24040RN64Y':
            want = 'fp16' if small else 'dynamic'
        else:
            want = k['int8_mode'] if (small and pd.notna(r.accdrop_int8) and r.accdrop_int8 <= k['loss']) else 'dynamic'
        order = [want, 'dynamic', 'fp16', 'fp32']
        pick.append(next(m for m in order if m in r.feas))
    s = V.summary(*V.evaluate(d, np.array(pick, dtype=object)))
    print(f"{c}: rules exact {s['exact']:.1f}  geo {s['geo']:.1f}  p90 {s['p90']:.1f}  ms {s['ms_mean']:.2f}")
