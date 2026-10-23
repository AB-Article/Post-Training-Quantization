#!/usr/bin/env python3
"""Fig. 1 of v85: (a) accelerator profiles, (b) within-device control, (c) cost
of one vs two vs per-device artifacts.

Drawn at the exact IEEEtran text width (516 pt = 7.14 in) with an 8 pt font and
saved without a tight bounding box, so \\includegraphics[width=\\linewidth]
renders it at scale 1.0 and no label falls below the paper's 8 pt caption size.
    python make_fig_v85.py   ->  ../figures/fig_v85_mechanism.png
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
prof = json.load(open('results/v85/accel_profile.json'))['profile']
# 2025 MediaTek part, measured with the same on-device benchmark_model
prof.update(json.load(open('results/flagship_2026/mtk_backend_profile.json')))
vd = json.load(open('results/v85/virtual_device.json'))['cifar10']
ka = json.load(open('results/v85/k_artifacts.json'))['cifar10']
oo = json.load(open('results/v83/one_or_many.json'))['cifar10']
TEXTWIDTH_IN = 516 / 72.27          # IEEEtran \textwidth
FS = 8                              # = caption size of the paper
plt.rcParams.update({'font.size': FS, 'font.family': 'DejaVu Sans'})
fig = plt.figure(figsize=(TEXTWIDTH_IN, 3.25))
gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.0, 1.0], wspace=0.46,
                      left=0.115, right=0.995, top=0.915, bottom=0.185)
# (a) heatmap
ax = fig.add_subplot(gs[0])
devs = ['Redmi 13', 'Pixel 4', 'Tab S7+', 'Tab S11']; bes = ['cpu', 'gpu', 'npu']; modes = ['fp16', 'mixed', 'dynamic', 'int8']
lab = {'fp16': 'FP16', 'mixed': 'Mixed', 'dynamic': 'Dyn.', 'int8': 'INT8'}
rows = [(d, b) for d in devs for b in bes]
Z = np.full((len(rows), 4), np.nan); N = np.zeros((len(rows), 4), int)
for i, (d, b) in enumerate(rows):
    for j, m in enumerate(modes):
        v = prof.get(f'{d}|{b}|{m}')
        if v: Z[i, j] = v['med']; N[i, j] = v['n']
im = ax.imshow(np.log2(np.clip(Z, 0.1, 4)), cmap='RdBu', norm=TwoSlopeNorm(0, -2.6, 1.0), aspect='auto')
for i in range(len(rows)):
    for j in range(4):
        if np.isnan(Z[i, j]): continue
        small = N[i, j] < 20
        ax.text(j, i, f'{Z[i,j]:.2f}' + ('*' if small else ''), ha='center', va='center', fontsize=7.2,
                color='white' if Z[i, j] < 0.45 else 'black', style='italic' if small else 'normal')
ax.set_xticks(range(4)); ax.set_xticklabels([lab[m] for m in modes])
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{d if b=="cpu" else ""} {b.upper()}'.strip() for d, b in rows], fontsize=7.2)
for k in (2.5, 5.5, 8.5): ax.axhline(k, color='k', lw=0.8)
ax.set_title('(a) Latency / FP32, same backend', fontsize=FS)
# (b) control
ax = fig.add_subplot(gs[1])
sc = ['all backends (real)', 'no GPU', 'no NPU', 'CPU only']; short = ['all', 'no\nGPU', 'no\nNPU', 'CPU\nonly']
cols = {'Redmi 13': '#c0392b', 'Pixel 4': '#2c7fb8', 'Tab S7+': '#7fcdbb'}
w = 0.26
# Panel (b): the within-device backend-removal control. The three main-campaign
# devices are CIFAR-10; the Tab S11 was only ever run on the ImageNet-100 freeze
# subset, so its bars come from that corpus and are hatched to say so. The
# comparable quantity across devices is the within-device trend, not the level.
mtkb = json.load(open('results/flagship_2026/mtk_virtual_device.json'))
devs_b = ['Redmi 13', 'Pixel 4', 'Tab S7+', 'Tab S11']
cols['Tab S11'] = '#eda100'
w = 0.20
for k, d in enumerate(devs_b):
    vals = ([vd[s][d]['dec_int'] for s in sc] if d != 'Tab S11'
            else [mtkb[s]['dec_int'] for s in sc])
    ax.bar(np.arange(4) + (k - 1.5) * w, vals, w, label=d, color=cols[d],
           hatch='////' if d == 'Tab S11' else None, edgecolor='white',
           linewidth=0.5)
ax.set_xticks(range(4)); ax.set_xticklabels(short)
ax.set_ylabel('Integer mode clearly fastest (%)'); ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=6.4, loc='upper center', ncol=2,
          columnspacing=0.8, handlelength=1.0, labelspacing=0.25)
ax.set_title('(b) Backends removed', fontsize=FS)
# (c) K artifacts
ax = fig.add_subplot(gs[2])
cl = ['<2 ms', '2-10 ms', '>=10 ms']; pol = [('Always dynamic', None), ('One artifact', 'K=1 single artifact'), ('Two (by vendor)', 'K=2 vendor split (Redmi | Pixel+Tab)')]
cc = ['#bdbdbd', '#e08e45', '#2c7fb8']
for k, (nm, key) in enumerate(pol):
    vals = [oo['always dynamic INT8']['model ' + c]['geo'] for c in cl] if key is None else [ka[key]['model ' + c]['geo'] for c in cl]
    b = ax.bar(np.arange(3) + (k - 1) * 0.27, vals, 0.27, label=nm, color=cc[k])
    for x, v in zip(np.arange(3) + (k - 1) * 0.27, vals): ax.text(x, v * 1.08, f'{v:.0f}' if v >= 9.5 else f'{v:.1f}', ha='center', fontsize=6.8)
ax.set_yscale('log'); ax.set_ylim(0.2, 400); ax.set_xticks(range(3)); ax.set_xticklabels([c.replace('>=', '$\\geq$') for c in cl], rotation=20, ha='right')
ax.set_xlabel('Model latency class'); ax.set_ylabel('Geometric-mean regret (%)')
ax.legend(frameon=False, fontsize=7, loc='upper right', handlelength=1.0,
          labelspacing=0.25)
ax.set_title('(c) One vs two artifacts', fontsize=FS)
fig.savefig('../figures/fig_v85_mechanism.png', dpi=600)
print('ok')
