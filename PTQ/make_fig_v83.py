#!/usr/bin/env python3
"""Fig. 2: (a) constrained best-mode win rate by device on the released CIFAR-10
pool; (b) geometric-mean latency regret by model latency class (best feasible
latency on the model's slowest device) for four policies; (c) share of valid
CIFAR-10 runs in which each TFLite backend is the fastest, by device; (d) the
ImageNet-100 counterpart of (a).

The figure is drawn at the exact width of the IEEEtran text block (516 pt =
7.14 in) with an 8 pt font and saved without a tight bounding box, so that
\\includegraphics[width=\\linewidth] renders it at scale 1.0 and every label in
the figure is set at the same 8 pt as the paper's captions.

Inputs: results/v83/dataset_full_cifar10.csv, results/v83/picks_A.csv,
        results/v83/dataset_full_imagenet100.csv, results/v83/runs_long.csv
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import collections
import csv
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

TEXTWIDTH_IN = 516 / 72.27          # IEEEtran \textwidth
FS = 8                              # = caption size of the paper

MODES = ['fp32', 'fp16', 'mixed', 'dynamic', 'int8']
LAB = {'fp32': 'FP32', 'fp16': 'FP16', 'mixed': 'Mixed',
       'dynamic': 'Dynamic', 'int8': 'Static INT8'}
COL = {'fp32': '#4477AA', 'fp16': '#EE7733', 'mixed': '#009988',
       'dynamic': '#CC3311', 'int8': '#AA3377'}
DEV = [('24040RN64Y', 'Redmi 13'), ('Pixel 4', 'Pixel 4'), ('SM-T970', 'Tab S7+')]
BACKEND = [('cpu', 'CPU', '#88CCEE'), ('gpu', 'GPU', '#44AA99'), ('npu', 'NPU', '#DDCC77')]

plt.rcParams.update({'font.size': FS, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, ax = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 2.22),
                       gridspec_kw={'width_ratios': [0.92, 0.95, 0.86, 1.18]})
# (a) is a grid and (b) a curve: no shared axis


def winrate_both(c10, im100, axis):
    """Best-mode win rate by device, both corpora, as a labelled grid.

    Thirty grouped bars in a 1.6 in panel are unreadable at print size whatever
    the resolution, so the same thirty numbers are drawn as a grid: rows are
    modes, columns are device x corpus, the cell carries the win rate and the
    shading encodes it. Nothing is lost and every value can be read."""
    cols, head = [], []
    for k, nm in DEV:
        for frame, tag in ((c10, 'C'), (im100, 'I')):
            cols.append([100 * (frame[frame.device == k].label == m).mean()
                         for m in MODES])
            head.append(tag)
    Z = np.array(cols).T                      # modes x (device x corpus)
    axis.imshow(Z, cmap='Blues', vmin=0, vmax=55, aspect='auto')
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            axis.text(j, i, f'{Z[i, j]:.0f}', ha='center', va='center',
                      fontsize=5.8,
                      color='white' if Z[i, j] > 30 else '#1a1a19')
    axis.set_xticks(range(len(head)))
    axis.set_xticklabels(['C10', 'IN100'] * 3, fontsize=5.0, rotation=90)
    short = {'Redmi 13': 'Redmi 13', 'Pixel 4': 'Pixel 4', 'Tab S7+': 'Tab S7+'}
    for x, (_, nm) in zip((0.5, 2.5, 4.5), DEV):
        axis.annotate(short[nm], xy=(x, 0), xycoords=('data', 'axes fraction'),
                      xytext=(0, -18), textcoords='offset points',
                      ha='center', va='top', fontsize=6.0, annotation_clip=False)
    axis.set_yticks(range(len(MODES)))
    axis.set_yticklabels([LAB[m] for m in MODES], fontsize=6.4)
    axis.set_xticks(np.arange(-.5, len(head), 1), minor=True)
    axis.set_yticks(np.arange(-.5, len(MODES), 1), minor=True)
    axis.grid(which='minor', color='white', lw=0.8)
    axis.tick_params(which='minor', length=0)
    for b in (1.5, 3.5):
        axis.axvline(b, color='k', lw=0.9)
    axis.set_title('(a) Best-mode win rate (%)', fontsize=FS)


# (a) win rate, both corpora -------------------------------------------------
d = pd.read_csv('results/v83/dataset_full_cifar10.csv')
im = pd.read_csv('results/v83/dataset_full_imagenet100.csv')
winrate_both(d, im, ax[0])

# (b) calibration curve: exact match and regret vs benchmarked models ---------
import re as _re
few = {}
for line in open('results/v83/log_B.txt'):
    m = _re.match(r'\s*(\S.*?)\s+few-shot k=(\d+)\s+exact\s+([\d.]+)\s+geo\s+([\d.]+)',
                  line)
    if m:
        few.setdefault(m.group(1).strip(), []).append(
            (int(m.group(2)), float(m.group(3)), float(m.group(4))))
KS = sorted({k for v in few.values() for k, _, _ in v})
DEVCOL = {'Redmi 13': '#c0392b', 'Pixel 4': '#2c7fb8', 'Tab S7+': '#7fcdbb'}
axr = ax[1].twinx() if False else ax[1]
for dev, pts in few.items():
    pts.sort()
    axr.plot([k for k, _, _ in pts], [e for _, e, _ in pts], '-o', ms=3.4,
             lw=1.4, color=DEVCOL.get(dev, '#444444'), label=dev)
    axr.plot([k for k, _, _ in pts], [g for _, _, g in pts], '--s', ms=3.0,
             lw=1.1, color=DEVCOL.get(dev, '#444444'), alpha=0.75)
axr.set_xscale('log')
axr.set_xticks(KS)
axr.set_xticklabels([str(k) for k in KS])
axr.set_xticks([], minor=True)          # drop the log minor ticks/labels
axr.xaxis.set_minor_formatter(plt.NullFormatter())
axr.set_xlabel('Benchmarked models $k$', fontsize=FS)
axr.set_ylabel('Exact match / regret (%)')
axr.set_ylim(0, 75)
axr.grid(axis='y', color='#e5e5e5', lw=0.6)
axr.set_axisbelow(True)
axr.set_title('(b) Calibrating a new device', fontsize=FS)
axr.legend(fontsize=6.4, frameon=False, loc='center right', handlelength=1.1,
           labelspacing=0.2)
axr.annotate('exact match', xy=(KS[0], few['Tab S7+'][0][1]), fontsize=6.2,
             xytext=(2, 5), textcoords='offset points', color='#444444')
axr.annotate('regret', xy=(KS[0], few['Redmi 13'][0][2]), fontsize=6.2,
             xytext=(2, -9), textcoords='offset points', color='#444444')

# (b) regret by latency class ------------------------------------------------
p = pd.read_csv('results/v83/picks_A.csv')
bins, bl = [0, 2, 10, 1e9], ['<2 ms', '2–10 ms', '≥10 ms']
p['model_ms'] = p.groupby('model').best_ms.transform('max')
p['band'] = pd.cut(p.model_ms, bins, labels=bl, right=False)
POL = [('global dynamic', 'Always dynamic INT8', '#BBBBBB'),
       ('per-device order, feasibility-aware', 'Per-device order', '#888888'),
       ('per-arch oracle majority (in-sample)', 'Per-arch. majority (oracle)', '#444444'),
       ('learned +dev (GBT, masked)', 'Selector (GBT)', '#0077BB')]
for j, (k, n, c) in enumerate(POL):
    v = [100 * (np.exp(np.log(p[p.band == b][k + '|ratio']).mean()) - 1) for b in bl]
    ax[3].bar(np.arange(3) + (j - 1.5) * 0.2, v, 0.18, color=c, label=n)
cnt = p.groupby('band', observed=True).model.nunique()
ax[3].set_xticks(range(3))
ax[3].set_xticklabels(bl, rotation=20, ha='right')
ax[3].set_yscale('log')
ax[3].set_ylim(1, 900)
ax[3].set_yticks([1, 3, 10, 30, 100])
ax[3].set_yticklabels(['1', '3', '10', '30', '100'])
ax[3].set_ylabel('Geo.-mean regret (%)')
ax[3].set_title('(d) Regret by latency class', fontsize=FS)
ax[3].legend(fontsize=7, frameon=False, loc='upper right', ncol=1,
             bbox_to_anchor=(1.04, 1.06), handlelength=1.0, labelspacing=0.25)
ax[3].grid(axis='y', color='#e5e5e5', lw=0.6)
ax[3].set_axisbelow(True)

# (c) fastest backend share --------------------------------------------------
r = pd.read_csv('results/v83/runs_long.csv')
r = r[(r.corpus == 'cifar10') & (r.valid.astype(str).isin(['True', 'true', '1']))]
cols = ['cpu_ms', 'gpu_ms', 'npu_ms']
r = r.dropna(subset=cols, how='all').copy()
r['fastest'] = r[cols].idxmin(axis=1).str.replace('_ms', '', regex=False)
NAME = dict(DEV)
share = (r.groupby([r.device.map(NAME), 'fastest']).size()
         .unstack(fill_value=0).pipe(lambda t: (100 * t.T / t.T.sum()).T))
# The Tab S11 is the only 2025-2026 device with per-backend timings; add it.
mtk = collections.Counter()
_ml = collections.defaultdict(dict)
for _r in csv.DictReader(open('results/flagship_2026/raw/mediatek_rows.csv')):
    if _r['status'] != 'ok':
        continue
    try:
        _v = float(_r['latency_ms_avg'])
    except ValueError:
        continue
    if _v > 0.1:
        _ml[(_r['model_name'], _r['quant_mode'])][_r['backend']] = _v
for _d in _ml.values():
    if _d:
        mtk[min(_d, key=_d.get)] += 1
_tot = sum(mtk.values())
share.loc['Tab S11'] = {k: 100 * mtk[k] / _tot for k in ('cpu', 'gpu', 'npu')}
bottom = np.zeros(4)
order = [n for _, n in DEV] + ['Tab S11']
for key, lab, col in BACKEND:
    v = np.array([share.loc[n, key] if key in share.columns else 0.0 for n in order])
    ax[2].bar(range(4), v, 0.66, bottom=bottom, color=col, edgecolor='white', lw=0.6)
    for x, (b, h) in enumerate(zip(bottom, v)):
        if h >= 9:
            ax[2].text(x, b + h / 2, f'{lab}\n{h:.0f}', ha='center', va='center',
                       fontsize=7, linespacing=0.95)
    bottom += v
ax[2].set_xticks(range(4))
ax[2].set_xticklabels(order, rotation=20, ha='right', fontsize=7)
ax[2].set_ylim(0, 100)
ax[2].set_yticks([0, 25, 50, 75, 100])
ax[2].set_ylabel('Share of runs (%)')
ax[2].set_title('(c) Which backend wins', fontsize=FS)
ax[2].set_axisbelow(True)

fig.tight_layout()
fig.savefig('../figures/fig_v83_device_and_regret.png', dpi=600)
print('ok  figure is %.2f in wide' % TEXTWIDTH_IN)

# numbers plotted in panels (b) and (c), saved for the record
with open('results/v83/fig2b_model_bands.txt', 'w') as fh:
    for k, n, _ in POL:
        g = [100 * (np.exp(np.log(p[p.band == b][k + '|ratio']).mean()) - 1) for b in bl]
        m = [p[p.band == b][k + '|ms'].mean() for b in bl]
        fh.write(f'{n:32s} geo% ' + ' '.join(f'{x:6.1f}' for x in g)
                 + '   ms ' + ' '.join(f'{x:5.2f}' for x in m) + '\n')
    fh.write('models per class: ' + str(cnt.to_dict()) + '\n')
    fh.write('\nfastest-backend share (%), CIFAR-10 valid runs:\n')
    fh.write(share.round(1).to_string() + '\n')
