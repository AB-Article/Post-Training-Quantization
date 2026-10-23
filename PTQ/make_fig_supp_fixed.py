#!/usr/bin/env python3
"""Regenerate the supplementary figures that disagreed with the verified tables.

The supplementary figure set was produced by the authors' snapshot pipeline,
which is not part of this folder, so most of it cannot be rebuilt here. Four
figures, however, printed values that contradict result files committed under
``results/`` -- a reader comparing figure and table would see two different
numbers for the same quantity. Those four are regenerated here from the
committed outputs, so they can no longer drift:

  ../figures/fig08_best_mode_overall.png     Fig. S11  constrained win rates
  ../figures/fig16_policy_match.png          Fig. S12  policy exact-match rates
  ../figures/fig14_input_resolution.png      Fig. S4   input-resolution coverage
  ../figures/fig10_dcor_best_mode.png        Fig. S13  CIFAR-10 distance corr.
  ../figures/fig_im100_dcor_best_mode.png    Fig. S24  ImageNet-100 distance corr.

What was wrong, and the authority used:

* S11 showed 7.2 / 36.0 / 13.3 / 35.6 / 8.0 for FP32/FP16/mixed/dynamic/INT8.
  ``results/v83/log_A.txt`` and supplementary Table S5 both give
  8.6 / 32.1 / 13.0 / 35.4 / 10.9. S12's "global default" bar inherited the
  same stale 36.0.
* S4's in-image title said "max 512 px" and the axis stopped there;
  ``results/v83/misc_checks.txt`` records a maximum input side of 800 px.
* S13 and S24 plotted distance correlations that do not appear in
  ``results/v83/dcor_released.txt``, and S24 drew ImageNet-100 input
  resolution as a zero bar although that file records it as not computed.

    python make_fig_supp_fixed.py
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import ast
import csv
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FS = 8
COL = 3.5
plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans"})
# validated categorical slots
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK = "#1a1a19"
GRID = "#e3e3e0"


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)


def _logA(path="results/v83/log_A.txt"):
    out = {}
    for line in open(path):
        m = re.match(r"^(.*?)\s{2,}exact\s+([\d.]+)", line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out


# ---------------------------------------------------------- S11 win rates
MODES = ["fp32", "fp16", "mixed", "dynamic", "int8"]
LAB = {"fp32": "FP32", "fp16": "FP16", "mixed": "Mixed",
       "dynamic": "Dynamic INT8", "int8": "Static INT8"}
A = _logA()
share = {m: A[f"global {m}"] for m in MODES}
fig, ax = plt.subplots(figsize=(COL, 2.25))
v = [share[m] for m in MODES]
ax.bar(range(5), v, 0.62, color=C1, edgecolor="white", linewidth=0.6)
for i, y in enumerate(v):
    ax.text(i, y + 0.7, f"{y:.1f}", ha="center", fontsize=6.6, color=INK)
ax.set_xticks(range(5))
ax.set_xticklabels([LAB[m] for m in MODES], fontsize=6.6)
ax.set_ylabel("Share of model\u2013device pairs (%)", fontsize=FS)
ax.set_ylim(0, max(v) + 7)
ax.set_title(r"Constrained best PTQ mode ($\tau=0.99$), CIFAR-10", fontsize=FS)
ax.yaxis.grid(True, color=GRID, lw=0.6)
_style(ax)
fig.tight_layout()
fig.savefig("../figures/fig08_best_mode_overall.png", dpi=300)
plt.close(fig)
print("ok  fig08_best_mode_overall.png ", [f"{x:.1f}" for x in v])

# ------------------------------------------------------- S12 policy match
POL = [("global dynamic", "Global default\n(dynamic INT8)"),
       ("global order, feasibility-aware", "Global order,\nfeasibility-aware"),
       ("per-device order, feasibility-aware", "Per-device order,\nfeasibility-aware"),
       ("learned offline (GBT, masked)", "Learned selector\n(offline features)"),
       ("per-arch oracle majority (in-sample)", "Per-architecture\noracle")]
vals = [A[k] for k, _ in POL]
fig, ax = plt.subplots(figsize=(COL, 2.45))
ax.barh(range(len(POL)), vals, 0.6, color=C1, edgecolor="white", linewidth=0.6)
for i, y in enumerate(vals):
    ax.text(y + 1.0, i, f"{y:.1f}", va="center", fontsize=6.6, color=INK)
ax.set_yticks(range(len(POL)))
ax.set_yticklabels([lab for _, lab in POL], fontsize=6.2)
ax.invert_yaxis()
ax.set_xlabel("Exact match with the constrained optimum (%)", fontsize=FS)
ax.set_xlim(0, max(vals) + 10)
ax.xaxis.grid(True, color=GRID, lw=0.6)
_style(ax)
fig.tight_layout()
fig.savefig("../figures/fig16_policy_match.png", dpi=300)
plt.close(fig)
print("ok  fig16_policy_match.png      ", [f"{x:.1f}" for x in vals])

# --------------------------------------------------- S4 input resolution
px = Counter()
for r in csv.DictReader(open("results/v83/dataset_full_cifar10.csv")):
    try:
        px[int(float(r["in_w"]))] += 1
    except (TypeError, ValueError):
        pass
sides = sorted(px)
tot = sum(px.values())
fig, ax = plt.subplots(figsize=(COL, 2.25))
ax.bar(range(len(sides)), [px[s] for s in sides], 0.72, color=C1,
       edgecolor="white", linewidth=0.6)
ax.set_xticks(range(len(sides)))
ax.set_xticklabels([str(s) for s in sides], fontsize=6.2, rotation=90)
ax.set_xlabel("Input side (px)", fontsize=FS)
ax.set_ylabel("Model\u2013device pairs", fontsize=FS)
med = sorted(s for s in sides for _ in range(px[s]))[tot // 2]
ax.set_title(f"Input resolutions (median {med} px, max {max(sides)} px)",
             fontsize=FS)
ax.yaxis.grid(True, color=GRID, lw=0.6)
_style(ax)
fig.tight_layout()
fig.savefig("../figures/fig14_input_resolution.png", dpi=300)
plt.close(fig)
print(f"ok  fig14_input_resolution.png   median {med} px, max {max(sides)} px")

# ------------------------------------------------------ S13/S24 distance corr.
raw = open("results/v83/dcor_released.txt").read()
dcor = {}
for corpus in ("cifar10", "imagenet100"):
    m = re.search(corpus + r"\s+(\{.*?\})\n", raw, re.S)
    d = ast.literal_eval(re.sub(r"np\.float64\(([^)]*)\)", r"\1", m.group(1)))
    dcor[corpus] = {k: v for k, v in d.items() if v is not None}
for corpus, out, title in [
        ("cifar10", "../figures/fig10_dcor_best_mode.png",
         "CIFAR-10: dependence on the best-mode label"),
        ("imagenet100", "../figures/fig_im100_dcor_best_mode.png",
         "ImageNet-100: dependence on the best-mode label")]:
    d = dict(sorted(dcor[corpus].items(), key=lambda kv: -kv[1]))
    fig, ax = plt.subplots(figsize=(COL, 2.20))
    ax.barh(range(len(d)), list(d.values()), 0.6, color=C1,
            edgecolor="white", linewidth=0.6)
    for i, y in enumerate(d.values()):
        ax.text(y + 0.008, i, f"{y:.2f}", va="center", fontsize=6.6, color=INK)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(list(d), fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("Distance correlation", fontsize=FS)
    ax.set_xlim(0, max(d.values()) + 0.08)
    ax.set_title(title, fontsize=FS)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"ok  {out.split('/')[-1]:30s}", {k: round(v, 2) for k, v in d.items()})
