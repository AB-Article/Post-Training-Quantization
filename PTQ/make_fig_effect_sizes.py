#!/usr/bin/env python3
"""Supplementary effect-size figures, regenerated from the released records.

Replaces the two bar charts that previously came from the authors' snapshot
pipeline and had no generator in this folder:

    ../figures/fig15_effect_sizes.png          (CIFAR-10,     Fig. S10)
    ../figures/fig_im100_effect_sizes.png      (ImageNet-100, Fig. S18)

Both now read ``results/v83/paired_tests.json``, so they cannot drift from
Tables S4 and S11, which are computed by the same script. The ImageNet-100
panel in particular had to be regenerated: its effect sizes were replaced with
values recomputed from the released records, and the old figure still showed
the superseded ones with the opposite sign on the Redmi 13.

Fixes carried over from the audit of the originals: the legend is placed
outside the plot area instead of on top of the tallest bars, every bar carries
its value so identity never rests on colour alone, a zero line is drawn because
some values are negative, and the output is saved at 300 dpi.

    python make_fig_effect_sizes.py
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = json.load(open("results/v83/paired_tests.json"))

MODES = ["fp16", "mixed", "dynamic", "int8"]
MODE_LAB = {"fp16": "FP16", "mixed": "Mixed", "dynamic": "Dynamic INT8",
            "int8": "Static INT8"}
DEVICES = ["Redmi 13", "Pixel 4", "Tab S7+"]
# validated categorical slots 1, 2, 3 (see the project's dataviz palette)
DEV_COL = {"Redmi 13": "#2a78d6", "Pixel 4": "#eb6834", "Tab S7+": "#1baf7a"}

COLWIDTH_IN = 3.5          # IEEEtran single column
FS = 8
plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans"})


def draw(corpus, out, title):
    cells = R[corpus]["cells"]
    fig, ax = plt.subplots(figsize=(COLWIDTH_IN, 2.30))
    x = np.arange(len(MODES))
    w = 0.265
    lo, hi = 0.0, 0.0
    for k, dev in enumerate(DEVICES):
        vals = [cells[f"{dev}|{m}"]["r"] for m in MODES]
        lo, hi = min(lo, min(vals)), max(hi, max(vals))
        ax.bar(x + (k - 1) * w, vals, w, label=dev, color=DEV_COL[dev],
               edgecolor="white", linewidth=0.6)
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + (0.018 if v >= 0 else -0.018), f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=5.6, color="#1a1a19")
    ax.axhline(0, color="#1a1a19", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LAB[m] for m in MODES], fontsize=7)
    ax.set_ylabel("Rank-biserial $r$ vs FP32", fontsize=FS)
    ax.set_ylim(min(-0.28, lo - 0.12), hi + 0.22)
    ax.set_title(title, fontsize=FS, pad=16)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), columnspacing=1.3,
              handlelength=1.0, handletextpad=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e3e3e0", lw=0.6)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("ok ", out)


draw("cifar10", "../figures/fig15_effect_sizes.png",
     "CIFAR-10: effect size vs FP32 (positive $=$ faster)")
draw("imagenet100", "../figures/fig_im100_effect_sizes.png",
     "ImageNet-100: effect size vs FP32 (positive $=$ faster)")
