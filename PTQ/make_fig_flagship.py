#!/usr/bin/env python3
"""Fig. 3: the 2025-2026 flagship spot-check across all six measured devices.

(a) Share of the frozen 150-model subset for which each PTQ mode is the fastest
    *feasible* one under the 1% accuracy budget, for all six devices. This is
    the panel that shows the new hardware next to the original three, and it
    replaces the former per-mode table.
(b) Median relative regret of shipping one device's choice on another, as an
    ordered 6x6 matrix. Devices are grouped by measurement harness; cells that
    cross harnesses are hatched, because a same-device control shows the
    harness alone costs about as much agreement as the vendor cut.

Drawn at the exact IEEEtran text width (516 pt = 7.14 in) with an 8 pt font and
saved without a tight bounding box, matching make_fig_v85.py so that
\\includegraphics[width=\\linewidth] renders it at scale 1.0.

Colors are the validated categorical slots 1,2,3,4,7 for the five modes and the
blue sequential ramp for the matrix (see the project's dataviz palette); the
categorical set passes the colorblind-separation check, and every segment above
8% also carries a direct numeric label so identity is never colour-alone.

    python make_fig_flagship.py   ->  ../figures/fig_flagship.png
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

R = json.load(open("results/flagship_2026/flagship_corrected.json"))

MODES = ["fp32", "fp16", "dynamic", "int8", "mixed"]
MODE_LAB = {"fp32": "FP32", "fp16": "FP16", "dynamic": "Dynamic INT8",
            "int8": "Static INT8", "mixed": "Mixed"}
# validated categorical slots 1, 2, 3, 4, 7
MODE_COL = {"fp32": "#2a78d6", "fp16": "#eb6834", "dynamic": "#1baf7a",
            "int8": "#eda100", "mixed": "#6250d6"}
DEV_LAB = {
    "USB_Redmi13": "Redmi 13\nHelio G91",
    "USB_Pixel4": "Pixel 4\nSD 855",
    "USB_TabS7": "Tab S7+\nSD 865+",
    "MTK_D9400": "Tab S11\nD9400",
    "S25": "S25\n8 Elite",
    "S26": "S26\n8 Elite G5",
}
STACK_LAB = {"usb": "USB harness", "benchmark_model": "on-device",
             "aihub": "AI Hub"}

order = R["device_order"]
stack = R["device_stack"]

TEXTWIDTH_IN = 516 / 72.27
FS = 8
plt.rcParams.update({"font.size": FS, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.46))
gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.30,
                      left=0.072, right=0.985, top=0.800, bottom=0.255)

# ---------------------------------------------------------------- (a) shares
ax = fig.add_subplot(gs[0])
x = np.arange(len(order))
bottom = np.zeros(len(order))
for m in MODES:
    vals = np.array([R["winrate_constrained"][d]["pct"].get(m, 0.0)
                     for d in order])
    ax.bar(x, vals, 0.72, bottom=bottom, label=MODE_LAB[m],
           color=MODE_COL[m], edgecolor="white", linewidth=0.8)
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v >= 8.0:                      # direct label: identity not colour-alone
            ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center",
                    fontsize=6.6, color="white", fontweight="bold")
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels([DEV_LAB[d] for d in order], fontsize=6.6)
ax.set_ylim(0, 100)
ax.set_ylabel("Fastest feasible mode (% of models)", fontsize=FS)
ax.set_title("(a) Winner share under the 1% accuracy budget", fontsize=FS,
             pad=5)
ax.tick_params(axis="y", labelsize=7)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_linewidth(0.6)
ax.spines["bottom"].set_linewidth(0.6)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#e3e3e0", lw=0.6)
# harness brackets under the device labels
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=7, ncol=5,
           loc="upper center", bbox_to_anchor=(0.5, 1.012),
           columnspacing=1.4, handlelength=1.0, handletextpad=0.45)

# --------------------------------------------------------------- (b) regret
ax = fig.add_subplot(gs[1])
n = len(order)
Z = np.full((n, n), np.nan)
same = np.zeros((n, n), bool)
for i, a in enumerate(order):
    for j, b in enumerate(order):
        c = R["regret_matrix"].get(f"{a}|{b}")
        if c is None:
            continue
        Z[i, j] = c["median_pct"]
        same[i, j] = c["same_stack"]
blues = LinearSegmentedColormap.from_list(
    "seqblue", ["#eef5fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6",
                "#1c5cab", "#0d366b"])
norm = SymLogNorm(linthresh=1.0, vmin=0, vmax=140, base=10)
ax.imshow(Z, cmap=blues, norm=norm, aspect="auto")
for i in range(n):
    for j in range(n):
        if np.isnan(Z[i, j]):
            continue
        if i == j:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#f0efec",
                                   edgecolor="none", zorder=2))
            ax.text(j, i, "\u2014", ha="center", va="center", fontsize=7,
                    color="#6b6b68", zorder=3)
            continue
        if not same[i, j]:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   hatch="////", edgecolor="white",
                                   linewidth=0.0, zorder=2, alpha=0.55))
        dark = norm(Z[i, j]) > 0.62
        ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center", fontsize=6.8,
                color="white" if dark else "#1a1a19", zorder=3,
                path_effects=[pe.withStroke(
                    linewidth=1.6,
                    foreground="#0d366b" if dark else "white")])
short = {"USB_Redmi13": "Redmi 13", "USB_Pixel4": "Pixel 4",
         "USB_TabS7": "Tab S7+", "MTK_D9400": "Tab S11", "S25": "S25",
         "S26": "S26"}
ax.set_xticks(range(n))
ax.set_xticklabels([short[d] for d in order], fontsize=6.6, rotation=38,
                   ha="right")
ax.set_yticks(range(n))
ax.set_yticklabels([short[d] for d in order], fontsize=6.6)
ax.set_xlabel("shipped on", fontsize=7, labelpad=1.5)
ax.set_ylabel("mode chosen for", fontsize=7, labelpad=1.5)
ax.set_title("(b) Median regret (%) of one shared artifact", fontsize=FS,
             pad=5)
# harness-group separators
bounds = []
for k in range(1, n):
    if stack[order[k]] != stack[order[k - 1]]:
        bounds.append(k - 0.5)
for bd in bounds:
    ax.axhline(bd, color="#1a1a19", lw=1.0)
    ax.axvline(bd, color="#1a1a19", lw=1.0)
ax.set_xticks(np.arange(-.5, n, 1), minor=True)
ax.set_yticks(np.arange(-.5, n, 1), minor=True)
ax.grid(which="minor", color="white", lw=0.8)
ax.tick_params(which="minor", length=0)
for s in ax.spines.values():
    s.set_linewidth(0.6)

fig.savefig("../figures/fig_flagship.png", dpi=600)
print("ok  ../figures/fig_flagship.png")
print("    panel (a) devices:", ", ".join(order))
print("    panel (b) cross-harness cells hatched:",
      int((~same).sum() - 0), "of", n * n - n)
