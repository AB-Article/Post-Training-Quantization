#!/usr/bin/env python3
"""Generators for the descriptive figure set that previously had none.

``figures/`` ships 22 plots that appear in neither document; every quantity they
show is tabulated in the supplement and they are kept only for traceability.
Eighteen of them had no generating script in this package, so they could not be
rebuilt.  This module draws all eighteen from the committed records, in the
house style of ``make_all_figures.py``:

    CIFAR-10   fig03 accuracy vs latency      fig05 accuracy drop vs FP32
               fig07 model-family comparison  fig09 best mode by device
               fig11 size vs latency          fig12 latency percentiles
               fig17 replication
    ImageNet-100 mirrors
               accuracy_vs_latency_scatter    accuracy_drop_vs_fp32
               model_family_comparison        best_mode_by_device
               size_vs_latency                latency_percentiles
               best_mode_overall              backend_distribution
               mean_latency_by_mode_device    mean_accuracy_by_mode_device
               tau_sensitivity

Inputs: ``results/v83/dataset_full_<corpus>.csv`` and
``results/v83/runs_long.csv``.  ``fig17_replication`` additionally reads
``results/v83/fig17_replication_values.json`` (see that file for why).

Provenance note: the PNGs that shipped before this module existed were drawn by
the authors' earlier snapshot pipeline, whose record set the released records
under ``results/v83/`` have since superseded -- the same migration the
supplement describes in Section S-VI.  The plots produced here therefore track
the released records and the supplementary tables, and differ by a few tenths of
a point from those earlier images.  None of these figures is included in
``main.tex`` or ``supplementary.tex``; they are kept only for traceability.

    python make_fig_extra.py        # or via make_all_figures.py
"""
from __future__ import annotations

import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / ".." / "figures"
R83 = ROOT / "results" / "v83"

MODES = ["fp32", "fp16", "mixed", "dynamic", "int8"]
MODE_LAB = {"fp32": "FP32", "fp16": "FP16", "mixed": "Mixed",
            "dynamic": "Dynamic INT8", "int8": "Static INT8"}
MODE_COL = {"fp32": "#2a78d6", "fp16": "#eb6834", "mixed": "#6250d6",
            "dynamic": "#1baf7a", "int8": "#eda100"}
DEV_MAP = {"24040RN64Y": "Redmi 13", "Pixel 4": "Pixel 4", "SM-T970": "Tab S7+"}
DEV_ORDER = ["Redmi 13", "Pixel 4", "Tab S7+"]
DEV_COL = {"Redmi 13": "#2a78d6", "Pixel 4": "#eb6834", "Tab S7+": "#1baf7a"}
CORPUS_LAB = {"cifar10": "CIFAR-10", "imagenet100": "ImageNet-100"}
TAU = 0.01                      # relative accuracy budget, as in predictor.py
INK, GRID = "#1a1a19", "#e3e3e0"


# --------------------------------------------------------------- scaffolding
def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(labelsize=7, colors=INK)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=300)
    plt.close(fig)
    print(f"ok  {name}")


def _mode_legend(ax, modes=MODES, **kw):
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MODE_COL[m]) for m in modes]
    ax.legend(handles, [MODE_LAB[m] for m in modes], fontsize=6.5,
              frameon=False, ncol=kw.pop("ncol", 2), **kw)


FAM_ORDER = ["named", "unq", "rag", "moe"]


def load(corpus):
    """Committed labelled table.  ``acc_*`` and ``accdrop_*`` are stored as
    fractions; they are converted to percent and percentage points here.  The
    ``label`` column is the constrained winner at tau = 0.99 and is used as-is
    wherever a figure shows best-mode rates, so the plots agree with the tables
    rather than re-deriving the label."""
    d = pd.read_csv(R83 / f"dataset_full_{corpus}.csv")
    d["dev"] = d.device.map(DEV_MAP)
    for m in MODES:
        d[f"acc_pct_{m}"] = 100 * d[f"acc_{m}"]
        if m != "fp32":
            d[f"accchg_{m}"] = -100 * d[f"accdrop_{m}"]
    return d


def runs(corpus):
    r = pd.read_csv(R83 / "runs_long.csv")
    r = r[(r.corpus == corpus) & r.valid].copy()
    r["dev"] = r.device.map(DEV_MAP)
    r["ms"] = r.duration / 1e6
    return r


def feasible(row, tau=TAU):
    """Modes within the accuracy budget of the matched FP32 graph."""
    ref, out = row["acc_fp32"], ["fp32"]
    for m in MODES[1:]:
        a = row.get(f"acc_{m}")
        if pd.notna(a) and a >= ref * (1 - tau):
            out.append(m)
    return out


def winner(row, tau=TAU):
    """Fastest mode inside the budget; None when nothing was measured."""
    cand = {m: row[f"lat_{m}"] for m in feasible(row, tau) if pd.notna(row[f"lat_{m}"])}
    if not cand:
        return None
    best = min(cand.values())
    for m in MODES:                       # ties resolve to the least quantized mode
        if m in cand and cand[m] == best:
            return m
    return None


# ------------------------------------------------------------------ figures
def accuracy_vs_latency(corpus, name):
    d = load(corpus)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), sharey=True)
    for ax, dev in zip(axes, DEV_ORDER):
        s = d[d.dev == dev]
        for m in MODES:
            ax.scatter(s[f"lat_{m}"] / 1e6, s[f"acc_pct_{m}"], s=3, alpha=0.35,
                       c=MODE_COL[m], edgecolors="none", rasterized=True)
        ax.set_xscale("log")
        ax.set_title(dev, fontsize=8, color=INK)
        ax.set_xlabel("Latency (ms, log)", fontsize=7.5)
        _style(ax)
        ax.grid(axis="x", color=GRID, lw=0.6)
    axes[0].set_ylabel("Converted-graph accuracy (%)", fontsize=7.5)
    _mode_legend(axes[0], loc="lower right", ncol=1)
    fig.suptitle(f"{CORPUS_LAB[corpus]}: accuracy versus on-device latency",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, name)


def accuracy_drop(corpus, name):
    """Common-support pool: only rows with all four quantized accuracies, so the
    four bars are means over the same architectures.  On that pool the mixed and
    static-INT8 means coincide, which is the result misc_checks.py reports."""
    d = load(corpus).dropna(subset=[f"accdrop_{m}" for m in MODES[1:]])
    modes = MODES[1:]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    w = 0.8 / len(modes)
    for i, m in enumerate(modes):
        vals = [d[d.dev == dev][f"accchg_{m}"].mean() for dev in DEV_ORDER]
        ax.bar(np.arange(3) + i * w - 0.4 + w / 2, vals, w,
               color=MODE_COL[m], label=MODE_LAB[m])
    ax.set_xticks(range(3))
    ax.set_xticklabels(DEV_ORDER, fontsize=7.5)
    ax.set_ylabel("Mean accuracy change vs FP32 (pp)", fontsize=7.5)
    ax.axhline(0, color=INK, lw=0.6)
    _style(ax)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    ax.set_title(f"{CORPUS_LAB[corpus]}: PTQ accuracy change relative to the "
                 "matched FP32 graph", fontsize=8.5, color=INK)
    _save(fig, name)


def family_comparison(corpus, name):
    d = load(corpus)
    fams = [f for f in FAM_ORDER if (d.family == f).any()]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    w = 0.8 / len(MODES)
    for i, m in enumerate(MODES):
        off = i * w - 0.4 + w / 2
        axes[0].bar(np.arange(len(fams)) + off,
                    [d[d.family == f][f"acc_pct_{m}"].mean() for f in fams], w,
                    color=MODE_COL[m])
        axes[1].bar(np.arange(len(fams)) + off,
                    [d[d.family == f][f"lat_{m}"].mean() / 1e6 for f in fams], w,
                    color=MODE_COL[m])
    for ax, lab in zip(axes, ("Mean accuracy (%)", "Mean latency (ms)")):
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels(fams, fontsize=7.5)
        ax.set_ylabel(lab, fontsize=7.5)
        _style(ax)
    _mode_legend(axes[1], loc="upper right")
    fig.suptitle(f"{CORPUS_LAB[corpus]}: accuracy and latency by model family",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, name)


def best_mode_by_device(corpus, name):
    d = load(corpus)
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    w = 0.8 / len(MODES)
    for i, m in enumerate(MODES):
        vals = [100 * (d[d.dev == dev].label == m).mean() for dev in DEV_ORDER]
        ax.bar(np.arange(3) + i * w - 0.4 + w / 2, vals, w,
               color=MODE_COL[m], label=MODE_LAB[m])
    ax.set_xticks(range(3))
    ax.set_xticklabels(DEV_ORDER, fontsize=7.5)
    ax.set_ylabel("Best-mode win rate (%)", fontsize=7.5)
    _style(ax)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    ax.set_title(rf"{CORPUS_LAB[corpus]}: device-conditioned best PTQ mode "
                 rf"($\tau = {1 - TAU:.2f}$)", fontsize=8.5, color=INK)
    _save(fig, name)


def best_mode_overall(corpus, name):
    d = load(corpus)
    share = [100 * (d.label == m).mean() for m in MODES]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.bar(range(len(MODES)), share, 0.62, color=[MODE_COL[m] for m in MODES])
    for i, v in enumerate(share):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([MODE_LAB[m] for m in MODES], fontsize=7.5)
    ax.set_ylabel("Share of model-device pairs (%)", fontsize=7.5)
    ax.set_ylim(0, max(share) * 1.18)
    _style(ax)
    ax.set_title(rf"{CORPUS_LAB[corpus]}: fastest feasible mode "
                 rf"($\tau = {1 - TAU:.2f}$)", fontsize=8.5, color=INK)
    _save(fig, name)


def size_vs_latency(corpus, name):
    d = load(corpus)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for m in MODES:
        ax.scatter(d.size_fp32_kib, d[f"lat_{m}"] / 1e6, s=4, alpha=0.3,
                   c=MODE_COL[m], edgecolors="none", label=MODE_LAB[m],
                   rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FP32 model size (KiB, log)", fontsize=7.5)
    ax.set_ylabel("On-device latency (ms, log)", fontsize=7.5)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    leg = ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    for h in leg.legend_handles:
        h.set_alpha(1)
        h.set_sizes([18])
    ax.set_title(f"{CORPUS_LAB[corpus]}: model size versus on-device latency",
                 fontsize=8.5, color=INK)
    _save(fig, name)


def latency_percentiles(corpus, name):
    r = runs(corpus)
    pcts = [("median", 50), ("p90", 90), ("p95", 95), ("p99", 99)]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    w = 0.8 / len(DEV_ORDER)
    for i, dev in enumerate(DEV_ORDER):
        v = [np.percentile(r[r.dev == dev].ms, p) for _, p in pcts]
        xs = np.arange(len(pcts)) + i * w - 0.4 + w / 2
        ax.bar(xs, v, w, color=DEV_COL[dev], label=dev)
        for x, y in zip(xs, v):
            ax.text(x, y + max(v) * 0.015, f"{y:.0f}", ha="center", fontsize=6.5,
                    color=INK)
    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels([n for n, _ in pcts], fontsize=7.5)
    ax.set_ylabel("Latency (ms)", fontsize=7.5)
    _style(ax)
    ax.legend(fontsize=6.5, frameon=False, ncol=3)
    ax.set_title(f"{CORPUS_LAB[corpus]}: latency percentiles, pooled over PTQ modes",
                 fontsize=8.5, color=INK)
    _save(fig, name)


def mean_by_mode_device(corpus, name, what):
    """what = 'lat' (mean latency, ms) or 'acc' (mean converted-graph accuracy)."""
    d = load(corpus)
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    w = 0.8 / len(MODES)
    for i, m in enumerate(MODES):
        col = f"lat_{m}" if what == "lat" else f"acc_pct_{m}"
        vals = [d[d.dev == dev][col].mean() / (1e6 if what == "lat" else 1)
                for dev in DEV_ORDER]
        ax.bar(np.arange(3) + i * w - 0.4 + w / 2, vals, w,
               color=MODE_COL[m], label=MODE_LAB[m])
    ax.set_xticks(range(3))
    ax.set_xticklabels(DEV_ORDER, fontsize=7.5)
    ax.set_ylabel("Mean latency (ms)" if what == "lat" else "Mean top-1 accuracy (%)",
                  fontsize=7.5)
    _style(ax)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    kind = "latency" if what == "lat" else "converted-graph accuracy"
    ax.set_title(f"{CORPUS_LAB[corpus]}: mean {kind} by mode and device",
                 fontsize=8.5, color=INK)
    _save(fig, name)


def backend_distribution(corpus, name):
    r = runs(corpus)
    cols = {"CPU": "cpu_ms", "GPU": "gpu_ms", "NPU": "npu_ms"}
    sub = r.dropna(subset=list(cols.values()), how="all").copy()
    arr = sub[list(cols.values())].to_numpy(dtype=float)
    fastest = np.full(len(sub), -1)
    ok = ~np.all(np.isnan(arr), axis=1)
    fastest[ok] = np.nanargmin(arr[ok], axis=1)
    sub["backend"] = [list(cols)[i] if i >= 0 else None for i in fastest]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    w = 0.8 / len(cols)
    bcol = {"CPU": "#2a78d6", "GPU": "#eb6834", "NPU": "#1baf7a"}
    for i, b in enumerate(cols):
        vals = [100 * (sub[sub.dev == dev].backend == b).mean() for dev in DEV_ORDER]
        xs = np.arange(3) + i * w - 0.4 + w / 2
        ax.bar(xs, vals, w, color=bcol[b], label=b)
        for x, y in zip(xs, vals):
            ax.text(x, y + 1.2, f"{y:.0f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(DEV_ORDER, fontsize=7.5)
    ax.set_ylabel("Share of valid runs (%)", fontsize=7.5)
    ax.set_ylim(0, 100)
    _style(ax)
    ax.legend(fontsize=6.5, frameon=False, ncol=3)
    ax.set_title(f"{CORPUS_LAB[corpus]}: fastest backend over all valid runs",
                 fontsize=8.5, color=INK)
    _save(fig, name)


def tau_sensitivity(corpus, name):
    d = load(corpus)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    w = 0.38
    for i, (tau, lab, col) in enumerate([(None, r"$\tau = 0.99$", "#2a78d6"),
                                         (0.001, r"$\tau = 0.999$", "#eb6834")]):
        win = (d.label if tau is None else
               pd.Series([winner(r, tau) for _, r in d.iterrows()]).dropna())
        vals = [100 * (win == m).mean() for m in MODES]
        ax.bar(np.arange(len(MODES)) + i * w - w / 2, vals, w, color=col, label=lab)
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([MODE_LAB[m] for m in MODES], fontsize=7.5)
    ax.set_ylabel("Win rate (%)", fontsize=7.5)
    _style(ax)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"{CORPUS_LAB[corpus]}: sensitivity to the accuracy tolerance",
                 fontsize=8.5, color=INK)
    _save(fig, name)


def replication(name="fig17_replication.png"):
    """Both series are carried in the committed JSON beside this script.

    Neither is recomputable from the released record set: one is the authors'
    full-corpus snapshot (supplement Section S-VI) and the other is a historical
    202-architecture state of the public repository that the current records have
    superseded.  The JSON documents that provenance.
    """
    ref = json.load(open(R83 / "fig17_replication_values.json"))
    labels = ["best mode changes\nwith device", "architecture-only\nmatch rate"]
    series = [(ref["full_corpus"], "Full corpus (authors' snapshot)", "#2a78d6"),
              (ref["released_subset"],
               f"Released subset ($n = {ref['released_n']}$)", "#eb6834")]
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    w = 0.38
    for i, (vals, lab, col) in enumerate(series):
        xs = np.arange(2) + i * w - w / 2
        ax.bar(xs, vals, w, color=col, label=lab)
        for x, y in zip(xs, vals):
            ax.text(x, y + 1.5, f"{y:.1f}%", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Percent", fontsize=7.5)
    ax.set_ylim(0, 100)
    _style(ax)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax.set_title("CIFAR-10: released subset versus the full corpus",
                 fontsize=8.5, color=INK)
    _save(fig, name)


def main():
    accuracy_vs_latency("cifar10", "fig03_accuracy_vs_latency_scatter.png")
    accuracy_drop("cifar10", "fig05_accuracy_drop_vs_fp32.png")
    family_comparison("cifar10", "fig07_model_family_comparison.png")
    best_mode_by_device("cifar10", "fig09_best_mode_by_device.png")
    size_vs_latency("cifar10", "fig11_size_vs_latency.png")
    latency_percentiles("cifar10", "fig12_latency_percentiles.png")
    replication()

    c = "imagenet100"
    accuracy_vs_latency(c, "fig_im100_accuracy_vs_latency_scatter.png")
    accuracy_drop(c, "fig_im100_accuracy_drop_vs_fp32.png")
    family_comparison(c, "fig_im100_model_family_comparison.png")
    best_mode_by_device(c, "fig_im100_best_mode_by_device.png")
    best_mode_overall(c, "fig_im100_best_mode_overall.png")
    size_vs_latency(c, "fig_im100_size_vs_latency.png")
    latency_percentiles(c, "fig_im100_latency_percentiles.png")
    mean_by_mode_device(c, "fig_im100_mean_latency_by_mode_device.png", "lat")
    mean_by_mode_device(c, "fig_im100_mean_accuracy_by_mode_device.png", "acc")
    backend_distribution(c, "fig_im100_backend_distribution.png")
    tau_sensitivity(c, "fig_im100_tau_sensitivity.png")


if __name__ == "__main__":
    main()
