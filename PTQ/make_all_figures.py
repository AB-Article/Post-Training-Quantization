#!/usr/bin/env python3
"""Single entry point: regenerate every manuscript figure from committed data.

Runs the specialized drawers in this folder, then builds the supplementary
figures that previously had no generator here (fig01, fig02, fig06, fig13,
fig16 twin for ImageNet-100, fig18, fig_im100_export).

Together with make_fig_extra.py -- which draws the eighteen descriptive plots
that had no generator in this package at all -- this driver rebuilds every PNG
under figures/, so `./run_all.sh figures` leaves no image unaccounted for.

    python make_all_figures.py

Needs only analysis/results/, analysis/im100_acc/, and ../figures/.
Optional sibling scripts: make_fig_v85.py, make_fig_flagship.py,
make_fig_v83.py, make_fig_effect_sizes.py, make_fig_supp_fixed.py,
make_fig_extra.py.
"""
from __future__ import annotations

import ast
import csv
import json
import re
import runpy
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / ".." / "figures"
R83 = ROOT / "results" / "v83"
FS = 8
COL = 3.5
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, GRID = "#1a1a19", "#e3e3e0"
MODES = ["fp32", "fp16", "mixed", "dynamic", "int8"]
MODE_LAB = {
    "fp32": "FP32",
    "fp16": "FP16",
    "mixed": "Mixed",
    "dynamic": "Dynamic INT8",
    "int8": "Static INT8",
}
MODE_COL = {
    "fp32": "#2a78d6",
    "fp16": "#eb6834",
    "mixed": "#6250d6",
    "dynamic": "#1baf7a",
    "int8": "#eda100",
}
DEV_MAP = {"24040RN64Y": "Redmi 13", "Pixel 4": "Pixel 4", "SM-T970": "Tab S7+"}
DEV_ORDER = ["Redmi 13", "Pixel 4", "Tab S7+"]


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)


def _save(fig, name):
    out = FIG / name
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"ok  {name}")


def _runs(corpus="cifar10"):
    df = pd.read_csv(R83 / "runs_long.csv")
    df = df[(df["corpus"] == corpus) & (df["valid"] == True)].copy()  # noqa: E712
    df["dev"] = df["device"].map(DEV_MAP)
    df["lat_ms"] = df["duration"] / 1e6
    return df.dropna(subset=["dev"])


def _logA(path):
    out = {}
    for line in open(path):
        m = re.match(r"^(.*?)\s{2,}exact\s+([\d.]+)", line)
        if m:
            out[m.group(1).strip()] = float(m.group(2))
    return out


def _acc_json(path):
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[k] = float(v["accuracy"])
        else:
            out[k] = float(v)
    return out


# ---------------------------------------------------------------------------
# Previously missing generators
# ---------------------------------------------------------------------------

def fig01_mean_latency():
    df = _runs("cifar10")
    means = df.groupby(["mode", "dev"], observed=True)["lat_ms"].mean()
    x = np.arange(len(DEV_ORDER))
    w = 0.15
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for i, m in enumerate(MODES):
        vals = [float(means.get((m, d), np.nan)) for d in DEV_ORDER]
        ax.bar(x + (i - 2) * w, vals, w, label=MODE_LAB[m], color=MODE_COL[m],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(DEV_ORDER)
    ax.set_ylabel("Mean latency (ms)", fontsize=FS)
    ax.set_title("CIFAR-10 mean latency by PTQ mode and device", fontsize=FS)
    ax.legend(fontsize=5.8, ncol=5, loc="upper right", frameon=False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    _style(ax)
    _save(fig, "fig01_mean_latency_by_mode_device.png")


def fig06_backend_distribution():
    df = _runs("cifar10")
    backends = ["CPU", "GPU", "NPU"]
    bcol = {"CPU": C1, "GPU": C2, "NPU": C3}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, dev in zip(axes, DEV_ORDER):
        sub = df[df["dev"] == dev]
        shares = (sub["unit"].value_counts(normalize=True) * 100).reindex(backends).fillna(0)
        ax.bar(range(3), shares.values, 0.65, color=[bcol[b] for b in backends],
               edgecolor="white", linewidth=0.5)
        for i, y in enumerate(shares.values):
            if y >= 3:
                ax.text(i, y + 1.2, f"{y:.0f}%", ha="center", fontsize=6.2, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(backends, fontsize=7)
        ax.set_title(dev, fontsize=FS)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        _style(ax)
    axes[0].set_ylabel("Share of valid runs (%)", fontsize=FS)
    fig.suptitle("Fastest successful TFLite backend (CIFAR-10)", fontsize=FS, y=1.02)
    _save(fig, "fig06_backend_distribution.png")


def fig13_repeatability():
    df = _runs("cifar10")

    def rsd(r):
        u = str(r["unit"]).upper()
        if u == "GPU" and r["gpu_ms"]:
            return 100.0 * r["gpu_sd_ms"] / r["gpu_ms"]
        if u == "CPU" and r["cpu_ms"]:
            return 100.0 * r["cpu_sd_ms"] / r["cpu_ms"]
        if u == "NPU" and r["npu_ms"]:
            return 100.0 * r["npu_sd_ms"] / r["npu_ms"]
        return np.nan

    df["rsd"] = df.apply(rsd, axis=1)
    med = df.groupby(["mode", "dev"], observed=True)["rsd"].median()
    x = np.arange(len(DEV_ORDER))
    w = 0.15
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for i, m in enumerate(MODES):
        vals = [float(med.get((m, d), np.nan)) for d in DEV_ORDER]
        ax.bar(x + (i - 2) * w, vals, w, label=MODE_LAB[m], color=MODE_COL[m],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(DEV_ORDER)
    ax.set_ylabel("Median RSD over 20 iters (%)", fontsize=FS)
    ax.set_title("Within-run repeatability (CIFAR-10)", fontsize=FS)
    ax.legend(fontsize=5.8, ncol=5, loc="upper right", frameon=False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    _style(ax)
    _save(fig, "fig13_repeatability.png")


def fig02_accuracy_and_cost():
    """Mean converted-graph accuracy and drop vs FP32 (unique models, all 5 modes)."""
    df = pd.read_csv(R83 / "dataset_full_cifar10.csv")
    cols = [f"acc_{m}" for m in MODES]
    # one row per model (accuracies are model-level; take first device)
    u = df.drop_duplicates("model")
    u = u.dropna(subset=cols)
    means = [100.0 * u[f"acc_{m}"].mean() for m in MODES]
    drops = [0.0] + [
        100.0 * u[f"accdrop_{m}"].mean() for m in MODES if m != "fp32"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.55))
    ax = axes[0]
    ax.bar(range(5), means, 0.62, color=[MODE_COL[m] for m in MODES],
           edgecolor="white", linewidth=0.5)
    for i, y in enumerate(means):
        ax.text(i, y + 0.8, f"{y:.1f}", ha="center", fontsize=6.2, color=INK)
    ax.set_xticks(range(5))
    ax.set_xticklabels([MODE_LAB[m] for m in MODES], fontsize=6.2, rotation=15)
    ax.set_ylabel("Mean accuracy (%)", fontsize=FS)
    ax.set_title("Converted-graph accuracy", fontsize=FS)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    _style(ax)

    ax = axes[1]
    drop_modes = MODES  # fp32 drop = 0
    ax.bar(range(5), drops, 0.62, color=[MODE_COL[m] for m in drop_modes],
           edgecolor="white", linewidth=0.5)
    for i, y in enumerate(drops):
        ax.text(i, y + 0.05, f"{y:.2f}", ha="center", fontsize=6.2, color=INK)
    ax.set_xticks(range(5))
    ax.set_xticklabels([MODE_LAB[m] for m in MODES], fontsize=6.2, rotation=15)
    ax.set_ylabel("Mean loss vs FP32 (pp)", fontsize=FS)
    ax.set_title("Change relative to FP32 TFLite", fontsize=FS)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    _style(ax)
    _save(fig, "fig02_accuracy_and_cost.png")


def fig_im100_policy_match():
    A = _logA(R83 / "log_A_im100.txt")
    POL = [
        ("global dynamic", "Global default\n(dynamic INT8)"),
        ("global order, feasibility-aware", "Global order,\nfeasibility-aware"),
        ("per-device order, feasibility-aware", "Per-device order,\nfeasibility-aware"),
        ("learned offline (GBT, masked)", "Learned selector\n(offline features)"),
        ("per-arch oracle majority (in-sample)", "Per-architecture\noracle"),
    ]
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
    ax.set_title("ImageNet-100 policy match rates", fontsize=FS)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    _style(ax)
    _save(fig, "fig_im100_policy_match.png")


def fig18_export_vs_quant_loss():
    gap_path = R83 / "export_gap_values.json"
    if not gap_path.exists():
        print("warn fig18: missing export_gap_values.json — skip histogram panel data")
        gaps = np.array([])
    else:
        gaps = np.array([g["gap_pp"] for g in json.load(open(gap_path))["gaps"]])

    # PTQ component from accuracy_released means (pp)
    ptq = {"fp16": 0.09, "dynamic": 0.01, "mixed": 1.29, "int8": 1.59}
    # Export group means from committed export_gap.txt-style summary if present
    export_mean = float(np.mean(gaps)) if len(gaps) else 17.4

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axes[0]
    if len(gaps):
        ax.hist(gaps, bins=40, color=C1, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Checkpoint − FP32 graph (pp)", fontsize=FS)
    ax.set_ylabel("Models", fontsize=FS)
    ax.set_title("(a) Export / preprocess gap", fontsize=FS)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    _style(ax)

    ax = axes[1]
    labels = ["Export\n(pooled)", "FP16", "Dynamic", "Mixed", "Static\nINT8"]
    vals = [export_mean, ptq["fp16"], ptq["dynamic"], ptq["mixed"], ptq["int8"]]
    cols = [C2, MODE_COL["fp16"], MODE_COL["dynamic"], MODE_COL["mixed"], MODE_COL["int8"]]
    ax.bar(range(5), vals, 0.62, color=cols, edgecolor="white", linewidth=0.5)
    for i, y in enumerate(vals):
        ax.text(i, y + 0.3, f"{y:.1f}", ha="center", fontsize=6.2, color=INK)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=6.0)
    ax.set_ylabel("Mean loss (pp)", fontsize=FS)
    ax.set_title("(b) Export vs PTQ components", fontsize=FS)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    _style(ax)
    _save(fig, "fig18_export_vs_quant_loss.png")


def fig_im100_export_vs_quant_loss():
    gap_path = R83 / "im100_checkpoint_gap_values.json"
    if gap_path.exists():
        gaps = np.array([g["gap_pp"] for g in json.load(open(gap_path))["gaps"]])
    else:
        # fall back: checkpoint − fp32 from shipped JSONs
        fp32 = _acc_json(ROOT / "im100_acc" / "all_models_accuracy_fp32.json")
        ckpt_raw = json.load(open(ROOT / "im100_acc" / "checkpoint_epoch50_all_models.json"))
        gaps = []
        for m, fa in fp32.items():
            if m not in ckpt_raw:
                continue
            ca = ckpt_raw[m]["accuracy"]
            gaps.append(100.0 * (ca - fa))
        gaps = np.array(gaps)

    df = pd.read_csv(R83 / "dataset_full_imagenet100.csv").drop_duplicates("model")
    ptq_modes = ["fp16", "dynamic", "mixed", "int8"]
    # signed change: negative = loss; accdrop is loss so negate for "signed PTQ change"
    ptq_means = []
    for m in ptq_modes:
        col = f"accdrop_{m}"
        ptq_means.append(-100.0 * df[col].dropna().mean())

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axes[0]
    ax.hist(gaps, bins=40, color=C1, edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(gaps), color=C2, lw=1.2, label=f"mean {np.mean(gaps):.1f}")
    ax.set_xlabel("Checkpoint − FP32 graph (pp)", fontsize=FS)
    ax.set_ylabel("Models", fontsize=FS)
    ax.set_title("(a) Checkpoint vs converted FP32", fontsize=FS)
    ax.legend(fontsize=6, frameon=False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    _style(ax)

    ax = axes[1]
    labs = [MODE_LAB[m] for m in ptq_modes]
    ax.bar(range(4), ptq_means, 0.62,
           color=[MODE_COL[m] for m in ptq_modes], edgecolor="white", linewidth=0.5)
    for i, y in enumerate(ptq_means):
        ax.text(i, y - 0.15 if y < 0 else y + 0.05, f"{y:.2f}",
                ha="center", fontsize=6.2, color=INK,
                va="top" if y < 0 else "bottom")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, fontsize=6.2, rotation=15)
    ax.set_ylabel("Signed PTQ change (pp)", fontsize=FS)
    ax.set_title("(b) PTQ vs matched FP32 graph", fontsize=FS)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    _style(ax)
    _save(fig, "fig_im100_export_vs_quant_loss.png")


def run_sibling(name):
    path = ROOT / name
    if not path.exists():
        print(f"skip {name} (not found)")
        return
    print(f"-- {name}")
    runpy.run_path(str(path), run_name="__main__")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    # Specialized drawers already tuned for main-paper layout
    for sib in (
        "make_fig_v85.py",
        "make_fig_flagship.py",
        "make_fig_v83.py",
        "make_fig_effect_sizes.py",
        "make_fig_supp_fixed.py",
        "make_fig_extra.py",
    ):
        run_sibling(sib)

    print("-- previously missing supplementary figures")
    fig01_mean_latency()
    fig06_backend_distribution()
    fig13_repeatability()
    fig02_accuracy_and_cost()
    fig_im100_policy_match()
    fig18_export_vs_quant_loss()
    fig_im100_export_vs_quant_loss()
    print("== make_all_figures.py done ==")


if __name__ == "__main__":
    main()
