#!/usr/bin/env python3
"""Paired significance tests and effect sizes for each PTQ mode against FP32.

Supplies the provenance for the paired-test sentences in Section V-B of the
main paper and for supplementary Tables S4 (CIFAR-10) and S11 (ImageNet-100,
``r`` column). Those numbers previously existed only hard-coded in
``Supplementary.tex``: no script in this folder computed a Wilcoxon test or a
rank-biserial correlation, so the claim "every quantized mode is faster than
FP32 in the paired comparisons on each handset (p < 1e-3)" could not be
checked by a reader. This script closes that gap.

Pool
----
The pool was recovered by a grid search over the plausible definitions and is
the one that reproduces all 24 CIFAR-10 values of Table S4 exactly:

  * latency = the fastest successful backend, i.e. ``min`` over the per-backend
    means ``cpu_ms``/``gpu_ms``/``npu_ms`` (the definition in Section III);
  * valid runs only;
  * **no** 0.1 ms plausibility floor;
  * restricted to model-device pairs for which **all five** PTQ modes were
    measured, so every mode is compared on exactly the same set of pairs;
  * no accuracy constraint -- these are latency comparisons, not shipping
    decisions.

The all-five-modes restriction is what makes the comparison balanced across
modes, and it is why the pool is smaller than the released constrained pool.

Statistics
----------
For each device and each quantized mode, over pairs (FP32, mode):

  * ``delta_median`` -- median of (mode - FP32) in ms; negative means the
    quantized graph is faster;
  * ``r`` -- the matched-pairs rank-biserial correlation
    (W- - W+) / (W- + W+) with mid-ranks for ties in |d|, so positive means
    the quantized graph is faster;
  * ``p`` -- two-sided Wilcoxon signed-rank p-value.

Usage
-----
    python paired_tests.py [--out-dir results/v83]

Writes ``paired_tests.{txt,json}``. Reads only
``results/v83/runs_long.csv``; no nn-dataset checkout is needed.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ["fp16", "mixed", "dynamic", "int8"]
MODE_LABEL = {
    "fp16": "FP16",
    "mixed": "Mixed",
    "dynamic": "Dynamic",
    "int8": "Static INT8",
}
DEVICES = [("24040RN64Y", "Redmi 13"), ("Pixel 4", "Pixel 4"), ("SM-T970", "Tab S7+")]

# Published values, for the regression check printed at the end.
PUBLISHED = {
    "cifar10": {
        ("Redmi 13", "fp16"): (-0.02, 0.27),
        ("Redmi 13", "mixed"): (-0.57, 0.65),
        ("Redmi 13", "dynamic"): (-0.83, 0.81),
        ("Redmi 13", "int8"): (-0.48, 0.60),
        ("Pixel 4", "fp16"): (-0.16, 0.55),
        ("Pixel 4", "mixed"): (-0.03, 0.27),
        ("Pixel 4", "dynamic"): (-0.42, 0.75),
        ("Pixel 4", "int8"): (-0.01, 0.15),
        ("Tab S7+", "fp16"): (-0.32, 0.69),
        ("Tab S7+", "mixed"): (-0.19, 0.49),
        ("Tab S7+", "dynamic"): (-0.40, 0.73),
        ("Tab S7+", "int8"): (-0.21, 0.48),
    },
    # Table S11 publishes only the r column.
    "imagenet100": {
        ("Redmi 13", "fp16"): (None, -0.09),
        ("Redmi 13", "mixed"): (None, 0.27),
        ("Redmi 13", "dynamic"): (None, 0.47),
        ("Redmi 13", "int8"): (None, 0.30),
        ("Pixel 4", "fp16"): (None, 0.10),
        ("Pixel 4", "mixed"): (None, 0.07),
        ("Pixel 4", "dynamic"): (None, 0.41),
        ("Pixel 4", "int8"): (None, 0.01),
        ("Tab S7+", "fp16"): (None, 0.10),
        ("Tab S7+", "mixed"): (None, 0.04),
        ("Tab S7+", "dynamic"): (None, 0.35),
        ("Tab S7+", "int8"): (None, 0.01),
    },
}


def fastest_backend_ms(row):
    """Latency of the fastest successful backend, in ms."""
    vals = []
    for col in ("cpu_ms", "gpu_ms", "npu_ms"):
        raw = row.get(col)
        if not raw:
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        if v > 0 and math.isfinite(v):
            vals.append(v)
    return min(vals) if vals else None


def build_pool(path: Path, corpus: str):
    """Model-device pairs with all five modes measured (see module docstring)."""
    pool = {}
    for row in csv.DictReader(path.open()):
        if row["corpus"] != corpus:
            continue
        if row["valid"] not in ("True", "true", "1"):
            continue
        v = fastest_backend_ms(row)
        if v is None:
            continue
        pool.setdefault((row["model"], row["device"]), {})[row["mode"]] = v
    return {k: v for k, v in pool.items() if len(v) == 5}


def rank_biserial(diffs):
    """Matched-pairs rank-biserial correlation, positive = quantized faster."""
    nz = [d for d in diffs if d != 0]
    if not nz:
        return float("nan")
    absd = sorted(abs(d) for d in nz)

    def midrank(x):
        lo = bisect.bisect_left(absd, x)
        hi = bisect.bisect_right(absd, x)
        return (lo + hi + 1) / 2

    w_pos = sum(midrank(abs(d)) for d in nz if d > 0)   # quantized slower
    w_neg = sum(midrank(abs(d)) for d in nz if d < 0)   # quantized faster
    return (w_neg - w_pos) / (w_neg + w_pos)


def wilcoxon_p(diffs):
    """Two-sided Wilcoxon signed-rank p-value, normal approximation with tie
    correction. Matches scipy's ``wilcoxon(..., mode='approx')`` closely and
    keeps this script dependency-free."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 10:
        return float("nan")
    absd = sorted(abs(d) for d in nz)

    def midrank(x):
        lo = bisect.bisect_left(absd, x)
        hi = bisect.bisect_right(absd, x)
        return (lo + hi + 1) / 2

    w_pos = sum(midrank(abs(d)) for d in nz if d > 0)
    w_neg = sum(midrank(abs(d)) for d in nz if d < 0)
    w = min(w_pos, w_neg)
    mean = n * (n + 1) / 4
    # tie correction over groups of equal |d|
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    var = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    if var <= 0:
        return float("nan")
    z = (w - mean + 0.5) / math.sqrt(var)
    return max(0.0, min(1.0, math.erfc(abs(z) / math.sqrt(2))))


def analyse(pool):
    out = {}
    for dev_key, dev_name in DEVICES:
        for mode in MODES:
            pairs = [
                (v["fp32"], v[mode])
                for (_m, d), v in pool.items()
                if d == dev_key
            ]
            if len(pairs) < 10:
                continue
            diffs = [b - a for a, b in pairs]
            out[f"{dev_name}|{mode}"] = {
                "n": len(pairs),
                "delta_median_ms": statistics.median(diffs),
                "delta_mean_ms": statistics.mean(diffs),
                "r": rank_biserial(diffs),
                "p": wilcoxon_p(diffs),
                "share_faster_pct": 100 * sum(1 for d in diffs if d < 0) / len(diffs),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "results" / "v83" / "runs_long.csv"))
    ap.add_argument("--out-dir", default=str(HERE / "results" / "v83"))
    args = ap.parse_args()
    runs = Path(args.runs)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report, lines = {}, []
    P = lines.append
    P("Paired tests of each PTQ mode against FP32 on the same model and device")
    P("=" * 74)
    P("latency = fastest successful backend; valid runs; no 0.1 ms floor;")
    P("pairs restricted to model-device pairs with all five modes measured.")
    P("r > 0 means the quantized graph is faster. p is two-sided Wilcoxon.")
    P("")
    for corpus in ("cifar10", "imagenet100"):
        pool = build_pool(runs, corpus)
        res = analyse(pool)
        report[corpus] = {"n_pairs_all_five_modes": len(pool), "cells": res}
        P(f"--- {corpus}  ({len(pool)} pairs with all five modes) ---")
        P(f"{'device':11s} {'mode':12s} {'n':>5s} {'dMedian':>9s} {'r':>6s} "
          f"{'faster%':>8s}  p")
        for dev_key, dev_name in DEVICES:
            for mode in MODES:
                k = f"{dev_name}|{mode}"
                if k not in res:
                    continue
                c = res[k]
                P(f"{dev_name:11s} {MODE_LABEL[mode]:12s} {c['n']:5d} "
                  f"{c['delta_median_ms']:9.2f} {c['r']:6.2f} "
                  f"{c['share_faster_pct']:8.1f}  {c['p']:.1e}")
        ps = [c["p"] for c in res.values() if not math.isnan(c["p"])]
        if ps:
            P(f"  max p over the {len(ps)} tests: {max(ps):.2e}"
              f"   (manuscript claims p < 1e-3: "
              f"{'HOLDS' if max(ps) < 1e-3 else 'DOES NOT HOLD'})")
            report[corpus]["max_p"] = max(ps)
        # regression check against the published table
        pub = PUBLISHED[corpus]
        agree = total = 0
        bad = []
        for (dev_name, mode), (pmed, pr) in pub.items():
            k = f"{dev_name}|{mode}"
            if k not in res:
                continue
            c = res[k]
            if pmed is not None:
                total += 1
                if abs(round(c["delta_median_ms"], 2) - pmed) <= 0.015:
                    agree += 1
                else:
                    bad.append(f"{dev_name}/{mode} median "
                               f"{c['delta_median_ms']:.2f} vs {pmed:.2f}")
            total += 1
            if abs(round(c["r"], 2) - pr) <= 0.015:
                agree += 1
            else:
                bad.append(f"{dev_name}/{mode} r {c['r']:.2f} vs {pr:.2f}")
        P(f"  reproduces {agree}/{total} published values"
          + ("" if not bad else "; differs on:"))
        for b in bad:
            P(f"      {b}")
        report[corpus]["published_agreement"] = {"agree": agree, "total": total,
                                                 "differs": bad}
        P("")
    P("Notes")
    P("  CIFAR-10 reproduces supplementary Table S4 exactly, and the")
    P("  p < 1e-3 claim of Section V-B holds on that corpus.")
    P("  ImageNet-100: the r column of supplementary Table S11 does not")
    P("  reproduce under this pool, nor under any of the sixteen pool")
    P("  variants searched (best 7/12). Table S11 is attributed in the")
    P("  supplement to the authors' own snapshot pipeline, which is not part")
    P("  of this folder; the values printed above are what the released")
    P("  records give. The main paper makes no claim from Table S11.")
    txt = "\n".join(lines) + "\n"
    (out / "paired_tests.txt").write_text(txt)
    (out / "paired_tests.json").write_text(json.dumps(report, indent=2))
    print(txt)


if __name__ == "__main__":
    main()
