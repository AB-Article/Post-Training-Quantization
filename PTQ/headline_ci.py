#!/usr/bin/env python3
"""95% confidence intervals for the headline numbers of the main paper.

No new measurements: every interval is computed from the committed records
(`results/v83/dataset_full_*.csv`, `results/v83/picks_A*.csv`, and the frozen
flagship subset in `results/flagship_2026/`).

Two interval types, matching the ones already used in the manuscript:

* main campaign  -- architecture-clustered bootstrap (2,000 resamples of
  architectures with replacement, 2.5/97.5 percentiles), because each
  architecture contributes one row per device and the rows are not independent.
  This is the same procedure as `predictor_ci.py` / `robustness.py`.
* flagship spot-check -- Wilson score interval for shares (one row per model,
  so the rows are independent) and a model-level bootstrap for mean regret.

    python headline_ci.py > results/v85/headline_ci.txt   # also writes .json
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import collections
import json
import math

import numpy as np
import pandas as pd

import guideline as G
import v83_analysis as V

REPS = 2000
RNG = np.random.default_rng(0)
CORPORA = ("cifar10", "imagenet100")
MODES = V.MODES

# rules of thumb exactly as evaluated in rules_eval.py (main-paper Table IV)
RULES = {
    "cifar10": dict(size_col="size_dynamic_kib", thr=1180, int8_mode="mixed", loss=0.004),
    "imagenet100": dict(size_col="size_fp16_kib", thr=4095, int8_mode="int8", loss=0.003),
}


# --------------------------------------------------------------------------
# interval machinery
# --------------------------------------------------------------------------
def cluster_idx(models):
    """Index blocks, one per architecture, for the clustered bootstrap."""
    by = collections.defaultdict(list)
    for i, m in enumerate(models):
        by[m].append(i)
    return [np.asarray(v) for v in by.values()]


def cluster_ci(values, models, stat=np.mean, reps=REPS):
    """Architecture-clustered bootstrap percentile interval."""
    v = np.asarray(values, dtype=float)
    blocks = cluster_idx(models)
    n = len(blocks)
    draws = np.empty(reps)
    for b in range(reps):
        pick = RNG.integers(0, n, n)
        draws[b] = stat(v[np.concatenate([blocks[j] for j in pick])])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial share (percent)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def boot_ci(values, stat=np.mean, reps=REPS):
    """Ordinary (unclustered) bootstrap percentile interval."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    draws = np.array([stat(v[RNG.integers(0, n, n)]) for _ in range(reps)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def pct(x):
    return 100 * float(x)


# --------------------------------------------------------------------------
# main campaign
# --------------------------------------------------------------------------
def rules_pick(d, cfg):
    """Reproduces rules_eval.py's policy row by row."""
    out = []
    for r in d.itertuples():
        sz = getattr(r, cfg["size_col"])
        small = bool(pd.notna(sz) and sz <= cfg["thr"])
        if r.device != "24040RN64Y":
            want = "fp16" if small else "dynamic"
        else:
            low_loss = pd.notna(r.accdrop_int8) and r.accdrop_int8 <= cfg["loss"]
            want = cfg["int8_mode"] if (small and low_loss) else "dynamic"
        order = [want, "dynamic", "fp16", "fp32"]
        out.append(next(m for m in order if m in r.feas))
    return np.array(out, dtype=object)


def main_campaign():
    res = {}
    for c in CORPORA:
        d = G.load(c)
        models = d.model.values
        r = {"n_pairs": int(len(d)), "n_architectures": int(d.model.nunique())}

        # (1) share of pairs for which each mode is the fastest feasible one
        r["winner_share"] = {}
        for m in MODES:
            hit = (d.label.values == m).astype(float)
            lo, hi = cluster_ci(hit, models)
            r["winner_share"][m] = dict(pct=pct(hit.mean()), lo=100 * lo, hi=100 * hi)

        # (2) rules of thumb
        pick = rules_pick(d, RULES[c])
        hit = (pick == d.label.values).astype(float)
        lo, hi = cluster_ci(hit, models)
        r["rules_exact"] = dict(pct=pct(hit.mean()), lo=100 * lo, hi=100 * hi)

        # (3) learned selector, from the stored out-of-fold picks.  The
        # CIFAR-10 interval reproduces predictor_ci.py's [66.5, 70.0] to
        # within Monte Carlo error; the manuscript quotes that earlier run.
        tag = "" if c == "cifar10" else "_im100"
        p = pd.read_csv(f"results/v83/picks_A{tag}.csv")
        col = "learned +dev (GBT, masked)|hit"
        hit = p[col].to_numpy(float)
        lo, hi = cluster_ci(hit, p.model.values)
        r["selector_exact"] = dict(pct=pct(hit.mean()), lo=100 * lo, hi=100 * hi)

        res[c] = r
    return res


# --------------------------------------------------------------------------
# 2025-2026 flagship spot-check
# --------------------------------------------------------------------------
def flagship():
    import flagship_corrected as F
    from pathlib import Path

    fdir = F.HERE / "results" / "flagship_2026"
    lat, _meta = F.load_winners(fdir / "winners_by_device.csv")
    acc = F.load_accuracies(F.HERE / "im100_acc")
    tau = 0.01
    out = {}

    # INT8 share on the S25, unconstrained and under the 1% threshold
    for key, kw in (("pure", {}), ("constrained", dict(acc=acc, tau=tau))):
        w = F.winners(lat, "S25", **kw)
        n = len(w)
        k = sum(1 for m in w.values() if m == "int8")
        lo, hi = wilson(k, n)
        out[f"s25_int8_{key}"] = dict(pct=100 * k / n, lo=lo, hi=hi, k=k, n=n)

    # constrained winner agreement, S25 vs S26
    a = F.agreement(lat, "S25", "S26", acc=acc, tau=tau)
    lo, hi = wilson(a["agree"], a["n"])
    out["s25_s26_agreement"] = dict(pct=a["agree_pct"], lo=lo, hi=hi,
                                    k=a["agree"], n=a["n"])

    # mean regret of shipping one device's constrained winner on the other
    for ship, target in (("S25", "S26"), ("S26", "S25")):
        w = F.winners(lat, ship, acc=acc, tau=tau)
        rel = []
        for mdl, mode in w.items():
            if mdl not in lat[target]:
                continue
            L2 = lat[target][mdl]
            com = F.candidates(mdl, L2, acc, tau)
            if mode not in com or len(com) < 2:
                continue
            best = min(L2[x] for x in com)
            rel.append(100 * (L2[mode] - best) / best)
        lo, hi = boot_ci(rel)
        out[f"regret_{ship}_to_{target}"] = dict(mean_pct=float(np.mean(rel)),
                                                 lo=lo, hi=hi, n=len(rel))
    return out


def main():
    res = {"main_campaign": main_campaign(), "flagship": flagship()}

    print("Architecture-clustered bootstrap, 2,000 resamples, 95% percentile "
          "intervals (main campaign)")
    for c, r in res["main_campaign"].items():
        print(f"\n{c}  (n={r['n_pairs']} model-device pairs, "
              f"{r['n_architectures']} architectures)")
        for m in MODES:
            s = r["winner_share"][m]
            print(f"  fastest feasible = {m:8s} {s['pct']:5.1f}% "
                  f"[{s['lo']:.1f}, {s['hi']:.1f}]")
        for k in ("rules_exact", "selector_exact"):
            s = r[k]
            print(f"  {k:16s}        {s['pct']:5.1f}% "
                  f"[{s['lo']:.1f}, {s['hi']:.1f}]")

    print("\nFlagship spot-check (n=150 frozen ImageNet-100 models; Wilson "
          "intervals for shares, model bootstrap for mean regret)")
    for k, s in res["flagship"].items():
        v = s.get("pct", s.get("mean_pct"))
        print(f"  {k:24s} {v:5.2f}% [{s['lo']:.2f}, {s['hi']:.2f}]  n={s['n']}")

    with open("results/v85/headline_ci.json", "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
