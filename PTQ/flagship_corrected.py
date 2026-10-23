#!/usr/bin/env python3
"""Corrected analysis of the 2025-2026 flagship spot-check.

Re-derives every flagship number reported in the main paper and the
supplement from the CSVs under ``results/flagship_2026/``, fixing four
problems in the first-pass analysis:

1.  *Estimand.*  The first pass reported pure-latency winners.  The paper's
    estimand is the fastest mode within ``--tau`` of the matched converted
    FP32 accuracy.  Per-mode ImageNet-100 accuracies for all 150 freeze
    models are available in ``im100_acc/``, so the constrained winner is
    computed here and reported as primary.

2.  *Chance-corrected agreement.*  Raw winner agreement is inflated when one
    mode dominates (INT8 takes 91% of pure-latency wins on the AI Hub
    devices, so chance agreement alone is ~86-92%).  Cohen's kappa and the
    chance-agreement rate are reported alongside the raw percentage.

3.  *Regret and tie tolerance.*  The paper treats regret as primary and
    winner rates as fragile.  Relative regret and tie-tolerant agreement at
    tau in {2,5,10}% are computed for the flagship devices on the same
    footing as the main corpus.

4.  *Measurement-stack confound.*  ``raw/harness_20_paper_vs_aihub.csv``
    benchmarks the *same* physical devices (Pixel 4, Tab S7+) through both
    the paper's USB harness and Qualcomm AI Hub.  That control bounds how
    much of any cross-stack winner disagreement is attributable to the stack
    rather than to the hardware, and is required before the Snapdragon
    (AI Hub) vs MediaTek (benchmark_model) comparison can be interpreted.

Also emits the MediaTek delegate-coverage table, which supplies the
mechanism for the cross-vendor divergence, and a full accounting of
MediaTek run outcomes.

Usage
-----
    python flagship_corrected.py [--tau 0.01] [--out-dir results/flagship_2026]

Writes ``flagship_corrected.{txt,json}``, ``flagship_corrected_tables.tex``
(main-paper tables), ``flagship_corrected_supp.tex`` (supplement tables) and
``winners_constrained.csv`` into the output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

MODES = ["fp32", "fp16", "dynamic", "int8", "mixed"]
# Minimum plausible single-inference duration for the USB harness; matches the
# main campaign's exclusion rule.  Not applied to AI Hub fused latencies.
MIN_USB_MS = 0.1
MODE_TEX = {
    "fp32": "FP32",
    "fp16": "FP16",
    "dynamic": "dyn.",
    "int8": "INT8",
    "mixed": "mixed",
}
DEV_TEX = {
    "S25": "S25 (8 Elite)",
    "S26": "S26 (8 Elite Gen5)",
    "MTK_D9400": "Tab~S11 (D9400)",
    "USB_Pixel4": "USB Pixel~4",
    "USB_TabS7": "USB Tab~S7+",
    "USB_Redmi13": "USB Redmi~13",
}
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _f(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "-", ""}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) and v > 0 else None


def load_winners(path: Path):
    """device_tag -> model_name -> {mode: ms, ...} plus metadata."""
    lat = defaultdict(dict)
    meta = {}
    for r in csv.DictReader(path.open()):
        d = {m: _f(r.get("ms_" + m)) for m in MODES}
        lat[r["device_tag"]][r["model_name"]] = {
            m: v for m, v in d.items() if v is not None
        }
        meta.setdefault(
            r["model_name"],
            {
                "family": r.get("model_family", ""),
                "bucket": r.get("latency_bucket", ""),
                "in_mtk40": r.get("in_mtk40", "0") == "1",
            },
        )
    return dict(lat), meta


def load_accuracies(acc_dir: Path):
    acc = {}
    for m in MODES:
        p = acc_dir / f"all_models_accuracy_{m}.json"
        acc[m] = json.load(p.open()) if p.exists() else {}
    return acc


def feasible_set(model: str, acc, tau: float):
    """Modes within ``tau`` (absolute, fraction) of the converted FP32 graph."""
    if model not in acc["fp32"]:
        return None
    a32 = acc["fp32"][model]["accuracy"]
    out = set()
    for m in MODES:
        rec = acc[m].get(model)
        if rec is not None and rec["accuracy"] >= a32 - tau:
            out.add(m)
    return out


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def _argmin(L, cand):
    """Deterministic argmin over ``cand``.

    Candidates are always visited in ``MODES`` order, so exact latency ties
    resolve to the *least* quantized mode (FP32 > FP16 > dynamic > INT8 >
    mixed).  This is the conservative choice -- a more aggressive mode is
    only ever selected when it is strictly faster -- and it keeps the result
    independent of set-iteration order, which varies between Python
    processes.  Exact ties are frequent on the AI Hub devices, where FP32 and
    FP16 report identical latency for several models, so the rule is load
    bearing and is reported in ``tie_counts``.
    """
    order = [m for m in MODES if m in cand]
    if not order:
        return None
    best = order[0]
    for m in order[1:]:
        if L[m] < L[best]:
            best = m
    return best


def candidates(mdl, L, acc=None, tau=None):
    cand = set(L)
    if acc is not None:
        f = feasible_set(mdl, acc, tau)
        if not f:
            return set()
        cand &= f
    return cand


def winners(lat, dev, acc=None, tau=None):
    """Winner mode per model on ``dev``; constrained when acc/tau are given."""
    out = {}
    for mdl, L in lat[dev].items():
        cand = candidates(mdl, L, acc, tau)
        w = _argmin(L, cand) if cand else None
        if w is not None:
            out[mdl] = w
    return out


def tie_counts(lat, dev, acc=None, tau=None):
    """Models whose selected latency is matched exactly by another mode."""
    n = ties = 0
    pat = Counter()
    for mdl, L in lat[dev].items():
        cand = candidates(mdl, L, acc, tau)
        if len(cand) < 2:
            continue
        n += 1
        mn = min(L[m] for m in cand)
        k = tuple(m for m in MODES if m in cand and L[m] == mn)
        if len(k) > 1:
            ties += 1
            pat[k] += 1
    return {
        "n": n,
        "ties": ties,
        "tie_pct": 100 * ties / n if n else float("nan"),
        "patterns": {"+".join(k): v for k, v in pat.most_common()},
    }


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    ag = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] / n * cb[k] / n for k in sorted(set(a) | set(b)))
    k = (ag - pe) / (1 - pe) if pe < 1 else float("nan")
    return ag, pe, k


def agreement(lat, d1, d2, acc=None, tau=None):
    w1 = winners(lat, d1, acc, tau)
    w2 = winners(lat, d2, acc, tau)
    com = sorted(set(w1) & set(w2))
    a = [w1[m] for m in com]
    b = [w2[m] for m in com]
    ag, pe, k = kappa(a, b)
    return {
        "n": len(com),
        "agree": sum(1 for x, y in zip(a, b) if x == y),
        "agree_pct": 100 * ag if com else float("nan"),
        "chance_pct": 100 * pe if com else float("nan"),
        "kappa": k,
    }


def triple_agreement(lat, devs, acc=None, tau=None):
    ws = [winners(lat, d, acc, tau) for d in devs]
    com = sorted(set.intersection(*[set(w) for w in ws]))
    k = sum(1 for m in com if len({w[m] for w in ws}) == 1)
    return {
        "n": len(com),
        "agree": k,
        "agree_pct": 100 * k / len(com) if com else float("nan"),
    }


def tol_agreement(lat, d1, d2, tol, acc=None, tau=None):
    """Within-``tol`` winner *sets* intersect, restricted to modes measured on
    both devices (and feasible, when constrained)."""
    n = k = 0
    for mdl in sorted(set(lat[d1]) & set(lat[d2])):
        L1, L2 = lat[d1][mdl], lat[d2][mdl]
        com = set(L1) & set(L2)
        if acc is not None:
            f = feasible_set(mdl, acc, tau)
            if not f:
                continue
            com &= f
        if len(com) < 2:
            continue
        com = {m for m in MODES if m in com}
        m1 = min(L1[x] for x in com)
        m2 = min(L2[x] for x in com)
        s1 = {x for x in com if L1[x] <= m1 * (1 + tol)}
        s2 = {x for x in com if L2[x] <= m2 * (1 + tol)}
        n += 1
        k += 1 if (s1 & s2) else 0
    return {"n": n, "agree": k, "agree_pct": 100 * k / n if n else float("nan")}


def regret(lat, ship, target, acc=None, tau=None):
    """Relative latency penalty (%) of shipping ``ship``'s winner on
    ``target``, and the absolute penalty in ms."""
    w = winners(lat, ship, acc, tau)
    rel, absol = [], []
    for mdl, mode in w.items():
        if mdl not in lat[target]:
            continue
        L2 = lat[target][mdl]
        com = candidates(mdl, L2, acc, tau)
        if mode not in com or len(com) < 2:
            continue
        best = min(L2[x] for x in com)
        rel.append(100 * (L2[mode] - best) / best)
        absol.append(L2[mode] - best)
    if not rel:
        return None
    return {
        "n": len(rel),
        "median_pct": statistics.median(rel),
        "mean_pct": statistics.mean(rel),
        "p90_pct": sorted(rel)[int(0.9 * (len(rel) - 1))],
        "median_ms": statistics.median(absol),
        "mean_ms": statistics.mean(absol),
    }


# --------------------------------------------------------------------------
# MediaTek delegate coverage + run accounting
# --------------------------------------------------------------------------
def mtk_coverage(path: Path):
    rows = list(csv.DictReader(path.open()))
    status = Counter(r["status"] for r in rows)
    cell = Counter((r["quant_mode"], r["backend"], r["status"]) for r in rows)
    models = {r["model_name"] for r in rows}
    ok_models = defaultdict(set)
    ok_models_backend = defaultdict(set)
    for r in rows:
        if r["status"] == "ok":
            ok_models[r["quant_mode"]].add(r["model_name"])
            ok_models_backend[(r["quant_mode"], r["backend"])].add(r["model_name"])
    complete = set.intersection(*[ok_models[m] for m in MODES]) if models else set()
    return {
        "rows": len(rows),
        "n_models": len(models),
        "status": dict(status),
        "per_mode_backend_ok": {
            f"{m}|{b}": len(ok_models_backend[(m, b)])
            for m in MODES
            for b in ("cpu", "gpu", "npu")
        },
        "per_mode_any_backend_ok": {m: len(ok_models[m]) for m in MODES},
        "models_all_five_modes": len(complete),
        # The first-pass harness captured a benchmark_model usage error in
        # place of a version string; the field is published as "unrecorded".
        "runtime_version_recorded": not any(
            "ERROR" in (r.get("runtime_version") or "")
            or (r.get("runtime_version") or "").strip()
            in ("", "unrecorded")
            for r in rows
        ),
    }


# --------------------------------------------------------------------------
# same-device cross-stack control
# --------------------------------------------------------------------------
def spearman(d1, d2):
    com = sorted(set(d1) & set(d2))
    k = len(com)
    if k < 3:
        return None
    r1 = {m: i for i, m in enumerate(sorted(com, key=lambda x: d1[x]))}
    r2 = {m: i for i, m in enumerate(sorted(com, key=lambda x: d2[x]))}
    dsum = sum((r1[m] - r2[m]) ** 2 for m in com)
    return 1 - 6 * dsum / (k * (k * k - 1))


def stack_control(path: Path):
    """Same-device (or closest-sibling) control across the two measurement
    stacks.

    Only the Pixel 4 rows are a strict same-device control: the AI Hub side of
    the tablet rows is a Galaxy Tab S7 (SD 865) standing in for the paper's
    Tab S7+ (SD 865+), which the source CSV records in its ``notes`` column.
    The Pixel 4 figure is therefore the one to quote as the stack bound, and
    ``same_device`` marks which rows qualify.
    """
    rows = list(csv.DictReader(path.open()))
    out = {}
    for dev in sorted({r["device_paper_usb"] for r in rows}):
        sub = [r for r in rows if r["device_paper_usb"] == dev]
        aihub = sorted({r["device_aihub"] for r in sub})
        scored = [r for r in sub if r["match_aihub_vs_paper_tau099"] in ("Y", "N")]
        y = sum(1 for r in scored if r["match_aihub_vs_paper_tau099"] == "Y")
        ratios, rhos, lp = [], [], []
        for r in sub:
            a = _f(r.get("aihub_winner_latency_ms"))
            p = _f(r.get("paper_usb_tau099_latency_ms"))
            if a and p:
                ratios.append(p / a)
            da = {m: _f(r.get(f"aihub_{m}_us")) for m in MODES}
            dp = {m: _f(r.get(f"paper_usb_{m}_ms")) for m in MODES}
            da = {m: v / 1000.0 for m, v in da.items() if v}
            dp = {m: v for m, v in dp.items() if v}
            rho = spearman(da, dp)
            if rho is not None:
                rhos.append(rho)
            for m in sorted(set(da) & set(dp)):
                lp.append((math.log(da[m]), math.log(dp[m])))
        out[dev] = {
            "n_scored": len(scored),
            "match": y,
            "match_pct": 100 * y / len(scored) if scored else float("nan"),
            "usb_over_aihub_ms_ratio_median": statistics.median(ratios)
            if ratios
            else None,
            "mode_rank_spearman_median": statistics.median(rhos) if rhos else None,
            "mode_rank_spearman_mean": statistics.mean(rhos) if rhos else None,
            "n_models": len(sub),
            "aihub_device": "/".join(aihub),
            "same_device": all(
                "stand" not in (r.get("notes") or "")
                and "AI Hub device=" not in (r.get("notes") or "")
                for r in sub
            ),
        }
    # pooled
    scored = [r for r in rows if r["match_aihub_vs_paper_tau099"] in ("Y", "N")]
    y = sum(1 for r in scored if r["match_aihub_vs_paper_tau099"] == "Y")
    allr, allrho, lp = [], [], []
    for r in rows:
        a = _f(r.get("aihub_winner_latency_ms"))
        p = _f(r.get("paper_usb_tau099_latency_ms"))
        if a and p:
            allr.append(p / a)
        da = {m: _f(r.get(f"aihub_{m}_us")) for m in MODES}
        dp = {m: _f(r.get(f"paper_usb_{m}_ms")) for m in MODES}
        da = {m: v / 1000.0 for m, v in da.items() if v}
        dp = {m: v for m, v in dp.items() if v}
        rho = spearman(da, dp)
        if rho is not None:
            allrho.append(rho)
        for m in sorted(set(da) & set(dp)):
            lp.append((math.log(da[m]), math.log(dp[m])))
    n = len(lp)
    mx = sum(x for x, _ in lp) / n
    my = sum(y2 for _, y2 in lp) / n
    num = sum((x - mx) * (y2 - my) for x, y2 in lp)
    den = math.sqrt(
        sum((x - mx) ** 2 for x, _ in lp) * sum((y2 - my) ** 2 for _, y2 in lp)
    )
    out["pooled"] = {
        "n_scored": len(scored),
        "match": y,
        "match_pct": 100 * y / len(scored) if scored else float("nan"),
        "usb_over_aihub_ms_ratio_median": statistics.median(allr),
        "mode_rank_spearman_median": statistics.median(allrho),
        "mode_rank_spearman_mean": statistics.mean(allrho),
        "log_latency_pearson_r": num / den,
        "n_mode_pairs": n,
    }
    return out


# --------------------------------------------------------------------------
# LaTeX emitters
# --------------------------------------------------------------------------
def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    return f"{x:.{nd}f}"


def tex_main(res, tau):
    r"""No flagship table is reproduced in the main paper any more.

    The stack-matched numbers (S25 vs S26 agreement, kappa and regret) are
    stated in the running text of the flagship subsection, and the full
    four-comparison table -- including the two cross-stack rows, which are
    confounded by the measurement stack -- is supplementary Table S33, emitted
    by ``tex_supp``.  This file is kept so that the main paper's build does not
    break if an older revision still \input's it.
    """
    tp = int(round(tau * 100))
    a = res["agree_constrained"]["S25|S26"]
    r = res["regret_constrained"]["S25|S26"]
    return (
        "% Generated by analysis/flagship_corrected.py -- do not edit by hand.\n"
        "% Intentionally empty: the flagship agreement table moved to the\n"
        "% supplement (Table S33).  For reference, the stack-matched S25-vs-S26\n"
        f"% figures quoted in the main text at tau = {tp} percent are: "
        f"agreement {fmt(a['agree_pct'])}, chance {fmt(a['chance_pct'])}, "
        f"kappa {fmt(a['kappa'],3)},\n"
        f"% median regret {fmt(r['median_pct'],1)} "
        f"(mean {fmt(r['mean_pct'],1)}) percent.\n"
    )


def tex_supp(res, tau):
    """Emits, in order: S33 tie tolerance, S34 regret, S35 same-device stack
    control, S36 MediaTek delegate coverage.  The main paper cites these by
    number, so the order must not change without updating Paper.tex."""
    tp = int(round(tau * 100))
    L = ["% Generated by analysis/flagship_corrected.py -- do not edit by hand.",
         "% Table order is cited as S33-S37 from Paper.tex; keep it stable.\n"]
    # ---- S33: tie tolerance
    L.append(r"\begin{table}[!t]")
    L.append(r"\centering")
    L.append(
        rf"\caption{{Agreement on the ${tp}\%$-constrained winner, including "
        rf"the cross-stack comparisons that the main paper omits. $\kappa$ is "
        rf"Cohen's chance-corrected agreement, which de-inflates raw "
        rf"agreement where one mode dominates. The tie-tolerant columns count "
        rf"two devices as agreeing when their within-tolerance winner sets "
        rf"intersect; tolerance makes the within-family (S25/S26) cut exact "
        rf"but does not rescue any cross-stack comparison. Rows marked "
        rf"$\dagger$ cross the two measurement stacks and are therefore "
        rf"\emph{{descriptive only}}: a same-device stack control "
        rf"(Table~\ref{{tab:supp-stack-control}}) shows that changing the "
        rf"stack alone costs as much agreement as these rows report, so they "
        rf"must not be read as a vendor effect. All columns, "
        rf"including the exact one, are restricted to the modes measured and "
        rf"feasible on \emph{{both}} devices, so the exact column need not "
        rf"equal the per-device agreement reported in the main paper.}}"
    )
    L.append(r"\label{tab:supp-flagship-tol}")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{3pt}")
    L.append(r"\begin{tabular}{@{}lrrrrrr@{}}")
    L.append(r"\toprule")
    L.append(
        r"Comparison & $N$ & $\kappa$ & exact & "
        r"$\tau_{\mathrm{tie}}{=}2\%$ & $5\%$ & $10\%$ \\"
    )
    L.append(r"\midrule")
    for key, label, dag in res["main_agree_rows"]:
        row = res["tol_constrained"][key]
        a = res["agree_constrained"][key]
        nm = label + (r"$^\dagger$" if dag else "")
        L.append(
            f"{nm} & {row['0.02']['n']} & "
            f"{fmt(a['kappa'],3)} & {fmt(row['0.00']['agree_pct'])} & "
            f"{fmt(row['0.02']['agree_pct'])} & "
            f"{fmt(row['0.05']['agree_pct'])} & "
            f"{fmt(row['0.10']['agree_pct'])} \\\\"
        )
    t = res["triple_constrained"]
    L.append(r"\midrule")
    L.append(
        rf"S25+S26+Tab~S11 agree$^\dagger$ & {t['n']} & --- & "
        rf"{fmt(t['agree_pct'])} & --- & --- & --- \\"
    )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L.append("")
    # ---- S34: regret detail
    L.append(r"\begin{table}[!t]")
    L.append(r"\centering")
    L.append(
        rf"\caption{{Regret of shipping one device's ${tp}\%$-constrained "
        rf"winner on another, in relative (\%) and absolute (ms) terms. "
        rf"Within the Snapdragon family the penalty is negligible; across the "
        rf"Snapdragon--MediaTek cut it is large but, being cross-stack, is not "
        rf"separable from the measurement stack (Table~\ref"
        rf"{{tab:supp-stack-control}}).}}"
    )
    L.append(r"\label{tab:supp-flagship-regret}")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\begin{tabular}{@{}lrrrrr@{}}")
    L.append(r"\toprule")
    L.append(
        r"Ship $\rightarrow$ target & $N$ & Med.\ \% & Mean \% & "
        r"P90 \% & Med.\ ms \\"
    )
    L.append(r"\midrule")
    for key, label in res["regret_rows"]:
        r = res["regret_constrained"].get(key)
        if not r:
            continue
        L.append(
            f"{label} & {r['n']} & {fmt(r['median_pct'])} & "
            f"{fmt(r['mean_pct'])} & {fmt(r['p90_pct'])} & "
            f"{fmt(r['median_ms'],3)} \\\\"
        )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L.append("")
    # ---- S35: same-device cross-stack control
    sc = res["stack_control"]
    L.append(r"\begin{table}[!t]")
    L.append(r"\centering")
    L.append(
        r"\caption{Measurement-stack control. The paper's USB harness and "
        r"Qualcomm AI~Hub benchmark the same hardware on a 20-model subset "
        r"each ($18$ scorable). The Pixel~4 row is a strict same-device "
        r"control; on the tablet row a Galaxy Tab~S7 (SD~865) stood in on AI "
        r"Hub for the paper's Tab~S7$+$ (SD~865$+$), so it is a "
        r"closest-sibling rather than same-device comparison and the Pixel~4 "
        r"figure is the one quoted as the stack bound. Winner-mode agreement "
        r"across stacks is as low as the cross-vendor agreement of "
        r"Table~\ref{tab:supp-flagship-tol}, whereas absolute latency is "
        r"close to unbiased (median USB/AI~Hub ratio near unity): mode "
        r"\emph{rankings}, not milliseconds, are what fail to transfer.}"
    )
    L.append(r"\label{tab:supp-stack-control}")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{3.5pt}")
    L.append(r"\begin{tabular}{@{}llrrrr@{}}")
    L.append(r"\toprule")
    L.append(
        r"USB device & AI Hub device & $N$ & Match \% & ms ratio & "
        r"$\rho$ \\"
    )
    L.append(r"\midrule")
    for dev in sorted(k for k in sc if k != "pooled"):
        d = sc[dev]
        tag = "" if d["same_device"] else r"$^\ddagger$"
        L.append(
            f"{dev.replace('+',r'$+$')} & "
            f"{d['aihub_device'].replace('+',r'$+$')}{tag} & "
            f"{d['n_scored']} & {fmt(d['match_pct'])} & "
            f"{fmt(d['usb_over_aihub_ms_ratio_median'],2)} & "
            f"{fmt(d['mode_rank_spearman_median'],2)} \\\\"
        )
    p = sc["pooled"]
    L.append(r"\midrule")
    L.append(
        f"Pooled & --- & {p['n_scored']} & {fmt(p['match_pct'])} & "
        f"{fmt(p['usb_over_aihub_ms_ratio_median'],2)} & "
        f"{fmt(p['mode_rank_spearman_median'],2)} \\\\"
    )
    L.append(r"\bottomrule")
    L.append(r"\addlinespace[2pt]")
    L.append(r"\multicolumn{6}{@{}l@{}}{\footnotesize "
             r"$^\ddagger$not the same physical device.} \\")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L.append("")
    # S36: MediaTek delegate coverage
    cov = res["mtk_coverage"]
    L.append(r"\begin{table}[!t]")
    L.append(r"\centering")
    L.append(
        r"\caption{MediaTek Dimensity~9400 delegate coverage: number of the 40 "
        r"subset models that produced a valid measurement on each backend, by "
        r"PTQ mode. The NPU path accepts the calibrated integer graphs for "
        r"almost no model, so on this device integer modes execute on the CPU "
        r"or GPU. This is the mechanism behind the cross-vendor divergence in "
        r"winner modes.}"
    )
    L.append(r"\label{tab:supp-mtk-coverage}")
    L.append(r"\footnotesize")
    L.append(r"\begin{tabular}{@{}lrrrr@{}}")
    L.append(r"\toprule")
    L.append(r"Mode & CPU & GPU & NPU & Any \\")
    L.append(r"\midrule")
    for m in MODES:
        pm = cov["per_mode_backend_ok"]
        L.append(
            f"{MODE_TEX[m]} & {pm[m+'|cpu']} & {pm[m+'|gpu']} & "
            f"{pm[m+'|npu']} & {cov['per_mode_any_backend_ok'][m]} \\\\"
        )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    L.append("")
    L.append("")
    # ---- S37: full win-rate matrix, pure latency vs constrained
    L.append(r"\begin{table}[!t]")
    L.append(r"\centering")
    L.append(
        rf"\caption{{Winner share (\%) per PTQ mode on the freeze-150 subset, "
        rf"selecting by pure latency (lat.) and under the ${tp}\%$ accuracy "
        rf"budget. Dynamic INT8, the most frequent optimum on both full "
        rf"corpora, never wins on the AI~Hub devices; INT8's pure-latency "
        rf"dominance there is more than halved by the accuracy budget.}}"
    )
    L.append(r"\label{tab:supp-flagship-winrates-full}")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\begin{tabular}{@{}llrrrrr@{}}")
    L.append(r"\toprule")
    L.append(r"Device & Sel. & FP32 & FP16 & dyn. & INT8 & mixed \\")
    L.append(r"\midrule")
    for dev in ["S25", "S26", "MTK_D9400", "USB_Pixel4", "USB_TabS7",
                "USB_Redmi13"]:
        if dev not in res["winrate_pure"]:
            continue
        for tag, key in (("lat.", "winrate_pure"),
                         (rf"${tp}\%$", "winrate_constrained")):
            w = res[key][dev]
            cells = " & ".join(fmt(w["pct"].get(m, 0.0)) for m in MODES)
            name = DEV_TEX[dev] if tag == "lat." else ""
            L.append(f"{name} & {tag} & {cells} \\\\")
        L.append(r"\addlinespace[2pt]")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument(
        "--flagship-dir", default=str(HERE / "results" / "flagship_2026")
    )
    ap.add_argument("--acc-dir", default=str(HERE / "im100_acc"))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    fdir = Path(args.flagship_dir)
    out = Path(args.out_dir) if args.out_dir else fdir
    out.mkdir(parents=True, exist_ok=True)

    lat, meta = load_winners(fdir / "winners_by_device.csv")
    acc = load_accuracies(Path(args.acc_dir))
    tau = args.tau

    freeze = sorted(lat["S25"])
    res = {"tau": tau, "n_freeze": len(freeze)}

    # sampling design of the freeze list
    res["freeze_design"] = {
        "buckets": dict(Counter(meta[m]["bucket"] for m in freeze)),
        "families": dict(Counter(meta[m]["family"] for m in freeze)),
        "mtk40": sum(1 for m in freeze if meta[m]["in_mtk40"]),
    }

    # feasibility
    fc = Counter()
    fsize = Counter()
    for m in freeze:
        f = feasible_set(m, acc, tau)
        fsize[len(f) if f else 0] += 1
        for mode in f or ():
            fc[mode] += 1
    res["feasible_counts"] = {m: fc[m] for m in MODES}
    res["feasible_set_sizes"] = dict(sorted(fsize.items()))
    res["acc_coverage"] = sum(1 for m in freeze if m in acc["fp32"])

    # win rates
    for key, kw in (("winrate_pure", {}), ("winrate_constrained", {"acc": acc, "tau": tau})):
        res[key] = {}
        for dev in lat:
            w = winners(lat, dev, **kw)
            n = len(w)
            c = Counter(w.values())
            res[key][dev] = {
                "n": n,
                "pct": {m: 100 * c[m] / n for m in MODES} if n else {},
                "count": {m: c[m] for m in MODES},
            }

    # agreement rows reported in the main table
    res["main_agree_rows"] = [
        ("S25|S26", "S25 vs S26", False),
        ("S25|MTK_D9400", "S25 vs Tab~S11", True),
        ("S25|USB_Pixel4", "S25 vs Pixel~4", True),
        ("USB_Pixel4|USB_TabS7", "Pixel~4 vs Tab~S7+", False),
    ]
    res["regret_rows"] = [
        ("S25|S26", r"S25 $\rightarrow$ S26"),
        ("S26|S25", r"S26 $\rightarrow$ S25"),
        ("S25|MTK_D9400", r"S25 $\rightarrow$ Tab~S11"),
        ("MTK_D9400|S25", r"Tab~S11 $\rightarrow$ S25"),
        ("USB_Pixel4|S25", r"USB Pixel~4 $\rightarrow$ S25"),
        ("USB_Pixel4|USB_TabS7", r"USB Pixel~4 $\rightarrow$ Tab~S7+"),
    ]
    pairs = sorted(
        {k for k, _, _ in res["main_agree_rows"]} | {k for k, _ in res["regret_rows"]}
    )
    res["agree_pure"], res["agree_constrained"] = {}, {}
    res["regret_pure"], res["regret_constrained"] = {}, {}
    res["tol_constrained"], res["tol_pure"] = {}, {}
    for key in pairs:
        d1, d2 = key.split("|")
        res["agree_pure"][key] = agreement(lat, d1, d2)
        res["agree_constrained"][key] = agreement(lat, d1, d2, acc, tau)
        res["regret_pure"][key] = regret(lat, d1, d2)
        res["regret_constrained"][key] = regret(lat, d1, d2, acc, tau)
        res["tol_constrained"][key] = {
            f"{t:.2f}": tol_agreement(lat, d1, d2, t, acc, tau)
            for t in (0.0, 0.02, 0.05, 0.10)
        }
        res["tol_pure"][key] = {
            f"{t:.2f}": tol_agreement(lat, d1, d2, t)
            for t in (0.0, 0.02, 0.05, 0.10)
        }
    # Full ordered pairwise regret matrix, for the figure. Devices measured
    # through different harnesses are flagged so the figure can mark them.
    DEV_ORDER = ["USB_Redmi13", "USB_Pixel4", "USB_TabS7", "MTK_D9400",
                 "S25", "S26"]
    STACK = {"USB_Redmi13": "usb", "USB_Pixel4": "usb", "USB_TabS7": "usb",
             "MTK_D9400": "benchmark_model", "S25": "aihub", "S26": "aihub"}
    res["device_order"] = [d for d in DEV_ORDER if d in lat]
    res["device_stack"] = {d: STACK[d] for d in res["device_order"]}
    mat = {}
    for a in res["device_order"]:
        for b in res["device_order"]:
            if a == b:
                mat[f"{a}|{b}"] = {"n": None, "median_pct": 0.0,
                                   "same_stack": True}
                continue
            r = regret(lat, a, b, acc, tau)
            if r is None:
                continue
            r["same_stack"] = STACK[a] == STACK[b]
            mat[f"{a}|{b}"] = r
    res["regret_matrix"] = mat

    res["triple_pure"] = triple_agreement(lat, ["S25", "S26", "MTK_D9400"])
    res["triple_constrained"] = triple_agreement(
        lat, ["S25", "S26", "MTK_D9400"], acc, tau
    )

    # Winner latency distribution.  The main campaign excludes recorded
    # durations below MIN_USB_MS as implausible for a single inference (one
    # Redmi 13 record reports 0.0011 ms); that exclusion is applied here to
    # the USB devices only.  AI Hub returns a fused latency on a different
    # scale -- 48 of 150 S25 winners are legitimately below 0.1 ms -- so its
    # values are summarized as returned.
    res["winner_latency_ms"] = {}
    res["winner_latency_excluded"] = {}
    for dev in lat:
        w = winners(lat, dev, acc, tau)
        raw = [lat[dev][m][mode] for m, mode in w.items()]
        if dev.startswith("USB_"):
            v = sorted(x for x in raw if x >= MIN_USB_MS)
            res["winner_latency_excluded"][dev] = len(raw) - len(v)
        else:
            v = sorted(raw)
            res["winner_latency_excluded"][dev] = 0
        if v:
            res["winner_latency_ms"][dev] = {
                "n": len(v),
                "min": v[0],
                "q1": v[len(v) // 4],
                "median": statistics.median(v),
                "q3": v[3 * len(v) // 4],
                "max": v[-1],
            }

    res["tie_counts"] = {
        dev: {
            "pure": tie_counts(lat, dev),
            "constrained": tie_counts(lat, dev, acc, tau),
        }
        for dev in lat
    }

    res["mtk_coverage"] = mtk_coverage(fdir / "raw" / "mediatek_rows.csv")
    res["stack_control"] = stack_control(
        fdir / "raw" / "harness_20_paper_vs_aihub.csv"
    )

    # ---------------- outputs ----------------
    (out / "flagship_corrected.json").write_text(json.dumps(res, indent=2))
    (out / "flagship_corrected_tables.tex").write_text(tex_main(res, tau))
    (out / "flagship_corrected_supp.tex").write_text(tex_supp(res, tau))

    with (out / "winners_constrained.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            ["model_name", "model_family", "latency_bucket", "device_tag",
             "winner_mode_pure", "winner_mode_constrained", "feasible_modes"]
        )
        for dev in sorted(lat):
            wp = winners(lat, dev)
            wc = winners(lat, dev, acc, tau)
            for mdl in sorted(lat[dev]):
                f = feasible_set(mdl, acc, tau) or set()
                wr.writerow(
                    [mdl, meta[mdl]["family"], meta[mdl]["bucket"], dev,
                     wp.get(mdl, ""), wc.get(mdl, ""),
                     "|".join(m for m in MODES if m in f)]
                )

    lines = []
    P = lines.append
    P(f"Corrected flagship spot-check (tau = {tau:.2%} of matched FP32)")
    P("=" * 68)
    P(f"freeze models: {res['n_freeze']}  (accuracies available: "
      f"{res['acc_coverage']})")
    P(f"  latency buckets : {res['freeze_design']['buckets']}")
    P(f"  model families  : {res['freeze_design']['families']}")
    P(f"  MediaTek subset : {res['freeze_design']['mtk40']}")
    P("")
    P("Feasibility under the accuracy budget")
    P(f"  modes feasible per model: {res['feasible_set_sizes']}")
    for m in MODES:
        P(f"  {m:8s} feasible for {res['feasible_counts'][m]:3d}/"
          f"{res['n_freeze']} models")
    P("")
    P("Winner share (%)  [pure latency -> accuracy-constrained]")
    for dev in ["S25", "S26", "MTK_D9400", "USB_Pixel4", "USB_TabS7", "USB_Redmi13"]:
        if dev not in res["winrate_pure"]:
            continue
        a, b = res["winrate_pure"][dev], res["winrate_constrained"][dev]
        P(f"  {dev:12s} n={a['n']:3d}/{b['n']:3d}")
        for m in MODES:
            P(f"      {m:8s} {a['pct'].get(m,0):5.1f} -> "
              f"{b['pct'].get(m,0):5.1f}")
    P("")
    P("Winner-mode agreement")
    for key, label, dag in res["main_agree_rows"]:
        ap_, ac = res["agree_pure"][key], res["agree_constrained"][key]
        P(f"  {label}{' [cross-stack]' if dag else ''}")
        P(f"      pure        n={ap_['n']:3d} agree={ap_['agree_pct']:5.1f}% "
          f"chance={ap_['chance_pct']:5.1f}% kappa={ap_['kappa']:.3f}")
        P(f"      constrained n={ac['n']:3d} agree={ac['agree_pct']:5.1f}% "
          f"chance={ac['chance_pct']:5.1f}% kappa={ac['kappa']:.3f}")
    P(f"  triple S25+S26+MTK pure       : {res['triple_pure']['agree']}/"
      f"{res['triple_pure']['n']} = {res['triple_pure']['agree_pct']:.1f}%")
    P(f"  triple S25+S26+MTK constrained: "
      f"{res['triple_constrained']['agree']}/{res['triple_constrained']['n']} "
      f"= {res['triple_constrained']['agree_pct']:.1f}%")
    P("")
    P("Tie-tolerant agreement on the constrained winner (%)")
    for key, label, _ in res["main_agree_rows"]:
        row = res["tol_constrained"][key]
        P(f"  {label:34s} " + "  ".join(
            f"tau={t}: {row[t]['agree_pct']:5.1f}"
            for t in ("0.00", "0.02", "0.05", "0.10")))
    P("")
    P("Regret of shipping device A's constrained winner on device B")
    for key, label in res["regret_rows"]:
        r = res["regret_constrained"].get(key)
        if not r:
            continue
        P(f"  {key:24s} n={r['n']:3d} median={r['median_pct']:7.2f}% "
          f"mean={r['mean_pct']:8.2f}% p90={r['p90_pct']:8.2f}% "
          f"median_abs={r['median_ms']:.4f} ms")
    P("")
    P("Constrained-winner latency (ms)")
    for dev, d in res["winner_latency_ms"].items():
        P(f"  {dev:12s} n={d['n']:3d} min={d['min']:.4f} q1={d['q1']:.4f} "
          f"med={d['median']:.4f} q3={d['q3']:.4f} max={d['max']:.4f}")
    P("")
    P("Exact latency ties (selection resolved toward the least quantized mode)")
    for dev, d in res["tie_counts"].items():
        P(f"  {dev:12s} pure {d['pure']['ties']:3d}/{d['pure']['n']:3d}   "
          f"constrained {d['constrained']['ties']:3d}/"
          f"{d['constrained']['n']:3d}   "
          f"{d['constrained']['patterns'] or '{}'}")
    P("")
    cov = res["mtk_coverage"]
    P("MediaTek Dimensity 9400 run accounting")
    P(f"  rows={cov['rows']} models={cov['n_models']} status={cov['status']}")
    P(f"  models with all five modes measurable: "
      f"{cov['models_all_five_modes']}/{cov['n_models']}")
    P("  valid measurements per mode x backend (of 40 models):")
    for m in MODES:
        pm = cov["per_mode_backend_ok"]
        P(f"      {m:8s} cpu={pm[m+'|cpu']:2d} gpu={pm[m+'|gpu']:2d} "
          f"npu={pm[m+'|npu']:2d} any={cov['per_mode_any_backend_ok'][m]:2d}")
    P(f"  benchmark_model version string recorded: "
      f"{cov['runtime_version_recorded']}")
    P("")
    P("Same-device measurement-stack control (USB harness vs AI Hub)")
    for dev, d in res["stack_control"].items():
        P(f"  {dev:16s} n={d['n_scored']:2d} winner match="
          f"{d['match_pct']:5.1f}% ms ratio(med)="
          f"{d['usb_over_aihub_ms_ratio_median']:.2f} "
          f"mode-rank rho(med)={d['mode_rank_spearman_median']:.2f}")
    P(f"  pooled log-latency Pearson r = "
      f"{res['stack_control']['pooled']['log_latency_pearson_r']:.3f} "
      f"over {res['stack_control']['pooled']['n_mode_pairs']} mode pairs")
    txt = "\n".join(lines) + "\n"
    (out / "flagship_corrected.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
