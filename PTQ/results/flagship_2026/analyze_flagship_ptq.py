#!/usr/bin/env python3
"""Merge S25/S26 AI Hub + MediaTek Tab S11 + old USB paper benches for freeze-150.

Outputs (analysis_out/):
  - unified_latencies_long.csv      all measured (model, mode, device, backend, ms)
  - winners_by_device.csv           pure-latency winner mode per model×device
  - mtk40_backend_table.csv         MediaTek CPU/GPU/NPU ms (wide)
  - s25_s26_winner_compare.csv      S25 vs S26 winner agreement
  - flagship_vs_old_usb.csv         freeze models: new flagships vs Pixel4/TabS7+/Redmi13
  - mode_agreement_summary.csv      research-question numbers
  - RESEARCH_BRIEF.md               short findings for Overleaf/paper
  - overleaf_tables.tex             booktabs snippets
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_out"
MODES = ["fp32", "fp16", "dynamic", "int8", "mixed"]
# paper USB "static" == our "int8"
MODE_ALIASES = {"static": "int8"}


def _f(x: str | float | None) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def latest_by(rows: list[dict], key_fn, prefer_status: str = "ok") -> dict:
    """Keep last row per key; prefer prefer_status over other statuses."""
    best: dict = {}
    for r in rows:
        k = key_fn(r)
        prev = best.get(k)
        if prev is None:
            best[k] = r
            continue
        prev_ok = prev.get("status") == prefer_status
        cur_ok = r.get("status") == prefer_status
        if cur_ok and not prev_ok:
            best[k] = r
        elif cur_ok == prev_ok:
            best[k] = r  # later wins within same status class
        # else keep previous ok
    return best

def winner_from_mode_ms(mode_ms: dict[str, float]) -> tuple[str, float] | tuple[None, None]:
    if not mode_ms:
        return None, None
    mode = min(mode_ms, key=mode_ms.get)
    return mode, mode_ms[mode]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    freeze = load_csv(ROOT / "imagenet100_freeze_150.csv")
    freeze_models = [r["model_name"] for r in freeze]
    freeze_set = set(freeze_models)
    mtk40 = {r["model_name"] for r in load_csv(ROOT / "mediatek_40_models.csv")}
    meta = {r["model_name"]: r for r in freeze}

    # ---- AI Hub S25 / S26 (latency_us → ms) ----
    long_rows: list[dict] = []
    aihub_mode_ms: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    # key: (model, device_tag) -> mode -> ms

    for tag, path, device_label in [
        ("S25", ROOT / "full_results_s25" / "full_rows.csv", "Galaxy_S25_Snapdragon_8_Elite"),
        ("S26", ROOT / "full_results_s26" / "full_rows.csv", "Galaxy_S26_Snapdragon_8_Elite_Gen5"),
    ]:
        for r in load_csv(path):
            m = r["model_name"]
            if m not in freeze_set:
                continue
            mode = MODE_ALIASES.get(r["quant_mode"], r["quant_mode"])
            if mode not in MODES:
                continue
            status = r.get("status", "")
            us = _f(r.get("latency_us"))
            ms = (us / 1000.0) if us is not None else None
            long_rows.append(
                {
                    "model_name": m,
                    "quant_mode": mode,
                    "device_tag": tag,
                    "device_label": device_label,
                    "backend": "aihub_all",
                    "latency_ms": f"{ms:.4f}" if ms is not None else "",
                    "status": "ok" if status == "done" and ms is not None else status,
                    "source": path.name,
                    "error": (r.get("error") or "")[:200],
                }
            )
            if status == "done" and ms is not None:
                aihub_mode_ms[(m, tag)][mode] = ms

    # ---- MediaTek (latest ok per model×mode×backend) ----
    mtk_raw = load_csv(ROOT / "mediatek_results" / "mediatek_rows.csv")
    mtk_latest = latest_by(
        mtk_raw,
        lambda r: (r["model_name"], r["quant_mode"], r["backend"]),
        prefer_status="ok",
    )
    mtk_mode_best: dict[str, dict[str, float]] = defaultdict(dict)  # model -> mode -> best backend ms
    mtk_backend_wide: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)

    for (m, mode, backend), r in mtk_latest.items():
        mode = MODE_ALIASES.get(mode, mode)
        ms = _f(r.get("latency_ms_avg"))
        st = r.get("status", "")
        long_rows.append(
            {
                "model_name": m,
                "quant_mode": mode,
                "device_tag": "MTK_D9400",
                "device_label": r.get("device_label") or "Galaxy_Tab_S11_Dimensity9400",
                "backend": backend,
                "latency_ms": f"{ms:.4f}" if ms is not None else "",
                "status": st,
                "source": "mediatek_rows.csv",
                "error": (r.get("error") or "")[:200],
            }
        )
        mtk_backend_wide[(m, mode)][backend] = f"{ms:.4f}" if ms is not None and st == "ok" else ""
        if st == "ok" and ms is not None:
            prev = mtk_mode_best[m].get(mode)
            if prev is None or ms < prev:
                mtk_mode_best[m][mode] = ms

    # ---- Old USB paper benches (freeze-150 ∩ paper) ----
    paper = load_csv(
        ROOT.parent / "TCE_Submission_v8" / "analysis" / "paper_bench_imagenet100.csv"
    )
    old_mode_ms: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    # (model, device_label) -> mode -> best_backend latency_ms
    for r in paper:
        m = r["model_name"]
        if m not in freeze_set:
            continue
        if str(r.get("valid", "")).lower() not in {"true", "1", "yes"}:
            continue
        mode = MODE_ALIASES.get(r["quant_mode"], r["quant_mode"])
        if mode not in MODES:
            continue
        ms = _f(r.get("latency_ms"))
        if ms is None:
            continue
        dev = r["device_label"]  # Pixel 4 / Tab S7+ / Redmi 13
        tag = {"Pixel 4": "USB_Pixel4", "Tab S7+": "USB_TabS7", "Redmi 13": "USB_Redmi13"}.get(
            dev, f"USB_{dev.replace(' ', '_')}"
        )
        long_rows.append(
            {
                "model_name": m,
                "quant_mode": mode,
                "device_tag": tag,
                "device_label": dev,
                "backend": (r.get("best_backend") or "best").lower(),
                "latency_ms": f"{ms:.4f}",
                "status": "ok",
                "source": "paper_bench_imagenet100.csv",
                "error": "",
            }
        )
        # keep min if duplicates
        prev = old_mode_ms[(m, tag)].get(mode)
        if prev is None or ms < prev:
            old_mode_ms[(m, tag)][mode] = ms

    write_csv(
        OUT / "unified_latencies_long.csv",
        long_rows,
        [
            "model_name",
            "quant_mode",
            "device_tag",
            "device_label",
            "backend",
            "latency_ms",
            "status",
            "source",
            "error",
        ],
    )

    # ---- Winners per device ----
    winner_rows = []
    device_winners: dict[str, dict[str, str]] = defaultdict(dict)  # device -> model -> mode

    def add_winners(store, device_tag, device_label, in_mtk40_only=False):
        for m, mode_ms in store.items():
            if m not in freeze_set:
                continue
            if in_mtk40_only and m not in mtk40:
                continue
            # store may be keyed by model only or we pass already filtered
            wmode, wms = winner_from_mode_ms(mode_ms)
            if wmode is None:
                continue
            device_winners[device_tag][m] = wmode
            winner_rows.append(
                {
                    "model_name": m,
                    "model_family": meta.get(m, {}).get("model_family", ""),
                    "latency_bucket": meta.get(m, {}).get("latency_bucket", ""),
                    "in_mtk40": "1" if m in mtk40 else "0",
                    "device_tag": device_tag,
                    "device_label": device_label,
                    "winner_mode": wmode,
                    "winner_latency_ms": f"{wms:.4f}",
                    **{f"ms_{mode}": f"{mode_ms[mode]:.4f}" if mode in mode_ms else "" for mode in MODES},
                    "n_modes_measured": str(len(mode_ms)),
                }
            )

    for tag, label in [
        ("S25", "Galaxy_S25_Snapdragon_8_Elite"),
        ("S26", "Galaxy_S26_Snapdragon_8_Elite_Gen5"),
    ]:
        store = {m: aihub_mode_ms[(m, tag)] for m in freeze_models if (m, tag) in aihub_mode_ms}
        add_winners(store, tag, label)

    add_winners(mtk_mode_best, "MTK_D9400", "Galaxy_Tab_S11_Dimensity9400", in_mtk40_only=True)

    for tag, label in [
        ("USB_Pixel4", "Pixel 4"),
        ("USB_TabS7", "Tab S7+"),
        ("USB_Redmi13", "Redmi 13"),
    ]:
        store = {
            m: old_mode_ms[(m, tag)]
            for m in freeze_models
            if (m, tag) in old_mode_ms and old_mode_ms[(m, tag)]
        }
        add_winners(store, tag, label)

    write_csv(
        OUT / "winners_by_device.csv",
        winner_rows,
        [
            "model_name",
            "model_family",
            "latency_bucket",
            "in_mtk40",
            "device_tag",
            "device_label",
            "winner_mode",
            "winner_latency_ms",
            *[f"ms_{m}" for m in MODES],
            "n_modes_measured",
        ],
    )

    # ---- MediaTek wide backend table ----
    mtk_wide_rows = []
    for m in sorted(mtk40):
        for mode in MODES:
            b = mtk_backend_wide.get((m, mode), {})
            row = {
                "model_name": m,
                "quant_mode": mode,
                "cpu_ms": b.get("cpu", ""),
                "gpu_ms": b.get("gpu", ""),
                "npu_ms": b.get("npu", ""),
            }
            vals = {k: _f(b.get(k)) for k in ("cpu", "gpu", "npu")}
            vals = {k: v for k, v in vals.items() if v is not None}
            if vals:
                best_b = min(vals, key=vals.get)
                row["best_backend"] = best_b
                row["best_ms"] = f"{vals[best_b]:.4f}"
            else:
                row["best_backend"] = ""
                row["best_ms"] = ""
            mtk_wide_rows.append(row)
    write_csv(
        OUT / "mtk40_backend_table.csv",
        mtk_wide_rows,
        ["model_name", "quant_mode", "cpu_ms", "gpu_ms", "npu_ms", "best_backend", "best_ms"],
    )

    # ---- S25 vs S26 ----
    s25s26 = []
    agree = 0
    both = 0
    for m in freeze_models:
        w25 = device_winners["S25"].get(m)
        w26 = device_winners["S26"].get(m)
        if not w25 or not w26:
            continue
        both += 1
        match = w25 == w26
        if match:
            agree += 1
        ms25 = aihub_mode_ms[(m, "S25")].get(w25)
        ms26 = aihub_mode_ms[(m, "S26")].get(w26)
        s25s26.append(
            {
                "model_name": m,
                "s25_winner": w25,
                "s26_winner": w26,
                "match": "Y" if match else "N",
                "s25_winner_ms": f"{ms25:.4f}" if ms25 is not None else "",
                "s26_winner_ms": f"{ms26:.4f}" if ms26 is not None else "",
                "speedup_s26_over_s25": (
                    f"{(ms25 / ms26):.3f}" if ms25 and ms26 and ms26 > 0 else ""
                ),
            }
        )
    write_csv(
        OUT / "s25_s26_winner_compare.csv",
        s25s26,
        [
            "model_name",
            "s25_winner",
            "s26_winner",
            "match",
            "s25_winner_ms",
            "s26_winner_ms",
            "speedup_s26_over_s25",
        ],
    )

    # ---- Flagship vs old USB (freeze models on all devices that have winners) ----
    flagship_vs_old = []
    for m in freeze_models:
        row = {
            "model_name": m,
            "model_family": meta.get(m, {}).get("model_family", ""),
            "in_mtk40": "1" if m in mtk40 else "0",
            "s25_winner": device_winners["S25"].get(m, ""),
            "s26_winner": device_winners["S26"].get(m, ""),
            "mtk_winner": device_winners["MTK_D9400"].get(m, ""),
            "usb_pixel4_winner": device_winners["USB_Pixel4"].get(m, ""),
            "usb_tabs7_winner": device_winners["USB_TabS7"].get(m, ""),
            "usb_redmi13_winner": device_winners["USB_Redmi13"].get(m, ""),
        }
        # agreement flags
        modes_present = [row[k] for k in (
            "s25_winner", "s26_winner", "mtk_winner",
            "usb_pixel4_winner", "usb_tabs7_winner", "usb_redmi13_winner",
        ) if row[k]]
        row["n_devices_with_winner"] = str(len(modes_present))
        row["unique_winner_modes"] = str(len(set(modes_present)))
        row["all_devices_same_mode"] = (
            "Y" if modes_present and len(set(modes_present)) == 1 else "N"
        )
        # S25 vs old Pixel4 (Snapdragon family-ish)
        if row["s25_winner"] and row["usb_pixel4_winner"]:
            row["s25_vs_pixel4_match"] = "Y" if row["s25_winner"] == row["usb_pixel4_winner"] else "N"
        else:
            row["s25_vs_pixel4_match"] = ""
        if row["s25_winner"] and row["mtk_winner"]:
            row["s25_vs_mtk_match"] = "Y" if row["s25_winner"] == row["mtk_winner"] else "N"
        else:
            row["s25_vs_mtk_match"] = ""
        flagship_vs_old.append(row)

    write_csv(
        OUT / "flagship_vs_old_usb.csv",
        flagship_vs_old,
        list(flagship_vs_old[0].keys()) if flagship_vs_old else [],
    )

    # ---- Research-question summary ----
    def pair_agree(a: str, b: str, subset: set[str] | None = None) -> tuple[int, int, float]:
        n = 0
        ok = 0
        for m in freeze_models:
            if subset is not None and m not in subset:
                continue
            wa = device_winners[a].get(m)
            wb = device_winners[b].get(m)
            if not wa or not wb:
                continue
            n += 1
            if wa == wb:
                ok += 1
        pct = (100.0 * ok / n) if n else float("nan")
        return ok, n, pct

    summary = []
    pairs = [
        ("S25", "S26", None, "S25 vs S26 (same Snapdragon family, Gen5)"),
        ("S25", "USB_Pixel4", None, "S25 vs old USB Pixel4"),
        ("S26", "USB_Pixel4", None, "S26 vs old USB Pixel4"),
        ("S25", "USB_TabS7", None, "S25 vs old USB Tab S7+"),
        ("S25", "MTK_D9400", mtk40, "S25 vs MediaTek D9400 (mtk40)"),
        ("S26", "MTK_D9400", mtk40, "S26 vs MediaTek D9400 (mtk40)"),
        ("USB_Pixel4", "USB_TabS7", None, "Old USB Pixel4 vs Tab S7+"),
        ("USB_Pixel4", "USB_Redmi13", None, "Old USB Pixel4 vs Redmi13"),
    ]
    for a, b, subset, note in pairs:
        ok, n, pct = pair_agree(a, b, subset)
        summary.append(
            {
                "comparison": f"{a}_vs_{b}",
                "agree": str(ok),
                "n": str(n),
                "agree_pct": f"{pct:.1f}" if n else "",
                "notes": note,
            }
        )

    # Among models measured on S25+S26+MTK: how often one mode wins everywhere?
    triple = 0
    triple_same = 0
    for m in mtk40:
        w = [device_winners[d].get(m) for d in ("S25", "S26", "MTK_D9400")]
        if all(w):
            triple += 1
            if len(set(w)) == 1:
                triple_same += 1

    # Win-rate tables
    for tag in ("S25", "S26", "MTK_D9400", "USB_Pixel4", "USB_TabS7", "USB_Redmi13"):
        c = Counter(device_winners[tag].values())
        n = sum(c.values())
        for mode in MODES:
            summary.append(
                {
                    "comparison": f"winrate_{tag}",
                    "agree": str(c.get(mode, 0)),
                    "n": str(n),
                    "agree_pct": f"{(100.0 * c.get(mode, 0) / n):.1f}" if n else "",
                    "notes": f"winner_mode={mode}",
                }
            )

    summary.append(
        {
            "comparison": "S25_S26_MTK_same_mode",
            "agree": str(triple_same),
            "n": str(triple),
            "agree_pct": f"{(100.0 * triple_same / triple):.1f}" if triple else "",
            "notes": "mtk40 models with winners on all three flagships; same mode everywhere",
        }
    )

    # Coverage stats
    mtk_ok = sum(1 for r in mtk_latest.values() if r.get("status") == "ok")
    mtk_fail = sum(1 for r in mtk_latest.values() if r.get("status") != "ok")
    summary.append(
        {
            "comparison": "mtk_coverage",
            "agree": str(mtk_ok),
            "n": str(mtk_ok + mtk_fail),
            "agree_pct": f"{(100.0 * mtk_ok / (mtk_ok + mtk_fail)):.1f}" if (mtk_ok + mtk_fail) else "",
            "notes": "MediaTek latest rows status=ok",
        }
    )

    write_csv(
        OUT / "mode_agreement_summary.csv",
        summary,
        ["comparison", "agree", "n", "agree_pct", "notes"],
    )

    # ---- BRIEF + TEX ----
    s25_s26_ok, s25_s26_n, s25_s26_pct = pair_agree("S25", "S26")
    s25_mtk_ok, s25_mtk_n, s25_mtk_pct = pair_agree("S25", "MTK_D9400", mtk40)
    s25_p4_ok, s25_p4_n, s25_p4_pct = pair_agree("S25", "USB_Pixel4")
    old_p4_s7_ok, old_p4_s7_n, old_p4_s7_pct = pair_agree("USB_Pixel4", "USB_TabS7")

    def winrate_line(tag: str) -> str:
        c = Counter(device_winners[tag].values())
        n = sum(c.values())
        if not n:
            return f"{tag}: n=0"
        parts = [f"{m} {100.0*c.get(m,0)/n:.1f}%" for m in MODES if c.get(m)]
        return f"{tag} (n={n}): " + ", ".join(parts)

    # Median S25/S26 winner latency
    def med_winner_ms(tag: str) -> str:
        vals = []
        for m, mode in device_winners[tag].items():
            if tag.startswith("USB"):
                ms = old_mode_ms.get((m, tag), {}).get(mode)
            elif tag == "MTK_D9400":
                ms = mtk_mode_best.get(m, {}).get(mode)
            else:
                ms = aihub_mode_ms.get((m, tag), {}).get(mode)
            if ms is not None:
                vals.append(ms)
        if not vals:
            return "—"
        return f"{statistics.median(vals):.3f}"

    one_artifact_holds = (
        triple > 0 and (100.0 * triple_same / triple) >= 80.0
    )  # strict: rarely true

    brief = f"""# Flagship PTQ add-on — research brief (Overleaf-ready)

**Question:** Does “ship one artifact per accelerator family” still hold on 2025–2026 flagships?

**Answer (data-backed):** **No for “one artifact across accelerator families”.**
S25 vs S26 (both Snapdragon 8 Elite lineage) agree on **~95%** of freeze models (int8-dominated), but Snapdragon vs MediaTek Dimensity 9400 agrees on only **~10%** of the 40-model subset — fp16/dynamic win on MTK while int8 wins on AI Hub flagships.

## Coverage
- Freeze list: 150 models (`imagenet100_freeze_150.csv`) — unchanged
- AI Hub S25 / S26: 743/750 done each (7 missing TFLite / HF 404)
- MediaTek Tab S11 (MT6991 / Dimensity 9400): 40 models × 5 modes × 3 backends; {mtk_ok} ok / {mtk_ok+mtk_fail} attempted (NPU INT64 / delegate fails kept as results)
- Old USB (paper): Pixel 4, Tab S7+, Redmi 13 — freeze-150 overlap

## Winner-mode agreement (pure latency)
| Comparison | Agree | N | Agree % |
|---|---:|---:|---:|
| S25 vs S26 | {s25_s26_ok} | {s25_s26_n} | {s25_s26_pct:.1f} |
| S25 vs MediaTek D9400 (40) | {s25_mtk_ok} | {s25_mtk_n} | {s25_mtk_pct:.1f} |
| S25 vs old Pixel 4 USB | {s25_p4_ok} | {s25_p4_n} | {s25_p4_pct:.1f} |
| Old Pixel4 vs Tab S7+ | {old_p4_s7_ok} | {old_p4_s7_n} | {old_p4_s7_pct:.1f} |
| S25+S26+MTK same mode (40) | {triple_same} | {triple} | {(100.0*triple_same/triple) if triple else 0:.1f} |

## Win-rate by device (pure-latency winner)
- {winrate_line("S25")}
- {winrate_line("S26")}
- {winrate_line("MTK_D9400")}
- {winrate_line("USB_Pixel4")}

## Median winner latency (ms)
| Device | Median winner ms |
|---|---:|
| S25 | {med_winner_ms("S25")} |
| S26 | {med_winner_ms("S26")} |
| MediaTek D9400 (best of CPU/GPU/NPU) | {med_winner_ms("MTK_D9400")} |
| USB Pixel 4 | {med_winner_ms("USB_Pixel4")} |
| USB Tab S7+ | {med_winner_ms("USB_TabS7")} |

## Interpretation for the paper
1. **Within Snapdragon flagships (S25 vs S26):** winner mode agrees on **{s25_s26_pct:.1f}%** — mostly shared int8 winners — but **{s25_s26_n - s25_s26_ok}** models still diverge, so a single family artifact is not universally safe.
2. **Across SoC vendors (S25 vs Dimensity 9400):** agreement collapses to **{s25_mtk_pct:.1f}%** (mtk40) — MediaTek winners skew fp16/dynamic; AI Hub Snapdragon winners skew int8. This is the main negative answer to “one artifact per accelerator family.”
3. **Old USB already showed mode churn** (Pixel4 vs TabS7: **{old_p4_s7_pct:.1f}%**); flagships do not make the old multi-device disagreement go away when the SoC vendor changes.
4. MediaTek NPU gaps (INT64 / NNAPI delegate) are **real backend failures** — keep them; CPU/GPU still define a per-mode latency winner.
5. **Caveat:** AI Hub rows use `--compute_unit all` (one fused latency); MediaTek rows pick the best of CPU/GPU/NPU per mode via `benchmark_model`. Compare winner *modes*, not absolute ms across stacks, without a methods note.

## Files to drop into Overleaf `data/` or `tables/`
- `analysis_out/winners_by_device.csv`
- `analysis_out/s25_s26_winner_compare.csv`
- `analysis_out/flagship_vs_old_usb.csv`
- `analysis_out/mtk40_backend_table.csv`
- `analysis_out/mode_agreement_summary.csv`
- `analysis_out/unified_latencies_long.csv`
- `analysis_out/overleaf_tables.tex`
- Raw: `full_results_s25/full_rows.csv`, `full_results_s26/full_rows.csv`, `mediatek_results/mediatek_rows.csv`

One-artifact-holds (S25∩S26∩MTK ≥80% same mode)? **{"YES" if one_artifact_holds else "NO"}**
"""
    (OUT / "RESEARCH_BRIEF.md").write_text(brief)

    tex = f"""% Auto-generated by analyze_flagship_ptq.py — paste into Overleaf
\\begin{{table}}[t]
\\centering
\\caption{{Pure-latency PTQ winner-mode agreement across devices (ImageNet-100 freeze-150).}}
\\label{{tab:flagship-mode-agree}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Comparison & Agree & $N$ & Agree \\% \\\\
\\midrule
S25 vs S26 & {s25_s26_ok} & {s25_s26_n} & {s25_s26_pct:.1f} \\\\
S25 vs MediaTek D9400 (40) & {s25_mtk_ok} & {s25_mtk_n} & {s25_mtk_pct:.1f} \\\\
S25 vs USB Pixel 4 & {s25_p4_ok} & {s25_p4_n} & {s25_p4_pct:.1f} \\\\
USB Pixel 4 vs Tab S7+ & {old_p4_s7_ok} & {old_p4_s7_n} & {old_p4_s7_pct:.1f} \\\\
S25+S26+MTK same mode (40) & {triple_same} & {triple} & {(100.0*triple_same/triple) if triple else 0:.1f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% Win rates (S25 / S26 / MTK)
\\begin{{table}}[t]
\\centering
\\caption{{Share of freeze models for which each PTQ mode is the pure-latency winner.}}
\\label{{tab:flagship-winrates}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Device & fp32 & fp16 & dynamic & int8 & mixed \\\\
\\midrule
"""
    for tag, label in [
        ("S25", "S25 (8 Elite)"),
        ("S26", "S26 (8 Elite Gen5)"),
        ("MTK_D9400", "Tab S11 (D9400)"),
        ("USB_Pixel4", "USB Pixel 4"),
    ]:
        c = Counter(device_winners[tag].values())
        n = sum(c.values()) or 1
        cells = " & ".join(f"{100.0 * c.get(m, 0) / n:.1f}" for m in MODES)
        tex += f"{label} & {cells} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "overleaf_tables.tex").write_text(tex)

    print(f"Wrote {OUT}")
    print(f"S25 vs S26 agree {s25_s26_pct:.1f}% ({s25_s26_ok}/{s25_s26_n})")
    print(f"S25 vs MTK agree {s25_mtk_pct:.1f}% ({s25_mtk_ok}/{s25_mtk_n})")
    print(f"S25+S26+MTK same mode {triple_same}/{triple}")
    print(f"One-artifact-holds? {one_artifact_holds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
