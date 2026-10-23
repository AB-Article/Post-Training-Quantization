#!/usr/bin/env python3
"""Few-shot calibration of the selector on the three 2025-2026 devices.

`v83_analysis.py` section B answers "how many models must I benchmark on a new
device before the selector is useful?" for the three main-campaign handsets, by
holding one out and calibrating on k of its models. That question is sharper for
hardware that did not exist when the corpus was built, and the freeze-150 subset
makes it answerable: the Galaxy S25, Galaxy S26 and Galaxy Tab S11 each have a
constrained winner and a full latency row for the same 150 (40) models, and all
150 appear in the ImageNet-100 feature table.

Protocol, matching `v83_analysis.py` section B (iv) as closely as the data
allow:

  * train on every ImageNet-100 row of the three main-campaign devices, plus k
    freeze models benchmarked on the target device;
  * test on the remaining freeze models of the target device;
  * predict with the same feasibility-masked HistGradientBoosting selector, the
    same offline model-side features, and average over 5 seeds;
  * score exact match against the target device's accuracy-constrained winner,
    and geometric-mean regret against its best feasible latency.

k is capped by the subset: 25/50/100 for the S25 and S26 (150 models), 10/20/30
for the Tab S11 (40 models).

    python calibrate_new_devices.py   ->  results/flagship_2026/calibration_new_devices.{txt,json}
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))        # so `python -I` can still find predictor
import predictor as P  # noqa: E402
OUT = HERE / "results" / "flagship_2026"
MODES = ["fp32", "fp16", "dynamic", "int8", "mixed"]
FALLBACK = ["dynamic", "fp16", "fp32", "mixed", "int8"]
TAU = 0.01
TARGETS = {"S25": (25, 50, 100), "S26": (25, 50, 100), "MTK_D9400": (10, 20, 30)}
LABEL = {"S25": "Galaxy S25", "S26": "Galaxy S26", "MTK_D9400": "Tab S11"}
SEEDS = 5


def load_target_frames():
    """One frame per 2025-2026 device: offline features + that device's
    latencies, feasible set and constrained winner."""
    feat = pd.read_csv(HERE / "results" / "v83" / "dataset_full_imagenet100.csv")
    feat = feat.drop_duplicates("model").set_index("model")
    acc = {m: json.load((HERE / "im100_acc" /
                         f"all_models_accuracy_{m}.json").open()) for m in MODES}
    rows = list(csv.DictReader((OUT / "winners_by_device.csv").open()))
    out = {}
    for dev in TARGETS:
        recs = []
        for r in rows:
            if r["device_tag"] != dev:
                continue
            mdl = r["model_name"]
            if mdl not in feat.index or mdl not in acc["fp32"]:
                continue
            a32 = acc["fp32"][mdl]["accuracy"]
            feas = {m for m in MODES
                    if mdl in acc[m] and acc[m][mdl]["accuracy"] >= a32 - TAU}
            lat = {}
            for m in MODES:
                v = r.get("ms_" + m)
                if v:
                    lat[m] = float(v)
            cand = {m: lat[m] for m in MODES if m in lat and m in feas}
            if len(cand) < 2:
                continue
            rec = feat.loc[mdl].to_dict()
            rec["model"] = mdl
            rec["label"] = min((m for m in MODES if m in cand),
                               key=lambda m: cand[m])
            rec["_feas"] = feas
            for m in MODES:
                rec[f"lat_{m}"] = lat.get(m, np.nan)
            recs.append(rec)
        out[dev] = pd.DataFrame(recs).reset_index(drop=True)
    return feat, out


def masked_pick(model, frame, cols):
    pr = model.predict_proba(frame[cols])
    cls = list(model.classes_)
    pick = []
    for i in range(len(frame)):
        feas = frame.iloc[i]["_feas"]
        order = np.argsort(-pr[i])
        pick.append(next((cls[j] for j in order if cls[j] in feas),
                         cls[order[0]]))
    return pick


def score(frame, pick):
    hit, rat = [], []
    for i, r in frame.iterrows():
        cand = {m: r[f"lat_{m}"] for m in MODES if pd.notna(r[f"lat_{m}"])}
        p = pick[i]
        hit.append(p == r["label"])
        if p not in cand:
            p = next(x for x in FALLBACK if x in cand)
        rat.append(cand[p] / min(cand.values()))
    rat = np.asarray(rat)
    return (100 * np.mean(hit),
            100 * (np.exp(np.log(rat).mean()) - 1))


def main():
    feat, targets = load_target_frames()
    NA, CA = P.NUM_ARCH, P.CAT_ARCH
    train_all = pd.read_csv(HERE / "results" / "v83" /
                            "dataset_full_imagenet100.csv")
    report, lines = {}, []
    lines.append("Few-shot calibration of the selector on the 2025-2026 devices")
    lines.append("=" * 66)
    lines.append("train: all ImageNet-100 rows of the three main-campaign")
    lines.append("devices + k freeze models of the target; test: the rest.")
    lines.append(f"{SEEDS} seeds, feasibility-masked HGB, offline features.")
    lines.append("")
    for dev, ks in TARGETS.items():
        te_all = targets[dev]
        report[dev] = {"n_models": len(te_all), "k": {}}
        lines.append(f"--- {LABEL[dev]}  ({len(te_all)} freeze models) ---")
        lines.append(f"{'k':>5s} {'exact %':>9s} {'regret %':>9s} {'n test':>7s}")
        # k = 0 reference: no model of the target device is benchmarked
        m = P.model(NA, CA, "hgb").fit(train_all[NA + CA], train_all.label)
        e0, g0 = score(te_all, masked_pick(m, te_all, NA + CA))
        report[dev]["k"][0] = {"exact": e0, "geo": g0, "n_test": len(te_all)}
        lines.append(f"{0:5d} {e0:9.1f} {g0:9.1f} {len(te_all):7d}")
        # context: trivial baselines on the target device itself
        base = {}
        maj = te_all.label.value_counts().idxmax()
        for name, pick in [(f"always {maj} (target majority)",
                            [maj] * len(te_all)),
                           ("always dynamic INT8",
                            ["dynamic"] * len(te_all))]:
            e, g = score(te_all, pick)
            base[name] = {"exact": e, "geo": g}
        report[dev]["baselines"] = base
        for k in ks:
            es, gs, ns = [], [], []
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                cal = set(rng.choice(te_all.model.unique(), k, replace=False))
                trk = pd.concat([train_all, te_all[te_all.model.isin(cal)]],
                                ignore_index=True)
                tek = te_all[~te_all.model.isin(cal)].reset_index(drop=True)
                m = P.model(NA, CA, "hgb").fit(trk[NA + CA], trk.label)
                e, g = score(tek, masked_pick(m, tek, NA + CA))
                es.append(e)
                gs.append(g)
                ns.append(len(tek))
            report[dev]["k"][k] = {"exact": float(np.mean(es)),
                                   "geo": float(np.mean(gs)),
                                   "n_test": int(np.mean(ns))}
            lines.append(f"{k:5d} {np.mean(es):9.1f} {np.mean(gs):9.1f} "
                         f"{int(np.mean(ns)):7d}")
        for name, v in report[dev]["baselines"].items():
            lines.append(f"   baseline {name:34s} exact {v['exact']:5.1f}  "
                         f"regret {v['geo']:7.1f}")
        lines.append("")
    txt = "\n".join(lines) + "\n"
    (OUT / "calibration_new_devices.txt").write_text(txt)
    (OUT / "calibration_new_devices.json").write_text(json.dumps(report, indent=2))
    print(txt)


if __name__ == "__main__":
    main()
