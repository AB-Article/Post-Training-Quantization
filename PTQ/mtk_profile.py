#!/usr/bin/env python3
"""Per-backend profile and backend-removal control for the Dimensity 9400.

The Tab S11 is the only 2025-2026 device with per-backend timings: the two
AI Hub flagships were run with ``--compute_unit all``, which returns a single
fused latency. This script therefore extends panels (a) and (b) of Fig. 1 to a
fourth device, using exactly the definitions of ``accel_profile.py`` and
``virtual_device.py`` so the new row and bars are computed like the others:

  (a) ``mtk_backend_profile.json`` -- median latency of each mode relative to
      FP32 **on the same backend**, paired by model.
  (b) ``mtk_virtual_device.json`` -- share of models for which the best
      *integer* mode (mixed, dynamic, static INT8) is more than 5% faster than
      the best *floating-point* mode, as backends are withheld, under the same
      1% accuracy budget.

Caveat carried into the figure caption: the Tab S11 was only ever run on the
ImageNet-100 freeze subset, while the other three devices in Fig. 1 are
CIFAR-10. The quantity that is comparable across devices is therefore the
within-device trend, not the absolute level. Its bars in panel (b) are hatched
for that reason.

Reads only ``results/flagship_2026/raw/mediatek_rows.csv`` and ``im100_acc/``.

    python mtk_profile.py
"""
import collections
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ["fp32", "fp16", "dynamic", "int8", "mixed"]
INTEGER = {"mixed", "dynamic", "int8"}
BACKENDS = ("cpu", "gpu", "npu")
MIN_MS = 0.1          # same plausibility floor as the main campaign
TAU = 0.01            # 1% of the matched FP32 accuracy
SCENARIOS = {
    "all backends (real)": ["cpu", "gpu", "npu"],
    "no GPU": ["cpu", "npu"],
    "no NPU": ["cpu", "gpu"],
    "CPU only": ["cpu"],
}
OUT = HERE / "results" / "flagship_2026"


def load_latencies():
    """model -> (mode, backend) -> ms, over successful runs above the floor."""
    lat = collections.defaultdict(dict)
    for r in csv.DictReader((OUT / "raw" / "mediatek_rows.csv").open()):
        if r["status"] != "ok":
            continue
        try:
            v = float(r["latency_ms_avg"])
        except (TypeError, ValueError):
            continue
        if v > MIN_MS:
            lat[r["model_name"]][(r["quant_mode"], r["backend"])] = v
    return lat


def load_accuracies():
    acc = {}
    for m in MODES:
        acc[m] = json.load((HERE / "im100_acc" /
                            f"all_models_accuracy_{m}.json").open())
    return acc


def feasible(model, acc):
    if model not in acc["fp32"]:
        return set()
    a32 = acc["fp32"][model]["accuracy"]
    return {m for m in MODES
            if model in acc[m] and acc[m][model]["accuracy"] >= a32 - TAU}


def backend_profile(lat):
    prof = {}
    for b in BACKENDS:
        for m in MODES:
            if m == "fp32":
                continue
            rel = [d[(m, b)] / d[("fp32", b)] for d in lat.values()
                   if ("fp32", b) in d and (m, b) in d]
            if rel:
                prof[f"Tab S11|{b}|{m}"] = {"med": round(statistics.median(rel), 2),
                                            "n": len(rel)}
    return prof


def removal_control(lat, acc):
    out = {}
    for name, allowed in SCENARIOS.items():
        decisive = []
        for model, d in lat.items():
            feas = feasible(model, acc)
            if not feas:
                continue
            best_int = best_flt = float("inf")
            for m in MODES:
                if m not in feas:
                    continue
                vals = [d[(m, b)] for b in allowed if (m, b) in d]
                if not vals:
                    continue
                v = min(vals)
                if m in INTEGER:
                    best_int = min(best_int, v)
                else:
                    best_flt = min(best_flt, v)
            if best_int == float("inf") or best_flt == float("inf"):
                continue
            decisive.append(1.0 if best_int < 0.95 * best_flt else 0.0)
        out[name] = {"dec_int": 100 * sum(decisive) / len(decisive),
                     "n": len(decisive)}
    return out


def main():
    lat = load_latencies()
    acc = load_accuracies()
    prof = backend_profile(lat)
    ctrl = removal_control(lat, acc)
    json.dump(prof, (OUT / "mtk_backend_profile.json").open("w"), indent=2)
    json.dump(ctrl, (OUT / "mtk_virtual_device.json").open("w"), indent=2)
    print(f"{len(lat)} models with per-backend timings")
    print("(a) latency relative to FP32 on the same backend")
    for b in BACKENDS:
        row = "  ".join(
            f"{m}={prof[f'Tab S11|{b}|{m}']['med']:.2f}"
            f"(n={prof[f'Tab S11|{b}|{m}']['n']})"
            for m in MODES if m != "fp32" and f"Tab S11|{b}|{m}" in prof)
        print(f"    {b.upper():4s} {row}")
    print("(b) integer mode decisively fastest, by backend set")
    for k, v in ctrl.items():
        print(f"    {k:22s} {v['dec_int']:5.1f}%  n={v['n']}")
    print(f"-> {OUT}/mtk_backend_profile.json, mtk_virtual_device.json")


if __name__ == "__main__":
    main()
