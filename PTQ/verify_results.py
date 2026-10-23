#!/usr/bin/env python3
"""Re-run the device-free part of the pipeline and diff it against the
committed outputs.  Nothing in this checkout is modified: the analysis folder is
copied to a temporary directory, the scripts run there, and the files they
produce are compared with the ones shipped under ``analysis/results/``.

    python verify_results.py              # ~12 min on two cores
    python verify_results.py --quick      # skips the two long scripts (~2 min)
    python verify_results.py --keep       # leave the scratch copy in place

Exit status is 0 when every output matches exactly or within its documented
tolerance, 1 otherwise.  Stage 1 scripts that need a local nn-dataset checkout
are out of scope here and are listed as skipped.
"""
from __future__ import annotations

import argparse
import difflib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (script, argv tail, stdout destination relative to analysis/)
JOBS = [
    ("flagship_corrected.py", [], None),
    ("guideline.py", [], "results/v83/guideline.txt"),
    ("rules_eval.py", [], "results/v83/rules_eval.txt"),
    ("one_or_many.py", [], None),
    ("dcor_released.py", [], "results/v83/dcor_released.txt"),
    ("paired_tests.py", [], None),
    ("mtk_profile.py", [], None),
    ("calibrate_new_devices.py", [], None),
    ("accel_profile.py", [], "results/v85/accel_profile.txt"),
    ("virtual_device.py", [], "results/v85/virtual_device.txt"),
    ("overhead_model.py", [], "results/v85/overhead_model.txt"),
    ("k_artifacts.py", [], "results/v85/k_artifacts.txt"),
    ("cross_task.py", [], "results/v85/cross_task.txt"),
    ("named_models.py", [], "results/v85/named_models.txt"),
    ("robustness_v85.py", [], "results/v85/robustness_v85.txt"),
    ("headline_ci.py", [], "results/v85/headline_ci.txt"),
    ("selector_variants.py", [], "results/v85/selector_variants.txt"),   # long
    ("family_transfer.py", [], "results/v85/family_transfer.txt"),       # long
    ("make_supp_extra.py", [], None),
    ("make_supp_tables.py", [], None),
    ("make_tables_v85.py", [], None),
    ("make_supp_v85.py", [], None),
    ("make_all_figures.py", [], None),
]
LONG = {"selector_variants.py", "family_transfer.py"}

NEEDS_NND = ["load_runs.py", "load_runs_extra.py", "predictor.py --data-root",
             "v83_analysis.py", "robustness.py", "robustness_extra.py",
             "robustness_im100.py", "accuracy_released.py", "export_gap.py",
             "misc_checks.py"]

# Outputs whose values depend on scikit-learn's gradient-boosting internals.
# Both are produced by HistGradientBoosting* fitted on small per-device subsets;
# across scikit-learn releases their entries move by up to ~1.2 points, with no
# change to any conclusion.  Everything else must match byte for byte, or differ
# only by floating-point association in the last couple of digits.
TOLERATED = {
    "results/v85/family_transfer.txt",
    "results/v85/family_transfer.json",
    "results/v85/selector_variants.txt",
    "results/v85/selector_variants.json",
    "results/v85/supp_v85.tex",          # embeds Tables S28 and S29
}
TOLERANCE_PP = 2.0            # documented margin for the two files above
FLOAT_NOISE = 1e-9            # relative; summation order in numpy reductions

NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _numbers(path: Path):
    return [float(x) for x in NUMBER.findall(path.read_text(errors="replace"))]


def _skeleton(path: Path) -> str:
    """File content with every number blanked, to confirm only values moved."""
    return NUMBER.sub("#", path.read_text(errors="replace"))


def compare(committed: Path, produced: Path, tolerated: bool):
    """Return 'exact', 'float-noise', 'tolerated', or None when it is a real
    mismatch."""
    if committed.read_bytes() == produced.read_bytes():
        return "exact"
    if _skeleton(committed) != _skeleton(produced):
        return None
    a, b = _numbers(committed), _numbers(produced)
    if len(a) != len(b):
        return None
    worst = max((abs(x - y) / max(1.0, abs(x)) for x, y in zip(a, b)), default=0.0)
    if worst <= FLOAT_NOISE:
        return "float-noise"
    if tolerated and all(abs(x - y) <= TOLERANCE_PP for x, y in zip(a, b)):
        return "tolerated"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the two long scripts")
    ap.add_argument("--keep", action="store_true", help="keep the scratch copy")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="verify_results_"))
    dst = work / "analysis"
    shutil.copytree(HERE, dst, ignore=shutil.ignore_patterns("__pycache__"))
    (work / "figures").mkdir(exist_ok=True)
    reference = HERE / "results"

    print(f"scratch copy: {dst}")
    print(f"python {sys.version.split()[0]}")
    for mod in ("numpy", "scipy", "pandas", "sklearn", "matplotlib"):
        try:
            print(f"  {mod:12s} {__import__(mod).__version__}")
        except Exception:                                   # pragma: no cover
            print(f"  {mod:12s} MISSING")

    failures, tolerated, noise, ran = [], [], [], 0
    for script, argv, stdout_to in JOBS:
        if args.quick and script in LONG:
            print(f"-- {script:28s} skipped (--quick)")
            continue
        out = subprocess.run([sys.executable, script, *argv], cwd=dst,
                             capture_output=True, text=True)
        ran += 1
        if out.returncode != 0:
            failures.append(f"{script}: exited {out.returncode}\n{out.stderr[-800:]}")
            print(f"-- {script:28s} FAILED")
            continue
        if stdout_to:
            (dst / stdout_to).write_text(out.stdout)
        print(f"-- {script:28s} ok")

    print("\ncomparing produced files with the committed ones")
    for produced in sorted(dst.glob("results/**/*")):
        if not produced.is_file() or produced.name == "cv_folds.json":
            continue
        rel = produced.relative_to(dst / "results").as_posix()
        committed = reference / rel
        if not committed.exists():
            continue
        key = f"results/{rel}"
        verdict = compare(committed, produced, key in TOLERATED)
        if verdict == "exact":
            continue
        if verdict == "float-noise":
            noise.append(key)
            continue
        if verdict == "tolerated":
            tolerated.append(key)
            continue
        diff = list(difflib.unified_diff(
            committed.read_text(errors="replace").splitlines(),
            produced.read_text(errors="replace").splitlines(),
            "committed", "reproduced", lineterm="", n=0))[:12]
        failures.append(key + "\n" + "\n".join(diff))

    print(f"\n{ran} scripts run")
    if noise:
        print(f"{len(noise)} output(s) identical to within floating-point noise "
              f"(relative {FLOAT_NOISE:g}):")
        for t in noise:
            print(f"    {t}")
    if tolerated:
        print(f"{len(tolerated)} output(s) matched within the documented "
              f"{TOLERANCE_PP:.0f}-point tolerance:")
        for t in tolerated:
            print(f"    {t}")
    print("stage 1 scripts not covered here (need a local nn-dataset checkout): "
          + ", ".join(NEEDS_NND))

    if failures:
        print(f"\n{len(failures)} MISMATCH(ES):")
        for f in failures:
            print("  " + f.replace("\n", "\n    "))
    else:
        print("\nall reproduced outputs match the committed ones.")

    if args.keep:
        print(f"scratch copy kept at {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
