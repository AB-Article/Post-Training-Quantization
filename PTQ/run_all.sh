#!/usr/bin/env bash
# =====================================================================
# Reproduce manuscript numbers and figures from this analysis/ folder.
#
# Usage (from analysis/):
#   ./run_all.sh figures              # regenerate figures from committed results/
#   ./run_all.sh stage1 NND=/path     # main campaign (needs nn-dataset checkout)
#   ./run_all.sh stage2               # mechanism / transfer (needs stage1 outputs)
#   ./run_all.sh stage3               # 2025–2026 flagship spot check
#   ./run_all.sh all NND=/path        # stage1 + stage2 + stage3 + figures
#
# Requirements: Python 3 with numpy, matplotlib, pandas, scipy.
# Set PYTHON to a venv if needed, e.g. PYTHON=/path/to/venv/bin/python
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
NND=""
MODE="${1:-figures}"
shift || true
for arg in "$@"; do
  case "$arg" in
    NND=*) NND="${arg#NND=}" ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

need_nnd() {
  if [[ -z "$NND" || ! -d "$NND" ]]; then
    echo "Set NND=/path/to/nn-dataset (directory with ab/nn/stat/run/tflite/)." >&2
    exit 2
  fi
}

run_figures() {
  echo "== Regenerating all manuscript figures (make_all_figures.py) =="
  "$PYTHON" make_all_figures.py
  echo "== Figures done =="
}

run_stage1() {
  need_nnd
  echo "== Stage 1 — main campaign (NND=$NND) =="
  "$PYTHON" load_runs.py --data-root "$NND" --out results/v83/runs_long.csv
  "$PYTHON" predictor.py --data-root "$NND" --out-csv results/v83/dataset_full_cifar10.csv
  "$PYTHON" predictor.py --data-root "$NND" --run-subdir imagenet100 --acc-dir im100_acc \
    --out-csv results/v83/dataset_full_imagenet100.csv
  for S in A B C D; do
    "$PYTHON" v83_analysis.py --data-root "$NND" --sections "$S" > "results/v83/log_${S}.txt"
  done
  for S in A B; do
    "$PYTHON" v83_analysis.py --data-root "$NND" --acc-dir im100_acc \
      --csv results/v83/dataset_full_imagenet100.csv --sections "$S" --tag _im100 \
      > "results/v83/log_${S}_im100.txt"
  done
  "$PYTHON" v83_analysis.py --data-root "$NND" --acc-dir im100_acc \
    --csv results/v83/dataset_full_imagenet100.csv --sections C --corpus imagenet100 \
    --tag _im100 > results/v83/log_C_im100.txt
  "$PYTHON" robustness.py --data-root "$NND" > results/v83/robustness_root.txt
  "$PYTHON" robustness_extra.py --data-root "$NND" --tag cifar10_root --out-dir results/v83
  "$PYTHON" robustness_extra.py --data-root "$NND" --run-subdir imagenet100 --acc-dir im100_acc \
    --tag imagenet100_root --out-dir results/v83
  "$PYTHON" robustness_im100.py --data-root "$NND" --acc-dir im100_acc > results/v83/robustness_im100.txt
  "$PYTHON" guideline.py > results/v83/guideline.txt
  "$PYTHON" rules_eval.py > results/v83/rules_eval.txt
  "$PYTHON" one_or_many.py
  "$PYTHON" accuracy_released.py "$NND" im100_acc im100_acc/checkpoint_epoch50_all_models.json \
    > results/v83/accuracy_released.txt
  "$PYTHON" export_gap.py "$NND" > results/v85/export_gap.txt
  "$PYTHON" dcor_released.py > results/v83/dcor_released.txt
  "$PYTHON" misc_checks.py "$NND"
  "$PYTHON" paired_tests.py
  "$PYTHON" mtk_profile.py
  "$PYTHON" calibrate_new_devices.py
  "$PYTHON" make_supp_extra.py
  "$PYTHON" make_fig_effect_sizes.py
  "$PYTHON" make_fig_flagship.py
  "$PYTHON" predictor.py --csv results/v83/dataset_full_cifar10.csv --seeds 1 \
    | tail -3 > results/v83/permimp_cifar10.txt
  "$PYTHON" predictor.py --csv results/v83/dataset_full_imagenet100.csv --seeds 1 \
    | tail -3 > results/v83/permimp_imagenet100.txt
  "$PYTHON" make_fig_v83.py
  "$PYTHON" make_supp_tables.py
  "$PYTHON" headline_ci.py > results/v85/headline_ci.txt
  echo "== Stage 1 done =="
}

run_stage2() {
  echo "== Stage 2 — mechanism, granularity, transfer =="
  need_nnd
  "$PYTHON" load_runs_extra.py --data-root "$NND" --out results/v85/runs_extra.csv
  "$PYTHON" accel_profile.py > results/v85/accel_profile.txt
  "$PYTHON" virtual_device.py > results/v85/virtual_device.txt
  "$PYTHON" overhead_model.py > results/v85/overhead_model.txt
  "$PYTHON" k_artifacts.py > results/v85/k_artifacts.txt
  "$PYTHON" selector_variants.py > results/v85/selector_variants.txt
  "$PYTHON" family_transfer.py > results/v85/family_transfer.txt
  "$PYTHON" cross_task.py > results/v85/cross_task.txt
  "$PYTHON" named_models.py > results/v85/named_models.txt
  "$PYTHON" robustness_v85.py > results/v85/robustness_v85.txt
  "$PYTHON" make_tables_v85.py
  "$PYTHON" make_supp_v85.py
  "$PYTHON" make_fig_v85.py
  echo "== Stage 2 done =="
}

run_stage3() {
  echo "== Stage 3 — flagship spot check =="
  "$PYTHON" flagship_corrected.py
  echo "== Stage 3 done =="
}

case "$MODE" in
  figures) run_figures ;;
  stage1)  run_stage1 ;;
  stage2)  run_stage2 ;;
  stage3)  run_stage3 ;;
  all)
    run_stage1
    run_stage2
    run_stage3
    run_figures
    ;;
  *)
    echo "Usage: $0 {figures|stage1|stage2|stage3|all} [NND=/path/to/nn-dataset]" >&2
    exit 2
    ;;
esac
