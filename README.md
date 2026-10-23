# Device-Aware Post-Training Quantization for Consumer Android Devices: When to Share PTQ Artifacts Across Accelerator Families

# Analysis scripts and results

All scripts that produce the numbers in `main.tex` and `supplementary.tex`,
with their stored outputs under `results/`.

Entry point: `./run_all.sh figures` (figures from committed results) or
`./run_all.sh all NND=/path/to/nn-dataset` (full rebuild). See the top-level
`README.md` for data sources, stages, the script-to-manuscript map, and
statistical conventions.

`results/flagship_2026/README.md` documents the 2025–2026 spot-check data:
sampling design, coverage, known data defects, and the one open measurement
needed to make the cross-vendor comparison identifiable.

## Reproducibility helpers in this folder

| File | Role |
|---|---|
| `verify_results.py` | re-runs every device-free script in a scratch copy and diffs the output against the committed results; `--quick` skips the two long ones. Nothing in the checkout is modified. |
| `grouped_cv.py` | the architecture-grouped CV split, computed here from a frozen fold map (`results/cv_folds.json`) instead of `sklearn.model_selection.GroupKFold`, whose group-to-fold assignment changed in scikit-learn 1.6 and silently moved cross-validated numbers by up to 1.5 points. |
| `make_fig_extra.py` | draws the eighteen descriptive plots that had no generator in this package; `make_all_figures.py` calls it, so `./run_all.sh figures` rebuilds every PNG under `figures/` starting from an empty folder. |
| `_runctx.py` | lets every script run from any working directory, not only from `analysis/`. |

Every estimator is seeded and no code path depends on `PYTHONHASHSEED`, so the
committed outputs reproduce byte for byte. The two documented exceptions —
`family_transfer.py` and `selector_variants.py`, whose
`HistGradientBoosting` fits on small per-device subsets move by up to ~1.2
points across scikit-learn releases — are described at the top of
`requirements.txt` and are reported separately by `verify_results.py`.
