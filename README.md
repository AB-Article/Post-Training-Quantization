# Device-Aware Post-Training Quantization for Consumer Android Devices: When to Share PTQ Artifacts Across Accelerator Families

Reproduction package for the manuscript submitted to *IEEE Transactions on
Consumer Electronics*. Every number in `main.tex` and `supplementary.tex` is
produced by a script in `analysis/` and stored under `analysis/results/`.

Contents: `main.tex` (main manuscript, 8 pages), `supplementary.tex`
(23 pages), `CoverLetter.tex`, `bibmain.bib`, `IEEEtran.cls`, `IEEEtran.bst`,
`figures/`, and `analysis/` (scripts and stored outputs).

## One-command reproduction

From `analysis/`, after installing `analysis/requirements.txt`:

```bash
# Figures that have generators here (uses committed results/ only; no devices)
./run_all.sh figures

# Full numeric rebuild (needs a local nn-dataset checkout)
./run_all.sh all NND=/path/to/nn-dataset
```

`run_all.sh` also accepts `stage1`, `stage2`, and `stage3` individually.
Set `PYTHON=/path/to/venv/bin/python` if the system Python lacks the stack.

## Build

```bash
pdflatex main         && bibtex main         && pdflatex main
pdflatex supplementary && bibtex supplementary && pdflatex supplementary
pdflatex main         && pdflatex main
pdflatex supplementary && pdflatex supplementary
```

**The two documents cross-reference each other, so build them in that
alternating order.** Each imports the other's labels through the `xr` package:
the supplement reads `main.aux` under an `M-` prefix for the main paper's
section and table numbers, and the main paper reads `supplementary.aux` under
an `S-` prefix for every `S`-numbered table and figure it quotes. No
cross-document number is typed by hand on either side, so renumbering in
either document propagates to the other. Both `.aux` files are shipped, so a
standalone build of either document resolves the other's numbers on the first
pass; a missing `.aux` produces a loud build warning rather than silent `??`.
If a cross-document number looks stale, run the four-line sequence above
again: one extra alternating pass is enough to converge.

The cover letter is standalone and needs no bibliography:

```bash
pdflatex CoverLetter && pdflatex CoverLetter
```

## Inputs

| Data | Where | Used for |
|---|---|---|
| Per-run TFLite records (both corpora) | `github.com/ABrain-One/nn-dataset`, `ab/nn/stat/run/tflite/` (computed at commit `28092cc`) | all latency results |
| CIFAR-10 converted accuracies | same repo, `_work/stats/all_models_accuracy_<mode>.json` | constrained labels |
| ImageNet-100 converted accuracies | `huggingface.co/NN-Dataset/tflite/<mode>/img-classification_imagenet100_acc/all_models.json`, copied here as `im100_acc/all_models_accuracy_<mode>.json` | constrained labels |
| CIFAR-10 training records + transform sources | same repo, `ab/nn/stat/train/` and `ab/nn/transform/` | export-gap groups (`export_gap.py`) |
| ImageNet-100 checkpoint accuracies (epoch 50) | `huggingface.co/NN-Dataset/checkpoints-imagenet100-epoch-50/all_models.json`, copied as `im100_acc/checkpoint_epoch50_all_models.json` | checkpoint-to-graph gap |

`NND` below is a local checkout of nn-dataset; only `ab/nn/stat/run/tflite/`
and `_work/stats/` are needed.

## Reproducing the results

Prefer `./run_all.sh …` above. The same stages are spelled out below for
reviewers who want to run scripts one by one.

**All manuscript figures:** `python make_all_figures.py` (or `./run_all.sh figures`).
That driver calls the specialized drawers (`make_fig_v85.py`,
`make_fig_flagship.py`, `make_fig_v83.py`, `make_fig_effect_sizes.py`,
`make_fig_supp_fixed.py`) and also builds the supplementary panels that used
to lack a generator (`fig01`, `fig02`, `fig06`, `fig13`, `fig18`,
`fig_im100_policy_match`, `fig_im100_export_vs_quant_loss`). Export-gap
histograms read `results/v83/export_gap_values.json` and
`im100_checkpoint_gap_values.json` (committed with the package).

All commands run from `analysis/`. Stage 1 takes roughly 25 min of CPU in
total; in stage 2, `selector_variants.py` and `family_transfer.py` take about
45 min each on two cores and the rest about 15 min.

### Stage 1 — main campaign

```bash
python load_runs.py --data-root NND --out results/v83/runs_long.csv
python predictor.py --data-root NND --out-csv results/v83/dataset_full_cifar10.csv
python predictor.py --data-root NND --run-subdir imagenet100 --acc-dir im100_acc --out-csv results/v83/dataset_full_imagenet100.csv
# CIFAR-10: one call per section (file names v83_results_<S>.json, logs log_<S>.txt)
for S in A B C D; do python v83_analysis.py --data-root NND --sections $S > results/v83/log_$S.txt; done
# ImageNet-100 (constrained)
for S in A B; do python v83_analysis.py --data-root NND --acc-dir im100_acc --csv results/v83/dataset_full_imagenet100.csv --sections $S --tag _im100 > results/v83/log_${S}_im100.txt; done
python v83_analysis.py --data-root NND --acc-dir im100_acc --csv results/v83/dataset_full_imagenet100.csv --sections C --corpus imagenet100 --tag _im100 > results/v83/log_C_im100.txt
python robustness.py --data-root NND                                         > results/v83/robustness_root.txt
python robustness_extra.py --data-root NND --tag cifar10_root --out-dir results/v83
python robustness_extra.py --data-root NND --run-subdir imagenet100 --acc-dir im100_acc --tag imagenet100_root --out-dir results/v83
python robustness_im100.py --data-root NND --acc-dir im100_acc               > results/v83/robustness_im100.txt
python guideline.py                                                          > results/v83/guideline.txt
python rules_eval.py                                                         > results/v83/rules_eval.txt
python one_or_many.py                                                        # writes one_or_many.{txt,json}
python accuracy_released.py NND im100_acc im100_acc/checkpoint_epoch50_all_models.json > results/v83/accuracy_released.txt
python export_gap.py NND                                                     > results/v85/export_gap.txt
python dcor_released.py                                                      > results/v83/dcor_released.txt
python misc_checks.py NND                                                    # writes misc_checks.txt
python paired_tests.py                                                       # writes results/v83/paired_tests.{txt,json}; needs only runs_long.csv
python mtk_profile.py                                                        # Dimensity 9400 per-backend profile + backend-removal control (Fig. 1a,b)
python calibrate_new_devices.py                                              # selector transfer to S25/S26/Tab S11 (Table S40); ~2 min
python make_supp_extra.py                                                    # Tables S38 (permutation importance), S39 (winner latency)
python make_fig_effect_sizes.py                                              # ../figures/fig15_effect_sizes.png, fig_im100_effect_sizes.png (Figs. S10, S18)
python make_fig_flagship.py                                                  # ../figures/fig_flagship.png (Fig. 3 of the main paper)
python predictor.py --csv results/v83/dataset_full_cifar10.csv --seeds 1     | tail -3 > results/v83/permimp_cifar10.txt
python predictor.py --csv results/v83/dataset_full_imagenet100.csv --seeds 1 | tail -3 > results/v83/permimp_imagenet100.txt
python make_fig_v83.py                                                       # ../figures/fig_v83_device_and_regret.png
python make_supp_tables.py                                                   # supp_s6.tex, supp_new.tex, supp_im100.tex
python headline_ci.py       > results/v85/headline_ci.txt                    # 95% CIs quoted in the main paper (also .json); ~3 min
```

`fig2b_model_bands.txt` holds the numbers plotted in Fig. 2(b), written by
`make_fig_v83.py`. `v83_analysis.py` writes
`results/v83/v83_results_<section><tag>.json` and `picks_A<tag>.csv`.

### Stage 2 — mechanism, granularity and transfer

Run after stage 1; needs `results/v83/runs_long.csv` and
`results/v83/dataset_full_*.csv`. No new measurements: these read the same
released records.

```bash
python load_runs_extra.py --data-root NND --out results/v85/runs_extra.csv   # per-backend min durations + timestamps
python accel_profile.py     > results/v85/accel_profile.txt
python virtual_device.py    > results/v85/virtual_device.txt
python overhead_model.py    > results/v85/overhead_model.txt
python k_artifacts.py       > results/v85/k_artifacts.txt
python selector_variants.py > results/v85/selector_variants.txt    # long
python family_transfer.py   > results/v85/family_transfer.txt      # long
python cross_task.py        > results/v85/cross_task.txt
python named_models.py      > results/v85/named_models.txt
python robustness_v85.py    > results/v85/robustness_v85.txt
python make_tables_v85.py; python make_supp_v85.py; python make_fig_v85.py
```

### Stage 3 — 2025–2026 flagship spot check

```bash
python3 flagship_corrected.py        # tau = 1% of matched FP32; ~5 s, no device needed
```

`flagship_corrected.py` reads only the CSVs in `results/flagship_2026/` and the
per-mode ImageNet-100 accuracies in `im100_acc/`. It reports the
accuracy-constrained winner (the paper's estimand), chance-corrected agreement,
regret, tie tolerance, MediaTek delegate coverage, and the measurement-stack
control. Selection is deterministic: exact latency ties resolve toward the
least quantized mode, and no code path iterates a Python `set`, so the output
does not depend on `PYTHONHASHSEED`.

To rebuild the measurement joins themselves, run `python3
analyze_flagship_ptq.py` from a local `flagship_ptq_bench` checkout and re-copy
the CSVs into `results/flagship_2026/`. See
`results/flagship_2026/README.md` for the sampling design, coverage, known data
defects, and the one open measurement needed to make the cross-vendor
comparison identifiable.

## Script → manuscript map

Main-paper tables are numbered as in the submitted PDF: I positioning,
II devices, III PTQ accuracy loss, IV artifact-sharing cost, V rules of thumb,
VI selector transfer, VII benchmarking ladder.

| Manuscript item | Script | Result file(s) |
|---|---|---|
| Table III (PTQ loss, feasibility), checkpoint gap | `accuracy_released.py` | `accuracy_released.txt` |
| Table III lower panel (export gap by transform), Fig. S25 caption | `export_gap.py` | `results/v85/export_gap.txt` |
| Table IV (artifact-sharing cost) | `one_or_many.py`, `k_artifacts.py` | `one_or_many.{txt,json}`, `results/v85/k_artifacts.{txt,json}` |
| Table V (rules), Table S20 (trees) | `guideline.py`, `rules_eval.py` | `guideline.{txt,json}`, `rules_eval.txt` |
| Table VI (selector transfer), Table S29 | `family_transfer.py` | `results/v85/family_transfer.{txt,json}` |
| Table VII (ladder), selector text, Tables S12–S14, S17–S19 | `v83_analysis.py` §A–B | `log_A*.txt`, `log_B*.txt`, `v83_results_A*.json`, `v83_results_B*.json`, `picks_A*.csv` |
| Device dependence, ties, fixed backend (Section V-E; Tables S21, S27, S32) | `robustness.py`, `robustness_extra.py`, `robustness_v85.py`, `v83_analysis.py` §C | `robustness_root.txt`, `robustness_extra_cifar10_root.*`, `results/v85/robustness_v85.{txt,json}`, `v83_results_C.json` |
| ImageNet-100 robustness numbers | `robustness_im100.py`, `robustness_extra.py`, `v83_analysis.py` §C | `robustness_im100.txt`, `robustness_extra_imagenet100_root.*`, `v83_results_C_im100.json` |
| `tab:latonly` (latency-only selection) | `v83_analysis.py` §D | `log_D.txt`, `v83_results_D.json` |
| 95% confidence intervals quoted in brackets | `headline_ci.py` | `results/v85/headline_ci.{txt,json}` |
| Fig. 1(a), Table S22 | `accel_profile.py` | `results/v85/accel_profile.{txt,json}` |
| Fig. 1(b), Table S30 (backend-removal control) | `virtual_device.py`, `mtk_profile.py` | `results/v85/virtual_device.{txt,json}` |
| Fig. 1(c), Table S23 (two artifacts) | `k_artifacts.py` | `results/v85/k_artifacts.{txt,json}` |
| Fig. 2 (win rates, fastest backend, regret) | `make_fig_v83.py` | `../figures/fig_v83_device_and_regret.png`, `fig2b_model_bands.txt` |
| Fig. 3 (flagship spot check) | `make_fig_flagship.py` | `../figures/fig_flagship.png` |
| Absolute-regret quartiles, Table S24 | `overhead_model.py` | `results/v85/overhead_model.{txt,json}` |
| Latency-regression baseline, per-family selector, budget violations, Table S28 | `selector_variants.py` | `results/v85/selector_variants.{txt,json}` |
| Cross-task paragraph, Table S31 | `cross_task.py` | `results/v85/cross_task.{txt,json}` |
| Named architectures, Tables S25–S26 | `named_models.py` | `results/v85/named_models.{txt,json}`, `tab_named*.tex` |
| Paired tests and effect sizes (Section V-B; Tables S4, S11) | `paired_tests.py` | `paired_tests.{txt,json}` |
| Distance correlations | `dcor_released.py` | `dcor_released.txt` |
| Permutation importance (Table S38) | `predictor.py --csv … --seeds 1` | `permimp_{cifar10,imagenet100}.txt` |
| Misc. facts (identical mixed/INT8 accuracy, max input size, size–latency ρ, family shares) | `misc_checks.py` | `misc_checks.txt` |
| Supplementary Table S6 | `robustness_extra.py` | `supp_s6.tex` |
| LaTeX fragments `\input` by the supplement (Tables S6, S12–S20) | `make_supp_tables.py` | `supp_s6.tex`, `supp_new.tex`, `supp_im100.tex` |
| Flagship text and Tables S33–S37 | `flagship_corrected.py` | `results/flagship_2026/flagship_corrected.{txt,json}`, `flagship_corrected_supp.tex`, `winners_constrained.csv` |
| Selector transfer to the 2025–2026 devices (Table S40) | `calibrate_new_devices.py` | `results/flagship_2026/calibration_new_devices.{txt,json}` |

## Statistical conventions

Bracketed quantities in the main paper are 95% percentile intervals.

* **Main campaign** — architecture-clustered bootstrap, 2,000 resamples of
  architectures with replacement, 2.5/97.5 percentiles. Clustering is by
  architecture because each architecture contributes one row per device.
  Implemented in `headline_ci.py`, `predictor_ci.py`, `robustness.py` and
  `k_artifacts.py`.
* **Flagship spot check** — Wilson score intervals for shares, since each of
  the 150 frozen models contributes one row, and a model-level bootstrap for
  mean regret. Implemented in `headline_ci.py`.
* **Paired tests** (`paired_tests.py`) use fastest-backend latency over valid
  runs with no 0.1 ms floor, restricted to model–device pairs with all five
  modes measured and with no accuracy constraint. The `p < 10⁻³` claim in
  Section V-B is scoped to CIFAR-10; on ImageNet-100 several modes are
  indistinguishable from FP32 (max p = 0.57).

## Notes for anyone rebuilding the documents

* **`tab:ties` (device dependence, currently S31) is hand-maintained.** It
  lives in `Supplementary.tex` directly, immediately after the `\input` of
  `results/v85/supp_v85.tex`. Do not move it into that generated file:
  re-running `make_supp_v85.py` would delete it. Its number is no longer
  load-bearing, since both documents reference it by label.
* **The three Section S-XI tables are written inline in `Supplementary.tex`**
  (`tab:whopays`, `tab:robshipping`, `tab:named-digest` — currently S40–S42),
  not emitted by a script, although their values come from the committed
  outputs of `one_or_many.py`, `robustness_v85.py` and `named_models.py`.
  Refer to them by label, never by number: the labels no longer carry a
  hard-coded `S41`/`S42`/`S43`, so they survive renumbering.
* **The supplement refers to main-paper tables through `xr` labels**
  (`M-tab:devices`, `M-tab:acc`, `M-tab:oneormany`, `M-tab:rules`,
  `M-tab:transfer`, `M-tab:ladder`), never by hard-coded number, so main-paper
  renumbering propagates automatically.
* **The supplement is table-first.** A figure is included only where no table
  in the same document already states what it shows, so `figures/` now holds
  22 plots that appear in neither document and are kept only for
  traceability: the CIFAR-10 descriptive set
  (`fig03_accuracy_vs_latency_scatter.png`,
  `fig05_accuracy_drop_vs_fp32.png`, `fig07_model_family_comparison.png`,
  `fig08_best_mode_overall.png`, `fig09_best_mode_by_device.png`,
  `fig11_size_vs_latency.png`, `fig12_latency_percentiles.png`,
  `fig14_input_resolution.png`, `fig15_effect_sizes.png`,
  `fig17_replication.png`) and the ImageNet-100 mirror set (every
  `fig_im100_*.png` except `fig_im100_policy_match.png`,
  `fig_im100_dcor_best_mode.png` and `fig_im100_export_vs_quant_loss.png`).
  Every quantity they plotted is tabulated in the supplement, and the
  generating scripts are unchanged and still write them, so re-including one
  needs only a `\begin{figure}` block. The `*_orig.jpg` files are the
  uncropped author photographs.
* **Superseded flagship tables.** `overleaf_tables.tex` and
  `flagship_spotcheck_tables.tex` hold pure-latency results that the manuscript
  no longer references; `flagship_corrected_tables.tex` is a comment-only stub.
* The `v82` files in `analysis/results/` and `dataset_final.csv` are retained
  for traceability only and feed no current number.

## Cover letter

`CoverLetter.tex` is the letter to submit. It compiles to a single page
covering the fit with the journal, what is new, corresponding-author contact
and ORCIDs, the declarations, and the suggested reviewers.
