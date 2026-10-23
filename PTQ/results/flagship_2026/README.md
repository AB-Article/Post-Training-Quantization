# Flagship 2025–2026 PTQ spot-check (ImageNet-100 freeze-150)

Measurements supporting the main-paper subsection "2025–2026 Flagship
Spot-Check" and supplementary Section "2025–2026 Flagship Spot-Check Data":
latencies on Galaxy S25 / S26 (Qualcomm AI Hub, Snapdragon 8 Elite / Elite
Gen 5) and Galaxy Tab S11 (Dimensity 9400 / MT6991) via on-device
`benchmark_model`, joined against the paper's USB Pixel 4 / Tab S7+ /
Redmi 13 records.

## Two analysis layers

| Script | Role |
|---|---|
| `analyze_flagship_ptq.py` | builds the measurement joins (`winners_by_device.csv`, `unified_latencies_long.csv`, the raw merges). Regenerate from a `flagship_ptq_bench` checkout. |
| `../../flagship_corrected.py` | **produces every number quoted in the manuscript.** Runs off the CSVs in this folder plus `../../im100_acc/`; no device access needed. |

```bash
python3 analysis/flagship_corrected.py            # tau = 1% of matched FP32
```
Outputs `flagship_corrected.{txt,json}`, `flagship_corrected_supp.tex`
(supplementary Tables S33–S37), `winners_constrained.csv`, and
`flagship_corrected_tables.tex`, which is now an intentionally empty
comment-only stub: no flagship table is reproduced in the main paper any more,
and the stack-matched numbers are stated in the running text of the flagship
subsection instead.

`flagship_corrected.py` exists because the first-pass analysis differed from
the manuscript in four ways that a reviewer would reasonably challenge:

1. **Estimand.** It reported *pure-latency* winners, while the paper's
   estimand throughout is the fastest mode within 1% of the matched converted
   FP32 accuracy. Per-mode ImageNet-100 accuracies are available for all 150
   freeze models, so the constrained winner is now primary. This matters: INT8
   takes 91.3% of pure-latency wins on the S25 but only 48.7% under the
   budget, because INT8 is accuracy-feasible for just 79 of 150 models.
   Reported pure-latency-only, the subsection appeared to contradict the
   paper's own abstract ("INT8 fastest feasible for only 6–11% of pairs").
2. **Chance-corrected agreement.** Raw agreement is inflated when one mode
   dominates. S25-vs-S26 pure-latency agreement of 94.7% sits against 85.5%
   chance agreement (κ = 0.633); the constrained comparison is 74.7% against
   38.4% chance (κ = 0.589). κ and the chance rate are now reported.
3. **Regret and tie tolerance.** The paper treats regret as primary and
   winner rates as fragile. Both are now computed for the flagships on the
   same footing as the main corpora.
4. **Measurement-stack confound.** The S25-vs-Tab S11 comparison crosses two
   runtimes. `raw/harness_20_paper_vs_aihub.csv` benchmarks the *same*
   physical devices through both stacks and shows that the stack alone
   accounts for a disagreement as large as the "cross-vendor" effect, so the
   cross-vendor claim is reported descriptively and is not identifiable from
   these data. See "Open measurement" below.

Exact latency ties are resolved deterministically toward the least quantized
mode (FP32 ≻ FP16 ≻ dynamic ≻ INT8 ≻ mixed), so a more aggressive mode is
selected only when strictly faster. The rule is load bearing: AI Hub returns
identical FP32 and FP16 latency for 13 of 150 models on the S25 and 5 on the
S26. Selection never iterates a Python `set`, so results are independent of
`PYTHONHASHSEED`.

## Files
| File | Description |
|------|-------------|
| `imagenet100_freeze_150.csv` | frozen 150-model list — do not change |
| `mediatek_40_models.csv` | 40-model MediaTek subset of the freeze list |
| `winners_by_device.csv` | per-model × device latencies and pure-latency winner |
| `winners_constrained.csv` | pure vs. accuracy-constrained winner and feasible set |
| `s25_s26_winner_compare.csv` | S25 vs S26 agreement |
| `flagship_vs_old_usb.csv` | freeze-150 × new flagships + old USB winners |
| `mtk40_backend_table.csv` | MediaTek CPU/GPU/NPU (ms) |
| `mode_agreement_summary.csv` | pure-latency headline agreement |
| `unified_latencies_long.csv` | long-form join |
| `flagship_corrected.{txt,json}` | all manuscript numbers |
| `flagship_corrected_tables.tex` | comment-only stub; kept so an older revision that still `\input`s it will build |
| `flagship_corrected_supp.tex` | supplementary tables S33–S37 |
| `overleaf_tables.tex`, `flagship_spotcheck_tables.tex` | superseded first-pass tables, kept for traceability; **not** used by the manuscript |
| `RESEARCH_BRIEF.md` | findings summary |
| `raw/` | source measurement CSVs |

## Headline (accuracy-constrained, τ = 1%)
- **S25 → S26 regret: median 0.00%, mean 0.22%** (reverse 0.34%, P90 ≤ 1.1%);
  at a 2% tie tolerance the winner sets always intersect. One artifact per
  Snapdragon generation is close to free.
- **INT8 win share on S25: 91.3% by pure latency → 48.7% under the budget.**
  On these flagships the binding constraint is accuracy, not latency.
  Dynamic INT8 — the most frequent optimum on both full corpora — never wins
  on the AI Hub devices (0/150).
- **S25 vs Tab S11: 17.5% agreement (κ = 0.061)** — descriptive only; see the
  stack confound below.
- **MediaTek NPU accepts the calibrated integer graphs for 1 of 40 models**
  (FP16 34, dynamic 31), so integer modes there execute on CPU/GPU. This is
  the delegate-coverage mechanism behind the cross-vendor divergence.

## Sampling
The 150-model list was drawn from the released ImageNet-100 pool and frozen
before any 2025–2026 measurement, stratified by the model's latency band on
the paper devices (36 below 2 ms, 37 at 2–5 ms, 39 at 5–10 ms, 38 above
10 ms) and comprising 9 named, 76 retrieval-constructed and 65 LLM-generated
architectures. The MediaTek subset is 40 models of the same list under the
same stratification.

## Coverage
- AI Hub S25 / S26: 743/750 jobs each (7 missing TFLite / HF 404).
- MediaTek Tab S11: 774 logged invocations over 40 models × 5 modes × 3
  backends — 490 ok, 210 delegate/operator failures, 74 push failures; 38 of
  40 models yield a valid measurement in all five modes. Failures are
  retained as logged fails, not dropped.

## Open measurement
The one experiment that would make the cross-vendor claim identifiable is to
benchmark a Snapdragon flagship through the **same on-device
`benchmark_model` harness** used for Tab S11 (40 models × 5 modes ×
CPU/GPU/NPU is sufficient). Until then the Snapdragon–MediaTek comparison is
reported descriptively, with the same-device control as its bound.

## Known data defects
- `raw/mediatek_rows.csv`: as first collected, the `runtime_version` column
  held the usage error of a `benchmark_model` call issued without `--graph`
  instead of a version string. The field is published as `unrecorded`; the
  `benchmark_model` version for the Tab S11 sweep is unknown. Latency columns
  were unaffected.
- The released Redmi 13 records contain one model with a 0.0011 ms minimum
  duration, below any plausible single-inference time. It is treated as an
  invalid measurement wherever winner latencies are summarized.
- AI Hub rows use `--compute_unit all` and expose one fused latency, so the
  per-backend accelerator profile of the main paper cannot be re-derived on
  the S25/S26. Absolute milliseconds across stacks are roughly unbiased
  (median USB/AI Hub ratio 0.96, log-latency r = 0.775); it is the winner
  *modes* that do not transfer.
