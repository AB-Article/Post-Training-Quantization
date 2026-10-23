# Flagship PTQ add-on — research brief

All numbers from `python3 analysis/flagship_corrected.py` (τ = 1% of the
matched converted FP32 accuracy, the paper's estimand throughout). Full log in
`flagship_corrected.txt`.

**Question.** Does "ship one quantized artifact per accelerator family" hold on
2025–2026 flagships?

**Answer.** Within an accelerator family, yes, and more cleanly than on the
older hardware. Across SoC vendors, these data cannot answer it: that
comparison crosses measurement stacks and is confounded.

---

## 1. Within the Snapdragon family one artifact is nearly free

| Ship → target | N | Median regret | Mean | P90 |
|---|--:|--:|--:|--:|
| S25 → S26 | 150 | 0.00% | 0.22% | 0.72% |
| S26 → S25 | 150 | 0.00% | 0.34% | 1.09% |

At a 2% tie tolerance the two devices' constrained winner sets always
intersect (100%, N=150). Exact-match agreement is 74.7% (chance 38.4%,
κ = 0.589), lower than the regret figure suggests because AI Hub returns
identical FP32 and FP16 latency for 13 of 150 models. Report regret; the
winner rate is the fragile measure, exactly as in the main paper's robustness
section.

Pure-latency agreement is 94.7% but against 85.5% chance agreement
(κ = 0.633) — quote κ, not the raw percentage, since INT8 takes 91% of
pure-latency wins and inflates chance.

## 2. The binding constraint shifts from latency to accuracy

Winner share on S25 (%), pure latency → under the 1% budget:

| Mode | pure | constrained |
|---|--:|--:|
| FP32 | 4.0 | 34.7 |
| FP16 | 0.7 | 14.0 |
| dynamic | 0.0 | 0.0 |
| INT8 | 91.3 | 48.7 |
| mixed | 4.0 | 2.7 |

INT8 is accuracy-feasible for only **79 of 150** models (mixed 77; FP32/FP16
150; dynamic 148). So faster integer hardware does not make INT8 the default —
it moves the bottleneck to whether the calibrated graph is accurate enough to
ship. This is consistent with, not contrary to, the paper's headline that INT8
is the fastest *feasible* mode for only 6–11% of pairs on the older devices.

Dynamic INT8, the most frequent optimum on both full corpora (35–42%), **never
wins** on the AI Hub devices (0/150), most likely because the fused
`compute_unit=all` path does not place dynamic-range graphs on the HTP.

## 3. The cross-vendor comparison is confounded — do not claim it

`raw/harness_20_paper_vs_aihub.csv` benchmarks the **same physical devices**
(Pixel 4, Tab S7+) through both the paper's USB harness and AI Hub:

| Control | Value |
|---|--:|
| Winner-mode agreement across stacks, same device | **19.4%** (7/36) |
| Median within-model mode-ranking ρ | 0.50 |
| Median USB / AI Hub latency ratio | 0.96 |
| Log-latency Pearson r | 0.775 |

The stack alone produces a disagreement as large as the measured S25-vs-Tab
S11 agreement of **17.5%** (κ = 0.061). The cross-vendor figure therefore
cannot be attributed to the SoC vendor.

Note the direction: absolute milliseconds are roughly **unbiased** across
stacks; it is the winner **modes** that fail to transfer. The usual caveat
("compare modes, not ms") is backwards here.

**To make the claim identifiable:** benchmark a Snapdragon flagship through the
same on-device `benchmark_model` harness used for Tab S11 — 40 models × 5
modes × CPU/GPU/NPU is enough. That single run converts this add-on from a
liability into the paper's strongest external-validity evidence.

## 4. What *is* identifiable: delegate coverage

MediaTek Dimensity 9400, models with a valid measurement (of 40):

| Mode | CPU | GPU | NPU |
|---|--:|--:|--:|
| FP32 | 40 | 39 | 34 |
| FP16 | 40 | 39 | 34 |
| dynamic | 40 | 39 | 31 |
| INT8 | 38 | 38 | **1** |
| mixed | 38 | 38 | **1** |

The NPU path accepts the calibrated integer graphs for one model only, so on
this device integer modes execute on CPU or GPU and cannot win. Where an
accelerator does not cover a mode, that mode loses regardless of vendor — the
same mechanism as the paper's accelerator-profile section, and a result that
needs no cross-stack comparison.

## 5. Practical reading

Constrained-winner latency (median): S25 0.21 ms, S26 0.16 ms, Tab S11
1.75 ms, USB Pixel 4 3.87 ms. The whole flagship population sits inside the
paper's sub-2 ms band, but within-family absolute regret is below 0.001 ms. On
current flagships the family split is a throughput and energy argument, not a
frame-deadline one. Say so explicitly, or a reviewer will note that tens of
microseconds cannot matter at 60 fps.

## Tie handling

Exact ties resolve toward the least quantized mode (FP32 ≻ FP16 ≻ dynamic ≻
INT8 ≻ mixed), so a more aggressive mode is picked only when strictly faster.
Tie counts under the budget: S25 13/150, S26 5/150, Redmi 13 12/150, zero on
Pixel 4 / Tab S7+ / Tab S11. Selection is deterministic and independent of
`PYTHONHASHSEED`.

## Where these numbers land in the manuscript

- Main paper, subsection "2025–2026 Flagship Spot-Check" → Tables V–VI from
  `flagship_corrected_tables.tex`.
- Supplement, "2025–2026 Flagship Spot-Check Data" → Tables S33–S37 from
  `flagship_corrected_supp.tex`.
- Abstract, Conclusion and Limitations carry the within-family regret, the
  accuracy-constraint shift, and the non-identifiability of the cross-vendor
  cut.
