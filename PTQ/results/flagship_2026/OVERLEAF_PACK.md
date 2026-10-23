# Overleaf pack — what to upload

The manuscript now `\input`s the flagship tables directly, so nothing needs to
be pasted by hand. Upload the whole `analysis/` tree and the two generated
table files will be picked up.

## Required by the build
| File | Used by |
|------|---------|
| `flagship_corrected_tables.tex` | `Paper.tex` → Tables V–VI |
| `flagship_corrected_supp.tex` | `Supplementary.tex` → Tables S33–S37 |

Both are generated; do not edit them by hand. Regenerate with:
```bash
python3 analysis/flagship_corrected.py
```
It needs only this folder's CSVs and `analysis/im100_acc/` — no device access.
Table order in `flagship_corrected_supp.tex` is cited as S33–S37 from
`Paper.tex`, so keep it stable.

## Data (supplement / plots)
| File | Use |
|------|-----|
| `winners_constrained.csv` | per-model pure vs. constrained winner + feasible set |
| `winners_by_device.csv` | per-model × device latencies and pure-latency winner |
| `s25_s26_winner_compare.csv` | S25 vs S26 match / mismatch |
| `flagship_vs_old_usb.csv` | freeze-150 × new + old USB winners |
| `mtk40_backend_table.csv` | MediaTek CPU/GPU/NPU wide |
| `mode_agreement_summary.csv` | pure-latency headline agreement |
| `unified_latencies_long.csv` | full long-form join (large) |
| `flagship_corrected.{txt,json}` | every manuscript number, with provenance |

## Raw sources
- `raw/s25_full_rows.csv`, `raw/s26_full_rows.csv` — AI Hub jobs
- `raw/mediatek_rows.csv` — Tab S11 `benchmark_model` sweep
- `raw/harness_20_paper_vs_aihub.csv` — **same-device, two-stack control**;
  this is the file that bounds the measurement-stack confound and should stay
  in the release
- `imagenet100_freeze_150.csv` — the frozen model list

## Superseded
`overleaf_tables.tex` and `flagship_spotcheck_tables.tex` are the first-pass
pure-latency tables. They are kept for traceability only and are no longer
referenced by the manuscript — do not re-add them, or the paper will report
pure-latency winners that conflict with its own abstract. See `README.md`
for why.
