# Device-Aware Post-Training Quantization for Consumer Android Devices

**When to share PTQ artifacts across accelerator families**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23607212.svg)](https://doi.org/10.5281/zenodo.23607212)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-blue.svg)](LICENSE-DATA)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Post-training quantization is usually applied once, and the resulting artifact is
then shipped to every Android device. This repository contains the analysis code
and the committed numerical results behind a study that treats the numeric format
as an **output** rather than an input: given an accuracy budget, which of five
TensorFlow Lite configurations (FP32, FP16, mixed precision, dynamic INT8, static
INT8) is the fastest *feasible* one on a given device, how much is lost by sharing
one artifact across devices, and can that choice be predicted without per-model
on-device benchmarking.

Every number and every figure in the accompanying manuscript is produced by a
script in `PTQ/` and stored under `PTQ/results/`. Nothing here requires access to
a phone: the measurements are already released (see **Data sources**).

---

## What is in this repository

```
PTQ/
├── run_all.sh              driver: figures | stage1 | stage2 | stage3 | all
├── requirements.txt        dependencies and the two reproducibility caveats
├── verify_results.py       re-runs the device-free pipeline and diffs it
├── grouped_cv.py           version-independent architecture-grouped CV split
├── _runctx.py              lets any script run from any working directory
├── make_all_figures.py     redraws every figure (calls the specialised drawers)
├── <39 analysis scripts>   selectors, robustness, transfer, flagship spot check
├── im100_acc/              per-mode ImageNet-100 converted-graph accuracies
└── results/
    ├── v83/                main campaign: labelled tables, logs, LaTeX fragments
    ├── v85/                mechanism, granularity and transfer analyses
    ├── flagship_2026/      2025–2026 spot check (+ its own README on the design)
    └── cv_folds.json       frozen cross-validation folds (see Reproducibility)
```

The manuscript sources (`main.tex`, `supplementary.tex`) and the rendered
`figures/` are **not** part of this repository; the scripts here regenerate the
figures into a `figures/` directory beside `PTQ/`.

---

## Quick start

```bash
git clone https://github.com/AB-Article/Post-Training-Quantization.git
cd Post-Training-Quantization/PTQ
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

bash run_all.sh figures      # redraws all 35 figures from committed results (~10 s)
python3 verify_results.py --quick   # re-runs the pipeline and diffs it (minutes)
```

`verify_results.py` copies the folder to a temporary directory, re-runs every
script that needs no external data, and compares what they produce with the
committed files. Nothing in your checkout is modified. Drop `--quick` for the
full run (~15 min) including the two long-running scripts. Exit status is 0 when
everything matches.

> If `./run_all.sh` reports *Permission denied*, the executable bit was lost in
> transit; use `bash run_all.sh ...` or run `chmod +x run_all.sh` once.

---

## Full rebuild

Figures and stage 3 run from the committed results alone. Stages 1 and 2 recompute
the numbers from the raw per-run records and need a local checkout of
[`nn-dataset`](https://github.com/ABrain-One/nn-dataset) (only
`ab/nn/stat/run/tflite/` and `_work/stats/` are used):

```bash
bash run_all.sh stage1 NND=/path/to/nn-dataset   # main campaign      (~25 min)
bash run_all.sh stage2 NND=/path/to/nn-dataset   # mechanism, transfer
bash run_all.sh stage3                           # flagship spot check (~5 s)
bash run_all.sh all    NND=/path/to/nn-dataset   # everything, then figures
```

Set `PYTHON=/path/to/venv/bin/python` if the system interpreter lacks the stack.

---

## Data sources

No measurements are taken by this code; all inputs are public and versioned.

| Input | Location | Used for |
|---|---|---|
| Per-run TFLite records, both corpora | [`ABrain-One/nn-dataset`](https://github.com/ABrain-One/nn-dataset), `ab/nn/stat/run/tflite/`, commit **`28092cc`** | all latency results |
| CIFAR-10 converted-graph accuracies | same repository, `_work/stats/all_models_accuracy_<mode>.json` | accuracy budget |
| ImageNet-100 converted-graph accuracies | [`huggingface.co/NN-Dataset/tflite`](https://huggingface.co/NN-Dataset) | accuracy budget (copied into `PTQ/im100_acc/`) |
| ImageNet-100 checkpoints, epoch 50 | [`huggingface.co/NN-Dataset`](https://huggingface.co/NN-Dataset), `checkpoints-imagenet100-epoch-50` | checkpoint-to-graph gap |
| Training records and transform sources | same repository, `ab/nn/stat/train/`, `ab/nn/transform/` | export-gap groups |

Pin the commit when re-running stage 1: the records are a living dataset, and the
committed outputs correspond to `28092cc`.

---

## Reproducibility

**What is guaranteed.** Every estimator is seeded (`random_state=0`), no code path
depends on `PYTHONHASHSEED`, and the cross-validation folds are frozen. Running
`verify_results.py` reproduces every committed output byte for byte, except as
noted below.

**The frozen folds matter.** `sklearn.model_selection.GroupKFold` changed its
group-to-fold assignment in scikit-learn 1.6 (an internal sort became stable).
Each architecture contributes two or three rows here, so the fold weights are
almost entirely ties and the two orderings produce different splits — silently
moving cross-validated numbers by up to 1.5 points. `grouped_cv.py` therefore
computes the split from the frozen map in `results/cv_folds.json` instead, which
makes the results independent of the installed library versions. Do not replace
it with the scikit-learn class.

**Two documented exceptions.** `family_transfer.py` and `selector_variants.py` fit
`HistGradientBoosting` models on small per-device subsets, and scikit-learn's
gradient-boosting internals have changed between releases. Their entries can move
by up to about 1.2 points across versions; no conclusion depends on that margin.
`verify_results.py` reports them separately rather than as failures.

**Environment used for the committed outputs.** Python 3.13, numpy 2.5, scipy
1.18, pandas 3.0, scikit-learn 1.9, matplotlib 3.11. The floors in
`requirements.txt` (Python 3.9+, scikit-learn 1.1+) are the minimum that runs.

**Statistical conventions.** Bracketed quantities are 95% percentile intervals
from a bootstrap of 2,000 resamples **clustered by architecture**, because each
architecture contributes one row per device. Shares in the flagship spot check use
Wilson score intervals; its mean regret uses a model-level bootstrap.

---

## Scope of the evidence

Three devices carry the main campaign, with 43,519 valid latency measurements:

| Segment | Device | SoC | RAM | Android |
|---|---|---|---|---|
| Budget phone | Redmi 13 4G | Helio G91 Ultra (MT6769) | 6 GB | 16 |
| Legacy flagship | Pixel 4 | Snapdragon 855 (SM8150) | 6 GB | 13 |
| Flagship tablet | Galaxy Tab S7+ | Snapdragon 865+ (SM8250) | 8 GB | 13 |

Selected findings are then checked on a Galaxy S25, a Galaxy S26 and a Dimensity
9400 tablet, using a subset of 150 models frozen before those measurements. The
S25 and S26 numbers come from a second measurement stack, so cross-vendor
comparisons involving them are reported descriptively — see
`PTQ/results/flagship_2026/README.md` for the sampling design, the coverage, the
known data defects and the one open measurement.

Evaluated pools: 3,539 CIFAR-10 model–device pairs from 1,185 architectures, and
5,047 ImageNet-100 pairs from 1,692 architectures, drawn from one public
collection of trained networks (named CNNs, retrieval-constructed networks,
LLM-generated architectures and mixture-of-experts variants). A mode counts as
feasible when its converted graph keeps at least 99% of the accuracy of the
matched FP32 TensorFlow Lite graph (τ = 0.99); `misc_checks.py` and the robustness
scripts report the sensitivity to that threshold.

Results are specific to TensorFlow Lite, to image classification, and to these
six devices; generalisation beyond them is not established here.

---

## Citation

If you use this code or the committed results, please cite the archived release:

```bibtex
@software{hussain_ptq_artifacts,
  author  = {Hussain, Muhammad Ahsan and Timofte, Radu and Ignatov, Dmitry},
  title   = {Device-Aware Post-Training Quantization for Consumer Android
             Devices: analysis code and results},
  year    = {2026},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://github.com/AB-Article/Post-Training-Quantization}
}
```

The DOI above resolves to the concept record, which always points at the newest
version; each release also receives its own version DOI.

Please also cite the underlying measurement dataset,
[`ABrain-One/nn-dataset`](https://github.com/ABrain-One/nn-dataset), and the
accompanying article once it appears.

---

## License

Code in `PTQ/` is released under the MIT License (`LICENSE`). The committed
results under `PTQ/results/` and `PTQ/im100_acc/` are released under CC BY 4.0
(`LICENSE-DATA`). The upstream records carry their own terms; see
[`ABrain-One/nn-dataset`](https://github.com/ABrain-One/nn-dataset).
