# red-herring

[![CI](https://github.com/woodRock/red-herring/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/woodRock/red-herring/actions/workflows/ci.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Coverage ≥ 90%](https://img.shields.io/badge/coverage-%E2%89%A590%25-brightgreen.svg)](.github/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Open-set detection and calibrated uncertainty for GC-MS fish species authentication.

Real fraud often involves a species that isn't in the training set, and a closed-set
classifier will confidently mislabel it. This repo measures that failure and evaluates
open-set recognition and conformal prediction as fixes with guarantees a regulator can
act on.

## Data

> **The data is private.** The code is public on GitHub, but the GC-MS dataset is not
> included in this repository and is not publicly available. `data/` is git-ignored.

103 GC-MS runs (Xcalibur text dumps) of four NZ species — Snapper (SNA), Gurnard (GUR),
Tarakihi (TAR), Blue Cod (BCO) — across body parts, in `data/raw-subsetted-CVSs/`.
`red_herring.data.load_raw` turns each run into a `[200 RT bins, 1000 m/z]` chromatogram
following chroma-dcnn: highest-TIC scan per RT window, sqrt + L2 per bin. Windows are
defined on a shared 5–45 min range because runs start at either 3 or 5 min.

## Protocol

For each species held out as the unseen "fraud" species (`--n-unknown 2` holds out pairs):

1. Stratified group K-fold over the known species. Groups are individual + tissue, so
   technical replicates never straddle train and test (`--groups none` gives the
   optimistic, leaky comparison).
2. A quarter of each training fold (also grouped) is held out as the conformal
   calibration set.
3. Evaluate on the known test fold plus every sample of the held-out species.

## Methods

| Models | |
|---|---|
| `lr` | standardise → PCA(20) → logistic regression, on the mean RT projection |
| `svm` | RBF SVM with Platt probabilities |
| `rf` | random forest, 500 trees |
| `cnn` | ChromatogramCNN (ported from chroma-dcnn) on the full 2D chromatogram |

| Open-set scores | |
|---|---|
| `msp` | 1 − max softmax probability |
| `energy` | −logsumexp(logits) |
| `knn` | distance to the 3rd-nearest training embedding (L2-normalised) |
| `mahalanobis` | min class-conditional Mahalanobis distance, Ledoit–Wolf tied covariance |

| Conformal | |
|---|---|
| `cp_lac`, `cp_aps` (± `_mondrian`) | prediction sets with marginal (or per-species) coverage ≥ 1−α; an empty set flags a novel species |
| `cnd_<score>` | conformal novelty p-values: false-alarm rate on known species ≤ α |

## Run

```bash
uv run red-herring                      # all models, 3 seeds × 5 folds, α ∈ {0.05, 0.1, 0.2}
uv run red-herring --models lr cnn --seeds 1 --epochs 20 --out results/quick
```

First run parses the raw text (~20 s) and caches to `data/processed/chroma.npz`.
Results go to `results/`: `metrics.csv` (long format), `summary.csv`
(mean ± std), `predictions.csv.gz` (per-sample probabilities and scores for figures).

## Development

```bash
uv sync                      # install with dev tools
uv run pytest                # tests + coverage report (fails under 90%)
uv run ruff check src tests  # lint
uv run ruff format src tests # format
```

Tests use only synthetic data (generated chromatograms and miniature Xcalibur text
files), so they run without the private dataset.
