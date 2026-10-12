"""
Open-set species authentication experiment.

Protocol (per held-out species combination × seed × fold):
  1. Hold out `unknown` species entirely — the "fraud" species never seen in training.
  2. Grouped stratified K-fold over the known species; groups are individual + tissue
     so technical replicates (A/B/C) never straddle train and test.
  3. Split each training fold into proper-train and a grouped calibration set
     (known species only) for conformal calibration.
  4. Fit each model on proper-train; evaluate on known test fold + all unknown samples.

Outputs (in --out):
  metrics.csv         long format: one row per scenario/seed/fold/model/method/alpha/metric
  predictions.csv.gz  per-sample probabilities and open-set scores, for figures
  summary.csv         mean ± std over scenarios, seeds, and folds
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from red_herring.baselines import BASELINES, FEATURES
from red_herring.conformal import SET_SCORES, novelty_pvalues, prediction_sets
from red_herring.data import SPECIES, SpectraData, load_raw, open_set_split
from red_herring.deep import ChromaCNNConfig, DeepCNN, TrainConfig
from red_herring.metrics import (
    closed_set_metrics,
    conformal_set_metrics,
    novelty_detection_metrics,
    open_set_metrics,
    overconfidence_metrics,
)
from red_herring.openset import open_set_scores

MODELS = (*BASELINES, "cnn")
EVAL_SPLITS = ("cal", "test", "unknown")


@dataclass
class ExperimentConfig:
    models: tuple[str, ...] = MODELS
    n_unknown: int = 1
    seeds: int = 3
    folds: int = 5
    cal_splits: int = 4  # 1/cal_splits of each training fold is used for calibration
    alphas: tuple[float, ...] = (0.05, 0.1, 0.2)
    epochs: int = 100
    feature: str = "mean"
    grouped: bool = True  # False: every sample its own group (optimistic, leaky)


@dataclass
class Fold:
    unknown: str  # held-out species, e.g. "TAR" or "SNA+GUR"
    seed: int
    fold: int
    data: SpectraData  # open-set relabelled
    train: np.ndarray
    cal: np.ndarray
    test: np.ndarray
    unknown_idx: np.ndarray

    @property
    def meta(self) -> dict:
        return {
            "unknown": self.unknown,
            "seed": self.seed,
            "fold": self.fold,
            "n_train": len(self.train),
            "n_cal": len(self.cal),
            "n_test": len(self.test),
        }

    def indices(self, split: str) -> np.ndarray:
        splits = {
            "train": self.train,
            "cal": self.cal,
            "test": self.test,
            "unknown": self.unknown_idx,
        }
        return splits[split]


def make_model(name: str, seed: int, config: ExperimentConfig, data: SpectraData):
    if name == "cnn":
        return DeepCNN(
            seed=seed,
            train_cfg=TrainConfig(epochs=config.epochs),
            model_cfg=ChromaCNNConfig(mz_max=data.X.shape[2], num_classes=len(data.classes)),
        )
    return BASELINES[name](seed=seed, feature=config.feature)


def _grouped_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cv.split(np.zeros(len(y)), y, groups)


def iter_folds(
    data: SpectraData, config: ExperimentConfig, log: Callable[[str], None] = print
) -> Iterator[Fold]:
    """Yield every (held-out species, seed, fold) with proper-train/cal/test indices."""
    groups = data.groups if config.grouped else np.arange(len(data))
    for unknown in itertools.combinations(SPECIES, config.n_unknown):
        osd = open_set_split(data, list(unknown))
        scenario = "+".join(unknown)
        known_idx = np.flatnonzero(osd.y >= 0)
        unknown_idx = np.flatnonzero(osd.y < 0)

        for seed in range(config.seeds):
            outer = _grouped_splits(osd.y[known_idx], groups[known_idx], config.folds, seed)
            for fold, (tr, te) in enumerate(outer):
                train_full, test = known_idx[tr], known_idx[te]
                inner = _grouped_splits(
                    osd.y[train_full], groups[train_full], config.cal_splits, seed
                )
                fit_rel, cal_rel = next(inner)
                train, cal = train_full[fit_rel], train_full[cal_rel]
                if len(np.unique(osd.y[train])) < len(osd.classes):
                    log(
                        f"[{scenario}] seed={seed} fold={fold}: a known species is "
                        f"missing from proper-train, skipping"
                    )
                    continue
                yield Fold(scenario, seed, fold, osd, train, cal, test, unknown_idx)


def evaluate_fold(model, fold: Fold, alphas: tuple[float, ...]) -> tuple[list[dict], dict]:
    """Fit on proper-train; return metric rows and per-split outputs and scores."""
    X, y = fold.data.X, fold.data.y
    train = fold.train
    model.fit(X[train], y[train])
    out = {split: model.outputs(X[fold.indices(split)]) for split in ("train", *EVAL_SPLITS)}
    scores = {split: open_set_scores(out[split], out["train"], y[train]) for split in EVAL_SPLITS}
    y_cal, y_test = y[fold.cal], y[fold.test]

    rows: list[dict] = []

    def add(method: str, metrics: dict, alpha: float = np.nan) -> None:
        rows.extend(
            {"method": method, "alpha": alpha, "metric": k, "value": v} for k, v in metrics.items()
        )

    add("closed", closed_set_metrics(out["test"]["probs"], y_test))
    add("closed", overconfidence_metrics(out["unknown"]["probs"]))
    for name in scores["test"]:
        add(name, open_set_metrics(scores["test"][name], scores["unknown"][name]))

    for alpha in alphas:
        for set_score, mondrian in itertools.product(SET_SCORES, (False, True)):
            test_sets, unknown_sets = (
                prediction_sets(
                    out["cal"]["probs"], y_cal, out[split]["probs"], alpha, set_score, mondrian
                )
                for split in ("test", "unknown")
            )
            method = f"cp_{set_score}" + ("_mondrian" if mondrian else "")
            add(method, conformal_set_metrics(test_sets, y_test, unknown_sets), alpha)

        for name in scores["test"]:
            p_test, p_unknown = (
                novelty_pvalues(scores["cal"][name], scores[split][name])
                for split in ("test", "unknown")
            )
            add(f"cnd_{name}", novelty_detection_metrics(p_test, p_unknown, alpha), alpha)

    return rows, {"out": out, "scores": scores}


def prediction_rows(fold: Fold, raw: dict) -> list[dict]:
    rows = []
    for split in EVAL_SPLITS:
        probs = raw["out"][split]["probs"]
        for j, i in enumerate(fold.indices(split)):
            row = {
                "split": split,
                "sample_id": fold.data.sample_ids[i],
                "y": int(fold.data.y[i]),
                "pred": int(probs[j].argmax()),
                "max_prob": float(probs[j].max()),
            }
            row.update({f"score_{k}": float(v[j]) for k, v in raw["scores"][split].items()})
            rows.append(row)
    return rows


def run_experiment(
    data: SpectraData, config: ExperimentConfig, log: Callable[[str], None] = print
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (metrics, predictions) data frames for every fold and model."""
    metric_rows, pred_rows = [], []
    for fold in iter_folds(data, config, log):
        for name in config.models:
            t0 = time.time()
            model = make_model(name, fold.seed, config, fold.data)
            rows, raw = evaluate_fold(model, fold, config.alphas)
            meta = {**fold.meta, "model": name}
            metric_rows.extend({**meta, **r} for r in rows)
            pred_rows.extend({**meta, **r} for r in prediction_rows(fold, raw))
            log(
                f"[{fold.unknown}] seed={fold.seed} fold={fold.fold} {name:>4}  "
                f"train/cal/test={len(fold.train)}/{len(fold.cal)}/{len(fold.test)}  "
                f"{time.time() - t0:.1f}s"
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def summarise(metrics: pd.DataFrame, by_scenario: bool = False) -> pd.DataFrame:
    keys = ["model", "method", "alpha", "metric"] + (["unknown"] if by_scenario else [])
    grouped = metrics.groupby(keys, dropna=False)["value"]
    return grouped.agg(["mean", "std", "count"]).reset_index()


# title → (reported at a conformal alpha?, [(method, metric), ...])
HEADLINE_TABLES = {
    "Closed-set accuracy and confidence on unseen species": (
        False,
        [
            ("closed", "bal_acc"),
            ("closed", "ece"),
            ("closed", "unk_conf_mean"),
            ("closed", "unk_conf>=0.9"),
        ],
    ),
    "Open-set detection AUROC": (
        False,
        [(m, "auroc") for m in ("msp", "energy", "knn", "mahalanobis")],
    ),
    "Conformal prediction sets": (
        True,
        [
            ("cp_lac", "coverage"),
            ("cp_lac", "set_size"),
            ("cp_lac", "unknown_empty"),
            ("cp_lac_mondrian", "coverage"),
            ("cp_lac_mondrian", "unknown_empty"),
        ],
    ),
    "Conformal novelty detection, FPR guaranteed <= alpha": (
        True,
        [(f"cnd_{m}", k) for m in ("msp", "knn", "mahalanobis") for k in ("fpr", "tpr")],
    ),
}


def headline_tables(summary: pd.DataFrame, alpha: float = 0.1) -> dict[str, pd.DataFrame]:
    """Model × (method:metric) tables of 'mean±std' strings."""
    tables = {}
    for title, (at_alpha, columns) in HEADLINE_TABLES.items():
        s = summary[summary["alpha"].eq(alpha) if at_alpha else summary["alpha"].isna()]
        table = {}
        for method, metric in columns:
            sel = s[(s["method"] == method) & (s["metric"] == metric)].set_index("model")
            table[f"{method}:{metric}"] = (
                sel["mean"].map("{:.3f}".format) + "±" + sel["std"].map("{:.3f}".format)
            )
        tables[f"{title} (alpha={alpha})" if at_alpha else title] = pd.DataFrame(table)
    return tables


def save_results(metrics: pd.DataFrame, predictions: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    predictions.to_csv(out_dir / "predictions.csv.gz", index=False)
    summary = summarise(metrics)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", default="data/processed/chroma.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--n-unknown", type=int, default=1, help="species held out per scenario")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--cal-splits", type=int, default=4, help="1/cal_splits of train used for calibration"
    )
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument(
        "--feature",
        default="mean",
        choices=FEATURES,
        help="RT projection used by the classical baselines",
    )
    ap.add_argument(
        "--groups",
        default="tissue",
        choices=["tissue", "none"],
        help="tissue: replicates of an individual's tissue stay together (honest); "
        "none: plain stratified split (optimistic, for comparison)",
    )
    return ap.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        models=tuple(args.models),
        n_unknown=args.n_unknown,
        seeds=args.seeds,
        folds=args.folds,
        cal_splits=args.cal_splits,
        alphas=tuple(args.alphas),
        epochs=args.epochs,
        feature=args.feature,
        grouped=args.groups == "tissue",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data = load_raw(cache=args.cache)
    metrics, predictions = run_experiment(
        data, config_from_args(args), log=lambda msg: print(msg, flush=True)
    )
    summary = save_results(metrics, predictions, Path(args.out))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for title, table in headline_tables(summary).items():
            print(f"\n== {title} ==\n{table}")


if __name__ == "__main__":
    main()
