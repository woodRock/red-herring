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
  metrics.csv      long format: one row per scenario/seed/fold/model/method/alpha/metric
  predictions.csv  per-sample probabilities and open-set scores, for figures
  summary.csv      mean ± std over scenarios, seeds, and folds
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from red_herring.baselines import BASELINES
from red_herring.conformal import novelty_pvalues, prediction_sets
from red_herring.data import SPECIES, load_raw, open_set_split
from red_herring.deep import ChromaCNNConfig, DeepCNN, TrainConfig
from red_herring.metrics import (
    closed_set_metrics,
    conformal_set_metrics,
    novelty_detection_metrics,
    open_set_metrics,
    overconfidence_metrics,
)
from red_herring.openset import open_set_scores

MODELS = {**BASELINES, "cnn": DeepCNN}


def _make_model(name: str, seed: int, args: argparse.Namespace, n_classes: int):
    if name == "cnn":
        return DeepCNN(seed=seed, train_cfg=TrainConfig(epochs=args.epochs),
                       model_cfg=ChromaCNNConfig(num_classes=n_classes))
    return MODELS[name](seed=seed, feature=args.feature)


def _grouped_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, groups
    )


def evaluate_fold(model, X, y, train, cal, test, unk, alphas) -> tuple[list[dict], dict]:
    """Fit on `train`; return metric rows and raw outputs for one fold."""
    model.fit(X[train], y[train])
    out = {name: model.outputs(X[idx]) for name, idx in
           {"train": train, "cal": cal, "test": test, "unknown": unk}.items()}

    scores = {split: open_set_scores(out[split], out["train"], y[train])
              for split in ("cal", "test", "unknown")}
    rows = []

    def add(method, metrics, alpha=np.nan):
        rows.extend({"method": method, "alpha": alpha, "metric": k, "value": v}
                    for k, v in metrics.items())

    add("closed", closed_set_metrics(out["test"]["probs"], y[test]))
    add("closed", overconfidence_metrics(out["unknown"]["probs"]))

    for name in scores["test"]:
        add(name, open_set_metrics(scores["test"][name], scores["unknown"][name]))

    for alpha in alphas:
        for set_score, mondrian in itertools.product(("lac", "aps"), (False, True)):
            sets = {split: prediction_sets(out["cal"]["probs"], y[cal], out[split]["probs"],
                                           alpha, set_score, mondrian)
                    for split in ("test", "unknown")}
            method = f"cp_{set_score}" + ("_mondrian" if mondrian else "")
            add(method, conformal_set_metrics(sets["test"], y[test], sets["unknown"]), alpha)

        for name in scores["test"]:
            p_test = novelty_pvalues(scores["cal"][name], scores["test"][name])
            p_unk = novelty_pvalues(scores["cal"][name], scores["unknown"][name])
            add(f"cnd_{name}", novelty_detection_metrics(p_test, p_unk, alpha), alpha)

    return rows, {"out": out, "scores": scores}


def _prediction_rows(raw, data, idx_by_split, y) -> list[dict]:
    rows = []
    for split in ("cal", "test", "unknown"):
        probs = raw["out"][split]["probs"]
        for j, i in enumerate(idx_by_split[split]):
            row = {"split": split, "sample_id": data.sample_ids[i], "y": int(y[i]),
                   "pred": int(probs[j].argmax()), "max_prob": float(probs[j].max())}
            row.update({f"score_{k}": float(v[j]) for k, v in raw["scores"][split].items()})
            rows.append(row)
    return rows


def run(args: argparse.Namespace) -> None:
    data = load_raw(cache=args.cache)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.groups == "tissue":
        groups_all = np.array([f"{ind}|{tis}" for ind, tis in zip(data.individuals, data.tissues)])
    else:  # every sample its own group: replicates may leak across splits (optimistic)
        groups_all = np.arange(len(data))

    metric_rows, pred_rows = [], []
    for unknown in itertools.combinations(SPECIES, args.n_unknown):
        osd = open_set_split(data, list(unknown))
        scenario = "+".join(unknown)
        known_idx = np.flatnonzero(osd.y >= 0)
        unk_idx = np.flatnonzero(osd.y < 0)
        y = osd.y

        for seed in range(args.seeds):
            outer = _grouped_splits(y[known_idx], groups_all[known_idx], args.folds, seed)
            for fold, (tr, te) in enumerate(outer):
                train_full, test = known_idx[tr], known_idx[te]
                inner = _grouped_splits(y[train_full], groups_all[train_full], args.cal_splits, seed)
                fit_rel, cal_rel = next(inner)
                train, cal = train_full[fit_rel], train_full[cal_rel]
                if len(np.unique(y[train])) < len(osd.classes):
                    print(f"[{scenario}] seed={seed} fold={fold}: a known species is missing "
                          f"from proper-train, skipping")
                    continue
                idx_by_split = {"cal": cal, "test": test, "unknown": unk_idx}

                for model_name in args.models:
                    t0 = time.time()
                    model = _make_model(model_name, seed, args, len(osd.classes))
                    rows, raw = evaluate_fold(model, osd.X, y, train, cal, test, unk_idx, args.alphas)
                    meta = {"unknown": scenario, "seed": seed, "fold": fold, "model": model_name,
                            "n_train": len(train), "n_cal": len(cal), "n_test": len(test)}
                    metric_rows.extend({**meta, **r} for r in rows)
                    pred_rows.extend({**meta, **r} for r in _prediction_rows(raw, osd, idx_by_split, y))
                    print(f"[{scenario}] seed={seed} fold={fold} {model_name:>4}  "
                          f"train/cal/test={len(train)}/{len(cal)}/{len(test)}  {time.time() - t0:.1f}s")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions.csv.gz", index=False)

    summary = summarise(metrics)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print_headline(summary)


def summarise(metrics: pd.DataFrame, by_scenario: bool = False) -> pd.DataFrame:
    keys = ["model", "method", "alpha", "metric"] + (["unknown"] if by_scenario else [])
    return (metrics.groupby(keys, dropna=False)["value"]
            .agg(["mean", "std", "count"]).reset_index())


def print_headline(summary: pd.DataFrame, alpha: float = 0.1) -> None:
    def table(method_metric: list[tuple[str, str]], at_alpha: bool) -> pd.DataFrame:
        s = summary[summary["alpha"].eq(alpha)] if at_alpha else summary[summary["alpha"].isna()]
        cols = {}
        for method, metric in method_metric:
            sel = s[(s["method"] == method) & (s["metric"] == metric)].set_index("model")
            cols[f"{method}:{metric}"] = sel.apply(lambda r: f"{r['mean']:.3f}±{r['std']:.3f}", axis=1)
        return pd.DataFrame(cols)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n== Closed-set accuracy and confidence on unseen species ==")
        print(table([("closed", "bal_acc"), ("closed", "ece"),
                     ("closed", "unk_conf_mean"), ("closed", "unk_conf>=0.9")], False))
        print("\n== Open-set detection AUROC ==")
        print(table([(m, "auroc") for m in ("msp", "energy", "knn", "mahalanobis")], False))
        print(f"\n== Conformal prediction sets (alpha={alpha}) ==")
        print(table([("cp_lac", "coverage"), ("cp_lac", "set_size"),
                     ("cp_lac", "unknown_empty"), ("cp_lac_mondrian", "coverage"),
                     ("cp_lac_mondrian", "unknown_empty")], True))
        print(f"\n== Conformal novelty detection (alpha={alpha}): FPR guaranteed <= alpha ==")
        print(table([(f"cnd_{m}", k) for m in ("msp", "knn", "mahalanobis") for k in ("fpr", "tpr")], True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data/processed/chroma.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--n-unknown", type=int, default=1, help="species held out per scenario")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cal-splits", type=int, default=4, help="1/cal_splits of train used for calibration")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--feature", default="mean", choices=["mean", "max"],
                    help="RT projection used by the classical baselines")
    ap.add_argument("--groups", default="tissue", choices=["tissue", "none"],
                    help="tissue: replicates of an individual's tissue stay together (honest); "
                         "none: plain stratified split (optimistic, for comparison)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
