"""Closed-set, calibration, open-set, and conformal metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def closed_set_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    return {
        "bal_acc": balanced_accuracy_score(y, probs.argmax(axis=1)),
        "ece": expected_calibration_error(probs, y),
    }


def open_set_metrics(known_score: np.ndarray, unknown_score: np.ndarray) -> dict:
    """Unknown = positive class; higher score = more novel."""
    s = np.concatenate([known_score, unknown_score])
    is_unknown = np.r_[np.zeros(len(known_score)), np.ones(len(unknown_score))]
    # FPR on knowns at the threshold that catches 95% of unknowns.
    thr = np.quantile(unknown_score, 0.05)
    return {
        "auroc": roc_auc_score(is_unknown, s),
        "aupr": average_precision_score(is_unknown, s),
        "fpr95": float((known_score >= thr).mean()),
    }


def overconfidence_metrics(unknown_probs: np.ndarray, threshold: float = 0.9) -> dict:
    """How often a closed-set model confidently assigns an unseen species to a known one."""
    conf = unknown_probs.max(axis=1)
    return {"unk_conf_mean": float(conf.mean()), f"unk_conf>={threshold}": float((conf >= threshold).mean())}


def conformal_set_metrics(known_sets: np.ndarray, known_y: np.ndarray, unknown_sets: np.ndarray) -> dict:
    size_known = known_sets.sum(axis=1)
    singleton = size_known == 1
    return {
        "coverage": float(known_sets[np.arange(len(known_y)), known_y].mean()),
        "set_size": float(size_known.mean()),
        "singleton_rate": float(singleton.mean()),
        "known_empty": float((size_known == 0).mean()),        # false alarms
        "unknown_empty": float((unknown_sets.sum(axis=1) == 0).mean()),  # detections
        "unknown_set_size": float(unknown_sets.sum(axis=1).mean()),
    }


def novelty_detection_metrics(known_p: np.ndarray, unknown_p: np.ndarray, alpha: float) -> dict:
    return {"fpr": float((known_p <= alpha).mean()), "tpr": float((unknown_p <= alpha).mean())}
