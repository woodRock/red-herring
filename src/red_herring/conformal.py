"""
Split conformal prediction for closed-set sets and open-set (novelty) rejection.

Prediction sets (Vovk et al. 2005; Romano et al. 2020):
  LAC  s(x, y) = 1 - p_y(x)                       — smallest sets on average
  APS  s(x, y) = sum of p_k(x) over p_k >= p_y    — adaptive set sizes
  Marginal calibration guarantees P(y ∈ C(x)) >= 1 - α over known species;
  Mondrian (class-conditional) calibration guarantees it per species.
  An empty set means "consistent with no known species" → flagged as novel.

Conformal novelty detection (Bates et al. 2023):
  Any score where higher = more novel, calibrated on held-out known samples,
  gives p(x) = (1 + #{cal >= s(x)}) / (n + 1). Flagging p <= α bounds the
  false-alarm rate on known species by α — the guarantee a regulator can act on.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - α) quantile; inf when n is too small for α."""
    n = len(cal_scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if n == 0 or level > 1:
        return np.inf
    return float(np.quantile(cal_scores, level, method="higher"))


def lac_scores(probs: np.ndarray) -> np.ndarray:
    """[N, K] → [N, K] nonconformity for every candidate label."""
    return 1.0 - probs


def aps_scores(probs: np.ndarray) -> np.ndarray:
    """[N, K] → [N, K]: total mass of labels at least as likely as each label."""
    return (probs[:, None, :] * (probs[:, None, :] >= probs[:, :, None])).sum(axis=-1)


SET_SCORES = {"lac": lac_scores, "aps": aps_scores}


def prediction_sets(
    cal_probs: np.ndarray,
    cal_y: np.ndarray,
    test_probs: np.ndarray,
    alpha: float,
    score: str = "lac",
    mondrian: bool = False,
) -> np.ndarray:
    """Boolean [N_test, K] membership matrix of conformal prediction sets."""
    score_fn = SET_SCORES[score]
    cal_s = score_fn(cal_probs)[np.arange(len(cal_y)), cal_y]
    test_s = score_fn(test_probs)
    n_classes = test_probs.shape[1]
    if mondrian:
        q = np.array([conformal_quantile(cal_s[cal_y == k], alpha) for k in range(n_classes)])
    else:
        q = np.full(n_classes, conformal_quantile(cal_s, alpha))
    return test_s <= q[None, :]


def novelty_pvalues(cal_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Conformal p-values; small p = unlike every calibration (known) sample."""
    cal_sorted = np.sort(cal_scores)
    n_ge = len(cal_sorted) - np.searchsorted(cal_sorted, test_scores, side="left")
    return (1.0 + n_ge) / (len(cal_sorted) + 1.0)
