from __future__ import annotations

import numpy as np
import pytest

from red_herring.conformal import (
    aps_scores,
    conformal_quantile,
    lac_scores,
    novelty_pvalues,
    prediction_sets,
)


def test_conformal_quantile_uses_finite_sample_correction():
    scores = np.arange(1, 11, dtype=float)  # n = 10
    assert conformal_quantile(scores, alpha=0.2) == 9.0  # ceil(11 * 0.8) / 10 = 0.9
    assert conformal_quantile(scores, alpha=0.1) == 10.0  # ceil(11 * 0.9) / 10 = 1.0


@pytest.mark.parametrize("n", [0, 5])
def test_conformal_quantile_is_infinite_when_calibration_too_small(n):
    assert conformal_quantile(np.ones(n), alpha=0.1) == np.inf


def test_lac_scores():
    np.testing.assert_allclose(lac_scores(np.array([[0.7, 0.2, 0.1]])), [[0.3, 0.8, 0.9]])


def test_aps_scores_accumulate_mass_of_more_likely_labels():
    np.testing.assert_allclose(aps_scores(np.array([[0.2, 0.5, 0.3]])), [[1.0, 0.5, 0.8]])


def test_prediction_sets_membership_from_threshold():
    cal_probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6]] * 5)
    cal_y = np.array([0, 0, 1, 1] * 5)
    test_probs = np.array([[0.95, 0.05], [0.5, 0.5], [0.0, 1.0]])
    sets = prediction_sets(cal_probs, cal_y, test_probs, alpha=0.2)
    # LAC threshold is the 0.8-quantile of {0.1, 0.2, 0.3, 0.4} → 0.4: include p_y >= 0.6
    np.testing.assert_array_equal(sets, [[True, False], [False, False], [False, True]])


def test_mondrian_sets_use_per_class_thresholds():
    cal_probs = np.array([[0.5, 0.5]] * 10 + [[0.01, 0.99]] * 10)
    cal_y = np.array([0] * 10 + [1] * 10)  # LAC scores: class 0 → 0.5, class 1 → 0.01
    test_probs = np.array([[0.3, 0.7]])
    marginal = prediction_sets(cal_probs, cal_y, test_probs, alpha=0.1)
    mondrian = prediction_sets(cal_probs, cal_y, test_probs, alpha=0.1, mondrian=True)
    np.testing.assert_array_equal(marginal, [[False, True]])  # shared threshold 0.5
    np.testing.assert_array_equal(mondrian, [[False, False]])  # class 1 needs p >= 0.99


def _random_probs(rng, n, k):
    logits = rng.normal(size=(n, k)) * 2
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("score", ["lac", "aps"])
@pytest.mark.parametrize("mondrian", [False, True])
def test_prediction_sets_achieve_nominal_coverage(score, mondrian):
    """Labels drawn from the model's own probabilities: exchangeable, so coverage >= 1 - α."""
    rng = np.random.default_rng(0)
    k, alpha = 3, 0.1
    probs = _random_probs(rng, 4000, k)
    y = np.array([rng.choice(k, p=p) for p in probs])
    sets = prediction_sets(probs[:2000], y[:2000], probs[2000:], alpha, score, mondrian)
    coverage = sets[np.arange(2000), y[2000:]].mean()
    assert coverage >= 1 - alpha - 0.02


def test_novelty_pvalues_formula():
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    p = novelty_pvalues(cal, np.array([0.0, 2.0, 2.5, 10.0]))
    np.testing.assert_allclose(p, [5 / 5, 4 / 5, 3 / 5, 1 / 5])


def test_novelty_pvalues_control_false_alarm_rate():
    """The guarantee is marginal over calibration draws, so average over many."""
    rng = np.random.default_rng(1)
    alpha = 0.1
    fpr, tpr = [], []
    for _ in range(200):
        cal, known, novel = rng.normal(size=50), rng.normal(size=50), rng.normal(3, 1, size=50)
        fpr.append((novelty_pvalues(cal, known) <= alpha).mean())
        tpr.append((novelty_pvalues(cal, novel) <= alpha).mean())
    assert np.mean(fpr) <= alpha + 0.01
    assert np.mean(tpr) > 0.8
