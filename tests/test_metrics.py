from __future__ import annotations

import numpy as np
import pytest

from red_herring.metrics import (
    closed_set_metrics,
    conformal_set_metrics,
    expected_calibration_error,
    novelty_detection_metrics,
    open_set_metrics,
    overconfidence_metrics,
)


def test_ece_is_zero_when_confidence_matches_accuracy():
    probs = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.75, 0.25], [0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]
    )
    y = np.array([0, 1, 0, 0, 0, 1])
    assert expected_calibration_error(probs, y) == pytest.approx(0.0)


def test_ece_of_overconfident_wrong_predictions():
    probs = np.array([[0.95, 0.05]] * 4)
    assert expected_calibration_error(probs, np.ones(4, int)) == pytest.approx(0.95)


def test_closed_set_metrics():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    m = closed_set_metrics(probs, np.array([0, 1, 1, 1]))
    assert m["bal_acc"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert 0 <= m["ece"] <= 1


def test_open_set_metrics_perfect_separation():
    m = open_set_metrics(np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.9]))
    assert m == {"auroc": 1.0, "aupr": 1.0, "fpr95": 0.0}


def test_open_set_metrics_fpr95_counts_knowns_above_threshold():
    unknown = np.linspace(0.5, 1.0, 21)  # 5th percentile = 0.525
    known = np.array([0.1, 0.6, 0.7, 0.2])
    assert open_set_metrics(known, unknown)["fpr95"] == pytest.approx(0.5)


def test_overconfidence_metrics():
    m = overconfidence_metrics(np.array([[0.95, 0.05], [0.6, 0.4]]))
    assert m == {"unk_conf_mean": pytest.approx(0.775), "unk_conf>=0.9": 0.5}


def test_conformal_set_metrics():
    known_sets = np.array([[True, False], [True, True], [False, False], [False, True]])
    unknown_sets = np.array([[False, False], [True, False]])
    m = conformal_set_metrics(known_sets, np.array([0, 1, 0, 0]), unknown_sets)
    assert m == {
        "coverage": 0.5,
        "set_size": 1.0,
        "singleton_rate": 0.5,
        "known_empty": 0.25,
        "unknown_empty": 0.5,
        "unknown_set_size": 0.5,
    }


def test_novelty_detection_metrics():
    m = novelty_detection_metrics(np.array([0.5, 0.05, 0.9, 0.3]), np.array([0.01, 0.2]), 0.1)
    assert m == {"fpr": 0.25, "tpr": 0.5}
