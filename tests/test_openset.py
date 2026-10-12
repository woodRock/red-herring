from __future__ import annotations

import numpy as np
import pytest

from red_herring.openset import (
    energy_score,
    knn_score,
    mahalanobis_score,
    msp_score,
    open_set_scores,
)


def test_msp_score():
    np.testing.assert_allclose(msp_score(np.array([[0.9, 0.1], [0.5, 0.5]])), [0.1, 0.5])


def test_energy_score_is_negative_logsumexp():
    logits = np.array([[0.0, 0.0], [10.0, 0.0]])
    expected = -np.log(np.exp(logits).sum(axis=1))
    np.testing.assert_allclose(energy_score(logits), expected)
    assert energy_score(logits)[1] < energy_score(logits)[0]  # confident → less novel


@pytest.fixture
def clusters():
    rng = np.random.default_rng(0)
    train = np.r_[rng.normal([5, 0], 0.3, (30, 2)), rng.normal([0, 5], 0.3, (30, 2))]
    train_y = np.r_[np.zeros(30, int), np.ones(30, int)]
    near = np.array([[5.0, 0.1], [0.1, 5.0]])
    far = np.array([[-5.0, -5.0], [5.0, 5.0]])
    return train, train_y, near, far


def test_knn_score_ranks_far_points_as_novel(clusters):
    train, _, near, far = clusters
    assert knn_score(train, near).max() < knn_score(train, far).min()


def test_knn_score_with_fewer_references_than_k():
    train = np.array([[1.0, 0.0], [0.0, 1.0]])
    score = knn_score(train, np.array([[1.0, 0.0]]), k=5)
    assert score[0] == pytest.approx(np.sqrt(2))  # falls back to the furthest reference


def test_mahalanobis_score_ranks_far_points_as_novel(clusters):
    train, train_y, near, far = clusters
    assert (
        mahalanobis_score(train, train_y, near).max() < mahalanobis_score(train, train_y, far).min()
    )


def test_open_set_scores_depend_on_available_outputs(clusters):
    train, train_y, near, _ = clusters
    probs = np.full((2, 2), 0.5)
    assert set(open_set_scores({"probs": probs}, {}, train_y)) == {"msp"}

    outputs = {"probs": probs, "logits": np.zeros((2, 2)), "emb": near}
    scores = open_set_scores(outputs, {"emb": train}, train_y)
    assert set(scores) == {"msp", "energy", "knn", "mahalanobis"}
    assert all(v.shape == (2,) for v in scores.values())
