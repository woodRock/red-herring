"""
Open-set (novelty) scores — higher = more likely an unseen species.

  msp          1 - max softmax probability        (Hendrycks & Gimpel 2017)
  energy       -logsumexp(logits)                 (Liu et al. 2020)
  knn          distance to k-th nearest training embedding, L2-normalised
               (Sun et al. 2022)
  mahalanobis  min class-conditional Mahalanobis distance, tied shrinkage
               covariance (Lee et al. 2018)
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf


def msp_score(probs: np.ndarray) -> np.ndarray:
    return 1.0 - probs.max(axis=1)


def energy_score(logits: np.ndarray) -> np.ndarray:
    return -logsumexp(logits, axis=1)


def _l2(z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + eps)


def knn_score(train_emb: np.ndarray, emb: np.ndarray, k: int = 3) -> np.ndarray:
    ref, z = _l2(train_emb), _l2(emb)
    d = np.linalg.norm(z[:, None, :] - ref[None, :, :], axis=-1)
    k = min(k, ref.shape[0])
    return np.sort(d, axis=1)[:, k - 1]


def mahalanobis_score(train_emb: np.ndarray, train_y: np.ndarray, emb: np.ndarray) -> np.ndarray:
    classes = np.unique(train_y)
    means = np.stack([train_emb[train_y == c].mean(axis=0) for c in classes])
    centred = train_emb - means[np.searchsorted(classes, train_y)]
    precision = LedoitWolf().fit(centred).precision_
    diff = emb[:, None, :] - means[None, :, :]  # [N, K, d]
    d2 = np.einsum("nkd,de,nke->nk", diff, precision, diff)
    return d2.min(axis=1)


def open_set_scores(outputs: dict, train: dict, train_y: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute every score a model's outputs support.

    outputs / train : dicts with "probs" and optionally "logits", "emb"
    """
    scores = {"msp": msp_score(outputs["probs"])}
    if "logits" in outputs:
        scores["energy"] = energy_score(outputs["logits"])
    if "emb" in outputs:
        scores["knn"] = knn_score(train["emb"], outputs["emb"])
        scores["mahalanobis"] = mahalanobis_score(train["emb"], train_y, outputs["emb"])
    return scores
