"""
Classical baselines on a 1D projection of the chromatogram over RT:
  mean — average spectrum over RT bins (≈ chroma-dcnn's sum spectrum, X.npy); default
  max  — per-m/z maximum over RT bins (chroma-dcnn's max_proj)

Each model exposes fit(X, y) and outputs(X) → {"probs", ["logits"], ["emb"]}
so the open-set and conformal machinery is shared with the deep model.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def project(X: np.ndarray, feature: str) -> np.ndarray:
    """[N, n_bins, mz] → [N, mz]"""
    if feature == "mean":
        return X.mean(axis=1)
    if feature == "max":
        return X.max(axis=1)
    raise ValueError(f"Unknown feature {feature!r}")


class PCALogReg:
    """Standardise → PCA → multinomial logistic regression; PCA space is the embedding."""

    def __init__(self, n_components: int = 20, seed: int = 0, feature: str = "mean") -> None:
        self.n_components = n_components
        self.feature = feature
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PCALogReg":
        F = project(X, self.feature)
        self.scaler = StandardScaler().fit(F)
        n_comp = min(self.n_components, len(F) - 1)
        self.pca = PCA(n_components=n_comp, random_state=self.seed).fit(self.scaler.transform(F))
        self.clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
        self.clf.fit(self._embed(X), y)
        return self

    def _embed(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(project(X, self.feature)))

    def outputs(self, X: np.ndarray) -> dict:
        emb = self._embed(X)
        return {
            "probs": self.clf.predict_proba(emb),
            "logits": self.clf.decision_function(emb),
            "emb": emb,
        }


class SVMRBF:
    """Standardise → RBF SVM with Platt-scaled probabilities."""

    def __init__(self, seed: int = 0, feature: str = "mean") -> None:
        self.seed = seed
        self.feature = feature

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMRBF":
        F = project(X, self.feature)
        self.scaler = StandardScaler().fit(F)
        self.clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                       class_weight="balanced", random_state=self.seed)
        self.clf.fit(self.scaler.transform(F), y)
        return self

    def outputs(self, X: np.ndarray) -> dict:
        return {"probs": self.clf.predict_proba(self.scaler.transform(project(X, self.feature)))}


class RandomForest:
    def __init__(self, seed: int = 0, feature: str = "mean") -> None:
        self.seed = seed
        self.feature = feature

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        self.clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                          random_state=self.seed, n_jobs=-1)
        self.clf.fit(project(X, self.feature), y)
        return self

    def outputs(self, X: np.ndarray) -> dict:
        return {"probs": self.clf.predict_proba(project(X, self.feature))}


BASELINES = {"lr": PCALogReg, "svm": SVMRBF, "rf": RandomForest}
