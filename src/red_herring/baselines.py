"""
Classical baselines on a 1D projection of the chromatogram over RT:
  mean — average spectrum over RT bins (≈ chroma-dcnn's sum spectrum, X.npy); default
  max  — per-m/z maximum over RT bins (chroma-dcnn's max_proj)

Each model exposes fit(X, y) and outputs(X) → {"probs", ["logits"], ["emb"]}
so the open-set and conformal machinery is shared with the deep model.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = ("mean", "max")


def project(X: np.ndarray, feature: str) -> np.ndarray:
    """[N, n_bins, mz] → [N, mz]"""
    if feature == "mean":
        return X.mean(axis=1)
    if feature == "max":
        return X.max(axis=1)
    raise ValueError(f"Unknown feature {feature!r}; expected one of {FEATURES}")


class _ProjectionModel:
    def __init__(self, seed: int = 0, feature: str = "mean") -> None:
        self.seed = seed
        self.feature = feature

    def _features(self, X: np.ndarray) -> np.ndarray:
        return project(X, self.feature)


class PCALogReg(_ProjectionModel):
    """Standardise → PCA → multinomial logistic regression; PCA space is the embedding."""

    def __init__(self, seed: int = 0, feature: str = "mean", n_components: int = 20) -> None:
        super().__init__(seed, feature)
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> PCALogReg:
        F = self._features(X)
        self.scaler = StandardScaler().fit(F)
        n_components = min(self.n_components, len(F) - 1)
        self.pca = PCA(n_components=n_components, random_state=self.seed)
        self.pca.fit(self.scaler.transform(F))
        self.clf = LogisticRegression(class_weight="balanced", max_iter=5000)
        self.clf.fit(self._embed(X), y)
        return self

    def _embed(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(self._features(X)))

    def outputs(self, X: np.ndarray) -> dict:
        emb = self._embed(X)
        return {
            "probs": self.clf.predict_proba(emb),
            "logits": self.clf.decision_function(emb),
            "emb": emb,
        }


class SVMRBF(_ProjectionModel):
    """Standardise → RBF SVM with Platt-scaled probabilities."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> SVMRBF:
        F = self._features(X)
        self.scaler = StandardScaler().fit(F)
        svm = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")
        # Platt scaling on out-of-fold decision values, then one SVM refit on all data.
        self.clf = CalibratedClassifierCV(svm, method="sigmoid", cv=3, ensemble=False)
        self.clf.fit(self.scaler.transform(F), y)
        return self

    def outputs(self, X: np.ndarray) -> dict:
        return {"probs": self.clf.predict_proba(self.scaler.transform(self._features(X)))}


class RandomForest(_ProjectionModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForest:
        self.clf = RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=self.seed, n_jobs=-1
        )
        self.clf.fit(self._features(X), y)
        return self

    def outputs(self, X: np.ndarray) -> dict:
        return {"probs": self.clf.predict_proba(self._features(X))}


BASELINES = {"lr": PCALogReg, "svm": SVMRBF, "rf": RandomForest}
