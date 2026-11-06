"""Raga classifier: thin wrapper around scikit-learn with top-k output."""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import extract_features
from .pitch import PitchTrack


def build_estimator(kind: str = "svm"):
    if kind == "knn":
        # Bray-Curtis works well on normalised histograms.
        return KNeighborsClassifier(n_neighbors=5, weights="distance", metric="braycurtis")
    if kind == "svm":
        # Calibrated so the app can show meaningful confidence percentages.
        svc = SVC(kernel="rbf", C=10, class_weight="balanced")
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(svc, cv=3, ensemble=False))
    raise ValueError(f"unknown model kind: {kind}")


@dataclass
class RagaClassifier:
    kind: str = "svm"
    bins: int = 120
    use_transitions: bool = True
    estimator: object = field(default=None, repr=False)

    def featurize(self, track: PitchTrack) -> np.ndarray:
        return extract_features(track, bins=self.bins, use_transitions=self.use_transitions)

    def fit(self, X: np.ndarray, y: list[str]) -> "RagaClassifier":
        self.estimator = build_estimator(self.kind)
        self.estimator.fit(X, y)
        return self

    @property
    def classes(self) -> list[str]:
        return list(self.estimator.classes_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def top_k(self, track: PitchTrack, k: int = 3) -> list[tuple[str, float]]:
        proba = self.predict_proba(self.featurize(track)[None, :])[0]
        order = np.argsort(proba)[::-1][:k]
        return [(self.classes[i], float(proba[i])) for i in order]

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "RagaClassifier":
        return joblib.load(path)
