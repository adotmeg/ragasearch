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


def build_estimator(kind: str = "svm", min_class_count: int = 3):
    if kind == "knn":
        # Bray-Curtis works well on normalised histograms.
        return KNeighborsClassifier(n_neighbors=5, weights="distance", metric="braycurtis")
    if kind == "svm":
        # Calibrated so the app can show meaningful confidence percentages.
        # Calibration needs >= 2 examples per class in its internal split.
        svc = SVC(kernel="rbf", C=10, class_weight="balanced")
        if min_class_count < 2:
            raise ValueError("svm needs at least 2 training clips per raga; use --model knn or more data")
        cv = min(3, min_class_count)
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(svc, cv=cv, ensemble=False))
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
        from collections import Counter

        self.estimator = build_estimator(self.kind, min(Counter(y).values()))
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
