"""Turn recordings into (X, y, groups) clip datasets, train and evaluate."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from .data import Recording, get_pitch_track
from .features import clip_tracks
from .model import RagaClassifier


def build_clip_dataset(recs: list[Recording], clf: RagaClassifier, clip_seconds: float,
                       hop_seconds: float | None = None, cache_dir="cache/pitch", verbose=True):
    """Every recording -> many fixed-length clips -> feature rows.

    groups[i] is the artist of row i, so that splits never put the same
    artist (and therefore the same recording) on both sides.
    """
    X, y, groups, rec_ids = [], [], [], []
    for i, rec in enumerate(recs):
        track = get_pitch_track(rec, cache_dir=cache_dir)
        clips = clip_tracks(track, clip_seconds, hop_seconds)
        if verbose:
            print(f"  [{i + 1}/{len(recs)}] {rec.raga:<14} {rec.artist:<20} tonic={track.tonic_hz:6.1f} Hz  clips={len(clips)}")
        for c in clips:
            X.append(clf.featurize(c))
            y.append(rec.raga)
            groups.append(rec.artist)
            rec_ids.append(i)
    return np.array(X), np.array(y), np.array(groups), np.array(rec_ids)


def cross_validate(X, y, groups, kind="svm", n_splits=5, k=3, **clf_kwargs):
    """Group k-fold by artist. Returns top-1 / top-k accuracy and confusion matrix."""
    n_groups = len(set(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise ValueError("Need at least 2 distinct groups (artists/recordings) for cross-validation.")
    labels = sorted(set(y))
    top1 = topk = total = 0
    y_true, y_pred = [], []
    per_class = defaultdict(lambda: [0, 0])
    skipped = 0
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        train_labels = set(y[train_idx])
        # A raga sung by only one artist cannot be learned from the other folds.
        keep = np.array([lab in train_labels for lab in y[test_idx]])
        skipped += int((~keep).sum())
        test_idx = test_idx[keep]
        if len(test_idx) == 0:
            continue
        clf = RagaClassifier(kind=kind, **clf_kwargs).fit(X[train_idx], list(y[train_idx]))
        proba = clf.predict_proba(X[test_idx])
        classes = np.array(clf.classes)
        order = np.argsort(proba, axis=1)[:, ::-1]
        for row, true in zip(order, y[test_idx]):
            ranked = classes[row]
            top1 += ranked[0] == true
            topk += true in ranked[:k]
            total += 1
            per_class[true][0] += ranked[0] == true
            per_class[true][1] += 1
            y_true.append(true)
            y_pred.append(ranked[0])
    return {
        "n_clips": total,
        "n_skipped_unseen_raga": skipped,
        "top1": top1 / max(total, 1),
        f"top{k}": topk / max(total, 1),
        "per_class": {c: v[0] / v[1] for c, v in sorted(per_class.items())},
        "labels": labels,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels) if total else None,
    }


def format_confusion(cm, labels, width=9) -> str:
    short = [lab[: width - 1] for lab in labels]
    lines = [" " * width + "".join(f"{s:>{width}}" for s in short) + "   <- predicted"]
    for lab, row in zip(short, cm):
        lines.append(f"{lab:<{width}}" + "".join(f"{v:>{width}}" for v in row))
    return "\n".join(lines)
