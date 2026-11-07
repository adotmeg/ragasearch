"""Train the final model on all recordings and save it.

    python scripts/train.py data/synthetic/manifest.csv --clip 30 --out models/raga.joblib
"""
import argparse
from pathlib import Path

import _path  # noqa: F401

from ragaid.data import read_manifest
from ragaid.model import RagaClassifier
from ragaid.training import build_clip_dataset

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("--clip", type=float, default=30, help="training clip length (s); match your app's query length")
ap.add_argument("--hop", type=float, default=None, help="clip hop (s); smaller = more overlapping clips")
ap.add_argument("--model", choices=["svm", "knn"], default="svm")
ap.add_argument("--out", default="models/raga.joblib")
ap.add_argument("--cache", default="cache/pitch")
args = ap.parse_args()

clf = RagaClassifier(kind=args.model)
X, y, _, _ = build_clip_dataset(read_manifest(args.manifest), clf, args.clip, args.hop, cache_dir=args.cache)
clf.fit(X, list(y))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
clf.save(args.out)
print(f"trained {args.model} on {len(y)} clips, {len(clf.classes)} ragas -> {args.out}")
