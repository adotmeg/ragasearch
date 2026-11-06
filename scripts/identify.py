"""Identify the raga of an audio file.

    python scripts/identify.py some_clip.wav [--tonic 146.8] [--model models/raga.joblib]
"""
import argparse

import _path  # noqa: F401

from ragaid.model import RagaClassifier
from ragaid.pitch import analyze

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("--model", default="models/raga.joblib")
ap.add_argument("--tonic", type=float, default=None, help="Sa in Hz, if known")
ap.add_argument("-k", type=int, default=3)
args = ap.parse_args()

clf = RagaClassifier.load(args.model)
track = analyze(args.audio, tonic_hz=args.tonic)
print(f"tonic (Sa): {track.tonic_hz:.1f} Hz")
for i, (raga, p) in enumerate(clf.top_k(track, args.k), 1):
    print(f"  {i}. {raga:<20}{p:6.1%}")
