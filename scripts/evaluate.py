"""Artist-wise cross-validation, reported per query clip length.

    python scripts/evaluate.py data/synthetic/manifest.csv --clip 10 20 30 --model svm
"""
import argparse

import _path  # noqa: F401

from ragaid.data import read_manifest
from ragaid.model import RagaClassifier
from ragaid.training import build_clip_dataset, cross_validate, format_confusion

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("--clip", type=float, nargs="+", default=[10, 20, 30])
ap.add_argument("--model", choices=["svm", "knn"], default="svm")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--no-transitions", action="store_true")
ap.add_argument("--cache", default="cache/pitch")
ap.add_argument("--group-by", choices=["artist", "recording"], default="artist",
                help="artist (strict, default): test artists never seen in training. "
                     "recording: test recordings unseen, but the same artist may appear in training.")
ap.add_argument("--min-recordings", type=int, default=1, help="drop ragas with fewer recordings")
args = ap.parse_args()

from collections import Counter

recs = read_manifest(args.manifest)
counts = Counter(r.raga for r in recs)
recs = [r for r in recs if counts[r.raga] >= args.min_recordings]
opts = dict(use_transitions=not args.no_transitions)
print(f"split by {args.group_by}; {len(recs)} recordings, {len({r.raga for r in recs})} ragas, {len({r.artist for r in recs})} artists\n")
for secs in args.clip:
    X, y, g, rec_ids = build_clip_dataset(recs, RagaClassifier(kind=args.model, **opts), secs,
                                    cache_dir=args.cache, verbose=False)
    groups = g if args.group_by == "artist" else rec_ids
    res = cross_validate(X, y, groups, kind=args.model, n_splits=args.folds, **opts)
    print(f"=== {secs:g}s clips: {res['n_clips']} test clips | top-1 {res['top1']:.1%} | top-3 {res['top3']:.1%}")
    if res["n_skipped_unseen_raga"]:
        print(f"    ({res['n_skipped_unseen_raga']} clips skipped: raga absent from that fold's training data)")
    for c, acc in res["per_class"].items():
        print(f"    {c:<20}{acc:6.1%}")
    if res["confusion"] is not None:
        print(format_confusion(res["confusion"], res["labels"]))
    print()
