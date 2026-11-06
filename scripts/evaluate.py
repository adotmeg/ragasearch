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
args = ap.parse_args()

recs = read_manifest(args.manifest)
opts = dict(use_transitions=not args.no_transitions)
print(f"{len(recs)} recordings, {len({r.raga for r in recs})} ragas, {len({r.artist for r in recs})} artists\n")
for secs in args.clip:
    X, y, g, _ = build_clip_dataset(recs, RagaClassifier(kind=args.model, **opts), secs,
                                    cache_dir=args.cache, verbose=False)
    res = cross_validate(X, y, g, kind=args.model, n_splits=args.folds, **opts)
    print(f"=== {secs:g}s clips: {res['n_clips']} test clips | top-1 {res['top1']:.1%} | top-3 {res['top3']:.1%}")
    if res["n_skipped_unseen_raga"]:
        print(f"    ({res['n_skipped_unseen_raga']} clips skipped: raga absent from that fold's training artists)")
    for c, acc in res["per_class"].items():
        print(f"    {c:<20}{acc:6.1%}")
    if res["confusion"] is not None:
        print(format_confusion(res["confusion"], res["labels"]))
    print()
