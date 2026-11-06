"""Build a manifest from the Saraga dataset (via mirdata).

    python scripts/make_manifest_saraga.py --tradition hindustani --download
    python scripts/make_manifest_saraga.py --tradition carnatic --data-home /data/saraga_carnatic
"""
import argparse
from collections import Counter

import _path  # noqa: F401

from ragaid.data import saraga_manifest, write_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--tradition", choices=["hindustani", "carnatic"], default="hindustani")
ap.add_argument("--data-home", default=None)
ap.add_argument("--download", action="store_true", help="download audio + annotations (large!)")
ap.add_argument("--min-recordings", type=int, default=3, help="drop ragas with fewer recordings")
ap.add_argument("--out", default=None)
args = ap.parse_args()

recs = saraga_manifest(args.tradition, args.data_home, args.download)
counts = Counter(r.raga for r in recs)
recs = [r for r in recs if counts[r.raga] >= args.min_recordings]
out = args.out or f"data/saraga_{args.tradition}.csv"
write_manifest(recs, out)
print(f"{len(recs)} recordings, {len({r.raga for r in recs})} ragas -> {out}")
for raga, n in Counter(r.raga for r in recs).most_common():
    print(f"  {raga:<25}{n}")
