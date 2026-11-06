"""Compare automatic tonic estimates against annotated tonics in a manifest.

Tonic errors break everything downstream, so check this first on any new data.

    python scripts/check_tonic.py data/saraga_hindustani.csv
"""
import argparse
from collections import Counter

import _path  # noqa: F401
import numpy as np

from ragaid.data import read_manifest
from ragaid.pitch import analyze

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--tolerance", type=float, default=50, help="cents, octave-agnostic")
args = ap.parse_args()

recs = [r for r in read_manifest(args.manifest) if r.tonic_hz][: args.limit]
kinds = Counter()
for r in recs:
    est = analyze(r.path).tonic_hz
    err = (1200 * np.log2(est / r.tonic_hz)) % 1200  # 0..1200, octave folded
    err = min(err, 1200 - err) if err > 600 else err
    signed = (1200 * np.log2(est / r.tonic_hz) + 600) % 1200 - 600
    kind = ("ok" if abs(signed) <= args.tolerance else
            "Pa error (+700/-500)" if min(abs(signed - 700 + 1200) % 1200, abs(signed + 500)) <= args.tolerance else
            "Ma error (+500/-700)" if min(abs(signed - 500), abs(signed + 700)) <= args.tolerance else "other")
    kinds[kind] += 1
    print(f"{kind:<22} annotated={r.tonic_hz:6.1f}  estimated={est:6.1f}  {signed:+6.0f} cents  {r.path}")
print()
for k, n in kinds.most_common():
    print(f"{k:<22}{n:4d}  ({n / len(recs):.0%})")
