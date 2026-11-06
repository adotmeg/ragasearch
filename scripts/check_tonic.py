"""Compare automatic tonic estimates against annotated tonics in a manifest.

Tonic errors break everything downstream, so check this first on any new data.
Estimates are made on short excerpts, as the app would have to.

    python scripts/check_tonic.py data/saraga_hindustani.csv --seconds 30 120 -j 4
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import _path  # noqa: F401
import numpy as np

from ragaid.data import read_manifest
from ragaid.pitch import estimate_tonic, load_audio


def signed_cents(est, ref):
    """Octave-folded error in (-600, 600]."""
    return (1200 * np.log2(est / ref) + 600) % 1200 - 600


def classify(err, tol):
    if abs(err) <= tol:
        return "ok"
    if abs(err + 500) <= tol:
        return "picked Pa (-500)"
    if abs(err - 500) <= tol:
        return "picked Ma (+500)"
    return "other"


def work(args):
    path, ref, seconds, offset_frac = args
    import librosa

    total = librosa.get_duration(path=path)
    offset = max(0.0, total * offset_frac - seconds / 2)
    est = estimate_tonic(load_audio(path, offset=offset, duration=seconds))
    return path, ref, est


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--seconds", type=float, nargs="+", default=[30])
    ap.add_argument("--where", type=float, default=0.5, help="excerpt centre as a fraction of the recording")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tolerance", type=float, default=50, help="cents, octave-agnostic")
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    recs = [r for r in read_manifest(args.manifest) if r.tonic_hz and r.path][: args.limit]
    for secs in args.seconds:
        kinds = Counter()
        with ProcessPoolExecutor(args.jobs) as pool:
            results = list(pool.map(work, [(r.path, r.tonic_hz, secs, args.where) for r in recs]))
        for path, ref, est in results:
            err = signed_cents(est, ref)
            kind = classify(err, args.tolerance)
            kinds[kind] += 1
            if args.verbose or kind != "ok":
                print(f"  {kind:<18} annotated={ref:6.1f}  estimated={est:6.1f}  {err:+5.0f} c  {path.split('/')[-1]}")
        print(f"=== {secs:g}s excerpts, {len(recs)} recordings: " +
              ", ".join(f"{k} {n} ({n / len(recs):.0%})" for k, n in kinds.most_common()))
