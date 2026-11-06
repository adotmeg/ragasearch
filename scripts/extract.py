"""Run pitch tracking + tonic detection for every recording (cached).

    python scripts/extract.py data/synthetic/manifest.csv
"""
import argparse

import _path  # noqa: F401

from ragaid.data import get_pitch_track, read_manifest

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("--cache", default="cache/pitch")
ap.add_argument("--refresh", action="store_true")
args = ap.parse_args()

recs = read_manifest(args.manifest)
for i, rec in enumerate(recs):
    t = get_pitch_track(rec, args.cache, refresh=args.refresh)
    voiced = (t.f0_hz > 0).mean()
    print(f"[{i + 1}/{len(recs)}] {rec.path}  tonic={t.tonic_hz:.1f} Hz  voiced={voiced:.0%}")
