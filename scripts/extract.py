"""Run pitch tracking + tonic detection for every recording (cached).

    python scripts/extract.py data/synthetic/manifest.csv -j 4
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import _path  # noqa: F401

from ragaid.data import get_pitch_track, read_manifest


def work(rec, cache, refresh):
    t = get_pitch_track(rec, cache, refresh=refresh)
    return rec.path, t.tonic_hz, float((t.f0_hz > 0).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--cache", default="cache/pitch")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=1)
    args = ap.parse_args()

    recs = read_manifest(args.manifest)
    with ProcessPoolExecutor(args.jobs) as pool:
        futs = [pool.submit(work, r, args.cache, args.refresh) for r in recs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                path, tonic, voiced = f.result()
                print(f"[{i}/{len(recs)}] {path}  tonic={tonic:.1f} Hz  voiced={voiced:.0%}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(recs)}] FAILED: {e}", flush=True)
