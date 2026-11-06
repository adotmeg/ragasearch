"""Generate a toy dataset (4 ragas x N 'artists') to smoke-test the pipeline.

    python scripts/make_synthetic_demo.py --out data/synthetic --artists 6
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import soundfile as sf

from ragaid.data import Recording, write_manifest
from ragaid.synth import TOY_RAGAS, render

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic")
ap.add_argument("--artists", type=int, default=6)
ap.add_argument("--seconds", type=float, default=90)
ap.add_argument("--blank-tonic", action="store_true",
                help="leave tonic_hz empty so it is estimated (tests tonic detection)")
args = ap.parse_args()

rng = np.random.default_rng(0)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
recs = []
for a in range(args.artists):
    tonic = float(rng.uniform(130, 175))  # each "artist" sings at their own Sa
    for raga in TOY_RAGAS:
        path = out / f"artist{a}_{raga}.wav"
        sf.write(path, render(raga, args.seconds, tonic, seed=int(rng.integers(1 << 31))), 44100)
        recs.append(Recording(str(path.resolve()), raga, f"artist{a}", None if args.blank_tonic else tonic))
        print("wrote", path, f"(Sa = {tonic:.1f} Hz)")
write_manifest(recs, out / "manifest.csv")
print("manifest:", out / "manifest.csv")
