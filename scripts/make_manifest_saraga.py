"""Build a manifest from an unzipped Saraga 1.5 dataset.

    # Hindustani: ~4 GB download (Carnatic: saraga1.5_carnatic.zip, ~13 GB)
    curl -L -o saraga.zip "https://zenodo.org/records/4301737/files/saraga1.5_hindustani.zip?download=1"
    unzip saraga.zip -d ~/mir_datasets/saraga_hindustani
    python scripts/make_manifest_saraga.py ~/mir_datasets/saraga_hindustani --out data/saraga_hindustani.csv
"""
import argparse
from collections import Counter

import _path  # noqa: F401

from ragaid.data import saraga_manifest, write_manifest

ap = argparse.ArgumentParser()
ap.add_argument("root", help="folder containing the unzipped saraga1.5_* directory")
ap.add_argument("--min-recordings", type=int, default=1, help="drop ragas with fewer recordings")
ap.add_argument("--out", default="data/saraga.csv")
args = ap.parse_args()

recs = saraga_manifest(args.root)
counts = Counter(r.raga for r in recs)
recs = [r for r in recs if counts[r.raga] >= args.min_recordings]
write_manifest(recs, args.out)
print(f"{len(recs)} recordings, {len({r.raga for r in recs})} ragas, "
      f"{len({r.artist for r in recs})} artists -> {args.out}")
print(f"with tonic annotation: {sum(r.tonic_hz is not None for r in recs)}, "
      f"with pitch file: {sum(r.pitch_path is not None for r in recs)}")
for raga, n in Counter(r.raga for r in recs).most_common():
    arts = len({r.artist for r in recs if r.raga == raga})
    print(f"  {raga:<28}{n:3d} recordings  {arts:2d} artists")
