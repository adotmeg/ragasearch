"""Datasets as a simple CSV manifest, plus a per-recording pitch cache.

Manifest columns:
    path       audio file path
    raga       label
    artist     used to split train/test so that no artist is in both
    tonic_hz   optional; if empty, the tonic is estimated automatically
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .pitch import PitchTrack, analyze


@dataclass
class Recording:
    path: str
    raga: str
    artist: str
    tonic_hz: float | None = None

    @property
    def cache_key(self) -> str:
        return hashlib.sha1(str(Path(self.path).resolve()).encode()).hexdigest()[:16]


def read_manifest(path) -> list[Recording]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tonic = row.get("tonic_hz") or ""
            out.append(Recording(row["path"], row["raga"].strip(), (row.get("artist") or "unknown").strip(),
                                 float(tonic) if tonic.strip() else None))
    return out


def write_manifest(recs: list[Recording], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "raga", "artist", "tonic_hz"])
        for r in recs:
            w.writerow([r.path, r.raga, r.artist, "" if r.tonic_hz is None else f"{r.tonic_hz:.3f}"])


def get_pitch_track(rec: Recording, cache_dir="cache/pitch", refresh: bool = False) -> PitchTrack:
    """Pitch tracking is the slow step, so cache one .npz per recording."""
    cache = Path(cache_dir) / f"{rec.cache_key}.npz"
    if cache.exists() and not refresh:
        track = PitchTrack.load(cache)
        if rec.tonic_hz is not None:
            track.tonic_hz = rec.tonic_hz
        return track
    track = analyze(rec.path, tonic_hz=rec.tonic_hz)
    cache.parent.mkdir(parents=True, exist_ok=True)
    track.save(cache)
    return track


# --------------------------------------------------------------------------- Saraga

def _first_name(items) -> str | None:
    for it in items or []:
        name = it.get("name") or it.get("common_name")
        if name:
            return name.strip()
    return None


def saraga_manifest(tradition: str = "hindustani", data_home: str | None = None,
                    download: bool = False) -> list[Recording]:
    """Build a manifest from the Saraga dataset via mirdata.

    The audio is large (tens of GB). `download=True` fetches it into
    `data_home` (default ~/mir_datasets/saraga_<tradition>). Tracks with no
    raga, or several ragas (ragamalika), are skipped.
    """
    import mirdata

    ds = mirdata.initialize(f"saraga_{tradition}", data_home=data_home)
    if download:
        ds.download()
    recs = []
    for tid, track in ds.load_tracks().items():
        if not track.audio_path or not Path(track.audio_path).exists():
            continue
        meta = track.metadata or {}
        ragas = meta.get("raags") or meta.get("raaga") or meta.get("ragas") or []
        names = {(r.get("name") or r.get("common_name") or "").strip() for r in ragas} - {""}
        if len(names) != 1:
            continue
        artist = _first_name(meta.get("album_artists")) or _first_name(meta.get("artists")) or "unknown"
        tonic = None
        try:
            tonic = float(track.tonic) if track.tonic else None
        except Exception:
            pass
        recs.append(Recording(track.audio_path, names.pop(), artist, tonic))
    return recs
