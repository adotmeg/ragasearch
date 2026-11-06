"""Tonic-normalised melodic features.

Two complementary views of a melody:

1. Pitch-class distribution (PCD): how much time is spent at each pitch
   (relative to Sa), folded into one octave at fine resolution (default
   120 bins = 10 cents). Captures the scale, note emphasis and, at fine
   resolution, intonation (e.g. the low komal Re of Bhairav vs. Todi).

2. Svara transition matrix: how the melody moves between the 12 svaras.
   This is a cheap first step toward phrase information (aaroh/avaroh,
   vakra movements) that separates ragas that share a scale.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .pitch import PitchTrack

SVARAS = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]


def to_cents(f0_hz: np.ndarray, tonic_hz: float) -> np.ndarray:
    """Voiced f0 values in cents above the tonic (unvoiced frames dropped)."""
    voiced = f0_hz[f0_hz > 0]
    return 1200.0 * np.log2(voiced / tonic_hz)


def pitch_class_distribution(cents: np.ndarray, bins: int = 120, smooth_bins: float = 1.0) -> np.ndarray:
    """Octave-folded, circularly smoothed, L1-normalised histogram."""
    if len(cents) == 0:
        return np.full(bins, 1.0 / bins)
    # Centre bins on the svaras: shift by half a bin so 0 cents falls in bin 0's middle.
    folded = np.mod(cents + 600.0 / bins, 1200.0)
    hist, _ = np.histogram(folded, bins=bins, range=(0, 1200))
    hist = hist.astype(np.float64)
    if smooth_bins > 0:
        hist = gaussian_filter1d(hist, smooth_bins, mode="wrap")
    return hist / max(hist.sum(), 1e-12)


def svara_sequence(cents: np.ndarray, min_frames: int = 15, tolerance: float = 35.0) -> np.ndarray:
    """Quantise the contour to stable svaras (0-11).

    A frame counts if it lies within `tolerance` cents of a semitone; a note
    is kept only if it is held for `min_frames` consecutive frames. Glides
    (meend) between notes are therefore skipped, which is what we want for
    note-to-note transitions.
    """
    if len(cents) == 0:
        return np.array([], dtype=int)
    nearest = np.round(cents / 100.0)
    stable = np.abs(cents - nearest * 100.0) <= tolerance
    pc = np.mod(nearest, 12).astype(int)
    pc[~stable] = -1

    notes = []
    run_val, run_len = pc[0], 0
    for v in pc:
        if v == run_val:
            run_len += 1
        else:
            if run_val >= 0 and run_len >= min_frames:
                notes.append(run_val)
            run_val, run_len = v, 1
    if run_val >= 0 and run_len >= min_frames:
        notes.append(run_val)

    # Collapse repeats (S S S R -> S R).
    out = [n for i, n in enumerate(notes) if i == 0 or n != notes[i - 1]]
    return np.asarray(out, dtype=int)


def transition_matrix(seq: np.ndarray) -> np.ndarray:
    """12x12 row-normalised svara transition probabilities, flattened."""
    m = np.zeros((12, 12))
    for a, b in zip(seq[:-1], seq[1:]):
        m[a, b] += 1
    m += 1e-3  # smoothing so unseen transitions are not exactly zero
    m /= m.sum(axis=1, keepdims=True)
    return m.ravel()


def frames_for(track: PitchTrack, seconds: float) -> int:
    return max(1, int(round(seconds / track.hop_seconds)))


def extract_features(track: PitchTrack, bins: int = 120, use_transitions: bool = True) -> np.ndarray:
    """Feature vector for one pitch track (or clip of one)."""
    cents = to_cents(track.f0_hz, track.tonic_hz)
    feats = [pitch_class_distribution(cents, bins=bins)]
    if use_transitions:
        # Hold time of ~45 ms to count as a note, independent of the hop size.
        seq = svara_sequence(cents, min_frames=frames_for(track, 0.045))
        feats.append(transition_matrix(seq))
    return np.concatenate(feats)


def clip_tracks(track: PitchTrack, clip_seconds: float, hop_seconds: float | None = None,
                min_voiced_fraction: float = 0.3) -> list[PitchTrack]:
    """Cut a long pitch track into fixed-length clips (simulating app queries)."""
    hop_seconds = hop_seconds or clip_seconds
    n = frames_for(track, clip_seconds)
    step = frames_for(track, hop_seconds)
    clips = []
    for start in range(0, max(1, len(track.f0_hz) - n + 1), step):
        f0 = track.f0_hz[start:start + n]
        if np.mean(f0 > 0) < min_voiced_fraction:
            continue  # mostly silence / percussion solo
        clips.append(PitchTrack(times=track.times[start:start + n], f0_hz=f0, tonic_hz=track.tonic_hz))
    return clips
