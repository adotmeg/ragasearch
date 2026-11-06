"""Synthetic 'performances' for smoke-testing the pipeline end to end.

These are NOT real music. Each toy raga has a scale, a few characteristic
phrases (pakad) and note weights; we render a voice-like tone with glides
plus a tanpura-like drone (Sa-Pa-Sa) so tonic detection has something to
lock on to. Use this to check the plumbing, then switch to real data.
"""
from __future__ import annotations

import numpy as np

# Svara offsets in semitones from Sa: S=0 r=1 R=2 g=3 G=4 m=5 M=6 P=7 d=8 D=9 n=10 N=11
TOY_RAGAS = {
    # Bhoop: S R G P D (pentatonic)
    "Bhoop": {"notes": [0, 2, 4, 7, 9], "weights": [1.0, 0.8, 1.4, 1.0, 1.0],
              "phrases": [[4, 2, 0, -3, 0], [0, 2, 4, 7, 4, 2, 0], [7, 9, 7, 4, 2, 4, 2, 0]]},
    # Deshkar: same notes as Bhoop, but D and P emphasised, G-R-S avoided
    "Deshkar": {"notes": [0, 2, 4, 7, 9], "weights": [1.0, 0.3, 0.8, 1.5, 1.4],
                "phrases": [[9, 7, 4, 7, 9, 7], [0, 2, 4, 7, 9, 12], [4, 7, 9, 7, 4, 0]]},
    # Malkauns: S g m d n
    "Malkauns": {"notes": [0, 3, 5, 8, 10], "weights": [1.0, 1.0, 1.4, 1.0, 0.9],
                 "phrases": [[5, 3, 5, 8, 5, 3, 0], [-2, -4, -2, 0], [3, 5, 8, 10, 8, 5]]},
    # Yaman: N R G M P D N (tivra Ma)
    "Yaman": {"notes": [0, 2, 4, 6, 7, 9, 11], "weights": [1.0, 0.9, 1.4, 0.8, 0.9, 0.8, 1.2],
              "phrases": [[-1, 2, 4, 2, 0], [4, 6, 9, 7, 6, 4], [-1, 2, 4, 6, 7]]},
}


def render(raga: str, seconds: float = 60.0, tonic_hz: float = 146.83, sr: int = 44100,
           seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spec = TOY_RAGAS[raga]
    notes, w = np.array(spec["notes"]), np.array(spec["weights"])
    w = w / w.sum()

    # Build a sequence of (semitone, duration) events: phrases mixed with weighted wandering.
    events = []
    t = 0.0
    while t < seconds:
        if rng.random() < 0.5:
            seq = spec["phrases"][rng.integers(len(spec["phrases"]))]
        else:
            seq = rng.choice(notes, size=rng.integers(3, 7), p=w) + 12 * rng.choice([0, 0, 0, -1, 1], size=None)
            seq = list(np.atleast_1d(seq))
        for n in seq:
            d = float(rng.uniform(0.25, 0.9))
            events.append((n, d))
            t += d
        if rng.random() < 0.3:
            events.append((None, float(rng.uniform(0.2, 0.6))))  # breath
            t += events[-1][1]

    n_samples = int(seconds * sr)
    semis = np.full(n_samples, np.nan)
    pos = 0
    for n, d in events:
        k = int(d * sr)
        if n is not None:
            semis[pos:pos + k] = n
        pos += k
        if pos >= n_samples:
            break
    semis = semis[:n_samples]

    voiced = ~np.isnan(semis)
    filled = np.where(voiced, semis, 0.0)
    # Glide between notes (meend), plus a slight vibrato.
    glide = int(0.06 * sr)
    kernel = np.ones(glide) / glide
    smooth = np.convolve(filled, kernel, mode="same")
    smooth = np.where(voiced, smooth, filled)
    vibrato = 0.15 * np.sin(2 * np.pi * 5.5 * np.arange(n_samples) / sr)
    f0 = tonic_hz * 2 ** ((smooth + vibrato * voiced) / 12)

    phase = 2 * np.pi * np.cumsum(f0) / sr
    voice = sum((0.6 / h) * np.sin(h * phase) for h in range(1, 6))
    env = np.convolve(voiced.astype(float), np.ones(int(0.02 * sr)) / int(0.02 * sr), mode="same")
    voice *= env

    tt = np.arange(n_samples) / sr
    drone = sum(a * np.sin(2 * np.pi * f * tt) for f, a in
                [(tonic_hz / 2, 0.10), (tonic_hz * 0.75, 0.06), (tonic_hz, 0.08), (tonic_hz * 2, 0.03)])
    audio = 0.5 * voice + drone + 0.003 * rng.standard_normal(n_samples)
    return (audio / np.max(np.abs(audio)) * 0.8).astype(np.float32)
