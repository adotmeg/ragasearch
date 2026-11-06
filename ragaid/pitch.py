"""Audio loading, predominant pitch tracking and tonic (Sa) identification.

Essentia is used when available (it has algorithms built specifically for
Indian art music). Otherwise we fall back to librosa's pYIN and a simple
pitch-histogram tonic guess, which is far less reliable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import essentia
    import essentia.standard as es

    essentia.log.warningActive = False
    HAVE_ESSENTIA = True
except ImportError:  # pragma: no cover
    HAVE_ESSENTIA = False

SAMPLE_RATE = 44100
HOP_SIZE = 128  # ~2.9 ms per pitch frame at 44.1 kHz


@dataclass
class PitchTrack:
    times: np.ndarray  # seconds, shape (n,)
    f0_hz: np.ndarray  # Hz, 0 where unvoiced, shape (n,)
    tonic_hz: float

    @property
    def hop_seconds(self) -> float:
        return float(self.times[1] - self.times[0]) if len(self.times) > 1 else HOP_SIZE / SAMPLE_RATE

    def save(self, path) -> None:
        np.savez_compressed(path, times=self.times, f0_hz=self.f0_hz, tonic_hz=self.tonic_hz)

    @classmethod
    def load(cls, path) -> "PitchTrack":
        d = np.load(path)
        return cls(times=d["times"], f0_hz=d["f0_hz"], tonic_hz=float(d["tonic_hz"]))


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any audio file as mono float32 at `sr`."""
    import librosa

    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def extract_pitch(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Predominant melody f0 (Hz, 0 = unvoiced) and frame times (s)."""
    if HAVE_ESSENTIA and sr == SAMPLE_RATE:
        eq = es.EqualLoudness(sampleRate=sr)(audio)
        f0, conf = es.PredominantPitchMelodia(
            sampleRate=sr, hopSize=HOP_SIZE, frameSize=2048, minFrequency=60, maxFrequency=1200
        )(eq)
        f0 = np.asarray(f0, dtype=np.float32)
        f0[np.asarray(conf) <= 0] = 0.0
        times = np.arange(len(f0)) * HOP_SIZE / sr
        return times, f0

    import librosa

    hop = 512
    f0, voiced, _ = librosa.pyin(audio, fmin=60, fmax=1200, sr=sr, hop_length=hop, frame_length=2048)
    f0 = np.where(voiced, f0, 0.0).astype(np.float32)
    f0 = np.nan_to_num(f0)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    return times, f0


def estimate_tonic(audio: np.ndarray, sr: int = SAMPLE_RATE, f0_hz: np.ndarray | None = None) -> float:
    """Estimate the tonic (Sa) frequency in Hz.

    Essentia's TonicIndianArtMusic (Salamon et al. 2012) uses the drone
    (tanpura/shruti box), so it works best on recordings that contain one.
    """
    if HAVE_ESSENTIA and sr == SAMPLE_RATE:
        return float(es.TonicIndianArtMusic(sampleRate=sr, minTonicFrequency=100, maxTonicFrequency=375)(audio))

    # Fallback: the most frequent pitch class, mapped into a typical Sa range.
    # Crude; supply the tonic manually if you can.
    if f0_hz is None:
        _, f0_hz = extract_pitch(audio, sr)
    voiced = f0_hz[f0_hz > 0]
    if len(voiced) == 0:
        raise ValueError("No voiced frames found; cannot estimate tonic.")
    cents = 1200 * np.log2(voiced / 55.0)
    hist, edges = np.histogram(np.mod(cents, 1200), bins=120, range=(0, 1200))
    peak_cents = edges[np.argmax(hist)] + 5
    tonic = 55.0 * 2 ** (peak_cents / 1200)
    while tonic < 100:
        tonic *= 2
    return float(tonic)


def analyze(path_or_audio, tonic_hz: float | None = None, sr: int = SAMPLE_RATE) -> PitchTrack:
    """Full front end: audio -> PitchTrack (pitch contour + tonic)."""
    audio = load_audio(path_or_audio, sr) if isinstance(path_or_audio, str) else path_or_audio
    times, f0 = extract_pitch(audio, sr)
    if tonic_hz is None:
        tonic_hz = estimate_tonic(audio, sr, f0)
    return PitchTrack(times=times, f0_hz=f0, tonic_hz=float(tonic_hz))
