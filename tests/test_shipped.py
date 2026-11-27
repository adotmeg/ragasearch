"""Guards for what gets deployed: the committed model and audio decoding."""

import shutil
import subprocess
from pathlib import Path

import pytest

from ragaid.model import RagaClassifier
from ragaid.pitch import analyze, load_audio
from ragaid.synth import render

SHIPPED = Path(__file__).resolve().parents[1] / "models" / "raga_saraga_hindustani_v1.joblib"


def test_shipped_model_loads_and_predicts():
    # Fails if a dependency upgrade breaks unpickling of the committed model.
    clf = RagaClassifier.load(SHIPPED)
    assert len(clf.classes) == 21 and "Yaman Kalyan" in clf.classes
    track = analyze(render("Yaman", 15, 150.0, seed=1), tonic_hz=150.0)
    top = clf.top_k(track, 3)
    assert len(top) == 3 and abs(sum(p for _, p in clf.top_k(track, 21)) - 1) < 1e-6


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_m4a_decodes_via_ffmpeg(tmp_path):
    path = tmp_path / "memo.m4a"  # iPhone voice memos are AAC in .m4a, which libsndfile can't read
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=220:d=3", "-c:a", "aac", str(path)]
    subprocess.run(cmd, check=True)
    audio = load_audio(str(path))
    assert abs(len(audio) / 44100 - 3) < 0.1
    assert len(load_audio(str(path), offset=1.0, duration=1.0)) == 44100
