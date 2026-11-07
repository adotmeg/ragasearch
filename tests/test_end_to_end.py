"""Synthetic audio -> pitch -> features -> classifier -> API. Slow-ish (~1 min)."""
import io

import numpy as np
import pytest
import soundfile as sf

from ragaid.model import RagaClassifier
from ragaid.pitch import PitchTrack, analyze
from ragaid.synth import render

SR = 44100
TONIC = 150.0


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    clf = RagaClassifier(kind="svm")
    X, y = [], []
    for raga in ["Bhoop", "Malkauns", "Yaman"]:
        for seed in range(3):
            track = analyze(render(raga, 40, TONIC * (1 + 0.05 * seed), seed=seed), tonic_hz=TONIC * (1 + 0.05 * seed))
            n = len(track.f0_hz) // 4
            for i in range(4):
                sl = slice(i * n, (i + 1) * n)
                X.append(clf.featurize(PitchTrack(track.times[sl], track.f0_hz[sl], track.tonic_hz)))
                y.append(raga)
    clf.fit(np.array(X), y)
    path = tmp_path_factory.mktemp("m") / "raga.joblib"
    clf.save(path)
    return path


def test_pitch_tracker_follows_melody():
    track = analyze(render("Yaman", 10, TONIC, seed=1), tonic_hz=TONIC)
    voiced = track.f0_hz[track.f0_hz > 0]
    assert len(voiced) > 0.5 * len(track.f0_hz)
    # Most voiced frames should sit on Yaman svaras (S R G M P D N).
    cents = np.mod(1200 * np.log2(voiced / TONIC), 1200)
    nearest = np.mod(np.round(cents / 100), 12)
    assert np.isin(nearest, [0, 2, 4, 6, 7, 9, 11]).mean() > 0.85


def test_classifier_on_unseen_performance(trained):
    clf = RagaClassifier.load(trained)
    tonic = 140.0  # a "singer" at a pitch not in training
    for raga in ["Bhoop", "Malkauns", "Yaman"]:
        track = analyze(render(raga, 20, tonic, seed=99), tonic_hz=tonic)
        assert clf.top_k(track, 1)[0][0] == raga


def test_api_identify(trained, monkeypatch):
    from fastapi.testclient import TestClient

    import api.main as api_main

    monkeypatch.setenv("RAGA_MODEL", str(trained))
    api_main.model.cache_clear()
    client = TestClient(api_main.app)

    buf = io.BytesIO()
    sf.write(buf, render("Malkauns", 15, 160.0, seed=7), SR, format="WAV")
    r = client.post("/identify", files={"audio": ("q.wav", buf.getvalue(), "audio/wav")}, data={"tonic_hz": "160"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predictions"][0]["raga"] == "Malkauns"
    assert body["tonic_source"] == "user"
    assert len(body["svara_profile"]) == 12

    buf = io.BytesIO()
    sf.write(buf, np.zeros(SR * 3, dtype=np.float32), SR, format="WAV")
    r = client.post("/identify", files={"audio": ("silence.wav", buf.getvalue(), "audio/wav")}, data={"tonic_hz": "160"})
    assert r.status_code == 422


def test_web_app_and_raga_info(trained, monkeypatch):
    from fastapi.testclient import TestClient

    import api.main as api_main

    monkeypatch.setenv("RAGA_MODEL", str(trained))
    api_main.model.cache_clear()
    client = TestClient(api_main.app)

    page = client.get("/")
    assert page.status_code == 200 and "Raga Search" in page.text
    assert client.get("/app.js").status_code == 200
    ragas = client.get("/ragas").json()
    assert ragas["Yaman"]["svaras"] is None or isinstance(ragas["Yaman"]["svaras"], str)
    assert set(ragas) == {"Bhoop", "Malkauns", "Yaman"}

    buf = io.BytesIO()
    sf.write(buf, render("Yaman", 6, 150.0, seed=3), SR, format="WAV")
    r = client.post("/identify", files={"audio": ("q.wav", buf.getvalue(), "audio/wav")}, data={"tonic_hz": "900"})
    assert r.status_code == 422  # implausible tonic rejected
