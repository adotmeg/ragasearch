"""HTTP API + web app.

    RAGA_MODEL=models/raga.joblib uvicorn api.main:app --host 0.0.0.0 --port 8000

    open http://localhost:8000            (the app; the microphone needs localhost or HTTPS)
    curl -F "audio=@clip.wav" http://localhost:8000/identify
    curl -F "audio=@clip.m4a" -F "tonic_hz=146.8" http://localhost:8000/identify
"""
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from ragaid.features import SVARAS, pitch_class_distribution, to_cents
from ragaid.model import RagaClassifier
from ragaid.pitch import analyze, load_audio

app = FastAPI(title="Raga ID", version="0.2.0")
MAX_SECONDS = 120  # longer uploads are trimmed to their first MAX_SECONDS
RAGA_INFO = json.loads((Path(__file__).resolve().parents[1] / "ragaid" / "ragas.json").read_text())


@lru_cache(maxsize=1)
def model() -> RagaClassifier:
    path = os.environ.get("RAGA_MODEL", "models/raga.joblib")
    if not Path(path).exists():
        raise RuntimeError(f"model not found at {path}; run scripts/train.py first")
    return RagaClassifier.load(path)


def info(raga: str) -> dict:
    d = RAGA_INFO.get(raga, {})
    return {"display": d.get("display", raga), "thaat": d.get("thaat"), "time": d.get("time"),
            "svaras": d.get("svaras")}


@app.get("/health")
def health():
    return {"ok": True, "ragas": model().classes}


@app.get("/ragas")
def ragas():
    return {r: info(r) for r in model().classes}


@app.post("/identify")
async def identify(audio: UploadFile = File(...), tonic_hz: float | None = Form(None), k: int = Form(3)):
    if tonic_hz is not None and not 50 <= tonic_hz <= 500:
        raise HTTPException(status_code=422, detail="tonic_hz must be between 50 and 500")
    suffix = Path(audio.filename or "clip.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp.flush()
        try:
            samples = load_audio(tmp.name, duration=MAX_SECONDS)
        except Exception as e:  # undecodable / unsupported format
            raise HTTPException(status_code=422, detail=f"could not decode audio: {e}")
    if len(samples) < 44100 * 2:
        raise HTTPException(status_code=422, detail="clip too short; record at least a few seconds")
    if np.sqrt(np.mean(samples ** 2)) < 1e-4:
        raise HTTPException(status_code=422, detail="no clear melody found; try a louder or longer clip")
    try:
        track = analyze(samples, tonic_hz=tonic_hz)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"could not analyse audio: {e}")

    voiced = float((track.f0_hz > 0).mean()) if len(track.f0_hz) else 0.0
    if voiced < 0.2:
        raise HTTPException(status_code=422, detail="no clear melody found; try a louder or longer clip")

    # Svara usage, so the app can show which notes were heard.
    pcd = pitch_class_distribution(to_cents(track.f0_hz, track.tonic_hz), bins=12, smooth_bins=0)
    return {
        "tonic_hz": round(track.tonic_hz, 2),
        "tonic_source": "user" if tonic_hz else "estimated",
        "duration_s": round(len(samples) / 44100, 1),
        "voiced_fraction": round(voiced, 2),
        "predictions": [{"raga": r, "confidence": round(p, 4), **info(r)} for r, p in model().top_k(track, k)],
        "svara_profile": {s: round(float(v), 4) for s, v in zip(SVARAS, pcd)},
    }


# The web app (mounted last so the API routes above take precedence).
app.mount("/", StaticFiles(directory=Path(__file__).resolve().parents[1] / "app", html=True), name="app")
