"""HTTP API for the mobile app.

    RAGA_MODEL=models/raga.joblib uvicorn api.main:app --reload

    curl -F "audio=@clip.wav" http://localhost:8000/identify
    curl -F "audio=@clip.m4a" -F "tonic_hz=146.8" http://localhost:8000/identify
"""
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from ragaid.features import SVARAS, pitch_class_distribution, to_cents
from ragaid.model import RagaClassifier
from ragaid.pitch import analyze

app = FastAPI(title="Raga ID", version="0.1.0")
MAX_SECONDS = 120


@lru_cache(maxsize=1)
def model() -> RagaClassifier:
    path = os.environ.get("RAGA_MODEL", "models/raga.joblib")
    if not Path(path).exists():
        raise RuntimeError(f"model not found at {path}; run scripts/train.py first")
    return RagaClassifier.load(path)


@app.get("/health")
def health():
    return {"ok": True, "ragas": model().classes}


@app.post("/identify")
async def identify(audio: UploadFile = File(...), tonic_hz: float | None = Form(None), k: int = Form(3)):
    suffix = Path(audio.filename or "clip.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp.flush()
        try:
            track = analyze(tmp.name, tonic_hz=tonic_hz)
        except Exception as e:  # undecodable audio, silence, ...
            raise HTTPException(status_code=422, detail=f"could not analyse audio: {e}")

    duration = float(track.times[-1]) if len(track.times) else 0.0
    if duration > MAX_SECONDS:
        raise HTTPException(status_code=413, detail=f"clip longer than {MAX_SECONDS}s")
    voiced = float((track.f0_hz > 0).mean()) if len(track.f0_hz) else 0.0
    if voiced < 0.2:
        raise HTTPException(status_code=422, detail="no clear melody found; try a louder or longer clip")

    # Svara usage, so the app can show which notes were heard.
    pcd = pitch_class_distribution(to_cents(track.f0_hz, track.tonic_hz), bins=12, smooth_bins=0)
    return {
        "tonic_hz": round(track.tonic_hz, 2),
        "tonic_source": "user" if tonic_hz else "estimated",
        "duration_s": round(duration, 1),
        "voiced_fraction": round(voiced, 2),
        "predictions": [{"raga": r, "confidence": round(p, 4)} for r, p in model().top_k(track, k)],
        "svara_profile": {s: round(float(v), 4) for s, v in zip(SVARAS, pcd)},
    }
