# raga-id

Identify the raga of a Hindustani or Carnatic music clip: a baseline pipeline, an evaluation harness and an HTTP API for a mobile app.

```
audio ─► predominant pitch (Melodia) ─► tonic / Sa (TonicIndianArtMusic) ─► cents relative to Sa
      ─► features: 120-bin pitch-class distribution + 12×12 svara transition matrix
      ─► calibrated SVM (or kNN) ─► top-k ragas with confidence
```

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Quick start (synthetic toy data)

The synthetic data checks the plumbing only. It is not real music.

```bash
python scripts/make_synthetic_demo.py --artists 6          # 4 toy ragas x 6 "artists"
python scripts/extract.py data/synthetic/manifest.csv      # pitch + tonic, cached in cache/pitch
python scripts/evaluate.py data/synthetic/manifest.csv --clip 10 20 30
python scripts/train.py data/synthetic/manifest.csv --clip 20
python scripts/identify.py some_clip.wav [--tonic 146.8]
RAGA_MODEL=models/raga.joblib uvicorn api.main:app --reload
curl -F "audio=@some_clip.wav" localhost:8000/identify
pytest
```

## Real data

**Saraga** (open, with tonic and raga annotations), via mirdata:

```bash
python scripts/make_manifest_saraga.py --tradition hindustani --download   # large download
python scripts/check_tonic.py data/saraga_hindustani.csv --limit 30        # how good is auto-Sa?
python scripts/evaluate.py data/saraga_hindustani.csv --clip 10 30 60
```

**Your own recordings:** write a CSV with columns `path,raga,artist,tonic_hz` (leave `tonic_hz` empty to estimate it automatically).

## Evaluation rules
- Splits are **by artist** (GroupKFold). Clips from one concert must never appear in both train and test, or accuracy is wildly inflated.
- Report accuracy **as a function of clip length**. That curve decides how long the app must listen.
- Measure **tonic accuracy separately** (`check_tonic.py`). Most real-world failures are Sa detected as Pa or Ma.

## Layout
| | |
|---|---|
| `ragaid/pitch.py` | audio loading, pitch tracking, tonic estimation (Essentia, with a librosa fallback) |
| `ragaid/features.py` | cents, pitch-class distribution, svara sequence and transitions, clipping |
| `ragaid/model.py` | `RagaClassifier` (fit / top_k / save / load) |
| `ragaid/data.py` | CSV manifests, pitch cache, Saraga loader |
| `ragaid/training.py` | clip datasets, artist-wise cross-validation, confusion matrix |
| `ragaid/synth.py` | toy raga renderer used by the demo and tests |
| `api/main.py` | FastAPI `/identify` and `/health` |

## Roadmap
1. **Baseline on Saraga.** Get real numbers. Inspect the confusion matrix for raga pairs that share a scale.
2. **Tonic robustness.** Evaluate on Saraga. Add a melody-based fallback and let users set or correct Sa in the app.
3. **Source separation** (Demucs) before pitch tracking, for accompanied recordings.
4. **Sequence model** (1D CNN or transformer on the tonic-normalised pitch contour, or TDMS features) for phrase-level discrimination.
5. **Recording fingerprinting** (Shazam-style) for exact matches against a catalogue.
6. **App.** Stream audio and show a live top-3 that sharpens over time, plus a "Wrong? Tell us" feedback loop.
