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

**Saraga 1.5** (open, with tonic and raga annotations and precomputed pitch):

```bash
curl -L -o saraga.zip "https://zenodo.org/records/4301737/files/saraga1.5_hindustani.zip?download=1"
unzip saraga.zip -d ~/mir_datasets/saraga_hindustani
python scripts/make_manifest_saraga.py ~/mir_datasets/saraga_hindustani --min-recordings 2 --out data/saraga_hindustani.csv
python scripts/check_tonic.py data/saraga_hindustani.csv --seconds 30 -j 4   # how good is auto-Sa?
python scripts/evaluate.py data/saraga_hindustani.csv --clip 10 30 60
```

Baseline results are in [`docs/results_saraga_hindustani.md`](docs/results_saraga_hindustani.md). Summary: with 21 ragas and test artists the model never saw, top-1/top-3 accuracy is 52.6%/71.9% on 30-second clips. Automatic Sa detection is correct 88% of the time on 30-second clips.

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
1. ~~**Baseline on Saraga.**~~ Done; see `docs/results_saraga_hindustani.md`. **More data** is next: at least 5 recordings by at least 3 artists per raga.
2. **Tonic robustness.** Evaluate on Saraga. Add a melody-based fallback and let users set or correct Sa in the app.
3. **Source separation** (Demucs) before pitch tracking, for accompanied recordings.
4. **Sequence model** (1D CNN or transformer on the tonic-normalised pitch contour, or TDMS features) for phrase-level discrimination.
5. **Recording fingerprinting** (Shazam-style) for exact matches against a catalogue.
6. **App.** Stream audio and show a live top-3 that sharpens over time, plus a "Wrong? Tell us" feedback loop.
