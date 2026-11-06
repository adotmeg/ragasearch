# Baseline results: Saraga 1.5 Hindustani

Date: 2026-11-06 · Model: tonic-normalised 120-bin pitch-class distribution plus a 12×12 svara transition matrix, classified with a calibrated RBF SVM.

## Data
- Saraga 1.5 Hindustani contains 108 tracks. 93 of them have exactly one raga, spread over **61 ragas** by **10 album artists**. All 93 have an annotated tonic and a precomputed pitch track.
- **39 of the 61 ragas have only one recording** and can't be evaluated on unseen recordings. The evaluation set is the **21 ragas with at least 2 recordings** (53 recordings): `data/saraga_hindustani.csv`.
- Pitch comes from Saraga's precomputed tracks and the tonic from its annotations. The section on tonic detection below measures how well the app could estimate Sa itself.

## 1. Raga accuracy (5-fold, annotated tonic)

Split **by artist** (test artists never appear in training):

| Query length | Top-1 | Top-3 |
|---|---|---|
| 10 s | 40.8% | 64.8% |
| 30 s | 52.6% | 71.9% |
| 60 s | 56.7% | 72.9% |

Split **by recording** (the same artist may appear in training): 10 s 39.9% / 63.6%, 30 s 43.0% / 63.3%, 60 s 57.4% / 75.5%, 120 s 57.6% / 72.3%.

Chance level is about 5% (21 classes). Test clips whose raga had no training data in that fold are excluded (the scripts report how many).

### Model and feature variants (split by artist, top-1 / top-3)

| Variant | 30 s | 60 s |
|---|---|---|
| **SVM, pitch distribution + transitions** | **52.6 / 71.9** | **56.7 / 72.9** |
| SVM, pitch distribution only | 50.5 / 72.4 | 55.2 / 74.0 |
| kNN, pitch distribution + transitions | 41.5 / 60.0 | 51.2 / 67.1 |
| kNN, pitch distribution only | 47.3 / 66.5 | 53.3 / 71.8 |

### Where it fails (60 s, split by artist)
Strong: Shree 96%, Miya Malhar 88%, Marwa 83%, Jog 81%, Lalat 78%.
Weak: Ramdasi Malhar 5%, Abhogi 1%, Bihag 0%, Malkauns 0%, Yaman Kalyan 0%.

Top confusions are musically related ragas:
- Ramdasi Malhar → Miya Malhar (62%)
- Bilaskhani Todi → Todi (50%)
- Multani → Shree (53%), Todi → Shree (40%)
- Hameer → Bihag (33%)
- Malkauns → Jog (52%)

The weakest ragas are almost all ones with only 2 recordings. In each fold the model has learned them from a single performance.

## 2. Automatic tonic (Sa) detection

Essentia `TonicIndianArtMusic` on an excerpt taken from the middle of each recording, compared with the annotation (octave-agnostic, ±50 cents tolerance):

| Excerpt | Correct | Picked Pa | Picked Ma | Other |
|---|---|---|---|---|
| 30 s | **88%** | 6% | 2% | 3% |
| 120 s | **95%** | 2% | 2% | 1% |

On real recordings with a tanpura, tonic detection is good. A rough end-to-end estimate for a 30-second query is 0.88 × 52.6% ≈ **46% top-1**.

## Takeaways
1. **Data is the bottleneck, not the method.** Errors concentrate on ragas with very few recordings. Next: add the Saraga Carnatic set (a separate model), CompMusic's raga datasets, and labelled recordings of our own. The target is at least 5 recordings by at least 3 artists per raga.
2. **The app should listen for 30–60 s** and show the top 3, which is right about 72% of the time at that length. Let users correct Sa, because a wrong Sa makes the raga prediction useless.
3. **Phrase modelling** is the next modelling step. A sequence model on the pitch contour is the fix for the confusions between related ragas (Todi/Bilaskhani Todi, Miya/Ramdasi Malhar).

## Reproduce
```bash
curl -L -o saraga.zip "https://zenodo.org/records/4301737/files/saraga1.5_hindustani.zip?download=1"
unzip saraga.zip -d ~/mir_datasets/saraga_hindustani && rm -rf ~/mir_datasets/saraga_hindustani/__MACOSX
python scripts/make_manifest_saraga.py ~/mir_datasets/saraga_hindustani --min-recordings 2 --out data/saraga_hindustani.csv
python scripts/extract.py data/saraga_hindustani.csv -j 4
python scripts/evaluate.py data/saraga_hindustani.csv --clip 10 30 60 --group-by artist
python scripts/check_tonic.py data/saraga_hindustani.csv --seconds 30 120 -j 4
```
