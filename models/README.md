# Model card: `raga_saraga_hindustani_v1.joblib`

| | |
|---|---|
| Task | Raga classification from a 30–60 s audio excerpt (top-k with confidence) |
| Classes | 21 Hindustani ragas (see `/ragas` in the API) |
| Input features | Tonic-normalised 120-bin pitch-class distribution (10-cent bins) + 12×12 svara transition matrix (264 dims) |
| Model | RBF SVM (C=10, class-balanced) with probability calibration (scikit-learn 1.9.1) |
| Training data | Saraga 1.5 Hindustani: the 53 single-raga recordings of ragas with ≥ 2 recordings (10 artists), cut into non-overlapping 30 s clips (3,018 clips). Pitch from Saraga's precomputed tracks, tonic from its annotations. |
| Train command | `python scripts/train.py data/saraga_hindustani.csv --clip 30 --hop 30 --out models/raga_saraga_hindustani_v1.joblib` |

## Performance
5-fold cross-validation, split by artist (test artists are never in training), annotated tonic:

| Query | Top-1 | Top-3 |
|---|---|---|
| 10 s | 40.8% | 64.8% |
| 30 s | 52.6% | 71.9% |
| 60 s | 56.7% | 72.9% |

Automatic tonic detection is correct on 88% of 30 s excerpts. A wrong tonic usually makes the raga prediction wrong, which is why the app lets users correct Sa. Full analysis: [`docs/results_saraga_hindustani.md`](../docs/results_saraga_hindustani.md).

## Limitations
- Only 21 ragas, from 10 artists, mostly vocal khayal with tanpura. Expect worse results on instrumental music, film songs, or ragas outside the list. The model always answers with one of its 21 ragas.
- Confuses closely related ragas (e.g. Todi / Bilaskhani Todi, Miyan ki Malhar / Ramdasi Malhar, Marwa / Lalit).
- The file is a scikit-learn pickle: load it only with the pinned versions in `requirements.txt`, and only from a trusted source.

## License and attribution
Trained on Saraga, so this model is licensed **CC BY-NC-SA 4.0** (non-commercial, share-alike).

> Bozkurt, B., Srinivasamurthy, A., Gulati, S., & Serra, X. *Saraga: research datasets of Indian Art Music.* Zenodo. https://doi.org/10.5281/zenodo.4301737
