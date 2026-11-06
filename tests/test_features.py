import numpy as np

from ragaid.features import pitch_class_distribution, svara_sequence, to_cents, transition_matrix
from ragaid.pitch import PitchTrack
from ragaid.features import clip_tracks


def test_to_cents_octave_and_unvoiced():
    c = to_cents(np.array([100.0, 0.0, 200.0, 150.0]), tonic_hz=100.0)
    np.testing.assert_allclose(c, [0, 1200, 1200 * np.log2(1.5)])


def test_pcd_peaks_on_scale_notes_and_is_transposition_invariant():
    # Same melody (S G P) sung at two different tonics -> identical PCD.
    notes = np.repeat([0, 4, 7], 100)
    a = pitch_class_distribution(to_cents(146.8 * 2 ** (notes / 12), 146.8), bins=12, smooth_bins=0)
    b = pitch_class_distribution(to_cents(220.0 * 2 ** (notes / 12), 220.0), bins=12, smooth_bins=0)
    np.testing.assert_allclose(a, b)
    assert set(np.nonzero(a > 0.1)[0]) == {0, 4, 7}
    assert abs(a.sum() - 1) < 1e-9


def test_svara_sequence_skips_glides_and_repeats():
    held = lambda n, k=20: [n * 100.0] * k
    cents = np.array(held(0) + list(np.linspace(0, 400, 10)) + held(4) + held(4) + held(7, 5) + held(0))
    # P is held only 5 frames -> ignored; S S collapse.
    assert list(svara_sequence(cents, min_frames=15)) == [0, 4, 0]


def test_transition_matrix_rows_normalised():
    m = transition_matrix(np.array([0, 2, 4, 2, 0])).reshape(12, 12)
    np.testing.assert_allclose(m.sum(axis=1), 1)
    assert m[0, 2] > 0.9


def test_clip_tracks_drops_silent_clips():
    f0 = np.concatenate([np.full(1000, 200.0), np.zeros(1000)])
    t = PitchTrack(times=np.arange(2000) * 0.01, f0_hz=f0, tonic_hz=100)
    clips = clip_tracks(t, clip_seconds=5)
    assert len(clips) == 2  # 4 clips of 5 s, last two are silent
