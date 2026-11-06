import numpy as np

from ragaid.data import Recording, get_pitch_track, read_manifest, write_manifest


def test_manifest_roundtrip_with_and_without_optional_columns(tmp_path):
    recs = [Recording("a.wav", "Yaman", "A", 146.8, "a.pitch.txt"), Recording("b.wav", "Bhoop", "B")]
    write_manifest(recs, tmp_path / "m.csv")
    assert read_manifest(tmp_path / "m.csv") == [Recording("a.wav", "Yaman", "A", 146.8, "a.pitch.txt"),
                                                 Recording("b.wav", "Bhoop", "B", None, None)]
    # Older manifests without a pitch_path column still load.
    (tmp_path / "old.csv").write_text("path,raga,artist,tonic_hz\nc.wav,Kafi,C,\n")
    assert read_manifest(tmp_path / "old.csv") == [Recording("c.wav", "Kafi", "C")]


def test_precomputed_pitch_file_is_used(tmp_path):
    pitch = tmp_path / "x.pitch.txt"
    np.savetxt(pitch, np.c_[np.arange(5) * 0.01, [0, 150, -1, 200, 0]], delimiter="\t")
    rec = Recording(str(tmp_path / "missing.wav"), "Yaman", "A", 150.0, str(pitch))
    t = get_pitch_track(rec, cache_dir=tmp_path / "cache")
    np.testing.assert_allclose(t.f0_hz, [0, 150, 0, 200, 0])  # negatives -> unvoiced
    assert t.tonic_hz == 150.0
    assert (tmp_path / "cache" / f"{rec.cache_key}.npz").exists()
