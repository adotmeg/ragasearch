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


def test_saraga_scanner(tmp_path):
    import json

    from ragaid.data import saraga_manifest

    def track(release, name, raags, album_artist="A. Singer"):
        d = tmp_path / "saraga1.5_hindustani" / release / name
        d.mkdir(parents=True)
        (d / f"{name}.json").write_text(json.dumps({
            "raags": [{"common_name": r, "name": r + "ā"} for r in raags],
            "album_artists": [{"name": album_artist}], "artists": []}))
        (d / f"{name}.mp3.mp3").write_bytes(b"")
        (d / f"{name}.pitch.txt").write_text("0.0\t0.0\n")
        (d / f"{name}.ctonic.txt").write_text("146.5\n")
        (d / f"._{name}.json").write_bytes(b"\x00\x05junk")  # macOS resource fork

    track("Album 1", "Raag Yaman", ["yaman  kalyan"])
    track("Album 1", "Ragamala", ["Bhoop", "Des"])  # several ragas -> skipped
    track("Album 2", "Raag Bhoop", ["Bhoop"], "B. Singer")
    recs = sorted(saraga_manifest(tmp_path), key=lambda r: r.raga)
    assert [(r.raga, r.artist, r.tonic_hz) for r in recs] == [("Bhoop", "B. Singer", 146.5),
                                                               ("Yaman Kalyan", "A. Singer", 146.5)]
    assert all(r.pitch_path.endswith(".pitch.txt") and r.path.endswith(".mp3.mp3") for r in recs)
