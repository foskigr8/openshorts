"""load_track_names: the ground-truth loader added for the ASD bake-off
(eval/run_asd_bakeoff.py). Pure JSON handling, no video/GPU involved.
"""
from eval import ground_truth


def test_load_track_names_missing_file_fails_open(tmp_path):
    assert ground_truth.load_track_names(str(tmp_path / "nope.json")) == {}


def test_load_track_names_converts_string_keys_to_int(tmp_path):
    p = tmp_path / "gt.json"
    p.write_text('{"track_names": {"0": "host", "1": "guest"}}')
    assert ground_truth.load_track_names(str(p)) == {0: "host", 1: "guest"}


def test_load_track_names_missing_key_is_empty(tmp_path):
    p = tmp_path / "gt.json"
    p.write_text('{"speaker_names": {"A": "host"}}')
    assert ground_truth.load_track_names(str(p)) == {}


def test_load_track_names_skips_non_integer_keys(tmp_path):
    p = tmp_path / "gt.json"
    p.write_text('{"track_names": {"0": "host", "not_a_number": "guest"}}')
    assert ground_truth.load_track_names(str(p)) == {0: "host"}


def test_load_speaker_names_and_track_names_coexist(tmp_path):
    p = tmp_path / "gt.json"
    p.write_text('{"speaker_names": {"A": "host"}, "track_names": {"0": "host"}}')
    assert ground_truth.load_speaker_names(str(p)) == {"A": "host"}
    assert ground_truth.load_track_names(str(p)) == {0: "host"}
