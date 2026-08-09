"""Tests for the optional face-ID enrichment layer."""

import face_id


def _trajectory(ranges_by_name):
    return {"identities": [
        {"name": name, "on_screen": ranges, "speaking": []}
        for name, ranges in ranges_by_name.items()
    ]}


def test_available_false_without_db(monkeypatch):
    monkeypatch.delenv("FACE_ID_DB", raising=False)
    assert face_id.available() is False


def test_available_false_with_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("FACE_ID_DB", str(tmp_path / "nope"))
    assert face_id.available() is False


def test_ranges_from_samples_groups_with_gap_tolerance():
    samples = [
        {"timestamp": 0.0}, {"timestamp": 0.5}, {"timestamp": 1.0},
        {"timestamp": 10.0}, {"timestamp": 11.0},
    ]
    ranges = face_id._ranges_from_samples(samples, gap_tolerance=2.0)
    assert ranges == [[0.0, 1.0], [10.0, 11.0]]


def test_build_face_trajectory_collapses_identifications():
    idents = [
        {"timestamp": 0.0, "name": "Joe", "confidence": 0.9, "bbox": [1, 2, 3, 4]},
        {"timestamp": 0.5, "name": "Joe", "confidence": 0.8, "bbox": [1, 2, 3, 4]},
        {"timestamp": 1.0, "name": "Elon", "confidence": 0.95, "bbox": [5, 6, 7, 8]},
    ]
    traj = face_id.build_face_trajectory(idents)
    by_name = {i["name"]: i for i in traj["identities"]}
    assert by_name["Joe"]["on_screen"] == [[0.0, 0.5]]
    assert by_name["Joe"]["confidence"] == 0.85
    assert by_name["Elon"]["on_screen"] == [[1.0, 1.0]]
    assert by_name["Joe"]["speaking"] == []


def test_overlap_seconds():
    assert face_id._overlap_seconds([[0, 10]], [[5, 8]]) == 3.0
    assert face_id._overlap_seconds([[0, 10]], [[12, 15]]) == 0.0
    assert face_id._overlap_seconds([[0, 5], [10, 15]], [[4, 12]]) == 3.0


def test_enrich_transcript_speakers_renames_high_coverage():
    transcript = {"segments": [
        {"start": 0.0, "end": 10.0, "text": "hello", "speaker": "S1"},
        {"start": 10.0, "end": 20.0, "text": "world", "speaker": "S2"},
    ]}
    traj = _trajectory({"Joe": [[0.0, 12.0]], "Elon": [[10.0, 20.0]]})
    enriched, mapping = face_id.enrich_transcript_speakers(transcript, traj)
    assert mapping == {"S1": "Joe", "S2": "Elon"}
    assert enriched["segments"][0]["speaker"] == "Joe"
    assert enriched["segments"][0]["speaker_named"] is True


def test_enrich_transcript_speakers_keeps_low_coverage_anonymous():
    # S1 speaks 0-10 but the named face is only on screen 8-9: 10% coverage.
    transcript = {"segments": [
        {"start": 0.0, "end": 10.0, "text": "hello", "speaker": "S1"},
    ]}
    traj = _trajectory({"Joe": [[8.0, 9.0]]})
    enriched, mapping = face_id.enrich_transcript_speakers(transcript, traj)
    assert mapping == {}
    assert enriched["segments"][0]["speaker"] == "S1"


def test_enrich_transcript_speakers_requires_min_turn_seconds():
    transcript = {"segments": [
        {"start": 0.0, "end": 2.0, "text": "hi", "speaker": "S1"},
    ]}
    traj = _trajectory({"Joe": [[0.0, 2.0]]})
    enriched, mapping = face_id.enrich_transcript_speakers(
        transcript, traj, min_turn_seconds=3.0)
    assert mapping == {}


def test_enrich_if_configured_noop_without_db(monkeypatch):
    monkeypatch.delenv("FACE_ID_DB", raising=False)
    transcript = {"segments": [{"start": 0, "end": 1, "text": "x", "speaker": "S1"}]}
    out, traj = face_id.enrich_if_configured(transcript, "/nonexistent.mp4")
    assert out is transcript
    assert traj is None


def test_enrich_if_configured_noop_on_missing_video(monkeypatch, tmp_path):
    monkeypatch.setenv("FACE_ID_DB", str(tmp_path))
    transcript = {"segments": [{"start": 0, "end": 1, "text": "x"}]}
    out, traj = face_id.enrich_if_configured(transcript, "/does/not/exist.mp4")
    assert out is transcript
    assert traj is None


def test_enrich_returns_a_mapping_the_picker_can_consume():
    """The second return value feeds the picker's named-speaker input, so it
    must be {label: name} — a trajectory dict silently never matches."""
    transcript = {"segments": [
        {"start": 0.0, "end": 10.0, "text": "hello", "speaker": "SPEAKER_00"}]}
    traj = _trajectory({"Joe": [[0.0, 10.0]]})
    enriched, mapping = face_id.enrich_transcript_speakers(transcript, traj)
    assert mapping == {"SPEAKER_00": "Joe"}
    # The enriched transcript carries the NAME on the segment (the picker's
    # `sp` field) instead of the anonymous diarized label.
    assert enriched["segments"][0]["speaker"] == "Joe"
    assert enriched["segments"][0].get("speaker_named") is True


def test_identify_faces_respects_the_scan_limit(monkeypatch):
    """max_seconds must bound the scan — a full stream is tens of GPU-minutes."""
    import inspect
    sig = inspect.signature(face_id.identify_faces_in_video)
    assert sig.parameters["max_seconds"].default == 900.0


# --- Device selection -------------------------------------------------------
#
# The integration guide is explicit: on a dual-GPU host, put InsightFace on
# GPU 1 and leave GPU 0 to the render pipeline. Face ID runs BEFORE clip
# selection, so sharing device 0 stalls the stage everything else waits on.


def _fake_torch(monkeypatch, count, available=True):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: available,
                                   device_count=lambda: count)))


def test_dual_gpu_puts_face_id_on_the_last_device(monkeypatch):
    monkeypatch.delenv("FACE_ID_CTX", raising=False)
    _fake_torch(monkeypatch, 2)
    assert face_id.default_ctx_id() == 1


def test_single_gpu_uses_device_zero(monkeypatch):
    monkeypatch.delenv("FACE_ID_CTX", raising=False)
    _fake_torch(monkeypatch, 1)
    assert face_id.default_ctx_id() == 0


def test_cpu_host_uses_device_zero(monkeypatch):
    monkeypatch.delenv("FACE_ID_CTX", raising=False)
    _fake_torch(monkeypatch, 0, available=False)
    assert face_id.default_ctx_id() == 0


def test_missing_torch_is_not_fatal(monkeypatch):
    import sys
    monkeypatch.delenv("FACE_ID_CTX", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)  # import raises
    assert face_id.default_ctx_id() == 0


def test_explicit_ctx_overrides_the_default(monkeypatch):
    monkeypatch.setenv("FACE_ID_CTX", "0")
    _fake_torch(monkeypatch, 4)
    assert face_id.default_ctx_id() == 0


def test_a_junk_ctx_falls_back_to_zero(monkeypatch, capsys):
    monkeypatch.setenv("FACE_ID_CTX", "gpu-one")
    _fake_torch(monkeypatch, 2)
    assert face_id.default_ctx_id() == 0
    assert "FACE_ID_CTX" in capsys.readouterr().out


# --- track matching helpers (ported from tests/test_face_id_framing.py) -----


def _tracks():
    return [
        {"track_id": 0, "frames": [
            {"timestamp": 1.0, "bbox": [100, 100, 200, 240]},
            {"timestamp": 1.4, "bbox": [102, 100, 202, 240]},
        ]},
        {"track_id": 1, "frames": [
            {"timestamp": 1.1, "bbox": [400, 120, 520, 280]},
        ]},
    ]


def test_find_track_at_timestamp_matches_by_iou():
    tracks = _tracks()
    # InsightFace-style [x1,y1,x2,y2] box overlapping track 0's frame at 1.0.
    assert face_id.find_track_at_timestamp(
        tracks, 1.05, [110, 110, 190, 230]) == 0
    assert face_id.find_track_at_timestamp(
        tracks, 1.1, [410, 130, 510, 270]) == 1


def test_find_track_at_timestamp_ignores_distant_frames():
    assert face_id.find_track_at_timestamp(
        _tracks(), 5.0, [110, 110, 190, 230]) is None


def test_merge_face_id_with_tracker_votes_and_ranges():
    idents = [
        {"timestamp": 1.0, "name": "Joe", "confidence": 0.9,
         "bbox": [110, 110, 190, 230]},
        {"timestamp": 1.4, "name": "Joe", "confidence": 0.88,
         "bbox": [112, 110, 192, 230]},
        {"timestamp": 1.1, "name": "Ann", "confidence": 0.7,
         "bbox": [410, 130, 510, 270]},
    ]
    merged = face_id.merge_face_id_with_tracker(_tracks(), idents)
    assert merged[0]["name"] == "Joe"
    assert merged[0]["on_screen"] == [[1.0, 1.4]]
    assert merged[1]["name"] == "Ann"


def test_merge_face_id_with_tracker_requires_agreement():
    # Two votes for different names on the same track -> nothing committed.
    idents = [
        {"timestamp": 1.0, "name": "Joe", "confidence": 0.9,
         "bbox": [110, 110, 190, 230]},
        {"timestamp": 1.4, "name": "Ann", "confidence": 0.9,
         "bbox": [112, 110, 192, 230]},
    ]
    assert face_id.merge_face_id_with_tracker(_tracks(), idents) == {}
