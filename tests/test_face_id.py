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


def test_enrich_returns_a_mapping_the_skill_can_consume():
    """The second return value feeds format_transcript_for_skill's `in` lookup,
    so it must be {label: name} — a trajectory dict silently never matches."""
    import viral_clip_finder as vcf

    transcript = {"segments": [
        {"start": 0.0, "end": 10.0, "text": "hello", "speaker": "SPEAKER_00"}]}
    traj = _trajectory({"Joe": [[0.0, 10.0]]})
    enriched, mapping = face_id.enrich_transcript_speakers(transcript, traj)
    assert mapping == {"SPEAKER_00": "Joe"}
    line = vcf.format_transcript_for_skill(enriched, face_identities=mapping)
    assert line.startswith("[00:00:00] Joe:")


def test_identify_faces_respects_the_scan_limit(monkeypatch):
    """max_seconds must bound the scan — a full stream is tens of GPU-minutes."""
    import inspect
    sig = inspect.signature(face_id.identify_faces_in_video)
    assert sig.parameters["max_seconds"].default == 900.0
