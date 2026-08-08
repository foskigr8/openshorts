"""Tests for clip_candidates.py — Stage 3 rebuild Phase 1 (Stage A, the
per-chunk candidate proposal map step).

These exercise the pure logic (boundary construction, marker insertion,
resolution, prompt/schema) without any network call — the same split the
rest of this session's work has used: plumbing is unit-tested here, real
judgment quality is eval/selection's job once wired to a live provider.
"""
import pytest

pydantic = pytest.importorskip("pydantic", reason="clip_candidates needs pydantic")

from clip_candidates import (
    BoundaryResolutionError,
    ChunkCandidate,
    ChunkCandidatesResponse,
    build_chunk_boundaries,
    build_chunk_prompt,
    build_chunks,
    resolve_candidates,
)


# ─── build_chunk_boundaries ──────────────────────────────────────────────────


def test_boundaries_outside_chunk_window_are_excluded():
    b = build_chunk_boundaries(10.0, 20.0, [0.0, 12.0, 25.0], [5.0, 18.0, 30.0])
    assert list(b.start_ids.values()) == [12.0]
    assert list(b.end_ids.values()) == [18.0]


def test_chunk_end_is_exclusive():
    """[chunk_start, chunk_end) — a boundary sitting exactly on the next
    chunk's start must not appear in both (double-counted across the
    overlap seam) or be silently dropped from either."""
    b = build_chunk_boundaries(0.0, 10.0, [0.0, 10.0], [5.0, 15.0])
    assert 10.0 not in b.start_ids.values()
    assert 0.0 in b.start_ids.values()


def test_ids_are_interleaved_by_timestamp_not_grouped_by_kind():
    """IDs must read in chronological order (start, end, start, end, ...)
    when they alternate in time, not all starts then all ends — a reader
    (or model) skimming markers left to right should see them count up in
    the order they actually appear in the transcript."""
    b = build_chunk_boundaries(0.0, 100.0, [0.0, 20.0], [10.0, 30.0])
    # timestamps in order: 0(s) 10(e) 20(s) 30(e) -> ids 0,1,2,3
    assert b.start_ids[0] == 0.0
    assert b.end_ids[1] == 10.0
    assert b.start_ids[2] == 20.0
    assert b.end_ids[3] == 30.0


def test_start_and_end_share_one_id_namespace():
    b = build_chunk_boundaries(0.0, 100.0, [0.0], [10.0])
    all_ids = set(b.start_ids) | set(b.end_ids)
    assert all_ids == {0, 1}  # no collision, no gap


def test_empty_chunk_produces_no_boundaries():
    b = build_chunk_boundaries(0.0, 5.0, [100.0], [110.0])
    assert b.start_ids == {}
    assert b.end_ids == {}


# ─── resolve_candidates ──────────────────────────────────────────────────────


def _boundaries():
    return build_chunk_boundaries(0.0, 100.0, [0.0, 50.0], [20.0, 70.0])


def test_valid_ids_resolve_to_real_timestamps():
    b = _boundaries()
    resp = ChunkCandidatesResponse(candidates=[
        ChunkCandidate(start_boundary_id=0, end_boundary_id=1, score=90,
                       clip_type="short", rationale="r", hook_payoff_summary="h"),
    ])
    out = resolve_candidates(resp, b)
    assert len(out) == 1
    assert out[0].start == 0.0 and out[0].end == 20.0
    assert out[0].score == 90


def test_illegal_start_id_raises():
    b = _boundaries()
    resp = ChunkCandidatesResponse(candidates=[
        ChunkCandidate(start_boundary_id=999, end_boundary_id=1, score=50,
                       clip_type="short", rationale="r", hook_payoff_summary="h"),
    ])
    with pytest.raises(BoundaryResolutionError, match="start_boundary_id"):
        resolve_candidates(resp, b)


def test_illegal_end_id_raises():
    b = _boundaries()
    resp = ChunkCandidatesResponse(candidates=[
        ChunkCandidate(start_boundary_id=0, end_boundary_id=999, score=50,
                       clip_type="short", rationale="r", hook_payoff_summary="h"),
    ])
    with pytest.raises(BoundaryResolutionError, match="end_boundary_id"):
        resolve_candidates(resp, b)


def test_end_id_naming_a_start_marker_is_illegal():
    """An end marker must come from the END namespace, not just any valid
    id — citing a start-marker id as the end is a real, catchable mistake,
    not a coincidental match."""
    b = _boundaries()  # start ids: {0: 0.0, 2: 50.0}, end ids: {1: 20.0, 3: 70.0}
    resp = ChunkCandidatesResponse(candidates=[
        ChunkCandidate(start_boundary_id=0, end_boundary_id=2, score=50,
                       clip_type="short", rationale="r", hook_payoff_summary="h"),
    ])
    with pytest.raises(BoundaryResolutionError, match="end_boundary_id"):
        resolve_candidates(resp, b)


def test_end_before_start_raises():
    b = build_chunk_boundaries(0.0, 100.0, [0.0, 50.0], [10.0, 90.0])
    # interleaved by time: 0.0(s)->id0, 10.0(e)->id1, 50.0(s)->id2, 90.0(e)->id3
    # so start_ids={0:0.0, 2:50.0}, end_ids={1:10.0, 3:90.0}. Both ids below
    # are legally-typed (a real start id, a real end id) but the end (10.0)
    # precedes the chosen start (50.0) in time -- must still be rejected.
    assert b.start_ids == {0: 0.0, 2: 50.0}
    assert b.end_ids == {1: 10.0, 3: 90.0}
    resp = ChunkCandidatesResponse(candidates=[
        ChunkCandidate(start_boundary_id=2, end_boundary_id=1, score=50,
                       clip_type="short", rationale="r", hook_payoff_summary="h"),
    ])
    with pytest.raises(BoundaryResolutionError, match="not after"):
        resolve_candidates(resp, b)


def test_empty_candidates_resolves_to_empty_list():
    b = _boundaries()
    resp = ChunkCandidatesResponse(candidates=[])
    assert resolve_candidates(resp, b) == []


# ─── schema null-tolerance ────────────────────────────────────────────────────


def test_null_optional_field_does_not_reject_the_response():
    """The exact lesson from the documented skill-engine incident: an
    explicit JSON null on an optional field must not fail the whole
    response over a field the caller doesn't require."""
    raw = {"candidates": [{
        "start_boundary_id": 0, "end_boundary_id": 1,
        "score": None, "clip_type": "short",
        "rationale": None, "hook_payoff_summary": "h",
    }]}
    parsed = ChunkCandidatesResponse.model_validate(raw)
    assert parsed.candidates[0].score == 0  # falls back to the field default
    assert parsed.candidates[0].rationale == ""


def test_missing_required_boundary_id_still_fails_loud():
    with pytest.raises(pydantic.ValidationError):
        ChunkCandidatesResponse.model_validate(
            {"candidates": [{"end_boundary_id": 1}]})


# ─── build_chunk_prompt ───────────────────────────────────────────────────────


def test_prompt_embeds_markers_and_preserves_json_shape_braces():
    b = build_chunk_boundaries(0.0, 20.0, [0.0], [10.0])
    segments = [{"s": 0.0, "e": 10.0, "t": "hello world", "sent": "0", "hl": False}]
    chunk = {"id": "w1", "start": 0.0, "end": 20.0}
    prompt = build_chunk_prompt(chunk, b, segments, max_candidates=3)
    assert "[0]" in prompt and "[1]" in prompt
    assert '"start_boundary_id"' in prompt  # literal JSON shape survived .format()
    assert "hello world" in prompt


def test_prompt_speaker_label_included_when_present():
    b = build_chunk_boundaries(0.0, 20.0, [0.0], [10.0])
    segments = [{"s": 0.0, "e": 10.0, "t": "hi", "sent": "0", "hl": False, "sp": "A"}]
    chunk = {"id": "w1", "start": 0.0, "end": 20.0}
    prompt = build_chunk_prompt(chunk, b, segments)
    assert "A: hi" in prompt


# ─── build_chunks ─────────────────────────────────────────────────────────────


def test_build_chunks_delegates_to_the_shared_windowing_primitive():
    transcript = {"segments": [
        {"start": 0.0, "end": 5.0, "text": "one"},
        {"start": 5.0, "end": 10.0, "text": "two"},
    ]}
    chunks = build_chunks(transcript, video_duration=10.0,
                          window_seconds=90, overlap_seconds=30)
    assert len(chunks) >= 1
    assert chunks[0]["start"] == 0.0
