"""face_spine.py's video/GPU path (extract_raw_tracks, InsightFace) needs
cv2 + onnxruntime + a real video and cannot run in a bare sandbox. But the
actual Phase 1 gate — "each real person is ONE track id across the whole
span, including scene cuts" — is decided entirely by merge_tracks_by_identity,
which is pure numpy over track dicts. These tests pin that logic directly,
independent of any detector.
"""
import numpy as np
import pytest

import face_spine as fs


def _unit(*components):
    v = np.array(components, dtype=float)
    return v / np.linalg.norm(v)


def _track(frames, embedding=None, n_boxes=None):
    n = n_boxes or len(frames)
    rec = {
        "frames": list(frames),
        "boxes": [(0.0, 0.0, 10.0, 10.0)] * n,
        "landmarks": [None] * n,
        "det_scores": [0.9] * n,
        "embeddings": [],
    }
    if embedding is not None:
        rec["embeddings"] = [embedding] * n
    return rec


def test_cosine_similarity_identical_vectors():
    v = _unit(1, 2, 3)
    assert fs._cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a, b = _unit(1, 0), _unit(0, 1)
    assert fs._cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)


def test_cosine_similarity_zero_vector_is_safe():
    z = np.zeros(3)
    assert fs._cosine_similarity(z, _unit(1, 0, 0)) == 0.0


def test_mean_embedding_empty_is_none():
    assert fs._mean_embedding([]) is None


def test_mean_embedding_normalizes():
    e = fs._mean_embedding([_unit(1, 0, 0), _unit(1, 0, 0)])
    assert np.linalg.norm(e) == pytest.approx(1.0)


def test_time_range_default_on_empty_frames():
    assert fs._time_range({"frames": []}) == (0.0, 0.0)


def test_time_range_from_frames():
    assert fs._time_range({"frames": [5.0, 1.0, 9.0]}) == (1.0, 9.0)


@pytest.mark.parametrize("a,b,expected", [
    ((0.0, 5.0), (5.0, 10.0), False),   # touching at the boundary: not overlapping
    ((0.0, 5.0), (4.9, 10.0), True),
    ((0.0, 5.0), (10.0, 15.0), False),
    ((0.0, 10.0), (2.0, 4.0), True),    # fully contained
])
def test_ranges_overlap(a, b, expected):
    assert fs._ranges_overlap(a, b) is expected


def test_merge_same_person_across_a_scene_cut():
    # Person appears 0-5s (before a cut), reappears 10-15s (after it) — two
    # raw track ids from the short-term tracker, same embedding both times.
    emb = _unit(1, 0, 0, 0)
    raw = {
        0: _track([0.0, 1.0, 2.0], embedding=emb),
        1: _track([10.0, 11.0, 12.0], embedding=emb),
    }
    merged = fs.merge_tracks_by_identity(raw, similarity_threshold=0.5)
    assert len(merged) == 1
    only = merged[0]
    assert sorted(only["raw_track_ids"]) == [0, 1]
    assert len(only["frames"]) == 6  # both tracks' detections combined


def test_no_merge_when_embeddings_differ():
    raw = {
        0: _track([0.0, 1.0], embedding=_unit(1, 0, 0)),
        1: _track([10.0, 11.0], embedding=_unit(0, 1, 0)),
    }
    merged = fs.merge_tracks_by_identity(raw, similarity_threshold=0.5)
    assert len(merged) == 2


def test_no_merge_when_tracks_overlap_in_time_even_with_identical_embeddings():
    # Same embedding vector (e.g. near-identical framing of two different
    # people is not realistic, but the point is the temporal veto must fire
    # regardless of similarity) at OVERLAPPING times -> must stay separate,
    # two people cannot both be track A at the same instant.
    emb = _unit(1, 0, 0)
    raw = {
        0: _track([0.0, 1.0, 2.0], embedding=emb),
        1: _track([1.5, 2.5, 3.5], embedding=emb),
    }
    merged = fs.merge_tracks_by_identity(raw, similarity_threshold=0.5)
    assert len(merged) == 2


def test_transitive_merge_across_three_raw_tracks():
    # A and C never directly co-occur close enough in the loop order to be
    # compared favorably on their own in some pathological embedding drift,
    # but B bridges them (A~B similar, B~C similar) -> all three must end up
    # in one identity via union-find's transitive closure.
    emb = _unit(0, 0, 1)
    raw = {
        0: _track([0.0, 1.0], embedding=emb),
        1: _track([10.0, 11.0], embedding=emb),
        2: _track([20.0, 21.0], embedding=emb),
    }
    merged = fs.merge_tracks_by_identity(raw, similarity_threshold=0.5)
    assert len(merged) == 1
    assert sorted(merged[0]["raw_track_ids"]) == [0, 1, 2]


def test_track_with_no_embeddings_stays_standalone():
    raw = {
        0: _track([0.0, 1.0], embedding=None),
        1: _track([10.0, 11.0], embedding=_unit(1, 0, 0)),
    }
    merged = fs.merge_tracks_by_identity(raw, similarity_threshold=0.5)
    # The embedding-less track cannot be matched to anything, so it must
    # survive as its own identity rather than being silently dropped.
    assert len(merged) == 2


def test_merged_output_is_densely_renumbered_and_time_sorted():
    emb_a, emb_b = _unit(1, 0), _unit(0, 1)
    raw = {
        5: _track([10.0], embedding=emb_b),   # appears second
        2: _track([0.0], embedding=emb_a),    # appears first
    }
    merged = fs.merge_tracks_by_identity(raw, similarity_threshold=0.5)
    assert set(merged.keys()) == {0, 1}
    assert merged[0]["frames"] == [0.0]
    assert merged[1]["frames"] == [10.0]


def test_merge_respects_custom_threshold():
    # Similarity ~0.87 between two embeddings that share one axis strongly
    # and differ on another — merges at a lenient threshold, not at a strict one.
    a = _unit(1, 0.3)
    b = _unit(1, -0.3)
    sim = fs._cosine_similarity(a, b)
    assert 0.8 < sim < 0.95
    raw = {0: _track([0.0], embedding=a), 1: _track([10.0], embedding=b)}
    assert len(fs.merge_tracks_by_identity(raw, similarity_threshold=0.5)) == 1
    assert len(fs.merge_tracks_by_identity(raw, similarity_threshold=0.99)) == 2
