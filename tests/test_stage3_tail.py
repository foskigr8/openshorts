"""Shared Stage 3 tail: overlap dedup on final boundaries (moved from the
deleted vision-fallback tests — dedup behavior is unchanged)."""
import pytest

main = pytest.importorskip("main")


def _clip(start, end, score):
    return {"start": start, "end": end, "predicted_score": score}


def test_dedup_overlapping_clips_keeps_higher_score():
    clips = [_clip(10.0, 40.0, 80), _clip(15.0, 45.0, 90)]
    out = main._dedup_overlapping_clips(clips)
    assert len(out) == 1
    assert out[0]["start"] == 15.0


def test_dedup_overlapping_clips_keeps_earlier_on_tie():
    clips = [_clip(10.0, 40.0, 80), _clip(15.0, 45.0, 80)]
    out = main._dedup_overlapping_clips(clips)
    assert len(out) == 1
    assert out[0]["start"] == 10.0


def test_dedup_overlapping_clips_chain_replaces_previous():
    # A better clip replaces the previous keeper, which may free another
    # earlier pick — the chain must settle on the best-scored set.
    clips = [_clip(0.0, 20.0, 70), _clip(10.0, 40.0, 60),
             _clip(30.0, 60.0, 95)]
    out = main._dedup_overlapping_clips(clips)
    assert len(out) == 2
    assert out[0]["start"] == 0.0
    assert out[1]["start"] == 30.0


def test_dedup_overlapping_clips_distinct_picks_untouched():
    clips = [_clip(0.0, 20.0, 70), _clip(25.0, 45.0, 60)]
    out = main._dedup_overlapping_clips(clips)
    assert len(out) == 2


def test_dedup_overlapping_clips_accepts_score_alias():
    clips = [{"start": 0.0, "end": 20.0, "score": 50},
             {"start": 10.0, "end": 30.0, "score": 90}]
    out = main._dedup_overlapping_clips(clips)
    assert len(out) == 1
    assert out[0]["start"] == 10.0
