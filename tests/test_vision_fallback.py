"""Round-5: VISION_CONFIRM_FALLBACK — when vision rejects every candidate,
the narrative picks ship anyway (user direction: "there's a whole lot
narrative that video could give"), unless the flag is explicitly off."""
import pytest

main = pytest.importorskip("main")


def _clips():
    return [{
        "start": 10.0, "end": 40.0, "predicted_score": 80, "clip_type": "short",
        "hook_type": "open question", "narrative_summary": "opens, resolves",
        "essential_span_note": "all essential", "keep_spans": [],
        "video_description_for_tiktok": "d", "video_description_for_instagram": "d",
        "video_title_for_youtube_short": "title", "viral_hook_text": "hook",
    }]


def _transcript():
    words = [
        {"word": " Hello", "start": 9.5, "end": 10.0},
        {"word": "world.", "start": 10.0, "end": 10.6},
        {"word": " Story", "start": 30.0, "end": 30.5},
        {"word": "done.", "start": 39.6, "end": 40.2},
    ]
    return {"language": "en", "segments": [
        {"start": 9.5, "end": 40.2, "text": "Hello world. Story done.",
         "words": words},
    ]}


def _run(monkeypatch, fallback):
    monkeypatch.setenv("VISION_CONFIRM_FALLBACK", "1" if fallback else "0")
    monkeypatch.setattr(
        main.deepseek_worker, "deepseek_select_narrative_clips",
        lambda *a, **kw: {"clips": _clips(), "cost_analysis": None})
    monkeypatch.setattr(main, "confirm_clip_with_vision",
                        lambda *a, **kw: False)  # reject everything
    monkeypatch.setattr(main.gemini_pool, "pool_from_env", lambda: ["key"])
    return main.get_viral_clips(_transcript(), 60.0, source_video_path="fake.mp4")


def test_fallback_on_ships_narrative_picks(monkeypatch):
    result = _run(monkeypatch, fallback=True)
    assert result is not None
    assert len(result["shorts"]) == 1


def test_fallback_off_returns_none(monkeypatch):
    assert _run(monkeypatch, fallback=False) is None


# ─── Overlap dedup for the shared clip tail ──────────────────────────────────


def _overlapping_clips():
    """Two picks of the same moment at different boundaries (the case the
    selector keeps producing): the later one is the better-scored cut."""
    return [{
        "start": 10.0, "end": 40.0, "predicted_score": 80, "clip_type": "short",
        "hook_type": "open question", "narrative_summary": "opens, resolves",
        "essential_span_note": "all essential", "keep_spans": [],
        "video_description_for_tiktok": "d", "video_description_for_instagram": "d",
        "video_title_for_youtube_short": "title", "viral_hook_text": "hook",
    }, {
        "start": 15.0, "end": 45.0, "predicted_score": 90, "clip_type": "short",
        "hook_type": "open question", "narrative_summary": "opens, resolves",
        "essential_span_note": "all essential", "keep_spans": [],
        "video_description_for_tiktok": "d", "video_description_for_instagram": "d",
        "video_title_for_youtube_short": "title", "viral_hook_text": "hook",
    }]


def test_narrative_overlapping_picks_deduped_end_to_end(monkeypatch):
    """Both engines' picks pass through the shared tail; overlapping picks
    must not both render. The skill engine dedups inside normalize_response,
    the narrative engine does not — the tail catches both."""
    monkeypatch.setenv("VISION_CONFIRM_FALLBACK", "1")
    monkeypatch.setattr(
        main.deepseek_worker, "deepseek_select_narrative_clips",
        lambda *a, **kw: {"clips": _overlapping_clips(), "cost_analysis": None})
    monkeypatch.setattr(main, "confirm_clip_with_vision",
                        lambda *a, **kw: True)  # approve both
    monkeypatch.setattr(main.gemini_pool, "pool_from_env", lambda: ["key"])
    result = main.get_viral_clips(_transcript(), 60.0, source_video_path="fake.mp4")
    assert result is not None
    assert len(result["shorts"]) == 1
    assert result["shorts"][0]["predicted_score"] == 90


def _narrative_clip(start, end, score):
    return {"start": start, "end": end, "predicted_score": score,
            "clip_type": "short", "keep_spans": []}


def test_dedup_overlapping_clips_keeps_higher_score():
    clips = [_narrative_clip(10.0, 40.0, 80), _narrative_clip(15.0, 45.0, 90)]
    out = main._dedup_overlapping_clips(clips)
    assert len(out) == 1
    assert out[0]["start"] == 15.0 and out[0]["predicted_score"] == 90


def test_dedup_overlapping_clips_keeps_earlier_on_tie():
    clips = [_narrative_clip(10.0, 40.0, 80), _narrative_clip(15.0, 45.0, 80)]
    out = main._dedup_overlapping_clips(clips)
    assert len(out) == 1
    assert out[0]["start"] == 10.0


def test_dedup_overlapping_clips_chain_replaces_previous():
    """A higher-scoring middle pick replaces the lower-scoring first, then
    out-scores the third — only the middle survives."""
    clips = [_narrative_clip(10.0, 40.0, 80), _narrative_clip(30.0, 60.0, 95),
             _narrative_clip(50.0, 80.0, 70)]
    out = main._dedup_overlapping_clips(clips)
    assert len(out) == 1
    assert out[0]["start"] == 30.0


def test_dedup_overlapping_clips_distinct_picks_untouched():
    clips = [_narrative_clip(10.0, 20.0, 50), _narrative_clip(30.0, 50.0, 70),
             _narrative_clip(60.0, 90.0, 90)]
    out = main._dedup_overlapping_clips(clips)
    assert [c["start"] for c in out] == [10.0, 30.0, 60.0]


def test_dedup_overlapping_clips_accepts_score_alias():
    """Skill-shaped clips carry score instead of predicted_score."""
    clips = [{"start": 10.0, "end": 40.0, "score": 50},
             {"start": 15.0, "end": 45.0, "score": 60}]
    out = main._dedup_overlapping_clips(clips)
    assert len(out) == 1
    assert out[0]["score"] == 60
