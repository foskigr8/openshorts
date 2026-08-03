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
