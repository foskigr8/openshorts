"""Stage 3 unified picker: count fulfillment, keep-looking loop, signals."""
import os

import pytest

import picker


def _transcript(segments=None):
    return {
        "language": "en",
        "segments": segments or [
            {"start": 0.0, "end": 10.0, "text": "Hello there.",
             "words": [{"word": " Hello", "start": 0.0, "end": 0.5},
                       {"word": "there.", "start": 0.5, "end": 1.0}]},
            {"start": 10.0, "end": 20.0, "text": "This is the story.",
             "words": [{"word": " This", "start": 10.0, "end": 10.5},
                       {"word": "is", "start": 10.5, "end": 10.8},
                       {"word": "the", "start": 10.8, "end": 11.1},
                       {"word": "story.", "start": 11.1, "end": 11.6}]},
            {"start": 20.0, "end": 30.0, "text": "And the payoff lands.",
             "words": [{"word": " And", "start": 20.0, "end": 20.4},
                       {"word": "the", "start": 20.4, "end": 20.7},
                       {"word": "payoff", "start": 20.7, "end": 21.3},
                       {"word": "lands.", "start": 21.3, "end": 21.9}]},
        ],
    }


def _clip(start, end, score=70, clip_type="short", span=(0, 10)):
    return {
        "start": start, "end": end, "predicted_score": score,
        "clip_type": clip_type, "hook_type": "open question",
        "narrative_summary": "opens, resolves", "essential_span_note": "all",
        "keep_spans": [{"start": span[0], "end": span[1]}],
        "video_description_for_tiktok": "d", "video_description_for_instagram": "d",
        "video_title_for_youtube_short": "title", "viral_hook_text": "hook",
    }


def _run(monkeypatch, responses, **kwargs):
    """Stub the Gemini call with a list of parsed responses (one per pass)."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def fake_call(api_key, prompt):
        calls.append(prompt)
        resp = responses[min(len(calls) - 1, len(responses) - 1)]
        return resp, None

    monkeypatch.setattr(picker, "_call_gemini", fake_call)
    result = picker.select_viral_clips(
        kwargs.get("transcript", _transcript()),
        kwargs.get("duration", 600.0),
        clip_count=kwargs.get("clip_count", 5),
        long_context_count=kwargs.get("long_context_count", 0),
        style_variant=kwargs.get("style_variant", "balanced"),
        context_blob=kwargs.get("context_blob"))
    return result, calls


def _parsed(clips, corrections=()):
    return {"clips": clips, "term_corrections": list(corrections)}


class TestCountFulfillment:
    def test_single_pass_fulfills_exactly(self, monkeypatch):
        result, calls = _run(monkeypatch, [
            _parsed([_clip(0, 30), _clip(40, 70), _clip(80, 110)]),
        ], clip_count=3)
        assert result["delivered"] == 3
        assert result["shortfall"] == 0
        assert len(calls) == 1

    def test_keep_looking_loop_fills_the_gap(self, monkeypatch):
        result, calls = _run(monkeypatch, [
            _parsed([_clip(0, 30), _clip(40, 70)]),          # pass 1: only 2
            _parsed([_clip(90, 120), _clip(130, 160)]),       # pass 2: 2 more
            _parsed([_clip(200, 230)]),                       # pass 3: 1 more
        ])
        assert result["delivered"] == 5
        assert result["shortfall"] == 0
        assert len(calls) == 3
        # Every pass asks for exactly what is still missing.
        assert "exactly 5 distinct" in calls[0] or "exactly 5" in calls[0]
        assert "ALREADY-PICKED SPANS" in calls[1]
        assert "exactly 3 more" in calls[1]

    def test_loop_is_bounded_and_surfaces_shortfall(self, monkeypatch):
        # The stub returns the same single clip every pass — dedup means it
        # can never reach 5; the loop must stop after MAX_PICKER_PASSES and
        # report the gap loudly instead of spinning forever.
        result, calls = _run(monkeypatch, [
            _parsed([_clip(0, 30)]),
        ])
        assert len(calls) == picker.MAX_PICKER_PASSES
        assert result["delivered"] == 1
        assert result["shortfall"] == 4

    def test_empty_first_pass_retries_never_gives_up(self, monkeypatch):
        # The old engines could return "no viral moment found" on a bad pass.
        # The picker treats an empty pass as a failure and keeps looking.
        result, calls = _run(monkeypatch, [
            _parsed([]),                                  # pass 1: nothing
            _parsed([_clip(0, 30), _clip(40, 70),
                     _clip(90, 120), _clip(130, 160)]),   # pass 2: the gems
            _parsed([_clip(200, 230)]),                   # pass 3: the last one
        ])
        assert result["delivered"] == 5
        assert result["shortfall"] == 0
        assert len(calls) == 3
        assert "No clips have been accepted yet" in calls[1]

    def test_physical_cap_is_surfaced_never_silent(self, monkeypatch, capsys):
        # 10 clips x 15s minimum = 150s of video, but the source is 60s.
        result, _ = _run(monkeypatch, [
            _parsed([_clip(0, 30), _clip(35, 55)]),
        ], clip_count=10, duration=60.0)
        out = capsys.readouterr().out
        assert "physical maximum" in out
        assert result["shortfall"] == 8

    def test_missing_clip_count_raises(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        with pytest.raises(ValueError):
            picker.select_viral_clips(_transcript(), 600.0, clip_count=None)


class TestPickerInputs:
    def test_assemblyai_signals_reach_the_prompt(self, monkeypatch):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Big claim.",
             "sentiment": "-", "highlight": True, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 9.0, "text": "And it lands.",
             "sentiment": "+", "highlight": False, "speaker": "SPEAKER_00"},
        ]
        _, calls = _run(monkeypatch, [_parsed([_clip(0, 20)])],
                        transcript=_transcript(segments))
        assert '"sent":"-"' in calls[0]
        assert '"hl":true' in calls[0]
        assert '"sp":"SPEAKER_00"' in calls[0]

    def test_context_blob_is_injected(self, monkeypatch):
        blob = {
            "summary": "Two contestants, one challenge.",
            "highlights": [{"start_s": 3.0, "end_s": 8.0,
                            "description": "the reveal", "type": "payoff"}],
            "lovable_moments": [{"start_s": 3.0, "end_s": 8.0,
                                 "description": "the reveal",
                                 "why_it_lands": "surprise"}],
        }
        _, calls = _run(monkeypatch, [_parsed([_clip(0, 20)])], context_blob=blob)
        assert "PRE-LOADED CONTEXT" in calls[0]
        assert "Two contestants, one challenge." in calls[0]

    def test_long_context_directive_only_when_requested(self, monkeypatch):
        _, calls = _run(monkeypatch, [_parsed([_clip(0, 20)])])
        assert "LONG-CONTEXT CLIPS" not in calls[0]
        _, calls2 = _run(monkeypatch, [_parsed([_clip(0, 20)])],
                         long_context_count=2)
        assert "LONG-CONTEXT CLIPS" in calls2[0]

    def test_overlapping_picks_are_deduped(self, monkeypatch):
        # Pass 1 proposes two overlapping picks; the higher-scored one wins.
        result, _ = _run(monkeypatch, [
            _parsed([_clip(0, 30, score=60), _clip(10, 40, score=90)]),
            _parsed([_clip(90, 120), _clip(130, 160),
                     _clip(200, 230), _clip(240, 270)]),
        ])
        spans = sorted((c["start"], c["end"]) for c in result["shorts"])
        assert (10.0, 40.0) in spans   # higher score kept
        assert (0.0, 30.0) not in spans  # lower score dropped
        assert result["delivered"] == 5
