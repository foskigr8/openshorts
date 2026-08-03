import json

import httpx
import pytest

import deepseek_worker as dw


# --- compact payload building ------------------------------------------

def _seg(start, end, text, sentiment=None, highlight=None):
    d = {"start": start, "end": end, "text": text}
    if sentiment is not None:
        d["sentiment"] = sentiment
    if highlight is not None:
        d["highlight"] = highlight
    return d


def test_compact_segment_uses_short_keys_and_rounds_timestamps():
    seg = _seg(1.23456, 4.5, "Hello there.", sentiment="+")
    compact = dw._compact_segment(seg)
    assert compact == {"s": 1.2, "e": 4.5, "t": "Hello there.", "sent": "+", "hl": False}


def test_compact_segment_defaults_missing_sentiment_to_neutral():
    seg = _seg(0.0, 1.0, "No sentiment here.")  # whisper-sourced, no AssemblyAI tag
    assert dw._compact_segment(seg)["sent"] == "0"


def test_compact_segment_carries_highlight_flag():
    seg = _seg(0.0, 1.0, "Standout moment.", highlight=True)
    assert dw._compact_segment(seg)["hl"] is True
    assert dw._compact_segment(_seg(0.0, 1.0, "Ordinary."))["hl"] is False


def test_build_compact_segments_drops_empty_text():
    transcript = {"segments": [_seg(0, 1, "  "), _seg(1, 2, "real text")]}
    compact = dw.build_compact_segments(transcript)
    assert len(compact) == 1
    assert compact[0]["t"] == "real text"


def test_merge_short_segments_combines_filler_into_previous():
    compact = [
        {"s": 0.0, "e": 1.0, "t": "actually this matters a lot", "sent": "0", "hl": False},
        {"s": 1.0, "e": 1.4, "t": "yeah", "sent": "+", "hl": False},  # short trailing filler, merges backward
    ]
    merged = dw._merge_short_segments(compact)
    assert len(merged) == 1
    assert merged[0]["t"] == "actually this matters a lot yeah"
    assert merged[0]["e"] == 1.4
    assert merged[0]["sent"] == "+"  # non-neutral tag preserved over the merge


def test_merge_short_segments_preserves_highlight_flag():
    compact = [
        {"s": 0.0, "e": 1.0, "t": "actually this matters a lot", "sent": "0", "hl": True},
        {"s": 1.0, "e": 1.4, "t": "yeah", "sent": "0", "hl": False},
    ]
    merged = dw._merge_short_segments(compact)
    assert merged[0]["hl"] is True


def test_no_word_level_data_in_compact_payload():
    transcript = {"segments": [_seg(0, 1, "hello", sentiment="+")]}
    compact = dw.build_compact_segments(transcript)
    payload = json.dumps(compact)
    assert "words" not in payload


def test_prompt_template_has_pronoun_trap_guard():
    # Item 7: the most common remaining hook failure is opening on a bare
    # pronoun with no referent inside the clip. The guard must survive.
    template = dw.NARRATIVE_PROMPT_TEMPLATE
    assert "THE PRONOUN TRAP" in template
    assert "rewind `start` EARLIER" in template
    assert "who/what every leading pronoun refers to" in template


def test_long_context_directive_off_by_default():
    assert dw._long_context_directive(0, 180) == ""
    assert dw._long_context_directive(-1, 180) == ""


def test_long_context_directive_requests_full_arcs():
    text = dw._long_context_directive(2, 180)
    assert "long_context" in text
    assert "up to 2" in text
    assert "arc" in text
    assert "180" in text


def test_long_context_count_threads_into_stage(monkeypatch):
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "key")
    transcript = {"language": "en", "segments": [_seg(0, 10, "a story")]}
    seen = {}

    def fake_stage(*a, **kw):
        seen["prompt"] = a[2]  # prompt is the third positional argument
        return {"clips": [_clip(clip_type="long_context")], "term_corrections": []}, None

    monkeypatch.setattr(dw, "_run_deepseek_stage", fake_stage)
    result = dw.deepseek_select_narrative_clips(transcript, 60.0, long_context_count=3)
    assert result is not None
    assert "LONG-CONTEXT CLIPS" in seen["prompt"]
    assert result["clips"][0]["clip_type"] == "long_context"


def test_unknown_clip_type_normalizes_to_short(monkeypatch):
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "key")
    transcript = {"language": "en", "segments": [_seg(0, 10, "a story")]}
    monkeypatch.setattr(
        dw, "_run_deepseek_stage",
        lambda *a, **kw: ({"clips": [_clip(clip_type="mystery")],
                           "term_corrections": []}, None))
    result = dw.deepseek_select_narrative_clips(transcript, 60.0)
    assert result["clips"][0]["clip_type"] == "short"


def test_style_variant_threads_directive_into_prompt(monkeypatch):
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "key")
    transcript = {"language": "en", "segments": [_seg(0, 10, "a story")]}
    seen = {}

    def fake_stage(*a, **kw):
        seen["prompt"] = a[2]
        return {"clips": [_clip()], "term_corrections": []}, None

    monkeypatch.setattr(dw, "_run_deepseek_stage", fake_stage)
    dw.deepseek_select_narrative_clips(transcript, 60.0, style_variant="high_energy")
    assert "HIGH-ENERGY HOOKS" in seen["prompt"]

    dw.deepseek_select_narrative_clips(transcript, 60.0, style_variant="story_driven")
    assert "STORY-DRIVEN" in seen["prompt"]

    dw.deepseek_select_narrative_clips(transcript, 60.0, style_variant="balanced")
    assert "STYLE VARIANT" not in seen["prompt"]


def test_compact_payload_is_actually_compact_json():
    # separators=(",", ":") in the real call site — verify our own serialization
    # convention here doesn't reintroduce whitespace overhead.
    compact = [{"s": 0.0, "e": 1.0, "t": "hi", "sent": "0"}]
    compact_str = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    assert " " not in compact_str.replace("hi", "")  # no spaces outside actual text content


# --- response parsing / schema validation -------------------------------

def _clip(start=1.0, end=20.0, **overrides):
    base = {
        "start": start, "end": end, "predicted_score": 80,
        "hook_type": "open question", "narrative_summary": "opens X, resolves X",
        "essential_span_note": "all essential",
        "video_description_for_tiktok": "desc #tag",
        "video_description_for_instagram": "desc #tag",
        "video_title_for_youtube_short": "title",
        "viral_hook_text": "hook text",
    }
    base.update(overrides)
    return base


def test_narrative_response_validates_well_formed_payload():
    payload = {"clips": [_clip()]}
    dw.NarrativeResponse.model_validate(payload)  # must not raise


def test_narrative_response_rejects_missing_field():
    payload = {"clips": [{"start": 1.0, "end": 20.0}]}  # missing required fields
    with pytest.raises(Exception):
        dw.NarrativeResponse.model_validate(payload)


def test_narrative_clip_defaults_to_short_type():
    clip = dw.NarrativeClipModel.model_validate(_clip())
    assert clip.clip_type == "short"


def test_narrative_clip_accepts_long_context_type():
    clip = dw.NarrativeClipModel.model_validate(_clip(clip_type="long_context"))
    assert clip.clip_type == "long_context"


# --- _run_deepseek_stage retry/parsing -----------------------------------

class _FakeResponse:
    def __init__(self, json_body, status_code=200, request=None):
        self._json_body = json_body
        self.status_code = status_code
        self.text = json.dumps(json_body)
        self.request = request

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=self.request, response=self)

    def json(self):
        return self._json_body


class _FakeClient:
    """Stands in for httpx.Client(...); `responses` is consumed in order so a
    test can script transient-failure-then-success sequences."""

    def __init__(self, responses):
        self._responses = responses

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, **kwargs):
        return self._responses.pop(0)


def _completion_payload(clips, prompt_tokens=1000, completion_tokens=200):
    return {
        "choices": [{"message": {"content": json.dumps({"clips": clips})}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def test_run_deepseek_stage_happy_path(monkeypatch):
    responses = [_FakeResponse(_completion_payload([_clip()]))]
    monkeypatch.setattr(dw.httpx, "Client", lambda timeout=None: _FakeClient(responses))
    monkeypatch.setattr(dw.time, "sleep", lambda s: None)

    parsed, cost = dw._run_deepseek_stage("key", "deepseek-v4-flash", "prompt", dw.NarrativeResponse)

    assert len(parsed["clips"]) == 1
    assert cost["input_tokens"] == 1000
    assert cost["output_tokens"] == 200
    assert cost["total_cost"] == pytest.approx(1000 / 1e6 * 0.14 + 200 / 1e6 * 0.28)


def test_run_deepseek_stage_retries_on_transient_status(monkeypatch):
    responses = [
        _FakeResponse({}, status_code=503),
        _FakeResponse(_completion_payload([_clip()])),
    ]
    monkeypatch.setattr(dw.httpx, "Client", lambda timeout=None: _FakeClient(responses))
    monkeypatch.setattr(dw.time, "sleep", lambda s: None)

    parsed, cost = dw._run_deepseek_stage("key", "deepseek-v4-flash", "prompt", dw.NarrativeResponse)
    assert len(parsed["clips"]) == 1


def test_run_deepseek_stage_gives_up_after_max_attempts(monkeypatch):
    responses = [_FakeResponse({}, status_code=503) for _ in range(3)]
    monkeypatch.setattr(dw.httpx, "Client", lambda timeout=None: _FakeClient(responses))
    monkeypatch.setattr(dw.time, "sleep", lambda s: None)

    with pytest.raises(httpx.HTTPStatusError):
        dw._run_deepseek_stage("key", "deepseek-v4-flash", "prompt", dw.NarrativeResponse)


def test_run_deepseek_stage_retries_on_malformed_json(monkeypatch):
    bad = _FakeResponse({
        "choices": [{"message": {"content": "not json at all"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })
    good = _FakeResponse(_completion_payload([_clip()]))
    responses = [bad, good]
    monkeypatch.setattr(dw.httpx, "Client", lambda timeout=None: _FakeClient(responses))
    monkeypatch.setattr(dw.time, "sleep", lambda s: None)

    parsed, cost = dw._run_deepseek_stage("key", "deepseek-v4-flash", "prompt", dw.NarrativeResponse)
    assert len(parsed["clips"]) == 1


# --- deepseek_select_narrative_clips (top-level entry point) ------------
# NOTE: DeepSeek was REMOVED from the equation (1-aug-2026, explicit user
# direction) — Gemini (NARRATIVE_GEMINI_API_KEY) is the sole provider now,
# so every test below configures that instead of DEEPSEEK_API_KEY.
# _run_deepseek_stage remains the (provider-agnostic) transport function.

def test_select_narrative_clips_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("NARRATIVE_GEMINI_API_KEY", raising=False)
    transcript = {"language": "en", "segments": [_seg(0, 5, "hello world")]}
    assert dw.deepseek_select_narrative_clips(transcript, 60.0) is None


def test_select_narrative_clips_returns_none_on_empty_transcript(monkeypatch):
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "key")
    assert dw.deepseek_select_narrative_clips({"segments": []}, 60.0) is None


def test_select_narrative_clips_single_pass_happy_path(monkeypatch):
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "key")
    transcript = {"language": "en", "segments": [_seg(0, 10, "a story with a hook and a payoff")]}
    monkeypatch.setattr(
        dw, "_run_deepseek_stage",
        lambda api_key, model_name, prompt, schema, **kw: ({"clips": [_clip()], "term_corrections": []}, {
            "input_tokens": 100, "output_tokens": 50, "total_cost": 0.001}))

    result = dw.deepseek_select_narrative_clips(transcript, 60.0)
    assert result is not None
    assert len(result["clips"]) == 1
    assert result["cost_analysis"]["total_cost"] == pytest.approx(0.001)


def test_select_narrative_clips_returns_none_when_stage_raises(monkeypatch):
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "key")
    transcript = {"language": "en", "segments": [_seg(0, 10, "hello world")]}

    def boom(*a, **kw):
        raise RuntimeError("gemini exploded")

    monkeypatch.setattr(dw, "_run_deepseek_stage", boom)
    assert dw.deepseek_select_narrative_clips(transcript, 60.0) is None


def test_select_narrative_clips_uses_chunked_fallback_for_huge_transcripts(monkeypatch):
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "key")
    monkeypatch.setattr(dw, "_MAX_SINGLE_PASS_ESTIMATED_TOKENS", 1)  # force the fallback path
    transcript = {"language": "en", "segments": [_seg(0, 10, "hello world, a fairly long sentence")]}

    called = {"chunked": False}

    def fake_chunked(*a, **kw):
        called["chunked"] = True
        return [_clip()], [], [{"input_tokens": 10, "output_tokens": 5, "total_cost": 0.0001}]

    monkeypatch.setattr(dw, "_chunked_narrative_selection", fake_chunked)
    result = dw.deepseek_select_narrative_clips(transcript, 60.0)
    assert called["chunked"] is True
    assert len(result["clips"]) == 1


# --- clip-count scaling (1-aug-2026: "a 1hr video, nothing less than 30
# clips" -- "usually 2-5" reliably undercounted long-form sources) --------

def test_target_clip_count_one_per_two_minutes():
    assert dw._target_clip_count(3600) == 30  # the exact user example
    assert dw._target_clip_count(600) == 5


def test_target_clip_count_floors_at_two_for_short_videos():
    assert dw._target_clip_count(30) == 2


def test_target_clip_count_override_wins_outright():
    assert dw._target_clip_count(3600, override=6) == 6
    assert dw._target_clip_count(30, override=10) == 10
    # Override is clamped at the same floor as the auto path.
    assert dw._target_clip_count(600, override=1) == 2


def test_candidate_count_directive_whole_video_mentions_the_floor():
    text = dw._candidate_count_directive(3600)
    assert "30" in text
    assert "chunk" not in text.lower()


def test_candidate_count_directive_override_is_a_hard_target():
    text = dw._candidate_count_directive(3600, override=7)
    assert "HARD TARGET" in text
    assert "7" in text
    assert "HARD FLOOR" not in text


def test_candidate_count_directive_override_per_chunk_mentions_total():
    text = dw._candidate_count_directive(3600, chunk_target=2, chunk_index=0,
                                         chunk_count=6, override=12)
    assert "HARD TARGET" in text
    assert "12" in text


def test_candidate_count_directive_per_chunk_mentions_both_numbers():
    text = dw._candidate_count_directive(3600, chunk_target=3, chunk_index=1, chunk_count=12)
    assert "chunk 2 of 12" in text
    assert "3" in text
    assert "30" in text  # still references the overall floor


def test_override_truncates_over_delivery_keeping_strongest(monkeypatch):
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "key")
    transcript = {"language": "en", "segments": [_seg(0, 10, "a story")]}
    weak_late = _clip(start=40.0, end=60.0, predicted_score=10)
    strong_late = _clip(start=80.0, end=100.0, predicted_score=95)
    medium = _clip(start=10.0, end=30.0, predicted_score=50)
    monkeypatch.setattr(
        dw, "_run_deepseek_stage",
        lambda *a, **kw: ({"clips": [weak_late, strong_late, medium],
                           "term_corrections": []}, None))

    result = dw.deepseek_select_narrative_clips(transcript, 60.0, clip_count=2)
    assert result is not None
    got = result["clips"]
    assert len(got) == 2
    # Keeps the strongest by score, restores narrative (start) order.
    assert [c["start"] for c in got] == [10.0, 80.0]


def test_no_override_keeps_all_clips(monkeypatch):
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "key")
    transcript = {"language": "en", "segments": [_seg(0, 10, "a story")]}
    clips = [_clip(start=i * 20.0, end=i * 20.0 + 15.0) for i in range(4)]
    monkeypatch.setattr(
        dw, "_run_deepseek_stage",
        lambda *a, **kw: ({"clips": clips, "term_corrections": []}, None))
    result = dw.deepseek_select_narrative_clips(transcript, 120.0)
    assert len(result["clips"]) == 4


def test_chunked_path_receives_override(monkeypatch):
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "key")
    monkeypatch.setattr(dw, "_MAX_SINGLE_PASS_ESTIMATED_TOKENS", 1)
    transcript = {"language": "en", "segments": [_seg(0, 10, "hello world, a fairly long sentence")]}
    seen = {}

    def fake_chunked(*a, **kw):
        seen["kw"] = kw
        return [_clip()], [], []

    monkeypatch.setattr(dw, "_chunked_narrative_selection", fake_chunked)
    dw.deepseek_select_narrative_clips(transcript, 3600.0, clip_count=8)
    assert seen["kw"].get("clip_count") == 8


def test_long_form_video_is_chunked_even_when_it_fits_a_single_call(monkeypatch):
    # The mechanical fix: a 1hr video's transcript easily fits under
    # _MAX_SINGLE_PASS_ESTIMATED_TOKENS, so token size alone never forced
    # chunking before -- LONG_FORM_SECONDS_THRESHOLD does it by duration.
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "key")
    transcript = {"language": "en", "segments": [_seg(0, 10, "short transcript, tiny payload")]}

    called = {"chunked": False}

    def fake_chunked(*a, **kw):
        called["chunked"] = True
        return [_clip()], [], [{"input_tokens": 10, "output_tokens": 5, "total_cost": 0.0001}]

    monkeypatch.setattr(dw, "_chunked_narrative_selection", fake_chunked)
    dw.deepseek_select_narrative_clips(transcript, 3600.0)  # 1 hour
    assert called["chunked"] is True


def test_short_video_stays_single_pass(monkeypatch):
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "key")
    transcript = {"language": "en", "segments": [_seg(0, 10, "short transcript")]}

    called = {"chunked": False}

    def fake_chunked(*a, **kw):
        called["chunked"] = True
        return [_clip()], [], []

    monkeypatch.setattr(dw, "_chunked_narrative_selection", fake_chunked)
    monkeypatch.setattr(
        dw, "_run_deepseek_stage",
        lambda *a, **kw: ({"clips": [_clip()], "term_corrections": []}, None))
    dw.deepseek_select_narrative_clips(transcript, 60.0)  # well under the threshold
    assert called["chunked"] is False


def test_chunked_selection_gives_each_chunk_a_minimum_target(monkeypatch):
    monkeypatch.setattr(dw, "build_transcript_windows",
                        lambda *a, **kw: [{"start": 0, "end": 300}] * 12)  # 12 chunks
    seen_directives = []

    def fake_stage(api_key, model_name, prompt, schema, **kw):
        seen_directives.append(prompt)
        return {"clips": [_clip()], "term_corrections": []}, None

    monkeypatch.setattr(dw, "_run_deepseek_stage", fake_stage)
    transcript = {"segments": [_seg(0, 300, "hello") for _ in range(12)]}
    dw._chunked_narrative_selection(transcript, 3600.0, "key", "model", "en", 180.0)
    assert len(seen_directives) == 12
    # 30 (overall target for 3600s) / 12 chunks = ceil(2.5) = 3 per chunk.
    assert all("AT LEAST 3 distinct candidates from THIS chunk" in p for p in seen_directives)


# --- provider candidates: Gemini-only ------------------------------------

def test_candidates_deepseek_key_alone_is_not_used(monkeypatch):
    # The whole point of the removal: a DEEPSEEK_API_KEY sitting in .env
    # must NOT bring DeepSeek back into the mix on its own.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dkey")
    monkeypatch.delenv("NARRATIVE_GEMINI_API_KEY", raising=False)
    assert dw._narrative_provider_candidates(None, None) == []


def test_candidates_gemini_only_when_configured(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dkey")  # present but must be ignored
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "gkey")
    monkeypatch.delenv("NARRATIVE_GEMINI_MODEL", raising=False)
    candidates = dw._narrative_provider_candidates(None, None)
    assert len(candidates) == 1
    provider, api_key, model_name, base_url = candidates[0]
    assert provider == "gemini"
    assert api_key == "gkey"
    assert model_name == dw.DEFAULT_NARRATIVE_GEMINI_MODEL
    assert base_url == dw.GEMINI_OPENAI_COMPAT_BASE


def test_candidates_list_empty_when_neither_configured(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("NARRATIVE_GEMINI_API_KEY", raising=False)
    assert dw._narrative_provider_candidates(None, None) == []


def test_candidates_gemini_model_is_env_overridable(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "gkey")
    monkeypatch.setenv("NARRATIVE_GEMINI_MODEL", "gemini-2.5-flash")
    candidates = dw._narrative_provider_candidates(None, None)
    assert candidates[0][2] == "gemini-2.5-flash"


def test_select_narrative_clips_uses_gemini(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("NARRATIVE_GEMINI_API_KEY", "gkey")
    transcript = {"language": "en", "segments": [_seg(0, 10, "a story with a hook and a payoff")]}
    seen = {}

    def fake_stage(api_key, model_name, prompt, schema, **kw):
        seen["api_key"] = api_key
        seen["model_name"] = model_name
        seen["base_url"] = kw.get("base_url")
        return {"clips": [_clip()], "term_corrections": []}, {
            "input_tokens": 10, "output_tokens": 5, "total_cost": 0.0001}

    monkeypatch.setattr(dw, "_run_deepseek_stage", fake_stage)
    result = dw.deepseek_select_narrative_clips(transcript, 60.0)
    assert result is not None
    assert seen["api_key"] == "gkey"
    assert seen["model_name"] == dw.DEFAULT_NARRATIVE_GEMINI_MODEL
    assert seen["base_url"] == dw.GEMINI_OPENAI_COMPAT_BASE
