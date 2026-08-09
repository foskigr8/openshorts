"""Retry behaviour of the Stage 3 picker's Gemini call.

Prod (22-jul-2026) lost 3 jobs to Gemini answering 200 with an empty body —
that raises while parsing, not while calling, so it used to escape the retry
loop and kill the job on the first blip. The old 2-pass stage's retry loop
now lives in picker._call_gemini (and context_layer._call_gemini): policy
blocks still fail fast; transient blips retry with backoff.
"""
import types

import pytest

import gemini_worker
import picker


class _FakeResponse:
    def __init__(self, parsed=None):
        self.parsed = parsed
        self.candidates = []
        self.usage_metadata = None

    @property
    def text(self):
        return "" if self.parsed is None else "{}"


_GOOD_PAYLOAD = {
    "clips": [{
        "start": 0, "end": 30, "predicted_score": 70, "clip_type": "short",
        "hook_type": "h", "narrative_summary": "s", "essential_span_note": "e",
        "video_description_for_tiktok": "d",
        "video_description_for_instagram": "d",
        "video_title_for_youtube_short": "t", "viral_hook_text": "v",
    }],
    "term_corrections": [],
}


def _stub_generate(monkeypatch, blips=0, raise_blocked=False):
    calls = {"n": 0}

    def fake_generate(client, model_name, prompt, config=None,
                      max_attempts=1, log=None):
        calls["n"] += 1
        if raise_blocked:
            return _FakeResponse(parsed=None)  # raise_if_blocked handles it
        if calls["n"] <= blips:
            return _FakeResponse(parsed=None)  # empty body -> parse retry
        return _FakeResponse(parsed=_GOOD_PAYLOAD)

    monkeypatch.setattr(picker.gemini_pool, "generate_with_fallback",
                        fake_generate)
    if raise_blocked:
        class _Reason:
            name = "PROHIBITED_CONTENT"

        class _PF:
            block_reason = _Reason()

        def blocked(response):
            raise gemini_worker.GeminiBlockedError(
                "Gemini blocked this video's content (PROHIBITED_CONTENT).")

        monkeypatch.setattr(picker.gemini_worker, "raise_if_blocked", blocked)
    return calls


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(picker.time, "sleep", lambda *_: None)


def test_recovers_from_a_single_empty_body(monkeypatch):
    calls = _stub_generate(monkeypatch, blips=1)
    parsed, _cost = picker._call_gemini("k", "prompt")
    assert calls["n"] == 2
    assert parsed["clips"][0]["start"] == 0


def test_recovers_from_two_consecutive_blips(monkeypatch):
    calls = _stub_generate(monkeypatch, blips=2)
    parsed, _cost = picker._call_gemini("k", "prompt")
    assert calls["n"] == 3
    assert parsed["clips"]


def test_gives_up_after_three_attempts(monkeypatch):
    calls = _stub_generate(monkeypatch, blips=99)
    with pytest.raises(ValueError) as exc:
        picker._call_gemini("k", "prompt")
    assert calls["n"] == 3
    assert "empty response body" in str(exc.value)


def test_non_transient_errors_are_not_retried(monkeypatch):
    def boom(client, model_name, prompt, config=None, max_attempts=1, log=None):
        raise ValueError("400 INVALID_ARGUMENT: bad request")

    monkeypatch.setattr(picker.gemini_pool, "generate_with_fallback", boom)
    with pytest.raises(ValueError):
        picker._call_gemini("k", "prompt")


def test_succeeds_without_retrying_when_the_first_call_is_fine(monkeypatch):
    calls = _stub_generate(monkeypatch, blips=0)
    picker._call_gemini("k", "prompt")
    assert calls["n"] == 1


def test_policy_block_fails_fast_without_retrying(monkeypatch):
    # Prod 23-jul-2026: PROHIBITED_CONTENT is deterministic — 3 retries just
    # burned quota and reported a misleading "empty response body".
    calls = _stub_generate(monkeypatch, blips=0, raise_blocked=True)
    with pytest.raises(gemini_worker.GeminiBlockedError) as exc:
        picker._call_gemini("k", "prompt")
    assert calls["n"] == 1
    assert "PROHIBITED_CONTENT" in str(exc.value)
