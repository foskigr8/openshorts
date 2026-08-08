"""Tests for main.py's Gemini Vision candidate-confirmation pass."""
import types

import pytest

import gemini_pool

# main pulls in cv2/scenedetect/yt-dlp at import time; skip where those aren't
# installed (matches tests/test_gemini_retry.py's convention).
main = pytest.importorskip("main")


class _FakeFiles:
    def __init__(self):
        self.deleted = []

    def upload(self, file):
        return types.SimpleNamespace(name="upload_123")

    def get(self, name):
        return types.SimpleNamespace(state=types.SimpleNamespace(name="ACTIVE"))

    def delete(self, name):
        self.deleted.append(name)


class _FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed
        self.candidates = []
        self.usage_metadata = None

    @property
    def text(self):
        return "{}" if self.parsed is not None else ""


class _FakeModels:
    def __init__(self, visual_parsed, context_parsed):
        self.visual_parsed = visual_parsed
        self.context_parsed = context_parsed
        self.calls = []

    def generate_content(self, model, contents, config):
        prompt = contents[1]
        self.calls.append(prompt)
        if "OPENING FRAME" in prompt:
            return _FakeResponse(self.visual_parsed)
        return _FakeResponse(self.context_parsed)


def _install_fake_client(monkeypatch, visual_parsed, context_parsed):
    files = _FakeFiles()
    models = _FakeModels(visual_parsed, context_parsed)

    class _FakeClient:
        def __init__(self, api_key=None, http_options=None):
            self.files = files
            self.models = models

    monkeypatch.setattr(main.genai, "Client", _FakeClient)
    return models, files


class _FakeModelsContextSequence:
    """Visual check always returns visual_parsed; successive context-check
    calls pop responses off context_parsed_sequence in order — needed to
    test the initial call vs. the bounded retry seeing different verdicts."""

    def __init__(self, visual_parsed, context_parsed_sequence):
        self.visual_parsed = visual_parsed
        self.context_parsed_sequence = list(context_parsed_sequence)
        self.calls = []

    def generate_content(self, model, contents, config):
        prompt = contents[1]
        self.calls.append(prompt)
        if "OPENING FRAME" in prompt:
            return _FakeResponse(self.visual_parsed)
        return _FakeResponse(self.context_parsed_sequence.pop(0))


def _install_fake_client_context_sequence(monkeypatch, visual_parsed, context_parsed_sequence):
    files = _FakeFiles()
    models = _FakeModelsContextSequence(visual_parsed, context_parsed_sequence)

    class _FakeClient:
        def __init__(self, api_key=None, http_options=None):
            self.files = files
            self.models = models

    monkeypatch.setattr(main.genai, "Client", _FakeClient)
    return models, files


@pytest.fixture(autouse=True)
def _no_sleep_no_ffmpeg(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    monkeypatch.setattr(main, "_rough_cut_candidate", lambda *a, **kw: "/tmp/fake_rough_cut.mp4")


def _candidate(start=10.0, end=40.0, narrative_summary="opens X, resolves X"):
    return {"start": start, "end": end, "narrative_summary": narrative_summary}


def _transcript():
    return {"segments": [{"start": 10.0, "end": 40.0, "text": "some words here"}]}


def test_no_pool_configured_approves_without_calling_gemini(monkeypatch):
    pool = gemini_pool.GeminiKeyPool([])
    candidate = _candidate()
    approved = main.confirm_clip_with_vision(pool, "model", "video.mp4", candidate, 100.0, _transcript())
    assert approved is True
    assert candidate["start"] == 10.0 and candidate["end"] == 40.0  # untouched


def test_both_checks_approve_with_no_deltas(monkeypatch):
    _install_fake_client(monkeypatch,
        visual_parsed={"approved": True, "suggested_start_delta": 0.0, "reason": "clean"},
        context_parsed={"approved": True, "narrative_resolved": True,
                        "suggested_end_delta": 0.0, "reason": "resolved"})
    pool = gemini_pool.GeminiKeyPool(["key1", "key2"])
    candidate = _candidate()
    approved = main.confirm_clip_with_vision(pool, "model", "video.mp4", candidate, 100.0, _transcript())
    assert approved is True
    assert candidate["start"] == 10.0 and candidate["end"] == 40.0


def test_visual_check_applies_start_delta_clamped(monkeypatch):
    _install_fake_client(monkeypatch,
        visual_parsed={"approved": True, "suggested_start_delta": 10.0, "reason": "bad open, shift a lot"},
        context_parsed={"approved": True, "narrative_resolved": True,
                        "suggested_end_delta": 0.0, "reason": "resolved"})
    pool = gemini_pool.GeminiKeyPool(["key1", "key2"])
    candidate = _candidate(start=10.0, end=40.0)
    main.confirm_clip_with_vision(pool, "model", "video.mp4", candidate, 100.0, _transcript())
    # delta clamped to +3, not the full +10 the model suggested
    assert candidate["start"] == pytest.approx(13.0)


def test_context_check_extends_end_when_unresolved(monkeypatch):
    _install_fake_client(monkeypatch,
        visual_parsed={"approved": True, "suggested_start_delta": 0.0, "reason": "fine"},
        context_parsed={"approved": True, "narrative_resolved": True,
                        "suggested_end_delta": 15.0, "reason": "needed more room to land the payoff"})
    pool = gemini_pool.GeminiKeyPool(["key1", "key2"])
    candidate = _candidate(start=10.0, end=40.0)
    main.confirm_clip_with_vision(pool, "model", "video.mp4", candidate, 100.0, _transcript())
    assert candidate["end"] == pytest.approx(55.0)


def test_end_extension_never_exceeds_safety_ceiling(monkeypatch):
    _install_fake_client(monkeypatch,
        visual_parsed={"approved": True, "suggested_start_delta": 0.0, "reason": "fine"},
        context_parsed={"approved": True, "narrative_resolved": True,
                        "suggested_end_delta": 400.0, "reason": "way too much"})
    pool = gemini_pool.GeminiKeyPool(["key1"])
    candidate = _candidate(start=0.0, end=40.0)
    main.confirm_clip_with_vision(
        pool, "model", "video.mp4", candidate, 1000.0, _transcript(), max_duration_ceiling=360.0)
    assert candidate["end"] - candidate["start"] <= 360.0
    assert candidate["end"] == 40.0  # extension rejected entirely since it would blow the ceiling


def test_visual_rejection_drops_the_candidate(monkeypatch):
    _install_fake_client(monkeypatch,
        visual_parsed={"approved": False, "suggested_start_delta": 0.0, "reason": "unusable footage"},
        context_parsed={"approved": True, "narrative_resolved": True,
                        "suggested_end_delta": 0.0, "reason": "fine"})
    pool = gemini_pool.GeminiKeyPool(["key1", "key2"])
    candidate = _candidate()
    approved = main.confirm_clip_with_vision(pool, "model", "video.mp4", candidate, 100.0, _transcript())
    assert approved is False


def test_context_rejection_drops_the_candidate(monkeypatch):
    _install_fake_client(monkeypatch,
        visual_parsed={"approved": True, "suggested_start_delta": 0.0, "reason": "fine"},
        context_parsed={"approved": False, "narrative_resolved": False,
                        "suggested_end_delta": 0.0, "reason": "no plausible resolution"})
    pool = gemini_pool.GeminiKeyPool(["key1", "key2"])
    candidate = _candidate()
    approved = main.confirm_clip_with_vision(pool, "model", "video.mp4", candidate, 100.0, _transcript())
    assert approved is False


def test_both_checks_use_different_keys_from_the_pool(monkeypatch):
    used_keys = []
    orig_acquire = gemini_pool.GeminiKeyPool.acquire

    def _tracking_acquire(self):
        key = orig_acquire(self)
        used_keys.append(key)
        return key

    monkeypatch.setattr(gemini_pool.GeminiKeyPool, "acquire", _tracking_acquire)
    _install_fake_client(monkeypatch,
        visual_parsed={"approved": True, "suggested_start_delta": 0.0, "reason": "fine"},
        context_parsed={"approved": True, "narrative_resolved": True,
                        "suggested_end_delta": 0.0, "reason": "fine"})
    pool = gemini_pool.GeminiKeyPool(["key1", "key2"])
    main.confirm_clip_with_vision(pool, "model", "video.mp4", _candidate(), 100.0, _transcript())
    assert len(used_keys) == 2
    assert set(used_keys) == {"key1", "key2"}  # both keys got used, not the same one twice


def test_gemini_error_fails_open_and_marks_key_bad(monkeypatch):
    class _BoomFiles:
        def upload(self, file):
            raise RuntimeError("upload failed")

    class _BoomClient:
        def __init__(self, api_key=None, http_options=None):
            self.files = _BoomFiles()
            self.models = None

    monkeypatch.setattr(main.genai, "Client", _BoomClient)
    pool = gemini_pool.GeminiKeyPool(["key1"])
    candidate = _candidate()
    approved = main.confirm_clip_with_vision(pool, "model", "video.mp4", candidate, 100.0, _transcript())
    assert approved is True  # fails open: no check ran successfully
    assert "key1" in pool._bad


def test_rough_cut_failure_fails_open(monkeypatch):
    monkeypatch.setattr(main, "_rough_cut_candidate",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("ffmpeg boom")))
    pool = gemini_pool.GeminiKeyPool(["key1"])
    candidate = _candidate()
    approved = main.confirm_clip_with_vision(pool, "model", "video.mp4", candidate, 100.0, _transcript())
    assert approved is True


def test_transcript_excerpt_joins_overlapping_segments():
    transcript = {"segments": [
        {"start": 0.0, "end": 5.0, "text": "before"},
        {"start": 10.0, "end": 20.0, "text": "in range one"},
        {"start": 20.0, "end": 30.0, "text": "in range two"},
        {"start": 40.0, "end": 50.0, "text": "after"},
    ]}
    excerpt = main._transcript_excerpt(transcript, 10.0, 30.0)
    assert excerpt == "in range one in range two"


def test_context_excerpt_marks_the_actual_proposed_boundary():
    transcript = {"segments": [
        {"start": 0.0, "end": 5.0, "text": "What do you want in a partner?"},
        {"start": 5.0, "end": 10.0, "text": "I only date ambitious people."},
    ]}
    excerpt = main._transcript_excerpt(transcript, 0.0, 10.0, proposed_start=5.0)
    assert excerpt == (
        "What do you want in a partner? [PROPOSED CLIP START] "
        "I only date ambitious people.")


def test_context_excerpt_marks_boundary_mid_segment_not_the_next_one():
    # Regression: an earlier version used `seg_end >= proposed_start`, which
    # fires on the FIRST segment as soon as its end reaches proposed_start —
    # wrong whenever proposed_start lands mid-segment (not just on an exact
    # segment edge): it must mark the segment that CONTAINS proposed_start,
    # not skip past it.
    transcript = {"segments": [
        {"start": 0.0, "end": 5.0, "text": "before"},
        {"start": 5.0, "end": 10.0, "text": "the reply starts partway through"},
        {"start": 10.0, "end": 15.0, "text": "after"},
    ]}
    excerpt = main._transcript_excerpt(transcript, 0.0, 15.0, proposed_start=7.0)
    assert excerpt == (
        "before [PROPOSED CLIP START] the reply starts partway through after")


class TestBoundedRetryAfterRescue:
    """The context check's own contract says has_real_hook/narrative_resolved
    come back false alongside a fix suggestion (the delta), not only when a
    fix is impossible — so the FIRST verdict alone can't be trusted as final
    once a delta was actually applied. Confirmed on real content (30-jul-2026):
    a candidate rescued to include the missing preceding question was still
    logged "still rejected" because nothing re-checked the fixed boundary.
    One bounded retry (context check only, run once) closes that gap."""

    def test_retry_approves_a_successfully_rescued_candidate(self, monkeypatch):
        _install_fake_client_context_sequence(
            monkeypatch,
            visual_parsed={"approved": True, "suggested_start_delta": 0.0, "reason": "fine"},
            context_parsed_sequence=[
                {"approved": False, "narrative_resolved": True, "has_real_hook": False,
                 "suggested_start_delta": -10.0, "suggested_end_delta": 0.0,
                 "reason": "missing the question this replies to"},
                {"approved": True, "narrative_resolved": True, "has_real_hook": True,
                 "suggested_start_delta": 0.0, "suggested_end_delta": 0.0,
                 "reason": "now includes the question, resolves cleanly"},
            ])
        pool = gemini_pool.GeminiKeyPool(["key1", "key2", "key3", "key4"])
        candidate = _candidate(start=20.0, end=40.0)
        approved = main.confirm_clip_with_vision(pool, "model", "video.mp4", candidate, 100.0, _transcript())
        assert approved is True
        assert candidate["start"] == 10.0  # rescue delta applied and kept
        assert "_rejection_reason" not in candidate

    def test_retry_still_rejects_when_the_rescue_did_not_fix_it(self, monkeypatch):
        _install_fake_client_context_sequence(
            monkeypatch,
            visual_parsed={"approved": True, "suggested_start_delta": 0.0, "reason": "fine"},
            context_parsed_sequence=[
                {"approved": False, "narrative_resolved": True, "has_real_hook": False,
                 "suggested_start_delta": -5.0, "suggested_end_delta": 0.0,
                 "reason": "missing setup"},
                {"approved": False, "narrative_resolved": True, "has_real_hook": False,
                 "suggested_start_delta": 0.0, "suggested_end_delta": 0.0,
                 "reason": "still needs the game-format intro from much earlier"},
            ])
        pool = gemini_pool.GeminiKeyPool(["key1", "key2", "key3", "key4"])
        candidate = _candidate(start=20.0, end=40.0)
        approved = main.confirm_clip_with_vision(pool, "model", "video.mp4", candidate, 100.0, _transcript())
        assert approved is False
        assert "retry" in candidate["_rejection_reason"]

    def test_no_retry_when_nothing_moved(self, monkeypatch):
        # Rejected with a zero delta (unfixable per the check's own contract)
        # — retrying would just repeat the exact same call for nothing.
        models, _ = _install_fake_client_context_sequence(
            monkeypatch,
            visual_parsed={"approved": True, "suggested_start_delta": 0.0, "reason": "fine"},
            context_parsed_sequence=[
                {"approved": False, "narrative_resolved": True, "has_real_hook": False,
                 "suggested_start_delta": 0.0, "suggested_end_delta": 0.0,
                 "reason": "no reasonable fix exists"},
            ])
        pool = gemini_pool.GeminiKeyPool(["key1", "key2", "key3", "key4"])
        candidate = _candidate(start=20.0, end=40.0)
        approved = main.confirm_clip_with_vision(pool, "model", "video.mp4", candidate, 100.0, _transcript())
        assert approved is False
        assert len(models.calls) == 2  # exactly visual + context, no retry call


class TestExtendKeepSpansToCoverBoundaries:
    """A hook/narrative rescue that pulls start earlier or pushes end later
    must also grow keep_spans, or the jump-cutter would silently drop
    exactly the content the rescue was for (confirmed 31-jul-2026)."""

    def test_start_rescue_extends_the_first_keep_span_backward(self):
        candidate = {"start": 100.0, "end": 140.0,
                     "keep_spans": [{"start": 102.0, "end": 140.0}]}
        main._extend_keep_spans_to_cover_boundaries(candidate)
        starts = [s["start"] for s in candidate["keep_spans"]]
        assert min(starts) == 100.0

    def test_end_rescue_extends_keep_spans_forward(self):
        candidate = {"start": 100.0, "end": 150.0,
                     "keep_spans": [{"start": 100.0, "end": 140.0}]}
        main._extend_keep_spans_to_cover_boundaries(candidate)
        ends = [s["end"] for s in candidate["keep_spans"]]
        assert max(ends) == 150.0

    def test_no_keep_spans_is_a_no_op(self):
        candidate = {"start": 100.0, "end": 140.0}
        main._extend_keep_spans_to_cover_boundaries(candidate)
        assert "keep_spans" not in candidate

    def test_no_boundary_change_leaves_keep_spans_untouched(self):
        candidate = {"start": 100.0, "end": 140.0,
                     "keep_spans": [{"start": 100.0, "end": 140.0}]}
        main._extend_keep_spans_to_cover_boundaries(candidate)
        assert candidate["keep_spans"] == [{"start": 100.0, "end": 140.0}]

    def test_full_confirm_pipeline_extends_keep_spans_on_hook_rescue(self, monkeypatch):
        _install_fake_client(monkeypatch,
            visual_parsed={"approved": True, "suggested_start_delta": 0.0, "reason": "fine"},
            context_parsed={"approved": True, "narrative_resolved": True, "has_real_hook": False,
                            "suggested_start_delta": -5.0, "suggested_end_delta": 0.0,
                            "reason": "missing the question"})
        pool = gemini_pool.GeminiKeyPool(["key1", "key2"])
        candidate = _candidate(start=100.0, end=140.0)
        candidate["keep_spans"] = [{"start": 102.0, "end": 140.0}]
        main.confirm_clip_with_vision(pool, "model", "video.mp4", candidate, 200.0, _transcript())
        assert candidate["start"] == 95.0
        starts = [s["start"] for s in candidate["keep_spans"]]
        assert min(starts) == 95.0, "keep_spans must cover the rescued lead-in"


# --- long-context routing -------------------------------------------------
# `clip_type` was written by the selector and then read by nothing, so a
# long-context candidate was reviewed by the tight-hook prompt (which demands
# a self-contained punchy claim in the opening seconds). That reviewer either
# pulled the start later to manufacture a hook or dropped the candidate, which
# is why asking for long clips still produced short ones.

def _long_candidate(**kw):
    c = _candidate(**kw)
    c["clip_type"] = "long_context"
    return c


def _approving_pair():
    return dict(
        visual_parsed={"approved": True, "suggested_start_delta": 0.0, "reason": "ok"},
        context_parsed={"approved": True, "narrative_resolved": True,
                        "suggested_end_delta": 0.0, "reason": "ok"},
    )


def test_long_context_candidate_uses_the_narrative_reviewer(monkeypatch):
    models, _ = _install_fake_client(monkeypatch, **_approving_pair())
    pool = gemini_pool.GeminiKeyPool(["k1", "k2"])
    main.confirm_clip_with_vision(pool, "m", "v.mp4", _long_candidate(), 300.0, _transcript())
    context_prompts = [p for p in models.calls if "OPENING FRAME" not in p]
    assert context_prompts, "context check never ran"
    joined = "\n".join(context_prompts)
    assert "ARC COMPLETENESS" in joined
    # The tight-short hook rule must NOT be applied to a long-context segment.
    assert "REAL HOOK / CONTEXT RESCUE" not in joined


def test_short_candidate_still_uses_the_tight_hook_reviewer(monkeypatch):
    models, _ = _install_fake_client(monkeypatch, **_approving_pair())
    pool = gemini_pool.GeminiKeyPool(["k1", "k2"])
    candidate = _candidate()
    candidate["clip_type"] = "short"
    main.confirm_clip_with_vision(pool, "m", "v.mp4", candidate, 300.0, _transcript())
    joined = "\n".join(p for p in models.calls if "OPENING FRAME" not in p)
    assert "REAL HOOK / CONTEXT RESCUE" in joined
    assert "ARC COMPLETENESS" not in joined


def test_untagged_candidate_defaults_to_the_tight_hook_reviewer(monkeypatch):
    models, _ = _install_fake_client(monkeypatch, **_approving_pair())
    pool = gemini_pool.GeminiKeyPool(["k1", "k2"])
    main.confirm_clip_with_vision(pool, "m", "v.mp4", _candidate(), 300.0, _transcript())
    joined = "\n".join(p for p in models.calls if "OPENING FRAME" not in p)
    assert "REAL HOOK / CONTEXT RESCUE" in joined
