import json
import sys
import types
from types import SimpleNamespace

import httpx
import pytest

import transcribe_backends as tb


class FakeFloat(float):
    """Stands in for numpy scalar types (float subclasses) from onnx-asr."""


def _seg(start, end, text, tokens, timestamps):
    return SimpleNamespace(
        start=FakeFloat(start), end=FakeFloat(end), text=text,
        tokens=tokens, timestamps=[FakeFloat(t) for t in timestamps],
    )


# --- token -> word reconstruction ------------------------------------------

def test_words_from_tokens_groups_by_leading_space():
    words = tb._words_from_tokens(
        [" T", "odo", " el", " mundo", "."],
        [0.0, 0.16, 0.32, 0.40, 0.60],
        seg_start=10.0, seg_end=12.0,
    )
    assert [w["word"] for w in words] == [" Todo", " el", " mundo."]
    # Absolute (segment-offset) start times.
    assert words[0]["start"] == pytest.approx(10.0)
    assert words[1]["start"] == pytest.approx(10.32)
    # End inferred from the next word's start; last word ends at segment end.
    assert words[0]["end"] == pytest.approx(10.32)
    assert words[2]["end"] == pytest.approx(11.0, abs=0.3)


def test_words_from_tokens_first_token_without_space_gets_one():
    words = tb._words_from_tokens([
        "Hola", " que", " tal"], [0.0, 0.5, 1.0], seg_start=0.0, seg_end=2.0)
    assert words[0]["word"] == " Hola"


def test_words_from_tokens_end_capped_after_long_silence():
    # Second word starts 5s later; first word's end must stay near its
    # own tokens (cap = last token + 0.6), not stretch across the gap.
    words = tb._words_from_tokens(
        [" uno", " dos"], [0.0, 5.0], seg_start=0.0, seg_end=6.0)
    assert words[0]["end"] <= 0.7


def test_words_from_tokens_all_floats_are_native():
    words = tb._words_from_tokens(
        [" a", "b", " c"], [0.0, 0.1, 0.2], seg_start=FakeFloat(1), seg_end=FakeFloat(2))
    for w in words:
        assert type(w["start"]) is float
        assert type(w["end"]) is float


# --- parakeet transcript assembly ------------------------------------------

@pytest.fixture
def fake_parakeet(monkeypatch):
    segs = [
        _seg(0.35, 2.43, "Todo el mundo habla.",
             [" Todo", " el", " mundo", " hab", "la", "."],
             [0.0, 0.32, 0.40, 0.56, 0.72, 0.80]),
        _seg(2.59, 5.57, "Pero, ¿qué es esto?",
             [" Pero", ",", " ¿", "qu", "é", " es", " esto", "?"],
             [0.0, 0.24, 0.32, 0.48, 0.64, 0.80, 1.04, 1.36]),
    ]
    model = SimpleNamespace(recognize=lambda path: iter(segs))
    monkeypatch.setattr(tb, "_get_parakeet_model", lambda: model)
    monkeypatch.setattr(tb, "_extract_wav", lambda path: "/tmp/fake.wav")
    monkeypatch.setattr(tb.os, "remove", lambda path: None)
    return segs


def test_parakeet_transcript_matches_contract(fake_parakeet, monkeypatch):
    monkeypatch.setattr(tb, "_detect_language", lambda text: "es")
    t = tb._transcribe_with_parakeet("video.mp4")

    assert t["text"] == "Todo el mundo habla. Pero, ¿qué es esto?"
    assert t["language"] == "es"
    assert len(t["segments"]) == 2

    seg = t["segments"][1]
    assert type(seg["start"]) is float and type(seg["end"]) is float
    # Continuations (",", "qu", "é") merged; leading spaces preserved.
    assert [w["word"] for w in seg["words"]] == [" Pero,", " ¿qué", " es", " esto?"]
    # Segment offset applied to word times.
    assert seg["words"][0]["start"] == pytest.approx(2.59)
    # Whole transcript is JSON-serializable (metadata json.dump path).
    json.dumps(t)


def test_parakeet_words_survive_merge_continuation_words(fake_parakeet, monkeypatch):
    from subtitles import merge_continuation_words
    monkeypatch.setattr(tb, "_detect_language", lambda text: "es")
    t = tb._transcribe_with_parakeet("video.mp4")
    for seg in t["segments"]:
        assert merge_continuation_words(seg["words"]) == seg["words"]


# --- fallback policy --------------------------------------------------------

def _valid_transcript(n_words=200, duration=120.0, language="es"):
    words = [
        {"word": f" w{i}", "start": i * duration / n_words,
         "end": (i + 1) * duration / n_words}
        for i in range(n_words)
    ]
    return {"text": "x " * n_words, "language": language,
            "segments": [{"start": 0.0, "end": duration, "text": "x", "words": words}]}


def test_fallback_reason_none_for_good_result():
    assert tb._parakeet_fallback_reason(_valid_transcript()) is None


def test_fallback_when_no_words():
    t = {"text": "", "language": "es",
         "segments": [{"start": 0, "end": 5, "text": "x", "words": []}]}
    assert "no words" in tb._parakeet_fallback_reason(t)


def test_fallback_when_language_unsupported():
    assert "outside" in tb._parakeet_fallback_reason(
        _valid_transcript(language="ja"))


def test_fallback_when_word_rate_absurdly_low():
    assert tb._parakeet_fallback_reason(
        _valid_transcript(n_words=5, duration=600.0)) is not None


def test_transcribe_media_falls_back_on_parakeet_exception(monkeypatch):
    def boom(path):
        raise RuntimeError("onnx exploded")

    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "parakeet")
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(tb, "_transcribe_with_parakeet", boom)
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: sentinel)
    assert tb.transcribe_media("video.mp4") is sentinel


def test_transcribe_media_default_is_whisper(monkeypatch):
    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.delenv("TRANSCRIBE_BACKEND", raising=False)
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(
        tb, "_transcribe_with_parakeet",
        lambda path: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: sentinel)
    assert tb.transcribe_media("video.mp4") is sentinel


def test_transcribe_media_raises_on_silent_video(monkeypatch):
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: False)
    with pytest.raises(tb.NoAudioError):
        tb.transcribe_media("silent.mp4")


# --- whisper singleton ------------------------------------------------------

@pytest.fixture
def fake_faster_whisper(monkeypatch):
    created = []

    class FakeModel:
        def __init__(self, model_size, device=None, compute_type=None):
            self.model_size = model_size
            self.device = device
            created.append(self)

        def transcribe(self, path, **params):
            segs = (s for s in [SimpleNamespace(
                start=0.0, end=1.0, text=" hola", words=None)])
            return segs, SimpleNamespace(language="es")

    fake_module = types.SimpleNamespace(WhisperModel=FakeModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr(tb, "_whisper_model", None)
    monkeypatch.setattr(tb, "_whisper_key", None)
    return created


def test_whisper_model_is_singleton(fake_faster_whisper, monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    a, _ = tb._get_whisper_model()
    b, _ = tb._get_whisper_model()
    assert a is b
    assert len(fake_faster_whisper) == 1


def test_whisper_model_rebuilds_when_env_changes(fake_faster_whisper, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "small")
    a, _ = tb._get_whisper_model()
    monkeypatch.setenv("WHISPER_MODEL", "large-v3-turbo")
    b, _ = tb._get_whisper_model()
    assert a is not b
    assert b.model_size == "large-v3-turbo"


def test_run_whisper_transcription_materializes_segments(fake_faster_whisper, monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    segments, info = tb.run_whisper_transcription("video.mp4")
    assert isinstance(segments, list)
    assert info.language == "es"


# --- assemblyai ---------------------------------------------------------

def _aa_word(text, start_ms, end_ms):
    return {"text": text, "start": start_ms, "end": end_ms, "confidence": 0.99}


def test_assemblyai_words_to_segments_uses_utterance_boundaries():
    utterances = [
        {
            "start": 1000, "end": 3000, "speaker": "A",
            "text": "Hello there friend.",
            "words": [_aa_word("Hello", 1000, 1400), _aa_word("there", 1450, 1800),
                      _aa_word("friend.", 1850, 2200)],
        },
        {
            "start": 3200, "end": 5000, "speaker": "A",
            "text": "What a day.",
            "words": [_aa_word("What", 3200, 3400), _aa_word("a", 3450, 3500),
                      _aa_word("day.", 3550, 3900)],
        },
    ]
    sentiment = [
        {"text": "Hello there friend.", "sentiment": "POSITIVE", "start": 1000, "end": 3000},
        {"text": "What a day.", "sentiment": "NEGATIVE", "start": 3200, "end": 5000},
    ]
    segments = tb._assemblyai_words_to_segments([], utterances, sentiment)

    assert len(segments) == 2
    assert segments[0]["start"] == pytest.approx(1.0)
    # end derives from the last word's own timestamp (2.2s), not the coarser
    # utterance-level "end" field (3.0s) — more precise, and consistent with
    # how the no-utterances path always worked.
    assert segments[0]["end"] == pytest.approx(2.2)
    assert segments[0]["text"] == "Hello there friend."
    assert [w["word"] for w in segments[0]["words"]] == [" Hello", " there", " friend."]
    assert segments[0]["sentiment"] == "+"
    assert segments[1]["sentiment"] == "-"
    # No highlights supplied — every segment defaults to not-highlighted.
    assert segments[0]["highlight"] is False
    assert segments[1]["highlight"] is False
    # Whole transcript stays JSON-serializable (metadata json.dump path).
    json.dumps(segments)


def test_assemblyai_words_to_segments_tags_auto_highlights():
    utterances = [
        {
            "start": 1000, "end": 3000, "speaker": "A", "text": "Hello there friend.",
            "words": [_aa_word("Hello", 1000, 1400), _aa_word("there", 1450, 1800),
                      _aa_word("friend.", 1850, 2200)],
        },
        {
            "start": 3200, "end": 5000, "speaker": "A", "text": "What a day.",
            "words": [_aa_word("What", 3200, 3400), _aa_word("a", 3450, 3500),
                      _aa_word("day.", 3550, 3900)],
        },
    ]
    highlights = [
        {"text": "there friend", "rank": 0.09, "count": 1,
         "timestamps": [{"start": 1450, "end": 2200}]},  # overlaps segment 0 only
    ]
    segments = tb._assemblyai_words_to_segments([], utterances, highlights_result=highlights)
    assert segments[0]["highlight"] is True
    assert segments[1]["highlight"] is False


def test_assemblyai_words_to_segments_tags_speaker_and_strips_transient_key():
    utterances = [
        {"start": 1000, "end": 2000, "speaker": "A", "text": "Hi.",
         "words": [_aa_word("Hi.", 1000, 2000)]},
        {"start": 2200, "end": 3200, "speaker": "B", "text": "Hey.",
         "words": [_aa_word("Hey.", 2200, 3200)]},
    ]
    segments = tb._assemblyai_words_to_segments([], utterances, sentiment_results=None)
    assert segments[0]["speaker"] == "A"
    assert segments[1]["speaker"] == "B"
    # The transient tagging key used to derive segment-level speaker must
    # never leak into the word dicts other pipeline code reads.
    for seg in segments:
        for w in seg["words"]:
            assert "_speaker" not in w


def test_assemblyai_words_to_segments_speaker_none_without_diarization():
    words = [_aa_word("hello", 0, 200)]
    segments = tb._assemblyai_words_to_segments(words, utterances=None, sentiment_results=None)
    assert segments[0]["speaker"] is None


def test_assemblyai_words_to_segments_long_utterance_shares_speaker_across_split_segments():
    # A single long utterance that gets internally split (word-count ceiling)
    # must propagate the SAME speaker to every resulting segment.
    words = [_aa_word(f"w{i}", i * 300, i * 300 + 250) for i in range(30)]
    utterances = [{"start": 0, "end": 9000, "speaker": "A",
                  "text": " ".join(f"w{i}" for i in range(30)), "words": words}]
    segments = tb._assemblyai_words_to_segments([], utterances, sentiment_results=None)
    assert len(segments) >= 2
    assert all(seg["speaker"] == "A" for seg in segments)


def test_assemblyai_words_to_segments_splits_a_single_long_utterance():
    # Regression test for prod 30-jul-2026: a continuous single-speaker
    # narrator (no speaker changes) came back as ONE AssemblyAI utterance
    # spanning an entire 208s video, leaving the narrative-selection stage
    # with no internal structure — it just handed back almost the whole
    # video. A single utterance with 30 words (over the 20-word ceiling)
    # must now still split into multiple segments.
    words = [_aa_word(f"w{i}", i * 300, i * 300 + 250) for i in range(30)]
    utterances = [{"start": 0, "end": 9000, "text": " ".join(f"w{i}" for i in range(30)), "words": words}]
    segments = tb._assemblyai_words_to_segments([], utterances, sentiment_results=None)

    assert len(segments) >= 2
    assert sum(len(s["words"]) for s in segments) == 30


def test_assemblyai_words_to_segments_splits_a_single_utterance_on_internal_pause():
    # A single utterance can also contain a real mid-speech pause (e.g. a
    # breath or a beat) well under the 20-word ceiling — that must still
    # force a split, not just the word-count limit.
    words = ([_aa_word("one", 0, 200), _aa_word("two", 250, 450)]
             + [_aa_word("three", 3000, 3300)])  # >1.2s gap mid-utterance
    utterances = [{"start": 0, "end": 3300, "text": "one two three", "words": words}]
    segments = tb._assemblyai_words_to_segments([], utterances, sentiment_results=None)
    assert len(segments) == 2


def test_assemblyai_words_to_segments_falls_back_without_utterances():
    words = [_aa_word(f"w{i}", i * 200, i * 200 + 150) for i in range(25)]
    segments = tb._assemblyai_words_to_segments(words, utterances=None, sentiment_results=None)

    # 25 words chunked at the 20-word ceiling -> at least 2 segments.
    assert len(segments) >= 2
    assert sum(len(s["words"]) for s in segments) == 25
    assert all(s["sentiment"] == "0" for s in segments)  # no sentiment data supplied


def test_assemblyai_words_to_segments_splits_on_silence_gap():
    words = [_aa_word("one", 0, 200), _aa_word("two", 250, 450),
             _aa_word("three", 3000, 3300)]  # >1.2s gap before "three"
    segments = tb._assemblyai_words_to_segments(words, utterances=None, sentiment_results=None)
    assert len(segments) == 2
    assert [w["word"] for w in segments[0]["words"]] == [" one", " two"]
    assert [w["word"] for w in segments[1]["words"]] == [" three"]


def test_assemblyai_words_to_segments_survive_merge_continuation_words():
    from subtitles import merge_continuation_words
    utterances = [{
        "start": 0, "end": 2000, "text": "Hello there.",
        "words": [_aa_word("Hello", 0, 400), _aa_word("there.", 450, 900)],
    }]
    segments = tb._assemblyai_words_to_segments([], utterances, None)
    for seg in segments:
        assert merge_continuation_words(seg["words"]) == seg["words"]


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=self)

    def json(self):
        return self._payload


class _FakeClient:
    """Stands in for httpx.Client(...) as a context manager. `responses` is a
    queue of _FakeResponse consumed in order across post()/get() calls, so a
    test can script upload -> submit -> poll -> poll -> ... in sequence."""

    def __init__(self, responses):
        self._responses = responses

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, **kwargs):
        return self._responses.pop(0)

    def get(self, url, **kwargs):
        return self._responses.pop(0)


def test_transcribe_with_assemblyai_happy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test_key")
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake video bytes")

    completed_payload = {
        "status": "completed",
        "text": "Hello there friend.",
        "language_code": "en",
        "words": [],
        "utterances": [{
            "start": 0, "end": 2000, "text": "Hello there friend.",
            "words": [_aa_word("Hello", 0, 400), _aa_word("there", 450, 800),
                      _aa_word("friend.", 850, 1200)],
        }],
        "sentiment_analysis_results": [
            {"text": "Hello there friend.", "sentiment": "POSITIVE", "start": 0, "end": 2000},
        ],
    }
    responses = [
        _FakeResponse({"upload_url": "https://cdn.assemblyai.com/upload/abc"}),  # upload
        _FakeResponse({"id": "transcript_123", "status": "queued"}),             # submit
        _FakeResponse(completed_payload),                                       # poll (done immediately)
    ]
    monkeypatch.setattr(tb.httpx, "Client", lambda timeout=None: _FakeClient(responses))

    transcript = tb._transcribe_with_assemblyai(str(media))

    assert transcript["language"] == "en"
    assert transcript["text"] == "Hello there friend."
    assert len(transcript["segments"]) == 1
    assert transcript["segments"][0]["sentiment"] == "+"
    json.dumps(transcript)  # matches the whisper/parakeet contract: JSON-serializable


def test_transcribe_with_assemblyai_raises_without_key(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        tb._transcribe_with_assemblyai("video.mp4")


def test_transcribe_with_assemblyai_raises_on_error_status(monkeypatch, tmp_path):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test_key")
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fake video bytes")

    responses = [
        _FakeResponse({"upload_url": "https://cdn.assemblyai.com/upload/abc"}),
        _FakeResponse({"id": "transcript_123", "status": "queued"}),
        _FakeResponse({"status": "error", "error": "Invalid audio file"}),
    ]
    monkeypatch.setattr(tb.httpx, "Client", lambda timeout=None: _FakeClient(responses))

    with pytest.raises(RuntimeError, match="Invalid audio file"):
        tb._transcribe_with_assemblyai(str(media))


def test_transcribe_media_uses_assemblyai_backend(monkeypatch):
    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "assemblyai")
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(tb, "_transcribe_with_assemblyai", lambda path: sentinel)
    monkeypatch.setattr(
        tb, "_transcribe_with_whisper",
        lambda path: (_ for _ in ()).throw(AssertionError("should not run")))
    assert tb.transcribe_media("video.mp4") is sentinel


def test_transcribe_media_falls_back_on_assemblyai_exception(monkeypatch):
    def boom(path):
        raise RuntimeError("assemblyai exploded")

    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "assemblyai")
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(tb, "_transcribe_with_assemblyai", boom)
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: sentinel)
    assert tb.transcribe_media("video.mp4") is sentinel


# --- Backend selection ------------------------------------------------------
#
# Measured on Kaggle 5-aug-2026 (job b86b8c5a): ASSEMBLYAI_API_KEY was set and
# local whisper ran anyway, because the backend is chosen by TRANSCRIBE_BACKEND
# and nothing set it. That cost 254s of a ~420s job AND all diarization, which
# is subject_policy's TIER_DIARIZED framing evidence.


def test_explicit_backend_always_wins(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "whisper")
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "k")
    assert tb._select_backend() == "whisper"


def test_assemblyai_preferred_when_key_set_and_backend_unset(monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_BACKEND", raising=False)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "k")
    assert tb._select_backend() == "assemblyai"


def test_whisper_remains_the_default_without_a_key(monkeypatch):
    monkeypatch.delenv("TRANSCRIBE_BACKEND", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    assert tb._select_backend() == "whisper"


def test_blank_backend_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "   ")
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "k")
    assert tb._select_backend() == "assemblyai"
