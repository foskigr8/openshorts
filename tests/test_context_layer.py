"""Pre-download context layer: 3-part prompt, URI call, blob I/O, failure."""
import json
from types import SimpleNamespace

import context_layer


class TestPromptContract:
    def test_three_niche_agnostic_sections(self):
        t = context_layer.CONTEXT_PROMPT_TEMPLATE
        assert "1. SUMMARY" in t and "2. HIGHLIGHTS" in t and "3. LOVABLE_MOMENTS" in t
        assert "never assume a niche" in t
        assert "audiovisual" in t

    def test_highlights_are_an_enumeration_not_a_summary(self):
        # The owner's feedback: a 90-minute episode returned only 7
        # highlights + 3 lovable moments — far too thin. The prompt must
        # force dense, runtime-scaled enumeration.
        t = context_layer.CONTEXT_PROMPT_TEMPLATE
        assert "ENUMERATION, not a summary" in t
        assert "40-80+" in t
        assert "fewer than 8" in t
        assert "DENSITY CHECK" in t


class TestKeyResolution:
    def test_dedicated_key_wins(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_GEMINI_API_KEY", "context-key")
        monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
        assert context_layer.resolve_api_key() == "context-key"

    def test_falls_back_to_primary(self, monkeypatch):
        monkeypatch.delenv("CONTEXT_GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
        assert context_layer.resolve_api_key() == "primary-key"


class TestAnalyzeUrl:
    def test_success_writes_blob(self, monkeypatch, tmp_path):
        blob = {
            "summary": "A challenge between two people.",
            "highlights": [{"start_s": 2.0, "end_s": 6.0,
                            "description": "the reveal", "type": "payoff"}],
            "lovable_moments": [],
            "video_duration_s": 300.0,
        }
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(context_layer, "_call_gemini",
                            lambda api_key, url, prompt: (blob, None))
        out = tmp_path / "gemini_context.json"
        result = context_layer.analyze_url("https://youtube.com/watch?v=abc", str(out))
        assert result["summary"] == blob["summary"]
        assert json.loads(out.read_text())["summary"] == blob["summary"]

    def test_uses_part_from_uri(self, monkeypatch):
        """The link is sent as a URI part, not an uploaded file."""
        seen = {}
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        class _FakeResponse:
            parsed = context_layer.ContextBlobResponse(
                summary="s", highlights=[], lovable_moments=[])

        def fake_generate(client, model_name, contents, config=None, max_attempts=1, log=None):
            seen["contents"] = contents
            return _FakeResponse()

        monkeypatch.setattr(context_layer.gemini_pool, "generate_with_fallback",
                            fake_generate)
        monkeypatch.setattr(context_layer.gemini_worker, "raise_if_blocked",
                            lambda resp: None)
        monkeypatch.setattr(context_layer.gemini_worker, "_calculate_cost_analysis",
                            lambda resp, model: None)
        context_layer.analyze_url("https://youtube.com/watch?v=abc")
        assert isinstance(seen["contents"][0],
                          context_layer.genai_types.Part)

    def test_failure_returns_none_without_raising(self, monkeypatch, capsys):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(context_layer, "_call_gemini",
                            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        assert context_layer.analyze_url("https://youtube.com/watch?v=abc") is None
        assert "transcript-only" in capsys.readouterr().out

    def test_private_url_rejected(self, monkeypatch, capsys):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        assert context_layer.analyze_url("http://localhost:8000/x") is None


class TestLoadContext:
    def test_missing_returns_none(self, tmp_path):
        assert context_layer.load_context(str(tmp_path / "nope.json")) is None

    def test_corrupt_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        assert context_layer.load_context(str(p)) is None

    def test_roundtrip(self, tmp_path):
        p = tmp_path / "ok.json"
        p.write_text(json.dumps({"summary": "x"}))
        assert context_layer.load_context(str(p)) == {"summary": "x"}


class TestWaitBudget:
    def test_full_budget_when_thread_just_started(self):
        # Cached re-run: the thread starts ~now, so it gets the full target.
        assert context_layer.wait_budget(0.0) == 35.0

    def test_remainder_scales_with_head_start(self):
        assert context_layer.wait_budget(10.0) == 25.0

    def test_never_below_floor(self):
        # Fresh run: the download+transcribe runway already exceeded the
        # target, so only the short floor remains (old 5s behaviour).
        assert context_layer.wait_budget(60.0) == 5.0

    def test_respects_custom_target_and_floor(self):
        assert context_layer.wait_budget(0.0, target=20.0, floor=3.0) == 20.0
        assert context_layer.wait_budget(50.0, target=20.0, floor=3.0) == 3.0


class TestTranscriptContextFallback:
    def test_compaction_keeps_speaker_times_and_skips_empty(self):
        segs = [
            {"start": 10, "end": 13, "speaker": "A",
             "words": [{"text": "hi"}, {"text": "there"}]},
            {"start": 14, "end": 16, "speaker": "B", "text": ""},
            {"start": 20, "end": 22, "speaker": None, "text": "no label"},
        ]
        out = context_layer._compact_transcript_segments(segs)
        assert "[10.0-13.0s] speaker A: hi there" in out
        assert "[20.0-22.0s]: no label" in out
        assert "speaker B" not in out  # empty segment skipped

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_FROM_TRANSCRIPT", "0")
        assert context_layer.build_context_from_transcript(
            {"segments": [{"start": 0, "end": 1, "text": "x"}]}) is None

    def test_no_segments_returns_none(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_FROM_TRANSCRIPT", "1")
        assert context_layer.build_context_from_transcript({}) is None
