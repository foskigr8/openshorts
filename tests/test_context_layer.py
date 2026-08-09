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
