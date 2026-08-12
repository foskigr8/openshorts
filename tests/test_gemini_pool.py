import pytest

import gemini_pool


class TestGeminiKeyPool:
    def test_empty_pool_is_falsy(self):
        pool = gemini_pool.GeminiKeyPool([])
        assert not pool
        assert pool.acquire() is None

    def test_dedupes_keys_preserving_order(self):
        pool = gemini_pool.GeminiKeyPool(["a", "b", "a", "c"])
        assert len(pool) == 3

    def test_ignores_falsy_entries(self):
        pool = gemini_pool.GeminiKeyPool(["a", "", None, "b"])
        assert len(pool) == 2

    def test_round_robins_across_keys(self):
        pool = gemini_pool.GeminiKeyPool(["a", "b"])
        seen = [pool.acquire() for _ in range(4)]
        assert seen == ["a", "b", "a", "b"]

    def test_skips_keys_marked_bad(self):
        pool = gemini_pool.GeminiKeyPool(["a", "b"])
        pool.mark_bad("a")
        seen = [pool.acquire() for _ in range(3)]
        assert seen == ["b", "b", "b"]

    def test_returns_none_when_every_key_is_bad(self):
        pool = gemini_pool.GeminiKeyPool(["a", "b"])
        pool.mark_bad("a")
        pool.mark_bad("b")
        assert pool.acquire() is None

    def test_single_key_pool_always_returns_it(self):
        pool = gemini_pool.GeminiKeyPool(["only"])
        assert [pool.acquire() for _ in range(3)] == ["only", "only", "only"]


class TestPoolFromEnv:
    def test_primary_key_only(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "primary")
        monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
        pool = gemini_pool.pool_from_env()
        assert len(pool) == 1
        assert pool.acquire() == "primary"

    def test_primary_plus_extra_keys(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "primary")
        monkeypatch.setenv("GEMINI_API_KEYS", "extra1,extra2")
        pool = gemini_pool.pool_from_env()
        assert len(pool) == 3

    def test_no_keys_configured_gives_empty_pool(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
        pool = gemini_pool.pool_from_env()
        assert not pool

    def test_extra_keys_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "primary")
        monkeypatch.setenv("GEMINI_API_KEYS", " extra1 , extra2 ")
        pool = gemini_pool.pool_from_env()
        assert len(pool) == 3


class TestModelFallbackChain:
    def _client(self, failures):
        """Fake Gemini client: raises the mapped exception per model, else
        returns a success marker. Records every model it was asked for."""
        class _Client:
            def __init__(self):
                self.calls = []
                self.models = _Models(self.calls, failures)

        class _Models:
            def __init__(self, calls, failures):
                self.calls = calls
                self.failures = failures

            def generate_content(self, model, contents, config=None):
                self.calls.append(model)
                exc = self.failures.get(model)
                if exc:
                    raise exc
                return f"ok:{model}"

        return _Client()

    def test_default_fallback_is_a_live_model(self, monkeypatch):
        monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
        assert gemini_pool.fallback_model_names() == ["gemini-3.5-flash"]

    def test_env_override_supports_a_chain(self, monkeypatch):
        monkeypatch.setenv("GEMINI_FALLBACK_MODEL",
                           "gemini-3.5-flash,gemini-3-flash-preview")
        assert gemini_pool.fallback_model_names() == [
            "gemini-3.5-flash", "gemini-3-flash-preview"]

    def test_transient_primary_switches_to_fallback(self, monkeypatch):
        monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
        import time
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        client = self._client({
            "gemini-3.1-flash-lite":
                Exception("503 UNAVAILABLE high demand"),
        })
        got = gemini_pool.generate_with_fallback(
            client, "gemini-3.1-flash-lite", "x")
        assert got == "ok:gemini-3.5-flash"
        assert client.calls == ["gemini-3.1-flash-lite", "gemini-3.5-flash"]

    def test_retired_model_in_chain_is_skipped(self, monkeypatch):
        # gemini-2.5-flash is retired (404 NOT_FOUND) — the chain must skip
        # it and reach the next live model instead of dying on the 404.
        monkeypatch.setenv("GEMINI_FALLBACK_MODEL",
                           "gemini-2.5-flash,gemini-3.5-flash")
        import time
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        client = self._client({
            "gemini-3.1-flash-lite":
                Exception("503 UNAVAILABLE high demand"),
            "gemini-2.5-flash": Exception(
                "404 NOT_FOUND: model no longer available to new users"),
        })
        got = gemini_pool.generate_with_fallback(
            client, "gemini-3.1-flash-lite", "x")
        assert got == "ok:gemini-3.5-flash"
        assert client.calls == [
            "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-3.5-flash"]

    def test_all_models_failing_raises_last_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_FALLBACK_MODEL", raising=False)
        import time
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        client = self._client({
            "gemini-3.1-flash-lite": Exception("503 UNAVAILABLE"),
            "gemini-3.5-flash": Exception("429 RESOURCE_EXHAUSTED"),
        })
        with pytest.raises(Exception, match="429"):
            gemini_pool.generate_with_fallback(
                client, "gemini-3.1-flash-lite", "x")
