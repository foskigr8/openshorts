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
