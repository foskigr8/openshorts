"""Per-source persistent store: video ID extraction + cache roundtrips."""
import os

import source_store


class TestVideoId:
    def test_watch_url(self):
        assert source_store.video_id_from_url(
            "https://www.youtube.com/watch?v=ua9Z0Lq3QVA") == "ua9Z0Lq3QVA"

    def test_shorts_url(self):
        assert source_store.video_id_from_url(
            "https://youtube.com/shorts/zCLzZ3_n1cs?si=abc") == "zCLzZ3_n1cs"

    def test_youtu_dot_be(self):
        assert source_store.video_id_from_url(
            "https://youtu.be/ua9Z0Lq3QVA") == "ua9Z0Lq3QVA"

    def test_embed_url(self):
        assert source_store.video_id_from_url(
            "https://www.youtube.com/embed/ua9Z0Lq3QVA") == "ua9Z0Lq3QVA"

    def test_non_youtube_returns_none(self):
        assert source_store.video_id_from_url("https://vimeo.com/12345") is None
        assert source_store.video_id_from_url(None) is None

    def test_same_video_different_urls_same_key(self):
        assert (source_store.source_key("https://youtu.be/ua9Z0Lq3QVA")
                == source_store.source_key(
                    "https://www.youtube.com/watch?v=ua9Z0Lq3QVA"))

    def test_non_youtube_keys_are_stable_and_distinct(self):
        a = source_store.source_key("https://vimeo.com/111")
        b = source_store.source_key("https://vimeo.com/222")
        assert a != b
        assert source_store.source_key("https://vimeo.com/111") == a


class TestCacheRoundtrip:
    def test_lookup_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOURCE_CACHE_DIR", str(tmp_path))
        assert source_store.lookup("https://youtu.be/abc123") is None

    def test_save_and_lookup_source(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOURCE_CACHE_DIR", str(tmp_path))
        src = tmp_path / "Ep1.mp4"
        src.write_bytes(b"fake-video-bytes")
        source_store.save_source("https://youtu.be/abc123", str(src), "Ep1")
        hit = source_store.lookup("https://youtu.be/abc123")
        assert hit is not None
        assert hit["title"] == "Ep1"
        assert hit["filename"] == "Ep1.mp4"
        assert hit["transcript"] is None
        assert os.path.exists(hit["video_path"])
        assert os.path.getsize(hit["video_path"]) == len(b"fake-video-bytes")

    def test_transcript_and_context_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOURCE_CACHE_DIR", str(tmp_path))
        src = tmp_path / "Ep1.mp4"
        src.write_bytes(b"x")
        source_store.save_source("https://youtu.be/abc123", str(src), "Ep1")
        transcript = {"language": "en", "segments": [
            {"start": 0.0, "end": 1.0, "text": "hi"}]}
        context = {"summary": "s", "highlights": [], "lovable_moments": []}
        source_store.save_transcript("https://youtu.be/abc123", transcript)
        source_store.save_context("https://youtu.be/abc123", context)
        hit = source_store.lookup("https://youtu.be/abc123")
        assert hit["transcript"] == transcript
        assert hit["context_blob"] == context

    def test_save_without_source_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOURCE_CACHE_DIR", str(tmp_path))
        source_store.save_transcript("https://youtu.be/abc123", {"segments": []})
        assert source_store.lookup("https://youtu.be/abc123") is None
