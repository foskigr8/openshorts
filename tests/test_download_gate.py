"""Pre-download client capability gate — pure stdlib, runs anywhere."""

from download_gate import ClientCannotServeFloor, can_serve_hd_floor


def _fmts(*entries):
    return {"formats": [dict(e) for e in entries]}


class TestCanServeHdFloor:
    def test_1080p_avc1_qualifies(self):
        assert can_serve_hd_floor(
            _fmts({"height": 1080, "vcodec": "avc1.640028"})) is True

    def test_only_360p_does_not(self):
        assert can_serve_hd_floor(
            _fmts({"height": 360, "vcodec": "avc1.4d401e"})) is False

    def test_missing_formats_is_conservative(self):
        # Can't judge from the probe -> let the download decide, never skip.
        assert can_serve_hd_floor({}) is True
        assert can_serve_hd_floor({"formats": []}) is True

    def test_audio_only_does_not_qualify(self):
        assert can_serve_hd_floor(_fmts(
            {"height": None, "vcodec": "none"},
            {"height": 1080, "vcodec": "none"})) is False

    def test_av1_1080p_does_not_qualify(self):
        # Turing's NVDEC has no AV1 decoder — the HD chain excludes it too.
        assert can_serve_hd_floor(
            _fmts({"height": 1440, "vcodec": "av01.0.05M.08"})) is False

    def test_10bit_vp9_does_not_qualify(self):
        assert can_serve_hd_floor(
            _fmts({"height": 1440, "vcodec": "vp09.00.40.08"})) is False

    def test_8bit_vp9_qualifies(self):
        assert can_serve_hd_floor(
            _fmts({"height": 1440, "vcodec": "vp09.00.10.08"})) is True

    def test_cap_excludes_above_source_max(self, monkeypatch):
        monkeypatch.setenv("SOURCE_MAX_HEIGHT", "1440")
        assert can_serve_hd_floor(
            _fmts({"height": 2160, "vcodec": "avc1.640033"})) is False
        monkeypatch.setenv("SOURCE_MAX_HEIGHT", "0")
        assert can_serve_hd_floor(
            _fmts({"height": 2160, "vcodec": "avc1.640033"})) is True

    def test_floor_respects_min_source_height_env(self, monkeypatch):
        monkeypatch.setenv("MIN_SOURCE_HEIGHT", "720")
        assert can_serve_hd_floor(
            _fmts({"height": 720, "vcodec": "avc1.4d401f"})) is True

    def test_exception_carries_best_height(self):
        exc = ClientCannotServeFloor(360)
        assert exc.best_height == 360
        assert "360" in str(exc)
