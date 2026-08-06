"""PART 6 download gate (6-aug-2026): verify what actually landed.

The downloader requests HD, but nothing downstream checked resolution or
bitrate — a low-bitrate 1080p stream (or a ~360p progressive fallback) shipped
silently and the reframe inherited the softness. These tests pin the
measurement and the floor logic.
"""
import os

import numpy as np
import pytest

import main


def test_quality_floor_flags_low_height():
    assert main._download_quality_floor(
        {"height": 480, "bitrate_mbps": 4.0}) is not None


def test_quality_floor_flags_low_bitrate():
    assert main._download_quality_floor(
        {"height": 1080, "bitrate_mbps": 0.8}) is not None


def test_quality_floor_passes_genuine_hd():
    assert main._download_quality_floor(
        {"height": 1080, "bitrate_mbps": 4.0}) is None


def test_hd_gate_raises_on_sub_hd_when_required():
    with pytest.raises(RuntimeError, match="HD download failed"):
        main._enforce_hd_gate({"height": 480, "bitrate_mbps": 4.0},
                              require_hd=True)


def test_hd_gate_warns_but_allows_when_not_required():
    assert main._enforce_hd_gate(
        {"height": 480, "bitrate_mbps": 4.0}, require_hd=False) is not None


def test_hd_gate_honors_allow_low_quality_escape_hatch(monkeypatch):
    monkeypatch.setenv("ALLOW_LOW_QUALITY_SOURCE", "1")
    assert main._enforce_hd_gate(
        {"height": 480, "bitrate_mbps": 4.0}, require_hd=True) is not None


def test_hd_gate_passes_genuine_hd():
    assert main._enforce_hd_gate(
        {"height": 1080, "bitrate_mbps": 4.0}, require_hd=True) is None


def test_probe_video_specs_reports_true_resolution(tmp_path):
    video = tmp_path / "tiny.mp4"
    writer = main.cv2.VideoWriter(
        str(video), main.cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    assert writer.isOpened()
    # Noisy frames so the encoded file is big enough for a real bitrate.
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for _ in range(5):
        writer.write(frame)
    writer.release()

    specs = main._probe_video_specs(str(video))
    assert specs is not None
    assert specs["width"] == 320 and specs["height"] == 240
    assert specs["duration_s"] == pytest.approx(0.5, abs=0.1)
    assert specs["bitrate_mbps"] > 0


def test_probe_video_specs_returns_none_for_missing_file():
    assert main._probe_video_specs("/nonexistent/file.mp4") is None
