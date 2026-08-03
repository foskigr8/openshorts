"""analyze_scenes_strategy() — TRACK vs GENERAL reframe-mode selection.

Regression cover for a real reaction-cam clip (30-jul-2026): a small
webcam-style face bubble reacting to an on-screen Instagram graphic detects
as exactly one face, so the old face-count-only heuristic picked TRACK and
tight-cropped around that tiny corner face — squeezing/cutting off the
actual on-screen content the reaction was about. The fix adds a
relative-face-size signal: a face that's tiny relative to the frame reads
as an overlay bubble, not a normally-framed single speaker, and should get
GENERAL (full-width blurred layout) instead.
"""
import types

import numpy as np
import pytest

main = pytest.importorskip("main")

FRAME_W, FRAME_H = 1920, 1080


class _Timecode:
    def __init__(self, frame):
        self._frame = frame

    def get_frames(self):
        return self._frame


class _FakeCapture:
    """Feeds a fixed sequence of frames regardless of which index is sought."""

    def __init__(self, n_frames=30, mean_brightness=128):
        self._n_frames = n_frames
        self._mean = mean_brightness
        self._pos = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return 30.0  # CAP_PROP_FPS

    def set(self, prop, value):
        self._pos = int(value)

    def read(self):
        if self._pos >= self._n_frames:
            return False, None
        frame = np.full((FRAME_H, FRAME_W, 3), self._mean, dtype=np.uint8)
        return True, frame

    def release(self):
        pass


def _install_fake_capture(monkeypatch, n_frames=30):
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda *_a, **_kw: _FakeCapture(n_frames))


def _scenes(n=1, frames_per_scene=10):
    return [(_Timecode(i * frames_per_scene), _Timecode((i + 1) * frames_per_scene))
            for i in range(n)]


def test_normally_framed_single_speaker_is_track(monkeypatch):
    _install_fake_capture(monkeypatch)
    # A talking-head face filling a third of the frame width.
    monkeypatch.setattr(main, "detect_face_candidates",
                         lambda frame: [{'box': [800, 200, 600, 600], 'score': 360000}])
    strategies = main.analyze_scenes_strategy("video.mp4", _scenes())
    assert strategies == ['TRACK']


def test_small_corner_reaction_bubble_is_general(monkeypatch):
    _install_fake_capture(monkeypatch)
    # A small circular reaction-cam bubble: ~10% of frame width.
    monkeypatch.setattr(main, "detect_face_candidates",
                         lambda frame: [{'box': [50, 700, 190, 190], 'score': 36100}])
    strategies = main.analyze_scenes_strategy("video.mp4", _scenes())
    assert strategies == ['GENERAL']


def test_no_faces_is_general(monkeypatch):
    _install_fake_capture(monkeypatch)
    monkeypatch.setattr(main, "detect_face_candidates", lambda frame: [])
    strategies = main.analyze_scenes_strategy("video.mp4", _scenes())
    assert strategies == ['GENERAL']


def test_group_shot_is_general(monkeypatch):
    _install_fake_capture(monkeypatch)
    monkeypatch.setattr(
        main, "detect_face_candidates",
        lambda frame: [{'box': [100, 200, 300, 300], 'score': 90000},
                        {'box': [900, 200, 300, 300], 'score': 90000}])
    strategies = main.analyze_scenes_strategy("video.mp4", _scenes())
    assert strategies == ['GENERAL']


def test_unopened_capture_defaults_all_scenes_to_track(monkeypatch):
    class _ClosedCapture(_FakeCapture):
        def isOpened(self):
            return False
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda *_a, **_kw: _ClosedCapture())
    strategies = main.analyze_scenes_strategy("video.mp4", _scenes(n=3))
    assert strategies == ['TRACK', 'TRACK', 'TRACK']
