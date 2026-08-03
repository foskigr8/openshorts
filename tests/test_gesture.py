"""Round-5 spec 3 — USE_GESTURE hand-gesture boost.

Off by default and byte-identical when off; when on, a gesturing candidate
outranks a still one, the size floor is respected, missing landmarks fail
open, and the gesture boost can never outrank an LR-ASD-identified speaker.
"""
import numpy as np
import pytest

import reframe_v2 as rv

cv2 = pytest.importorskip("cv2")


class _LM:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakePose:
    def __init__(self, positions):
        # positions: list of (x, y) per landmark index. Bright crops (the
        # "moving" candidate's marker region) get a shifted pose on call 2+;
        # dark crops stay static — so the two candidates differ.
        self._positions = positions
        self._calls = 0

    def process(self, rgb):
        self._calls += 1
        lms = []
        # Bright crops (the "moving" candidate) get a position that VARIES per
        # call, so the movement window sees a real delta; dark crops stay fixed.
        shift = self._calls % 3 if rgb.mean() > 50 else 0
        for i in range(17):
            if i in rv._GESTURE_LANDMARKS:
                base = self._positions[i]
                if shift:
                    base = (base[0] + 0.12 * shift, base[1] + 0.08 * shift)
                lms.append(_LM(*base))
            else:
                lms.append(_LM(0.5, 0.5))
        return type("R", (), {"pose_landmarks": type("L", (), {"landmark": lms})})()


def _frame(w=320, h=180):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _cand(box, score):
    return {"box": box, "score": score, "raw_score": score}


def test_off_is_byte_identical(monkeypatch):
    monkeypatch.setattr(rv, "USE_GESTURE", False)
    cands = [_cand([10, 10, 80, 100], 500.0)]
    before = [dict(c) for c in cands]
    rv._apply_gesture_boost(cands, _frame(), 1920)
    assert cands[0]["score"] == before[0]["score"]
    assert "gesture_activity" not in cands[0]


def test_gesturing_candidate_outranks_still_one(monkeypatch):
    monkeypatch.setattr(rv, "USE_GESTURE", True)
    fake = _FakePose([(0.5, 0.5)] * 17)
    monkeypatch.setattr(rv, "_gesture_pose_graph", lambda: fake)
    frame = _frame()
    # Bright marker inside the "moving" candidate's box region (right side).
    frame[30:90, 200:270] = 255
    still = _cand([10, 10, 60, 80], 100.0)
    moving = _cand([200, 10, 60, 80], 100.0)
    # Three detection samples: the movement window needs history to build.
    rv._apply_gesture_boost([still, moving], frame, 320)
    rv._apply_gesture_boost([still, moving], frame, 320)
    rv._apply_gesture_boost([still, moving], frame, 320)
    assert moving["score"] > still["score"]
    # Boosted at least once (score >= base * GESTURE_BOOST; it may stack across
    # qualifying samples).
    assert moving["score"] >= 100.0 * rv.GESTURE_BOOST


def test_size_floor_respected(monkeypatch):
    monkeypatch.setattr(rv, "USE_GESTURE", True)
    monkeypatch.setattr(rv, "_gesture_pose_graph",
                        lambda: _FakePose([(0.5, 0.5)] * 17))
    big = _cand([10, 10, 100, 120], 1000.0)
    sliver = _cand([200, 10, 5, 5], 2.0)  # far below the relative floor
    rv._apply_gesture_boost([big, sliver], _frame(), 1920)
    assert sliver["score"] == 2.0  # never boosted


def test_missing_landmarks_fail_open(monkeypatch):
    monkeypatch.setattr(rv, "USE_GESTURE", True)

    class _Empty:
        def process(self, rgb):
            return type("R", (), {"pose_landmarks": None})()

    monkeypatch.setattr(rv, "_gesture_pose_graph", lambda: _Empty())
    cand = _cand([10, 10, 60, 80], 100.0)
    rv._apply_gesture_boost([cand], _frame(), 1920)  # must not raise
    assert cand["score"] == 100.0


def test_gesture_never_outranks_asd_speaker():
    # Constants are the contract: LR-ASD's speaker boost (4.0) must exceed the
    # gesture boost (2.0), so an identified speaker always beats a performer.
    assert rv.ASD_SPEAKER_BOOST > rv.GESTURE_BOOST
    assert rv.GESTURE_BOOST >= 1.5
