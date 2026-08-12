"""Tests for smart_crop.PanelTracker — the AutoFlip-style per-panel glide
used by vertical splits. Pure numpy/stdlib, no cv2 or GPU needed."""

import pytest

from smart_crop import PanelTracker


def _tracker(aspect=1.0, **kwargs):
    return PanelTracker(1920, 1080, aspect, **kwargs)


class TestHeadroom:
    def test_crop_sits_below_the_hair_by_headroom(self):
        t = _tracker()
        t.crop = None  # first step snaps straight to the target
        crop = t.step((100, 200, 200, 300))
        x, y, w, h = crop
        assert y == pytest.approx(200 - 300 * 0.18)
        assert y + h >= 200 + 300  # face fully inside

    def test_target_preserves_panel_aspect(self):
        t = _tracker(aspect=1.184)
        t.crop = None
        x, y, w, h = t.step((100, 200, 200, 300))
        assert w / h == pytest.approx(1.184, abs=0.02)


class TestDeadZone:
    def test_micro_movement_does_not_pan_the_camera(self):
        t = _tracker().reset((500, 400, 400, 400))
        crop = t.step((680, 580, 40, 40))
        x, y, w, h = crop
        cx, cy = x + w / 2.0, y + h / 2.0
        # Pan held at the current center (zoom may still ease).
        assert cx == pytest.approx(700.0)
        assert cy == pytest.approx(600.0)
        assert w < 400

    def test_small_shift_inside_zone_is_ignored(self):
        t = _tracker().reset((500, 400, 400, 400))
        # 5px shift < dead zone (0.02 * 400 = 8px).
        crop = t.step((685, 585, 40, 40))
        x, y, w, h = crop
        assert x + w / 2.0 == pytest.approx(700.0)
        assert y + h / 2.0 == pytest.approx(600.0)


class TestSmoothing:
    def test_camera_glides_toward_target(self):
        t = _tracker().reset((500, 200, 900, 600))
        crop = t.step((1000, 250, 100, 100))
        x, y, w, h = crop
        cx, cy = x + w / 2.0, y + h / 2.0
        # Seed center (950, 500) -> target center (1050, 301): moved part
        # of the way, size eased toward 138 but not there yet.
        assert 950 < cx < 1050
        assert 301 < cy < 500
        assert 138 < w < 900

    def test_repeated_steps_converge(self):
        t = _tracker().reset((500, 200, 900, 600))
        for _ in range(300):
            t.step((1000, 250, 100, 100))
        x, y, w, h = t.crop
        assert x == pytest.approx(981, abs=4)
        assert y == pytest.approx(232, abs=4)


class TestContainmentProjection:
    def test_glide_never_clips_the_face(self):
        t = _tracker().reset((500, 200, 900, 600))
        for _ in range(60):
            x, y, w, h = t.step((1000, 250, 100, 100))
            assert x <= 1000 and y <= 250
            assert x + w >= 1100 and y + h >= 350


class TestContainmentValve:
    def test_face_leaving_the_crop_snaps_back(self):
        t = _tracker().reset((0, 0, 400, 400))
        crop = t.step((1500, 500, 100, 100))
        assert crop == (1481.0, 482.0, 138.0, 138.0)


class TestBoundaryClamping:
    def test_crop_never_pans_past_the_frame_edge(self):
        t = _tracker(aspect=2.0).reset((0, 0, 276, 138))
        x, y, w, h = t.step((1800, 500, 100, 100))
        assert x + w == pytest.approx(1920.0)
        assert x >= 0.0


class TestHardCutReset:
    def test_scene_cut_snaps_instantly(self):
        t = _tracker().reset((500, 400, 400, 400))
        crop = t.step((1500, 100, 100, 100), scene_cut=True)
        assert crop == (1481.0, 82.0, 138.0, 138.0)


class TestFallbacks:
    def test_lost_track_holds_then_relaxes_to_full_width(self):
        t = _tracker(lost_hold_frames=3).reset((500, 400, 400, 400))
        for _ in range(3):
            assert t.step(None) == (500.0, 400.0, 400.0, 400.0)
        crop = t.step(None)
        assert crop == (0.0, 0.0, 1920.0, 1080.0)

    def test_box_filling_the_frame_relaxes_to_full_width(self):
        t = _tracker().reset((0, 0, 400, 400))
        crop = t.step((100, 100, 1700, 900))
        assert crop == (0.0, 0.0, 1920.0, 1080.0)

    def test_first_step_with_no_box_uses_full_frame(self):
        t = _tracker()
        assert t.crop is None
        crop = t.step(None)
        assert crop == (0.0, 0.0, 1920.0, 1080.0)
