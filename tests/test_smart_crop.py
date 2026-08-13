"""Tests for smart_crop.PanelTracker — the AutoFlip-style per-panel glide
used by vertical splits. Pure numpy/stdlib, no cv2 or GPU needed."""

import pytest

from smart_crop import PanelTracker
from reframe_v3 import (
    DEFAULT_SIDE_MARGIN, DEFAULT_VERT_MARGIN, crop_rect_containing,
)


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
        t = _tracker()
        t.crop = None
        settled = t.step((680, 580, 40, 40))   # snap onto this face
        cx, cy = settled[0] + settled[2] / 2.0, settled[1] + settled[3] / 2.0
        # A 2px head bob: well inside the dead zone (0.02 * 270 = 5.4px).
        x, y, w, h = t.step((682, 582, 40, 40))
        assert x + w / 2.0 == pytest.approx(cx)
        assert y + h / 2.0 == pytest.approx(cy)

    def test_small_shift_inside_zone_is_ignored(self):
        t = _tracker()
        t.crop = None
        settled = t.step((680, 580, 40, 40))
        # 3px shift < dead zone (0.02 * 270 = 5.4px).
        crop = t.step((683, 583, 40, 40))
        assert crop == settled


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
        # Target is (807, 232, 486, 486); the glide settles inside the dead
        # zone around it rather than landing exactly on it.
        assert x == pytest.approx(807, abs=12)
        assert y == pytest.approx(232, abs=12)


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
        assert crop == (1307.0, 482.0, 486.0, 486.0)


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
        assert crop == (1307.0, 82.0, 486.0, 486.0)


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


class TestPanelFramingParity:
    """A split panel must frame its person the way a SINGLE shot would, and
    then be stacked — not zoom in on the face. The tracker used to run
    0.15/0.10 margins against the static engine's 0.55/0.35, which is what
    made split panels read as passport photos next to the rest of the clip.
    """

    def test_tracker_defaults_match_the_static_engine(self):
        t = _tracker()
        assert t.side_margin == DEFAULT_SIDE_MARGIN
        assert t.vert_margin == DEFAULT_VERT_MARGIN

    @pytest.mark.parametrize("box", [
        (800, 300, 160, 220),      # mid-shot face
        (400, 200, 300, 400),      # close subject
        (1500, 600, 90, 120),      # small, off to one side
    ])
    def test_panel_crop_is_never_tighter_than_the_static_crop(self, box):
        # 9:8 is the panel aspect for a 9:16 output split in two.
        panel_aspect = (9.0 / 16.0) * 2.0
        t = PanelTracker(1920, 1080, panel_aspect)
        t.crop = None
        _, _, w, h = t.step(box)
        static = crop_rect_containing(box, 1920, 1080, panel_aspect)
        assert w >= static[2] - 0.5
        assert h >= static[3] - 0.5

    def test_min_frac_floor_stops_a_passport_close_panel(self):
        # A small face box would ask for a ~150px-tall crop; the floor pulls
        # the camera back to a quarter of the source height.
        t = _tracker(min_height_frac=0.25)
        t.crop = None
        _, _, w, h = t.step((900, 500, 40, 55))
        assert h == pytest.approx(0.25 * 1080)

    def test_floor_never_tightens_a_crop_that_is_already_wider(self):
        loose = _tracker(min_height_frac=0.0)
        loose.crop = None
        floored = _tracker(min_height_frac=0.25)
        floored.crop = None
        box = (400, 200, 300, 400)
        assert floored.step(box)[3] >= loose.step(box)[3]
