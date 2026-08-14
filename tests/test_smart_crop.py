"""Tests for smart_crop.PanelTracker — the AutoFlip-style per-panel glide
used by vertical splits. Pure numpy/stdlib, no cv2 or GPU needed."""

import pytest

import framing_contract as fc
from smart_crop import PanelTracker, _recontain
from reframe_v3 import crop_rect_containing

PANEL_ASPECT = (9.0 / 16.0) * 2.0      # one half of a 9:16 output


def _tracker(aspect=1.0, **kwargs):
    return PanelTracker(1920, 1080, aspect, **kwargs)


class TestHeadroom:
    def test_crop_clears_the_hair_not_just_the_face_box(self):
        """The old assertion was `y == sy - sh*0.18`: the crop top pinned
        0.18 face-heights above the DETECTOR box. Real hair sits ~0.35 above
        it, so that anchor was inside the subject's head every time. The
        contract's headroom is measured against the head box instead."""
        t = _tracker()
        t.crop = None  # first step snaps straight to the target
        box = (100, 200, 200, 300)
        crop = t.step(box)
        assert fc.headroom_frac(crop, box, 1920, 1080) >= fc.HEADROOM_MIN
        head = fc.head_box(box, 1920, 1080)
        assert crop[1] <= head[1] + 0.5           # hair inside the crop
        assert crop[1] + crop[3] >= box[1] + box[3]

    @pytest.mark.parametrize("h", [40, 90, 150, 240, 380])
    def test_headroom_holds_at_every_subject_size(self, h):
        t = _tracker(aspect=PANEL_ASPECT)
        t.crop = None
        box = (800, 300, 0.8 * h, h)
        crop = t.step(box)
        assert (fc.headroom_frac(crop, box, 1920, 1080) >= fc.HEADROOM_MIN
                or crop[1] <= 0.5)

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
        # Target is (807, 127, 486, 486); the glide settles inside the dead
        # zone around it rather than landing exactly on it. The y moved up
        # from the old 232 because the anchor is now the eyeline at 0.34 of
        # the crop, not the face box top offset by 0.18 face-heights.
        assert x == pytest.approx(807, abs=12)
        assert y == pytest.approx(127, abs=12)


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
        # eyeline 542 - 0.34*486 = 376.8 (was 482 under the old anchor).
        assert crop == pytest.approx((1307.0, 376.76, 486.0, 486.0), abs=0.5)


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
        # eyeline 142 - 0.34*486 = -23, clamped to the frame top.
        assert crop == pytest.approx((1307.0, 0.0, 486.0, 486.0), abs=0.5)


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
    """A split panel must frame its person the way the static engine would,
    and then be stacked. The tracker and the static path are now the same
    function, so parity is exact rather than approximate — the two used to
    drift apart every time either side's margins were retuned.
    """

    @pytest.mark.parametrize("box", [
        (800, 300, 160, 220),      # mid-shot face
        (400, 200, 300, 400),      # close subject
        (1500, 600, 90, 120),      # small, off to one side
    ])
    def test_tracked_panel_matches_the_static_panel_exactly(self, box):
        t = PanelTracker(1920, 1080, PANEL_ASPECT)
        t.crop = None
        tracked = t.step(box)
        static = crop_rect_containing(box, 1920, 1080, PANEL_ASPECT,
                                      layout=fc.PANEL)
        assert tracked == pytest.approx(static, abs=0.5)

    @pytest.mark.parametrize("box", [
        (800, 300, 160, 220), (400, 200, 300, 400), (1500, 600, 90, 120),
    ])
    def test_a_tracked_panel_satisfies_the_contract(self, box):
        t = PanelTracker(1920, 1080, PANEL_ASPECT)
        t.crop = None
        crop = t.step(box)
        assert fc.check(crop, [box], fc.PANEL, 1920, 1080,
                        aspect=PANEL_ASPECT) == []

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


class TestAspectPreservingRecontain:
    """`_recontain` used to grow w and h independently. reframe_v3's panel
    renderer letterboxes anything more than 10% off its target aspect, so that
    drift IS the black bar between panels reported in clip 2."""

    def test_recontain_keeps_the_panel_aspect(self):
        rect = (800.0, 400.0, 400.0, 400.0 / PANEL_ASPECT)
        box = (700.0, 300.0, 500.0, 500.0)      # sticks out on every side
        x, y, w, h = _recontain(rect, box, 1920, 1080, PANEL_ASPECT)
        assert w / h == pytest.approx(PANEL_ASPECT, rel=0.01)
        assert x <= box[0] and y <= box[1]
        assert x + w >= box[0] + box[2] and y + h >= box[1] + box[3]

    def test_a_glide_never_drifts_off_aspect(self):
        t = PanelTracker(1920, 1080, PANEL_ASPECT).reset((0, 0, 500, 500 / PANEL_ASPECT))
        for _ in range(120):
            x, y, w, h = t.step((1400, 700, 120, 160))
            # The renderer's letterbox threshold is +/-10%; stay well inside it.
            assert 0.95 * PANEL_ASPECT <= w / h <= 1.05 * PANEL_ASPECT

    def test_the_glide_never_clips_the_head(self):
        t = PanelTracker(1920, 1080, PANEL_ASPECT).reset((0, 0, 500, 500 / PANEL_ASPECT))
        box = (1400, 700, 120, 160)
        head = fc.head_box(box, 1920, 1080)
        for _ in range(120):
            x, y, w, h = t.step(box)
            assert x <= head[0] + 0.5 and y <= head[1] + 0.5
            assert x + w >= head[0] + head[2] - 0.5
            assert y + h >= head[1] + head[3] - 0.5
