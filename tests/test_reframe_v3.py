"""Tests for reframe_v3 — the composition stage.

Geometry and attention weighting are pure numpy, so everything except the
split-screen delegation runs without opencv, onnxruntime or a GPU.
"""

import numpy as np
import pytest

from reframe_v3 import (
    BYSTANDER_SUPPRESSION,
    DEFAULT_HEAD_Y,
    LAYOUT_SINGLE,
    LAYOUT_SPLIT,
    LAYOUT_TWO_SHOT,
    ROLE_BYSTANDER,
    ROLE_REACTOR,
    ROLE_SPEAKER,
    VERTICAL_9_16,
    WeightedFace,
    attention_center,
    build_attention_map,
    contains,
    crop_rect_containing,
    decide_layout,
    union_box,
)

FRAME_W, FRAME_H = 1920, 1080


# ─── attention map ───────────────────────────────────────────────────────────


def test_speaker_region_is_boosted_above_saliency_peak():
    """A confidently bound speaker must outrank any purely visual peak."""
    sal = np.zeros((100, 200), dtype=np.float32)
    sal[40:60, 150:180] = 1.0  # bright background blob, right side

    composite = build_attention_map(
        sal, [WeightedFace((10, 40, 30, 20), ROLE_SPEAKER)]
    )
    assert composite[45, 20] == pytest.approx(3.0)
    assert composite[45, 20] > composite[45, 160]


def test_bystander_is_suppressed_multiplicatively():
    """The departure from upstream: suppression must SCALE DOWN.

    pyautoflip's get_composite_mask only ever raises values
    (`np.maximum(region, FACE_WEIGHT)`), so a bystander sitting on a bright
    background keeps their full saliency and still pulls the centre of mass.
    Scaling can actually take them out of contention.
    """
    sal = np.ones((100, 200), dtype=np.float32)

    composite = build_attention_map(
        sal, [WeightedFace((150, 40, 30, 20), ROLE_BYSTANDER)]
    )
    assert composite[45, 160] == pytest.approx(BYSTANDER_SUPPRESSION)
    assert composite[45, 160] < 1.0  # would be exactly 1.0 under maximum-only
    assert composite[45, 20] == pytest.approx(1.0)  # untouched elsewhere


def test_reactor_outranks_background_but_loses_to_speaker():
    sal = np.zeros((100, 200), dtype=np.float32)
    composite = build_attention_map(sal, [
        WeightedFace((10, 40, 30, 20), ROLE_SPEAKER),
        WeightedFace((100, 40, 30, 20), ROLE_REACTOR),
    ])
    speaker_v = composite[45, 20]
    reactor_v = composite[45, 110]
    assert reactor_v > 0.0
    assert speaker_v > reactor_v


def test_boost_wins_over_overlapping_suppression():
    """Role precedence is by weight, never by list order — a speaker box
    overlapping a bystander box must not be dimmed."""
    sal = np.ones((100, 200), dtype=np.float32)
    faces = [
        WeightedFace((20, 40, 40, 20), ROLE_BYSTANDER),
        WeightedFace((30, 40, 40, 20), ROLE_SPEAKER),
    ]
    composite = build_attention_map(sal, faces)
    assert composite[45, 45] == pytest.approx(3.0)  # overlap region

    # Order reversed: same result.
    assert build_attention_map(sal, list(reversed(faces)))[45, 45] == pytest.approx(3.0)


def test_out_of_bounds_boxes_are_clipped_not_crashed():
    sal = np.zeros((100, 200), dtype=np.float32)
    composite = build_attention_map(sal, [
        WeightedFace((-50, -50, 80, 80), ROLE_SPEAKER),
        WeightedFace((190, 90, 100, 100), ROLE_REACTOR),
        WeightedFace((500, 500, 10, 10), ROLE_SPEAKER),  # fully outside
    ])
    assert composite.shape == (100, 200)
    assert composite[10, 10] == pytest.approx(3.0)


def test_attention_center_follows_the_speaker_not_the_bright_blob():
    """The whole point of the module: attention, not brightness."""
    sal = np.zeros((100, 200), dtype=np.float32)
    sal[40:60, 150:180] = 1.0  # bright blob on the RIGHT

    composite = build_attention_map(
        sal, [WeightedFace((10, 40, 30, 20), ROLE_SPEAKER)]  # speaker on the LEFT
    )
    cx, _ = attention_center(composite)
    assert cx < 0.3


def test_attention_center_finds_a_reaction_with_no_speaker():
    """The case the owner named: nobody is speaking, something happened over
    there. Saliency alone must be enough to aim the camera."""
    sal = np.zeros((100, 200), dtype=np.float32)
    sal[40:60, 150:180] = 1.0

    cx, cy = attention_center(build_attention_map(sal, []))
    assert cx == pytest.approx(165 / 200, abs=0.05)
    assert cy == pytest.approx(50 / 100, abs=0.05)


def test_attention_center_ignores_a_broad_dim_field():
    """A large faint region must not outvote a small bright one on area
    alone — that is how a busy background drags the camera to frame centre."""
    sal = np.full((100, 200), 0.15, dtype=np.float32)
    sal[10:20, 10:25] = 1.0  # small bright patch, top-left

    cx, cy = attention_center(sal)
    assert cx < 0.25
    assert cy < 0.25


def test_attention_center_degenerate_maps_fall_back_to_centre():
    assert attention_center(np.zeros((10, 10), dtype=np.float32)) == (0.5, 0.5)
    assert attention_center(np.zeros((0, 0), dtype=np.float32)) == (0.5, 0.5)


# ─── crop geometry ───────────────────────────────────────────────────────────


def test_crop_has_the_target_aspect():
    crop = crop_rect_containing((100, 100, 100, 100), FRAME_W, FRAME_H)
    _, _, w, h = crop
    assert w / h == pytest.approx(VERTICAL_9_16, rel=0.01)


def test_crop_contains_its_subject():
    """The property the old engine failed — 'half a person' was a containment
    failure, so it is asserted rather than scored."""
    subject = (100, 100, 100, 100)
    assert contains(crop_rect_containing(subject, FRAME_W, FRAME_H), subject)


@pytest.mark.parametrize("subject", [
    (0, 0, 80, 80),                    # hard against top-left
    (1840, 1000, 80, 80),              # hard against bottom-right
    (900, 500, 60, 60),                # centre
    (900, 100, 300, 800),              # tall subject, forces full-height crop
    (50, 500, 40, 40),                 # small, near left edge
])
def test_crop_contains_subject_anywhere_in_frame(subject):
    crop = crop_rect_containing(subject, FRAME_W, FRAME_H)
    assert contains(crop, subject), f"{subject} escaped {crop}"


def test_crop_stays_inside_the_frame():
    for subject in [(0, 0, 50, 50), (1870, 1030, 50, 50), (960, 540, 100, 100)]:
        x, y, w, h = crop_rect_containing(subject, FRAME_W, FRAME_H)
        assert x >= 0 and y >= 0
        assert x + w <= FRAME_W + 0.5
        assert y + h <= FRAME_H + 0.5


def test_head_is_placed_high_not_centred():
    """Vertical composition — the thing upstream's crop window cannot express
    at all, since it always returns y=0 and full frame height."""
    subject = (900, 400, 100, 100)
    x, y, w, h = crop_rect_containing(subject, FRAME_W, FRAME_H)

    subject_cy = 450
    placement = (subject_cy - y) / h
    assert placement == pytest.approx(DEFAULT_HEAD_Y, abs=0.02)
    assert placement < 0.5  # strictly above centre


def test_full_height_crop_is_a_consequence_not_an_assumption():
    """A subject tall enough to need the whole frame height yields a
    full-height crop with no vertical freedom left — the degenerate case
    upstream hard-codes, reached here only when the geometry demands it."""
    _, y, _, h = crop_rect_containing((900, 100, 300, 800), FRAME_W, FRAME_H)
    assert h == pytest.approx(FRAME_H)
    assert y == pytest.approx(0.0)


def test_tighter_subject_gives_tighter_crop():
    """A punch-in must actually punch in — a small subject should not produce
    the same full-frame crop as a large one."""
    small = crop_rect_containing((900, 500, 60, 60), FRAME_W, FRAME_H)
    large = crop_rect_containing((900, 200, 300, 700), FRAME_W, FRAME_H)
    assert small[3] < large[3]


def test_crop_never_exceeds_a_small_frame():
    """Frame smaller than the ideal crop: aspect is preserved and the rect
    still fits."""
    x, y, w, h = crop_rect_containing((10, 10, 200, 200), 320, 240)
    assert w <= 320 and h <= 240
    assert x >= 0 and y >= 0
    assert w / h == pytest.approx(VERTICAL_9_16, rel=0.05)


# ─── helpers ─────────────────────────────────────────────────────────────────


def test_contains_rejects_a_subject_hanging_outside():
    crop = (100, 100, 200, 400)
    assert contains(crop, (150, 150, 50, 50))
    assert not contains(crop, (250, 150, 100, 50))   # off the right edge
    assert not contains(crop, (150, 450, 50, 100))   # off the bottom


def test_union_box_covers_both():
    assert union_box((10, 10, 20, 20), (100, 50, 30, 30)) == (10, 10, 120, 70)


def test_union_box_tolerates_missing_inputs():
    assert union_box(None, (10, 10, 20, 20)) == (10, 10, 20, 20)
    assert union_box(None, None) is None


# ─── layout decision ─────────────────────────────────────────────────────────


def test_one_subject_is_always_a_single():
    assert decide_layout([(900, 500, 100, 100)], [1.0], FRAME_W, FRAME_H) == LAYOUT_SINGLE


def test_two_close_live_subjects_are_a_two_shot():
    subjects = [(800, 400, 120, 120), (1000, 400, 120, 120)]
    assert decide_layout(subjects, [0.6, 0.4], FRAME_W, FRAME_H) == LAYOUT_TWO_SHOT


def test_two_far_live_subjects_split():
    subjects = [(200, 400, 120, 120), (1600, 400, 120, 120)]
    assert decide_layout(subjects, [0.55, 0.45], FRAME_W, FRAME_H) == LAYOUT_SPLIT


def test_far_apart_but_one_holds_the_floor_stays_single():
    """The editorial half of the decision. Geometrically this is a split; but
    dedicating half the frame to a silent listener wastes the format, so a
    single on whoever is talking is the better cut."""
    subjects = [(200, 400, 120, 120), (1600, 400, 120, 120)]
    assert decide_layout(subjects, [0.95, 0.02], FRAME_W, FRAME_H) == LAYOUT_SINGLE


def test_split_needs_a_genuine_exchange():
    """Just above and just below the second-speaker threshold."""
    subjects = [(200, 400, 120, 120), (1600, 400, 120, 120)]
    assert decide_layout(subjects, [0.7, 0.30], FRAME_W, FRAME_H) == LAYOUT_SPLIT
    assert decide_layout(subjects, [0.8, 0.20], FRAME_W, FRAME_H) == LAYOUT_SINGLE


def test_missing_subject_falls_back_to_single():
    assert decide_layout([(900, 500, 100, 100), None], [0.5, 0.5],
                         FRAME_W, FRAME_H) == LAYOUT_SINGLE


# ─── split-screen delegation ─────────────────────────────────────────────────


def test_split_centers_delegates_to_the_vendored_rule():
    pytest.importorskip("cv2", reason="vendored split-screen needs opencv")
    from reframe_v3 import split_centers

    subjects = [(200, 400, 120, 120), (1600, 400, 120, 120)]
    centers = split_centers(subjects, FRAME_W, FRAME_H)

    assert centers is not None and len(centers) == 2
    assert centers[0][0] < centers[1][0]           # left panel first
    assert all(0.0 <= c <= 1.0 for pt in centers for c in pt)


def test_split_centers_falls_back_when_the_vendored_rule_declines():
    """decide_layout can commit to a split for size reasons even when the two
    faces would technically fit one window; the fallback must still produce
    two panel centres rather than None."""
    pytest.importorskip("cv2", reason="vendored split-screen needs opencv")
    from reframe_v3 import split_centers

    subjects = [(880, 400, 60, 60), (980, 400, 60, 60)]  # close together
    centers = split_centers(subjects, FRAME_W, FRAME_H)

    assert centers is not None and len(centers) == 2
    assert centers[0][0] < centers[1][0]
