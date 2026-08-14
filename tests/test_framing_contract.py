"""The framing contract's assertions, as tests.

These are the tests that would have caught every failure in
PLAN_FRAMING_CONTRACT.md §2 before it rendered. The important ones are the
parametrised sweeps: the old engine's head-cut was not a bad tuning value, it
was a formula that failed for a whole REGION of inputs (any face box in a wide
panel aspect), so a single hand-picked example would have passed.
"""

import itertools

import pytest

import framing_contract as fc


FRAME_W, FRAME_H = 1920, 1080
V916 = 9.0 / 16.0
PANEL_ASPECT = V916 * 2.0        # 9:8 — one half of a 9:16 output

# Face heights from "distant person in a group" to "leaning into the lens".
FACE_HEIGHTS = [0.04, 0.06, 0.08, 0.10, 0.14, 0.20, 0.28, 0.35]


def face_at(cx_frac, cy_frac, h_frac, frame_w=FRAME_W, frame_h=FRAME_H):
    """A plausible detector face box: taller than it is wide, as SCRFD's are."""
    h = h_frac * frame_h
    w = 0.8 * h
    return (cx_frac * frame_w - w / 2.0, cy_frac * frame_h - h / 2.0, w, h)


# ---------------------------------------------------------------------------
# I1 — head, not face
# ---------------------------------------------------------------------------

def test_head_box_reaches_above_the_face_box():
    face = face_at(0.5, 0.4, 0.14)
    head = fc.head_box(face, FRAME_W, FRAME_H)
    assert head[1] < face[1], "the head box must include hair the detector misses"
    assert head[1] == pytest.approx(face[1] - fc.HEAD_DY * face[3])
    assert head[0] < face[0] and head[2] > face[2]


def test_head_box_is_clipped_to_the_frame():
    face = face_at(0.03, 0.03, 0.06)
    head = fc.head_box(face, FRAME_W, FRAME_H)
    assert head[0] >= 0 and head[1] >= 0
    assert head[0] + head[2] <= FRAME_W and head[1] + head[3] <= FRAME_H


# ---------------------------------------------------------------------------
# The regression that started all this: the old formula, reproduced
# ---------------------------------------------------------------------------

def _legacy_crop(face, aspect, frame_w=FRAME_W, frame_h=FRAME_H):
    """reframe_v3's pre-contract crop_rect_containing, verbatim in spirit."""
    sx, sy, sw, sh = face
    need_w = sw * (1.0 + 2.0 * 0.55)      # DEFAULT_SIDE_MARGIN
    need_h = sh * (1.0 + 2.0 * 0.35)      # DEFAULT_VERT_MARGIN
    crop_h = max(need_h, need_w / aspect)
    crop_w = crop_h * aspect
    cy = sy + sh / 2.0
    return (sx + sw / 2.0 - crop_w / 2.0, cy - crop_h * 0.36, crop_w, crop_h)


def test_the_legacy_panel_formula_really_did_cut_heads():
    """Documents the bug this module exists to remove (plan §3.1)."""
    face = face_at(0.5, 0.45, 0.14)
    legacy = _legacy_crop(face, PANEL_ASPECT)
    assert fc.headroom_frac(legacy, face, FRAME_W, FRAME_H) < 0, (
        "the legacy panel crop is supposed to start BELOW the hairline")


def test_the_contract_fixes_exactly_that_case():
    face = face_at(0.5, 0.45, 0.14)
    crop = fc.frame_panel(face, FRAME_W, FRAME_H, PANEL_ASPECT)
    assert fc.headroom_frac(crop, face, FRAME_W, FRAME_H) >= fc.HEADROOM_MIN


# ---------------------------------------------------------------------------
# I2 / I3 / I4 across the whole input region
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h_frac,cx,cy", list(itertools.product(
    FACE_HEIGHTS, [0.25, 0.5, 0.75], [0.35, 0.5])))
def test_single_never_cuts_the_head(h_frac, cx, cy):
    face = face_at(cx, cy, h_frac)
    crop = fc.frame_single(face, FRAME_W, FRAME_H, V916)
    room = fc.headroom_frac(crop, face, FRAME_W, FRAME_H)
    assert room >= fc.HEADROOM_MIN or crop[1] <= 0.5, (
        f"headroom {room:.3f} at face height {h_frac}")


@pytest.mark.parametrize("h_frac,cx,cy", list(itertools.product(
    FACE_HEIGHTS, [0.25, 0.5, 0.75], [0.35, 0.5])))
def test_panel_never_cuts_the_head(h_frac, cx, cy):
    face = face_at(cx, cy, h_frac)
    crop = fc.frame_panel(face, FRAME_W, FRAME_H, PANEL_ASPECT)
    room = fc.headroom_frac(crop, face, FRAME_W, FRAME_H)
    assert room >= fc.HEADROOM_MIN or crop[1] <= 0.5, (
        f"headroom {room:.3f} at face height {h_frac}")


@pytest.mark.parametrize("h_frac", FACE_HEIGHTS)
def test_eyeline_lands_on_target_when_the_crop_is_not_clamped(h_frac):
    face = face_at(0.5, 0.4, h_frac)
    crop = fc.frame_single(face, FRAME_W, FRAME_H, V916)
    if crop[1] <= 0.5 or crop[1] + crop[3] >= FRAME_H - 0.5:
        pytest.skip("crop clamped against a frame edge — eyeline is not free")
    assert fc.eye_frac(crop, face) == pytest.approx(fc.SINGLE_EYE_Y, abs=1e-6)


def test_the_two_layouts_agree_on_headroom():
    """The self-consistency check from the plan: single (0.155/0.22) and panel
    (0.30/0.34) are independent ratio pairs that must land on the same ~10%."""
    # A face height of 0.145 with the head high in frame is the one window
    # where NEITHER crop is floored by the blur limit nor clamped by a frame
    # edge, so both ratios are free to land exactly where they were derived.
    face = face_at(0.5, 0.25, 0.145)
    single = fc.headroom_frac(
        fc.frame_single(face, FRAME_W, FRAME_H, V916), face, FRAME_W, FRAME_H)
    panel = fc.headroom_frac(
        fc.frame_panel(face, FRAME_W, FRAME_H, PANEL_ASPECT), face,
        FRAME_W, FRAME_H)
    assert single == pytest.approx(0.101, abs=0.01)
    assert panel == pytest.approx(0.109, abs=0.01)
    assert abs(single - panel) < 0.02


def test_panel_face_reads_the_same_size_as_a_single():
    """0.30 of a panel == 0.15 of the output == what a single delivers. This is
    the argument for the split screen existing at all.

    Needs a face big enough that neither crop hits the blur floor — below
    ~0.135 of source height the panel is floored and reads smaller, which is
    the resolution budget (plan §5.7), not a contract violation.
    """
    face = face_at(0.5, 0.30, 0.145)
    single = fc.frame_single(face, FRAME_W, FRAME_H, V916)
    panel = fc.frame_panel(face, FRAME_W, FRAME_H, PANEL_ASPECT)
    assert fc.face_frac_of_output(single, face, fc.SINGLE) == pytest.approx(
        fc.face_frac_of_output(panel, face, fc.PANEL), abs=0.01)


# ---------------------------------------------------------------------------
# Aspect, framing, clamping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h_frac", FACE_HEIGHTS)
def test_crop_keeps_its_aspect_and_stays_in_frame(h_frac):
    for cx in (0.05, 0.5, 0.95):
        face = face_at(cx, 0.5, h_frac)
        x, y, w, h = fc.frame_single(face, FRAME_W, FRAME_H, V916)
        assert w / h == pytest.approx(V916, abs=1e-6)
        assert x >= -0.5 and y >= -0.5
        assert x + w <= FRAME_W + 0.5 and y + h <= FRAME_H + 0.5


def test_blur_floor_only_ever_widens_the_crop():
    tiny = face_at(0.5, 0.5, 0.04)
    crop = fc.frame_single(tiny, FRAME_W, FRAME_H, V916)
    assert crop[3] >= fc.min_crop_frac() * FRAME_H - 0.5
    # ...and the face therefore reads SMALLER than target. That is the source
    # telling us the camera was far away (plan §5.7), not a bug.
    assert fc.face_frac_of_output(crop, tiny, fc.SINGLE) < fc.SINGLE_FACE_FRAC


def test_achievable_face_frac_reports_the_resolution_budget():
    assert fc.achievable_face_frac(0.04 * FRAME_H, FRAME_W, FRAME_H, V916) < 0.10
    assert fc.achievable_face_frac(
        0.14 * FRAME_H, FRAME_W, FRAME_H, V916) == pytest.approx(
            fc.SINGLE_FACE_FRAC, abs=0.01)


# ---------------------------------------------------------------------------
# I6 — look room and the gap
# ---------------------------------------------------------------------------

def test_look_room_biases_toward_the_other_person():
    speaker = face_at(0.5, 0.45, 0.12)
    listener = face_at(0.68, 0.45, 0.12)
    centred = fc.frame_single(speaker, FRAME_W, FRAME_H, V916)
    d = fc.look_room_dir(speaker, [listener], centred[2])
    assert d > 0, "the listener is to the right, so the crop should shift right"
    shifted = fc.frame_single(speaker, FRAME_W, FRAME_H, V916, look_dir=d)
    assert shifted[0] > centred[0]


def test_look_room_never_takes_the_centre_off_the_head():
    speaker = face_at(0.5, 0.45, 0.10)
    for d in (-1.0, -0.5, 0.5, 1.0):
        crop = fc.frame_single(speaker, FRAME_W, FRAME_H, V916, look_dir=d)
        assert not fc.check(crop, [speaker], fc.SINGLE, FRAME_W, FRAME_H,
                            aspect=V916)


def test_a_crop_centred_on_the_gap_is_rejected():
    left = face_at(0.20, 0.45, 0.10)
    right = face_at(0.80, 0.45, 0.10)
    gap_crop = (FRAME_W / 2 - 300, 200, 600, 600 / V916)
    problems = fc.check(gap_crop, [left, right], fc.TWO_SHOT, FRAME_W, FRAME_H)
    assert any("I6" in p for p in problems)


def test_a_crop_with_nobody_in_it_is_rejected():
    far_away = face_at(0.9, 0.9, 0.05)
    empty = (0.0, 0.0, 600.0, 600.0 / V916)
    problems = fc.check(empty, [far_away], fc.SINGLE, FRAME_W, FRAME_H)
    assert any("I5" in p for p in problems)


def test_no_faces_at_all_is_rejected():
    assert fc.check((0.0, 0.0, 600.0, 1066.0), [], fc.SINGLE,
                    FRAME_W, FRAME_H) == ["I5 crop has no subject at all"]


# ---------------------------------------------------------------------------
# check() accepts what frame_subject produces — the round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h_frac,cx", list(itertools.product(
    [0.08, 0.10, 0.14, 0.20], [0.3, 0.5, 0.7])))
def test_frame_subject_output_always_passes_check(h_frac, cx):
    face = face_at(cx, 0.45, h_frac)
    single = fc.frame_single(face, FRAME_W, FRAME_H, V916)
    assert fc.check(single, [face], fc.SINGLE, FRAME_W, FRAME_H,
                    aspect=V916) == []


@pytest.mark.parametrize("h_frac,cx", list(itertools.product(
    [0.10, 0.14, 0.20], [0.3, 0.5, 0.7])))
def test_panel_passes_check_when_the_source_can_support_it(h_frac, cx):
    face = face_at(cx, 0.45, h_frac)
    panel = fc.frame_panel(face, FRAME_W, FRAME_H, PANEL_ASPECT)
    assert fc.check(panel, [face], fc.PANEL, FRAME_W, FRAME_H,
                    aspect=PANEL_ASPECT) == []


def test_a_face_too_small_to_panel_is_rejected_not_faked():
    """The resolution budget, enforced (plan §5.7).

    At 0.08 of source height the blur floor stops the panel crop from getting
    tight enough, so the face lands at 0.089 of output — under I4's 0.10. The
    contract is supposed to SAY SO rather than ship a mushy upscale: the caller
    then declines the split and shows the wide. A panel is not a way to
    manufacture subject size that the source never had.
    """
    face = face_at(0.5, 0.45, 0.08)
    panel = fc.frame_panel(face, FRAME_W, FRAME_H, PANEL_ASPECT)
    problems = fc.check(panel, [face], fc.PANEL, FRAME_W, FRAME_H,
                        aspect=PANEL_ASPECT)
    assert any("I4" in p for p in problems)
    # The same face IS fine as a single, because a single's crop gets twice the
    # output height to work with.
    single = fc.frame_single(face, FRAME_W, FRAME_H, V916)
    assert fc.check(single, [face], fc.SINGLE, FRAME_W, FRAME_H,
                    aspect=V916) == []


# ---------------------------------------------------------------------------
# I7 — a split is symmetric or it is not a split
# ---------------------------------------------------------------------------

def test_two_comparable_panels_are_admitted():
    a = face_at(0.20, 0.45, 0.12)
    b = face_at(0.80, 0.45, 0.12)
    panels = (fc.frame_panel(a, FRAME_W, FRAME_H, PANEL_ASPECT),
              fc.frame_panel(b, FRAME_W, FRAME_H, PANEL_ASPECT))
    assert fc.check_panels(panels, [a, b], FRAME_W, FRAME_H,
                           panel_aspect=PANEL_ASPECT) == []


def test_mismatched_panel_sizes_are_rejected():
    close = face_at(0.20, 0.45, 0.26)
    distant = face_at(0.80, 0.45, 0.05)
    panels = (fc.frame_panel(close, FRAME_W, FRAME_H, PANEL_ASPECT),
              fc.frame_panel(distant, FRAME_W, FRAME_H, PANEL_ASPECT))
    problems = fc.check_panels(panels, [close, distant], FRAME_W, FRAME_H,
                               panel_aspect=PANEL_ASPECT)
    assert any("I7" in p for p in problems)


def test_a_crowd_panel_is_rejected():
    a = face_at(0.20, 0.45, 0.12)
    b = face_at(0.78, 0.45, 0.12)
    crowd = [face_at(0.72, 0.45, 0.12), face_at(0.84, 0.45, 0.12),
             face_at(0.90, 0.45, 0.12)]
    panels = (fc.frame_panel(a, FRAME_W, FRAME_H, PANEL_ASPECT),
              fc.frame_panel(b, FRAME_W, FRAME_H, PANEL_ASPECT))
    problems = fc.check_panels(panels, [a, b], FRAME_W, FRAME_H,
                               panel_aspect=PANEL_ASPECT,
                               context=[(), crowd])
    assert any("crowd" in p for p in problems)


def test_a_head_cutting_panel_is_rejected():
    face = face_at(0.5, 0.45, 0.14)
    bad = _legacy_crop(face, PANEL_ASPECT)
    problems = fc.check(bad, [face], fc.PANEL, FRAME_W, FRAME_H)
    assert any("I2" in p for p in problems)


# ---------------------------------------------------------------------------
# The far-apart test — when the split is the RIGHT answer
# ---------------------------------------------------------------------------

def test_side_by_side_people_should_be_a_two_shot():
    a = face_at(0.44, 0.45, 0.14)
    b = face_at(0.58, 0.45, 0.14)
    assert fc.two_shot_holds_both([a, b], FRAME_W, FRAME_H, V916)


def test_far_apart_people_should_be_a_split():
    """Clip 4's failure: a two-shot 'holds' both by shrinking them to nothing.
    The size floor is what makes this return False and hand the moment to the
    split, which keeps both at 0.15."""
    a = face_at(0.12, 0.45, 0.10)
    b = face_at(0.88, 0.45, 0.10)
    assert not fc.two_shot_holds_both([a, b], FRAME_W, FRAME_H, V916)
    panels = (fc.frame_panel(a, FRAME_W, FRAME_H, PANEL_ASPECT),
              fc.frame_panel(b, FRAME_W, FRAME_H, PANEL_ASPECT))
    assert fc.check_panels(panels, [a, b], FRAME_W, FRAME_H,
                           panel_aspect=PANEL_ASPECT) == []
    for p, f in zip(panels, (a, b)):
        assert fc.face_frac_of_output(p, f, fc.PANEL) >= 0.10


def test_one_subject_never_needs_a_split():
    assert fc.two_shot_holds_both([face_at(0.5, 0.45, 0.12)],
                                  FRAME_W, FRAME_H, V916)
