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
    caption_output_args,
    contains,
    crop_rect_containing,
    decide_layout,
    union_box,
)
from reframe_v3 import ComposedShot, _render_regular

FRAME_W, FRAME_H = 1920, 1080


# ─── Phase 6: worker-GPU threading (gpu_affinity) ───────────────────────────


@pytest.fixture(autouse=True)
def _clean_ffmpeg_state(monkeypatch):
    import ffmpeg_utils
    import os
    monkeypatch.delenv("FFMPEG_ENCODER", raising=False)
    monkeypatch.delenv("GPU_RENDER", raising=False)
    ffmpeg_utils.reset_encoder_cache()
    ffmpeg_utils.reset_gpu_render_cache()
    yield
    ffmpeg_utils.reset_encoder_cache()
    ffmpeg_utils.reset_gpu_render_cache()


def _fake_ffmpeg_run(monkeypatch):
    """Capture the ffmpeg argv; never actually run ffmpeg."""
    import subprocess
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_render_regular_threads_worker_device_into_decode_and_encode(monkeypatch):
    import ffmpeg_utils
    import gpu_affinity
    monkeypatch.setenv("FFMPEG_ENCODER", "nvenc")
    monkeypatch.setattr(gpu_affinity, "current_device", lambda: "cuda:1")
    monkeypatch.setattr(ffmpeg_utils, "_probe_gpu_render", lambda: True)
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", lambda: True)
    captured = _fake_ffmpeg_run(monkeypatch)

    shots = [ComposedShot(0.0, 2.0, LAYOUT_SINGLE, (100.0, 0.0, 405.0, 720.0), [])]
    _render_regular("in.mp4", "out.mp4", shots, 1920, 1080, 405, 720)
    cmd = captured["cmd"]
    assert cmd.index("-hwaccel") < cmd.index("-i")
    assert "-hwaccel_device" in cmd and cmd[cmd.index("-hwaccel_device") + 1] == "1"
    assert "-gpu" in cmd and cmd[cmd.index("-gpu") + 1] == "1"


def test_render_regular_without_assignment_keeps_cpu_defaults(monkeypatch):
    import ffmpeg_utils
    import gpu_affinity
    monkeypatch.setenv("FFMPEG_ENCODER", "nvenc")
    monkeypatch.setattr(gpu_affinity, "current_device", lambda: None)
    monkeypatch.setattr(ffmpeg_utils, "_probe_gpu_render", lambda: True)
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", lambda: True)
    captured = _fake_ffmpeg_run(monkeypatch)

    shots = [ComposedShot(0.0, 2.0, LAYOUT_SINGLE, (100.0, 0.0, 405.0, 720.0), [])]
    _render_regular("in.mp4", "out.mp4", shots, 1920, 1080, 405, 720)
    cmd = captured["cmd"]
    assert "-hwaccel_device" not in cmd
    assert "-gpu" not in cmd


def test_caption_output_args_threads_worker_device(monkeypatch):
    import ffmpeg_utils
    monkeypatch.setenv("FFMPEG_ENCODER", "nvenc")
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", lambda: True)
    args = caption_output_args("ass='/tmp/subs.ass'", "/tmp/out_sub.mp4",
                               device="cuda:0")
    assert "-gpu" in args and args[args.index("-gpu") + 1] == "0"


def test_caption_output_args_without_device_omits_gpu_flag(monkeypatch):
    import ffmpeg_utils
    monkeypatch.setenv("FFMPEG_ENCODER", "nvenc")
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", lambda: True)
    args = caption_output_args("ass='/tmp/subs.ass'", "/tmp/out_sub.mp4")
    assert "-gpu" not in args


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


# ─── validation ──────────────────────────────────────────────────────────────

from reframe_v3 import ComposedShot, CompositionError, validate_composition  # noqa: E402


def _valid_shot(start=0.0, end=3.0):
    subject = (900, 500, 100, 100)
    return ComposedShot(
        start=start, end=end, layout=LAYOUT_SINGLE,
        crop=crop_rect_containing(subject, FRAME_W, FRAME_H),
        subjects=[subject],
    )


def test_a_well_formed_plan_validates():
    validate_composition([_valid_shot(), _valid_shot(3.0, 6.0)], FRAME_W, FRAME_H)


def test_uncontained_subject_raises():
    """The 'half a person' failure — the whole reason this stage exists."""
    shot = _valid_shot()
    shot.subjects = [(50, 500, 100, 100)]  # far left, crop is around x=900

    with pytest.raises(CompositionError, match="not contained"):
        validate_composition([shot], FRAME_W, FRAME_H)


def test_all_violations_are_reported_at_once():
    """A plan with several bad shots should take one run to diagnose."""
    bad_a = _valid_shot(0.0, 3.0)
    bad_a.subjects = [(50, 500, 100, 100)]
    bad_b = _valid_shot(3.0, 3.2)          # too short

    with pytest.raises(CompositionError) as exc:
        validate_composition([bad_a, bad_b], FRAME_W, FRAME_H)

    message = str(exc.value)
    assert "2 composition violation" in message
    assert "not contained" in message
    assert "min" in message


def test_short_shot_raises():
    with pytest.raises(CompositionError, match="duration"):
        validate_composition([_valid_shot(0.0, 0.4)], FRAME_W, FRAME_H)


def test_crop_escaping_the_frame_raises():
    shot = _valid_shot()
    shot.crop = (1800, 0, 607.5, 1080)  # runs off the right edge
    shot.subjects = []

    with pytest.raises(CompositionError, match="escapes frame"):
        validate_composition([shot], FRAME_W, FRAME_H)


def test_wrong_aspect_raises():
    shot = _valid_shot()
    shot.crop = (100, 100, 400, 400)  # square, not 9:16
    shot.subjects = []

    with pytest.raises(CompositionError, match="aspect"):
        validate_composition([shot], FRAME_W, FRAME_H)


def test_non_split_layout_without_a_crop_raises():
    shot = _valid_shot()
    shot.crop = None

    with pytest.raises(CompositionError, match="no crop rect"):
        validate_composition([shot], FRAME_W, FRAME_H)


def test_split_layout_is_exempt_from_containment():
    """Split panels hold their subjects separately, so a single containing
    rect is not a meaningful constraint — but duration still applies."""
    shot = ComposedShot(
        start=0.0, end=3.0, layout=LAYOUT_SPLIT, crop=None,
        subjects=[(200, 400, 120, 120), (1600, 400, 120, 120)],
    )
    validate_composition([shot], FRAME_W, FRAME_H)

    shot.end = 0.3
    with pytest.raises(CompositionError, match="duration"):
        validate_composition([shot], FRAME_W, FRAME_H)


def test_degenerate_crop_raises():
    shot = _valid_shot()
    shot.crop = (100, 100, 0, 0)
    shot.subjects = []

    with pytest.raises(CompositionError, match="degenerate"):
        validate_composition([shot], FRAME_W, FRAME_H)


# ─── Phase 5: render wiring ──────────────────────────────────────────────────

from reframe_v3 import (  # noqa: E402
    ComposedShot,
    _aspect_tuple,
    _compose_shot,
    _directive_dicts,
    _integer_crop,
    _regular_filtergraph,
    _speaker_shares,
    _track_nearest_x,
    attention_shifted_crop,
)
from shot_planner import SHOT_REACTION, SHOT_SINGLE, SHOT_WIDE, Shot  # noqa: E402


def _spine_two_tracks():
    """Two people present for the whole clip: speaker-ish at x=200, other at x=1400."""
    return {
        1: {"frames": [0.0, 1.0, 2.0], "boxes": [(200.0, 400.0, 120.0, 120.0)] * 3},
        2: {"frames": [0.0, 1.0, 2.0], "boxes": [(1400.0, 400.0, 120.0, 120.0)] * 3},
    }


def _saliency_at(box, frame_h=1080, frame_w=1920):
    sal = np.zeros((frame_h, frame_w), dtype=np.float32)
    x, y, w, h = (int(v) for v in box)
    sal[y:y + h, x:x + w] = 1.0
    return sal


def test_attention_shifted_crop_pulls_right_within_slack():
    """A reaction beside the speaker shifts the crop right, never losing containment."""
    frame_w, frame_h = 1920, 1080
    subject = (200.0, 400.0, 120.0, 120.0)
    base = crop_rect_containing(subject, frame_w, frame_h, VERTICAL_9_16)
    shifted = attention_shifted_crop(base, subject, 0.7604, frame_w, frame_h)

    assert shifted[0] > base[0]  # pulled toward the reaction
    assert contains(shifted, subject)
    assert shifted[2] == base[2] and shifted[3] == base[3]  # aspect untouched


def test_attention_shifted_crop_pulls_left_within_slack():
    frame_w, frame_h = 1920, 1080
    subject = (1600.0, 400.0, 120.0, 120.0)
    base = crop_rect_containing(subject, frame_w, frame_h, VERTICAL_9_16)
    shifted = attention_shifted_crop(base, subject, 0.1, frame_w, frame_h)

    assert shifted[0] < base[0]
    assert contains(shifted, subject)


def test_attention_shifted_crop_neutral_when_attention_on_subject():
    """Attention centred on the subject leaves the base composition alone."""
    frame_w, frame_h = 1920, 1080
    subject = (200.0, 400.0, 120.0, 120.0)
    base = crop_rect_containing(subject, frame_w, frame_h, VERTICAL_9_16)
    attention_x = (subject[0] + subject[2] / 2.0) / frame_w
    shifted = attention_shifted_crop(base, subject, attention_x, frame_w, frame_h)

    assert shifted[0] == pytest.approx(base[0])


def test_attention_shifted_crop_full_frame_has_no_slack():
    frame_w, frame_h = 1920, 1080
    full = (0.0, 0.0, float(frame_w), float(frame_h))
    base = crop_rect_containing(full, frame_w, frame_h, VERTICAL_9_16)
    shifted = attention_shifted_crop(base, full, 1.0, frame_w, frame_h)

    assert shifted == base


def test_compose_wide_shot_without_subjects_is_a_neutral_hold():
    """The planner emits WIDE shots on purpose (leading unbound seconds); v3
    must render them as a full-height centre crop, not fail the clip."""
    shot = Shot(0.0, 3.0, SHOT_WIDE, [], None)
    composed = _compose_shot(shot, {}, [None, None, None],
                             np.zeros((1080, 1920), dtype=np.float32),
                             1920, 1080, VERTICAL_9_16)

    assert composed.layout == LAYOUT_SINGLE
    assert composed.subjects == []
    assert composed.crop is not None
    x, y, w, h = composed.crop
    assert w / h == pytest.approx(VERTICAL_9_16)
    assert x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080
    validate_composition([composed], 1920, 1080, VERTICAL_9_16)


def test_compose_wide_shot_uses_provided_rect():
    shot = Shot(0.0, 3.0, SHOT_WIDE, [], (100.0, 200.0, 405.0, 720.0))
    composed = _compose_shot(shot, {}, [None, None, None],
                             np.zeros((1080, 1920), dtype=np.float32),
                             1920, 1080, VERTICAL_9_16)

    assert composed.crop == (100.0, 200.0, 405.0, 720.0)
    validate_composition([composed], 1920, 1080, VERTICAL_9_16)


def test_compose_shot_bystander_bright_spot_cannot_pull_the_crop():
    """A bright visual blob on a bystander is suppressed below the attention
    floor, so the crop stays on the speaker instead of being dragged away."""
    shot = Shot(0.0, 3.0, SHOT_SINGLE, [1], None)
    spine = _spine_two_tracks()
    saliency = _saliency_at((1400.0, 400.0, 120.0, 120.0))  # on the bystander
    composed = _compose_shot(shot, spine, [1, 1, 1], saliency, 1920, 1080, VERTICAL_9_16)

    speaker = spine[1]["boxes"][0]
    assert contains(composed.crop, speaker)
    # Attention stayed on the speaker, so the crop is the base composition.
    base = crop_rect_containing(speaker, 1920, 1080, VERTICAL_9_16)
    assert abs(composed.crop[0] - base[0]) <= 1.0


def test_compose_reaction_shot_pulls_toward_the_reaction():
    """Pop-the-balloon case: a bright visual off the reactor's face (the
    balloon) shifts the crop toward it while the reactor stays fully in shot."""
    shot = Shot(0.0, 3.0, SHOT_REACTION, [2], None)
    spine = _spine_two_tracks()
    saliency = _saliency_at((1600.0, 300.0, 100.0, 100.0))  # the balloon, off-face
    composed = _compose_shot(shot, spine, [1, 1, 1], saliency, 1920, 1080, VERTICAL_9_16)

    reactor = spine[2]["boxes"][0]
    assert contains(composed.crop, reactor)
    base = crop_rect_containing(reactor, 1920, 1080, VERTICAL_9_16)
    assert composed.crop[0] > base[0]  # frame moved right, toward the reaction


def test_compose_reaction_shot_no_reaction_stays_on_subject():
    shot = Shot(0.0, 3.0, SHOT_REACTION, [2], None)
    spine = _spine_two_tracks()
    saliency = _saliency_at(spine[2]["boxes"][0])  # attention on the reactor itself
    composed = _compose_shot(shot, spine, [1, 1, 1], saliency, 1920, 1080, VERTICAL_9_16)

    reactor = spine[2]["boxes"][0]
    base = crop_rect_containing(reactor, 1920, 1080, VERTICAL_9_16)
    assert contains(composed.crop, reactor)
    assert abs(composed.crop[0] - base[0]) <= 1.0


def test_speaker_shares_fractions():
    active = [1, 1, 2, 2, 1]
    shot = Shot(0.0, 5.0, SHOT_SINGLE, [1, 2], None)
    assert _speaker_shares(active, shot) == [0.6, 0.4]


def test_track_nearest_x_picks_the_key_subject():
    tracks = _spine_two_tracks()
    assert _track_nearest_x(tracks, 0.75, 1920) == 2
    assert _track_nearest_x(tracks, 0.12, 1920) == 1
    assert _track_nearest_x(tracks, 0.5, 1920) is None  # outside tolerance
    assert _track_nearest_x(tracks, None, 1920) is None


def test_integer_crop_is_even_and_in_frame():
    rect = (100.3, 50.7, 400.2, 300.9)
    x, y, w, h = _integer_crop(rect, 1920, 1080)
    assert w % 2 == 0 and h % 2 == 0
    assert x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080
    # The integer crop still covers the validated rect's region.
    assert x <= 100 and y <= 50 and x + w >= 500 and y + h >= 351


def test_regular_filtergraph_concats_every_shot():
    shots = [
        ComposedShot(0.0, 2.0, LAYOUT_SINGLE, (100.0, 0.0, 405.0, 720.0), []),
        ComposedShot(2.0, 4.0, LAYOUT_SINGLE, (700.0, 0.0, 405.0, 720.0), []),
    ]
    graph = _regular_filtergraph(shots, 1920, 1080, 405, 720)
    assert graph.count("trim=") == 2
    assert graph.count("crop=") == 2
    assert "concat=n=2:v=1:a=0[v]" in graph


def test_directive_dicts_accepts_dicts_and_models():
    raw = [{"start": 1.0, "end": 2.0, "x_position": 0.5, "reason": "causing_reaction"}]
    assert _directive_dicts(raw) == raw

    class _Stub:
        start, end, x_position, reason = 1.0, 2.0, 0.5, "referenced"

    assert _directive_dicts([_Stub()]) == [
        {"start": 1.0, "end": 2.0, "x_position": 0.5, "reason": "referenced"}
    ]

    class _Model:
        def model_dump(self):
            return {"start": 0.0, "end": 1.0, "x_position": 0.2, "reason": "speaking"}

    assert _directive_dicts([_Model()]) == [
        {"start": 0.0, "end": 1.0, "x_position": 0.2, "reason": "speaking"}
    ]
    assert _directive_dicts([{"start": 1.0}]) == []  # incomplete -> dropped


def test_aspect_tuple_maps_float_to_integer_ratio():
    assert _aspect_tuple(9.0 / 16.0) == (9, 16)
    assert _aspect_tuple(1.0) == (1, 1)
    assert abs(_aspect_tuple(0.75)[0] / _aspect_tuple(0.75)[1] - 0.75) < 1e-6
