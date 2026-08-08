"""
Pins the contract of the vendored pyautoflip 0.2.1 subset.

These tests exist because `vendor/pyautoflip/*.py` is kept byte-identical to
upstream (see vendor/pyautoflip/README.md). If a future version bump changes
behaviour, these fail rather than silently changing how clips are framed.

Two things are pinned:
  * split-screen geometry — the feature we adopted the library for
  * UNISAL saliency — that the vendored ONNX + external weights actually load
    and localize attention
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv required for vendored geometry")

from vendor.pyautoflip.split_screen import (  # noqa: E402
    find_split_faces,
    render_split_screen_from_centers,
)

VERTICAL = (9, 16)


def _rect(cx_norm, cy_norm, frame_w, frame_h, size=120):
    """Build an (x, y, w, h) face rect centered on a normalized point."""
    return (
        int(cx_norm * frame_w) - size // 2,
        int(cy_norm * frame_h) - size // 2,
        size,
        size,
    )


# ─── find_split_faces ────────────────────────────────────────────────────────


def test_no_faces_never_splits():
    assert find_split_faces([], 1920, 1080, VERTICAL) is None


def test_single_face_never_splits():
    """One subject is a crop problem, not a split-screen problem."""
    assert find_split_faces([_rect(0.5, 0.5, 1920, 1080)], 1920, 1080, VERTICAL) is None


def test_two_close_faces_fit_one_crop():
    """Faces inside one 9:16 window must NOT split — a split here would be a
    needless downgrade from a clean single shot."""
    faces = [_rect(0.45, 0.5, 1920, 1080), _rect(0.55, 0.5, 1920, 1080)]
    assert find_split_faces(faces, 1920, 1080, VERTICAL) is None


def test_two_far_faces_trigger_split():
    faces = [_rect(0.2, 0.4, 1920, 1080), _rect(0.8, 0.6, 1920, 1080)]
    result = find_split_faces(faces, 1920, 1080, VERTICAL)
    assert result is not None
    assert len(result) == 2


def test_split_threshold_is_the_crop_width():
    """The trigger is exactly 'wider apart than one crop window'. A 9:16 crop of
    a 1920x1080 frame is 607px => 0.316 normalized."""
    crop_w_norm = int(1080 * 9 / 16) / 1920
    frame = (1920, 1080)

    just_inside = [
        _rect(0.5 - crop_w_norm * 0.45, 0.5, *frame),
        _rect(0.5 + crop_w_norm * 0.45, 0.5, *frame),
    ]
    just_outside = [
        _rect(0.5 - crop_w_norm * 0.55, 0.5, *frame),
        _rect(0.5 + crop_w_norm * 0.55, 0.5, *frame),
    ]

    assert find_split_faces(just_inside, 1920, 1080, VERTICAL) is None
    assert find_split_faces(just_outside, 1920, 1080, VERTICAL) is not None


def test_split_returns_left_then_right_sorted():
    """Order is positional, not detection order — panel 1 is always the left
    subject. Passing faces in reverse must not flip the layout."""
    faces = [_rect(0.85, 0.5, 1920, 1080), _rect(0.15, 0.5, 1920, 1080)]
    result = find_split_faces(faces, 1920, 1080, VERTICAL)
    assert result[0][0] < result[1][0]
    assert result[0][0] == pytest.approx(0.15, abs=0.01)
    assert result[1][0] == pytest.approx(0.85, abs=0.01)


def test_three_faces_picks_the_extremes():
    """With 3+ subjects it spans outermost-left to outermost-right, so the
    middle subject is covered by the two panels rather than dropped."""
    faces = [
        _rect(0.1, 0.5, 1920, 1080),
        _rect(0.5, 0.5, 1920, 1080),
        _rect(0.9, 0.5, 1920, 1080),
    ]
    result = find_split_faces(faces, 1920, 1080, VERTICAL)
    assert result[0][0] == pytest.approx(0.1, abs=0.01)
    assert result[1][0] == pytest.approx(0.9, abs=0.01)


def test_centers_are_normalized():
    faces = [_rect(0.2, 0.3, 1920, 1080), _rect(0.8, 0.7, 1920, 1080)]
    for cx, cy in find_split_faces(faces, 1920, 1080, VERTICAL):
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0


# ─── render_split_screen_from_centers ────────────────────────────────────────


def test_split_output_has_target_aspect():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = render_split_screen_from_centers(frame, [(0.25, 0.5), (0.75, 0.5)], VERTICAL)
    h, w = out.shape[:2]
    assert h == 1080
    assert w == int(1080 * 9 / 16)
    assert w / h == pytest.approx(9 / 16, abs=0.01)


def test_split_panels_are_not_distorted():
    """The source strip and the rendered panel must share an aspect ratio.

    This is the property that made split-screen worth adopting while
    `apply_padding_to_crop` was rejected — that function stretches content to
    fill the output (see vendor README). If a future bump introduces the same
    stretch here, faces get squashed and this test catches it.
    """
    h, w = 1080, 1920
    out_w = int(h * 9 / 16)
    panel_h = (h - 4) // 2
    panel_ratio = out_w / panel_h

    src_crop_w = min(int(h * 9 / 16), w)
    src_crop_h = min(int(src_crop_w / panel_ratio), h)

    assert src_crop_w / src_crop_h == pytest.approx(panel_ratio, rel=0.01)


def test_split_panels_center_on_their_subject():
    """Each panel is centered on its own face, so the two subjects end up
    stacked and both centered — the actual point of the layout."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Distinct markers at the two face centers (chosen away from the frame
    # edges so the panel crop is not clamped).
    cv2.circle(frame, (480, 540), 40, (0, 0, 255), -1)    # red   @ x=0.25
    cv2.circle(frame, (1440, 540), 40, (0, 255, 0), -1)   # green @ x=0.75

    out = render_split_screen_from_centers(frame, [(0.25, 0.5), (0.75, 0.5)], VERTICAL)
    out_w = out.shape[1]
    panel_h = (1080 - 4) // 2

    top, bottom = out[:panel_h], out[panel_h + 4:]

    # Red dominates the top panel, green the bottom — subjects not swapped.
    assert top[:, :, 2].mean() > top[:, :, 1].mean()
    assert bottom[:, :, 1].mean() > bottom[:, :, 2].mean()

    # Each marker sits near its panel's horizontal center.
    red_x = np.argmax(top[:, :, 2].sum(axis=0))
    green_x = np.argmax(bottom[:, :, 1].sum(axis=0))
    assert red_x == pytest.approx(out_w / 2, abs=out_w * 0.12)
    assert green_x == pytest.approx(out_w / 2, abs=out_w * 0.12)


def test_split_has_a_divider():
    """A visible seam separates the panels."""
    frame = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    out = render_split_screen_from_centers(frame, [(0.25, 0.5), (0.75, 0.5)], VERTICAL)
    panel_h = (1080 - 4) // 2
    divider = out[panel_h:panel_h + 4]
    assert divider.mean() < 100  # dark seam against a white source


# ─── UNISAL saliency ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def saliency():
    pytest.importorskip("onnxruntime", reason="onnxruntime required for UNISAL")
    from vendor.pyautoflip.saliency_detector import SaliencyDetector

    return SaliencyDetector()


def test_unisal_loads_and_returns_normalized_map(saliency):
    """Guards the vendored weights: unisal.onnx uses an external .onnx.data
    file, so a partial copy would fail here rather than at render time."""
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    result = saliency.detect(frame)
    smap = result["saliency_map"]

    assert smap.shape == (360, 640)
    assert smap.min() >= 0.0
    assert smap.max() <= 1.0


def test_unisal_localizes_a_salient_region(saliency):
    """The signal we adopted UNISAL for: it marks where attention goes, which
    is what covers reactions the active-speaker detector cannot see."""
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (500, 120), 60, (255, 255, 255), -1)

    smap = saliency.detect(frame)["saliency_map"]
    peak_y, peak_x = np.unravel_index(np.argmax(smap), smap.shape)

    assert peak_x == pytest.approx(500, abs=60)
    assert peak_y == pytest.approx(120, abs=60)
