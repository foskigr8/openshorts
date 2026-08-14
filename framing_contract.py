"""The framing contract — the single owner of composition geometry.

WHY THIS MODULE EXISTS
----------------------
Before this, the engine's only hard guarantee was ``contains(crop, face_box)``:
is the detector's face rectangle inside the crop rectangle. That predicate is
satisfied by a crop with one pixel of headroom, by a crop centred on an empty
aisle between two people, and — critically — by a crop whose top edge runs
through the subject's hair, because **the detector box excludes hair**. Every
shipped framing failure passed ``contains()``.

So containment is not the contract. These ratios are, and they were measured
off footage that works (see PLAN_FRAMING_CONTRACT.md §1.1 — two reference
shorts, 695 sampled frames, YuNet at conf 0.7):

    face height  = 0.155 of frame height   (IQR 0.14 – 0.17)
    eyeline      = 0.22  of frame height   (IQR 0.20 – 0.24)
    face top     = 0.152 of frame height   → headroom is real and never zero

THE ARITHMETIC THIS REPLACES
-----------------------------
The old ``crop_rect_containing`` sized a crop from margins around the face box
and then placed the face *centre* at a fixed fraction of it::

    crop_h = max(h_f·1.70, w_f·2.10 / aspect)      # margins
    crop_y = face_centre_y − 0.36 · crop_h          # DEFAULT_HEAD_Y

For a 9:16 single the width term binds, crop_h ≈ 2.99·h_f, and there is ~7.5%
headroom — marginal, survives. For a 9:8 split panel the width term stops
binding, crop_h collapses to 1.70·h_f, and the hair (0.35·h_f above the box)
ends up **14% of the panel height ABOVE the crop top**. Every split panel this
engine ever rendered cut the head, by construction.

Here, size and position come from the same two ratios, so that cannot happen:

    crop_h = h_f / face_frac
    crop_y = eyeline_px − eye_y · crop_h

single (0.155 / 0.22) → crop_h = 6.45·h_f, headroom = +10.1% of crop
panel  (0.300 / 0.34) → crop_h = 3.33·h_f, headroom = +10.9% of panel

Two layouts, independent ratios, the same ~10% headroom. That agreement is the
check that the constants are self-consistent; ``tests/test_framing_contract.py``
asserts it numerically.

CAPTIONS ARE NOT THIS MODULE'S BUSINESS. Invariant I8 in the plan is
deliberately not implemented here — caption styling and placement are owned by
``subtitles.py`` and are left exactly as they are.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]   # (x, y, w, h) in source pixels
Rect = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# The measured constants. Changing these changes the LOOK. Changing anything
# else in this file is a bug fix.
# ---------------------------------------------------------------------------

#: Head box (hair + jaw) derived from a detector face box, as multiples of the
#: face box's own width/height: (left pad, top pad, width scale, height scale).
#: The top pad is the important one — it is the hair the detector never sees.
HEAD_DX, HEAD_DY, HEAD_SX, HEAD_SY = 0.30, 0.35, 1.60, 1.50

#: Eyeline as a fraction down the face box (matches the reference measurement).
EYELINE_IN_FACE = 0.42

#: SINGLE / TWO_SHOT: face height and eyeline as fractions of the crop.
SINGLE_FACE_FRAC = 0.155
SINGLE_EYE_Y = 0.22

#: Split panel: same two ratios, expressed against the PANEL height. A panel is
#: half the output, so 0.30 of a panel == 0.15 of the output frame — the same
#: subject size as a single. This is the whole argument for the split screen:
#: it is the only layout that keeps subject size when subjects are far apart.
PANEL_FACE_FRAC = 0.30
PANEL_EYE_Y = 0.34

#: I2 — crop top to head top, as a fraction of crop height. Target ~0.10.
HEADROOM_MIN = 0.06

#: I3 tolerances, as a fraction of crop height.
EYE_TOL_SINGLE = 0.06
EYE_TOL_PANEL = 0.08

#: I4 — the bound speaker's face height as a fraction of the OUTPUT frame.
FACE_FRAC_BAND = (0.10, 0.225)

#: I5 — a head this small does not count as "someone is on screen".
MIN_HEAD_FRAC = 0.06

#: Max horizontal offset from centre, as a fraction of crop width. This is what
#: reproduces the reference's over-the-shoulder composition: we cannot SELECT an
#: OTS shot (we get one source frame per moment), but biasing the crop toward
#: the person the speaker faces lets the listener occupy the near edge.
LOOK_ROOM = 0.12

#: A crop tighter than this fraction of source height upscales into mush.
DEFAULT_MIN_CROP_FRAC = 0.45

#: I6 — for TWO_SHOT / WIDE, some head centre must sit in this middle band.
CENTRE_BAND = 0.60

#: Layout names, mirrored from reframe_v3 so this module imports nothing.
SINGLE = "single"
TWO_SHOT = "two_shot"
PANEL = "panel"
WIDE = "wide"

#: How much of the OUTPUT frame this layout's crop occupies vertically. Used to
#: convert a face's fraction-of-crop into a fraction-of-output for I4.
OUTPUT_SCALE = {SINGLE: 1.0, TWO_SHOT: 1.0, PANEL: 0.5, WIDE: 1.0}

_EPS = 0.5   # pixels; everything here is pixel geometry, not sub-pixel


def min_crop_frac() -> float:
    """The blur floor, env-overridable (``MIN_CROP_FRAC``)."""
    try:
        return float(os.environ.get("MIN_CROP_FRAC", DEFAULT_MIN_CROP_FRAC))
    except (TypeError, ValueError):
        return DEFAULT_MIN_CROP_FRAC


# ---------------------------------------------------------------------------
# I1 — head, not face
# ---------------------------------------------------------------------------

def head_box(face: Box, frame_w: int, frame_h: int) -> Box:
    """The head (hair to jaw) implied by a detector face box, clipped to frame.

    I1. No composition function may consume a raw detector box: the box stops
    at the eyebrows, so a crop that 'contains' it can still shear the skull.
    """
    x, y, w, h = (float(v) for v in face)
    hx = x - HEAD_DX * w
    hy = y - HEAD_DY * h
    hw = HEAD_SX * w
    hh = HEAD_SY * h
    x0 = max(0.0, hx)
    y0 = max(0.0, hy)
    x1 = min(float(frame_w), hx + hw)
    y1 = min(float(frame_h), hy + hh)
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def eyeline_y(face: Box) -> float:
    """Absolute y of the subject's eyeline, in source pixels."""
    return float(face[1]) + EYELINE_IN_FACE * float(face[3])


def face_centre_x(face: Box) -> float:
    return float(face[0]) + float(face[2]) / 2.0


# ---------------------------------------------------------------------------
# THE crop function
# ---------------------------------------------------------------------------

def frame_subject(face: Box, frame_w: int, frame_h: int, aspect: float,
                  face_frac: float, eye_y: float,
                  look_dir: float = 0.0,
                  min_frac: Optional[float] = None) -> Rect:
    """The aspect-correct crop that frames ``face`` to the contract.

    Size comes from ``face_frac`` (how big the face should read), position from
    ``eye_y`` (where the eyeline should sit). Those are the only two inputs —
    margins are a consequence, not a control.

    ``look_dir`` in [-1, 1] biases the crop horizontally (negative = left).
    The shift is capped so the crop centre can never leave the head box, which
    is invariant I6; see ``look_room_dir`` for how the direction is chosen.
    """
    fx, fy, fw, fh = (float(v) for v in face)
    if fw <= 0 or fh <= 0:
        raise ValueError(f"degenerate face box {face!r}")

    floor_h = (min_crop_frac() if min_frac is None else float(min_frac)) * frame_h

    # Size: from the target face fraction, floored by the blur limit and
    # ceilinged by the source. Both clamps preserve the aspect.
    crop_h = max(fh / float(face_frac), floor_h)
    crop_h = min(crop_h, float(frame_h))
    crop_w = crop_h * aspect
    if crop_w > frame_w:
        crop_w = float(frame_w)
        crop_h = min(crop_w / aspect, float(frame_h))
        crop_w = crop_h * aspect

    # Position: eyeline at eye_y of the crop.
    crop_y = eyeline_y(face) - float(eye_y) * crop_h

    # Horizontal: centred on the face, then look-room, capped so the crop
    # centre stays inside the head box (I6).
    head = head_box(face, frame_w, frame_h)
    max_shift = 0.8 * max(head[2] / 2.0, _EPS)
    shift = max(-max_shift, min(float(look_dir) * LOOK_ROOM * crop_w, max_shift))
    crop_x = face_centre_x(face) - crop_w / 2.0 + shift

    crop_x = max(0.0, min(crop_x, frame_w - crop_w))
    crop_y = max(0.0, min(crop_y, frame_h - crop_h))
    return (crop_x, crop_y, crop_w, crop_h)


def frame_single(face: Box, frame_w: int, frame_h: int, aspect: float,
                 look_dir: float = 0.0, min_frac: Optional[float] = None) -> Rect:
    """A SINGLE / TWO_SHOT crop at the reference numbers."""
    return frame_subject(face, frame_w, frame_h, aspect,
                         SINGLE_FACE_FRAC, SINGLE_EYE_Y, look_dir, min_frac)


def frame_panel(face: Box, frame_w: int, frame_h: int, panel_aspect: float,
                look_dir: float = 0.0, min_frac: Optional[float] = None) -> Rect:
    """One split-screen panel, framed so the face lands at 0.15 of OUTPUT."""
    return frame_subject(face, frame_w, frame_h, panel_aspect,
                         PANEL_FACE_FRAC, PANEL_EYE_Y, look_dir, min_frac)


def look_room_dir(face: Box, others: Sequence[Box], crop_w: float,
                  gaze_dx: Optional[float] = None) -> float:
    """Which way to bias the crop, in [-1, 1].

    Toward the nearest other head within 1.5·crop_w — that is what pulls the
    listener's shoulder into the near edge and reproduces the reference's OTS
    composition without OTS coverage. Falling back to gaze direction (SCRFD's
    five landmarks give it for free: eye-midpoint x vs nose x), then to none.
    """
    cx = face_centre_x(face)
    best = None
    for other in others or ():
        if other is None:
            continue
        ox = face_centre_x(other)
        d = ox - cx
        if abs(d) < _EPS or abs(d) > 1.5 * crop_w:
            continue
        if best is None or abs(d) < abs(best):
            best = d
    if best is not None:
        return 1.0 if best > 0 else -1.0
    if gaze_dx is not None and abs(gaze_dx) > 1e-6:
        return max(-1.0, min(1.0, float(gaze_dx)))
    return 0.0


# ---------------------------------------------------------------------------
# Measurements a caller can act on
# ---------------------------------------------------------------------------

def headroom_frac(crop: Rect, face: Box, frame_w: int, frame_h: int) -> float:
    """Gap between the crop top and the head top, ÷ crop height (I2)."""
    head = head_box(face, frame_w, frame_h)
    if crop[3] <= 0:
        return -1.0
    return (head[1] - crop[1]) / crop[3]


def eye_frac(crop: Rect, face: Box) -> float:
    """Where the eyeline sits in the crop, 0 = top (I3)."""
    if crop[3] <= 0:
        return -1.0
    return (eyeline_y(face) - crop[1]) / crop[3]


def face_frac_of_output(crop: Rect, face: Box, layout: str) -> float:
    """Face height as a fraction of the OUTPUT frame height (I4)."""
    if crop[3] <= 0:
        return 0.0
    return (float(face[3]) / crop[3]) * OUTPUT_SCALE.get(layout, 1.0)


def achievable_face_frac(face_h: float, frame_w: int, frame_h: int,
                         aspect: float, layout: str = SINGLE,
                         min_frac: Optional[float] = None) -> float:
    """The best face size this source can actually deliver for this layout.

    The resolution budget (plan §5.7). ``crop_h`` cannot go below the blur
    floor, so a small/distant face simply cannot be framed to 0.155 — that is
    the source telling you the camera was far away, not a tuning problem.
    Callers use this to prefer a WIDE (or a different moment) instead of
    upscaling into mush.
    """
    floor_h = (min_crop_frac() if min_frac is None else float(min_frac)) * frame_h
    target = SINGLE_FACE_FRAC if layout != PANEL else PANEL_FACE_FRAC
    crop_h = max(float(face_h) / target, floor_h)
    crop_h = min(crop_h, float(frame_h))
    if crop_h * aspect > frame_w:
        crop_h = min(float(frame_w) / aspect, float(frame_h))
    if crop_h <= 0:
        return 0.0
    return (float(face_h) / crop_h) * OUTPUT_SCALE.get(layout, 1.0)


# ---------------------------------------------------------------------------
# The assertions
# ---------------------------------------------------------------------------

def check(crop: Rect, faces: Sequence[Box], layout: str,
          frame_w: int, frame_h: int,
          aspect: Optional[float] = None,
          subject_index: int = 0,
          aspect_tolerance: float = 0.02) -> List[str]:
    """Return every violated invariant for one composed crop. Empty == valid.

    ``faces[subject_index]`` is the bound subject — the person this crop is
    supposed to be about. The rest are context (used by I5/I6).

    Deliberately NOT checked here: captions (I8) and cut rhythm (I9). Captions
    are owned by ``subtitles.py`` and are out of scope for this rebuild.
    """
    problems: List[str] = []
    if crop is None:
        return ["I0 crop is None"]
    cx, cy, cw, ch = (float(v) for v in crop)
    if cw <= 0 or ch <= 0:
        return [f"I0 degenerate crop {tuple(round(v) for v in crop)}"]

    # Frame containment + aspect (kept from the old validator).
    if (cx < -_EPS or cy < -_EPS
            or cx + cw > frame_w + _EPS or cy + ch > frame_h + _EPS):
        problems.append(
            f"I0 crop {tuple(round(v) for v in crop)} escapes frame "
            f"{frame_w}x{frame_h}")
    if aspect is not None and abs(cw / ch - aspect) > aspect_tolerance:
        problems.append(f"I0 aspect {cw / ch:.4f} != target {aspect:.4f}")

    real = [f for f in (faces or ()) if f is not None]
    if not real:
        return problems + ["I5 crop has no subject at all"]

    subject = real[min(subject_index, len(real) - 1)]
    heads = [head_box(f, frame_w, frame_h) for f in real]
    inside = [h for h in heads
              if h[0] >= cx - _EPS and h[1] >= cy - _EPS
              and h[0] + h[2] <= cx + cw + _EPS
              and h[1] + h[3] <= cy + ch + _EPS]
    scale = OUTPUT_SCALE.get(layout, 1.0)

    # I5 — never frame nothing.
    if not any((h[3] / ch) * scale >= MIN_HEAD_FRAC for h in inside):
        problems.append(
            f"I5 no head ≥ {MIN_HEAD_FRAC:.2f} of output is fully inside the crop")

    if layout in (SINGLE, PANEL):
        head = head_box(subject, frame_w, frame_h)

        # I2 — headroom. A crop already flush with the frame top has no room
        # left to give; that is the source's framing, not ours.
        room = (head[1] - cy) / ch
        if room < HEADROOM_MIN - 1e-6 and cy > _EPS:
            problems.append(
                f"I2 headroom {room:+.3f} < {HEADROOM_MIN} "
                f"(head top {head[1]:.0f}, crop top {cy:.0f})")

        # I3 — eyeline. Exempt when the crop is clamped against a frame edge.
        clamped = cy <= _EPS or cy + ch >= frame_h - _EPS
        target = SINGLE_EYE_Y if layout == SINGLE else PANEL_EYE_Y
        tol = EYE_TOL_SINGLE if layout == SINGLE else EYE_TOL_PANEL
        rel = (eyeline_y(subject) - cy) / ch
        if not clamped and abs(rel - target) > tol:
            problems.append(
                f"I3 eyeline {rel:.3f} outside {target}±{tol}")

        # I4 — subject size.
        frac = (float(subject[3]) / ch) * scale
        lo, hi = FACE_FRAC_BAND
        if not (lo - 1e-6 <= frac <= hi + 1e-6):
            problems.append(
                f"I4 face height {frac:.3f} of output outside [{lo}, {hi}]")

        # I6 — never centre the gap. Exempt when the crop is clamped against a
        # side of the frame: a subject standing at the very edge cannot be
        # centred on, and the offset is then forced by geometry rather than
        # chosen. (Same reasoning as the eyeline exemption above. The failure
        # this invariant exists to catch — a crop parked on the empty space
        # BETWEEN two people — is never edge-clamped.)
        side_clamped = cx <= _EPS or cx + cw >= frame_w - _EPS
        mid = cx + cw / 2.0
        if (not side_clamped
                and not (head[0] - _EPS <= mid <= head[0] + head[2] + _EPS)):
            problems.append(
                f"I6 crop centre x {mid:.0f} is not on the subject's head "
                f"[{head[0]:.0f}, {head[0] + head[2]:.0f}]")
    else:
        # TWO_SHOT / WIDE — the centre legitimately falls between people, but
        # SOMEONE must be near the middle. This is clip 4's "crop on the gap".
        lo = cx + cw * (1.0 - CENTRE_BAND) / 2.0
        hi = cx + cw * (1.0 + CENTRE_BAND) / 2.0
        if not any(lo <= (h[0] + h[2] / 2.0) <= hi for h in heads):
            problems.append(
                f"I6 no head centre in the middle {CENTRE_BAND:.0%} of the crop")

    return problems


def check_panels(panels: Sequence[Rect], faces: Sequence[Box],
                 frame_w: int, frame_h: int,
                 panel_aspect: Optional[float] = None,
                 context: Optional[Sequence[Sequence[Box]]] = None,
                 max_size_ratio: float = 1.6,
                 max_heads_per_panel: int = 2) -> List[str]:
    """I7 — a split is symmetric or it is not a split.

    Each panel must independently satisfy I2/I3/I4/I5/I6, the two subjects must
    read at comparable size, and neither panel may be a crowd. A split that
    fails here is not rendered as a split — the caller falls back to a
    two-shot, then a look-room single, then the 4:3 wide.
    """
    problems: List[str] = []
    if not panels or len(panels) != 2:
        return ["I7 a vertical split needs exactly two panels"]
    if len(faces) < 2 or faces[0] is None or faces[1] is None:
        return ["I7 a vertical split needs two subjects"]

    for i, (panel, face) in enumerate(zip(panels, faces)):
        others = list((context or [(), ()])[i] or ())
        for p in check(panel, [face] + others, PANEL, frame_w, frame_h,
                       aspect=panel_aspect):
            problems.append(f"panel {i}: {p}")
        heads_in = 0
        for other in [face] + others:
            h = head_box(other, frame_w, frame_h)
            if (h[0] + h[2] / 2.0 >= panel[0]
                    and h[0] + h[2] / 2.0 <= panel[0] + panel[2]
                    and h[1] + h[3] / 2.0 >= panel[1]
                    and h[1] + h[3] / 2.0 <= panel[1] + panel[3]
                    and h[3] / max(panel[3], 1.0) * 0.5 >= MIN_HEAD_FRAC):
                heads_in += 1
        if heads_in > max_heads_per_panel:
            problems.append(
                f"panel {i}: I7 {heads_in} heads in one panel — that is a crowd, "
                f"not a participant")

    a = face_frac_of_output(panels[0], faces[0], PANEL)
    b = face_frac_of_output(panels[1], faces[1], PANEL)
    if a > 0 and b > 0:
        ratio = max(a, b) / min(a, b)
        if ratio > max_size_ratio + 1e-6:
            problems.append(
                f"I7 panel face sizes differ {ratio:.2f}× (> {max_size_ratio}) — "
                f"{a:.3f} vs {b:.3f} of output")
    return problems


def two_shot_holds_both(faces: Sequence[Box], frame_w: int, frame_h: int,
                        aspect: float, min_face_frac: float = 0.12) -> bool:
    """Can ONE crop hold both people *at a readable size*?

    This is the far-apart test, and the size floor is the point of it. A crop
    that technically contains both but shrinks them below ``min_face_frac`` is
    not holding both — that is the 4:3 wide that gave clips 2 and 4 their
    0.09-height faces, and a correct split beats it. Splitting is the RIGHT
    answer whenever this returns False.
    """
    real = [f for f in (faces or ()) if f is not None]
    if len(real) < 2:
        return True
    heads = [head_box(f, frame_w, frame_h) for f in real]
    x0 = min(h[0] for h in heads)
    y0 = min(h[1] for h in heads)
    x1 = max(h[0] + h[2] for h in heads)
    y1 = max(h[1] + h[3] for h in heads)
    need_w, need_h = x1 - x0, y1 - y0
    if need_w <= 0 or need_h <= 0:
        return False
    crop_h = max(need_h, need_w / aspect)
    if crop_h > frame_h or crop_h * aspect > frame_w:
        return False
    smallest = min(float(f[3]) for f in real)
    return smallest / crop_h >= min_face_frac
