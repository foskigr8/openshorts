"""Phase 4b of the framing-engine rebuild: composition.

WHY THIS EXISTS
---------------
`shot_planner.py` (Phase 4) decides WHO holds the frame and for how long. It
does not decide what the crop actually looks like — its `crop_rect_for_track`
returns the subject's own median box, which is a *subject* box, not a *camera*
box. This module turns the former into the latter.

Three problems are solved here that the previous engines did not solve:

1. WHO MATTERS WHEN NOBODY IS SPEAKING. Active-speaker detection answers "who
   is talking", which is the wrong question during a reaction — someone pops a
   balloon, someone winces, and the speaker is not the attraction. UNISAL
   saliency (vendored, trained on human eye-tracking) answers "where would a
   human look", which is the right question. The two are combined here into
   one weighted attention map rather than either being trusted alone.

2. VERTICAL COMPOSITION. Every crop the old engine produced was full-frame
   height — horizontal pan only — so a face could sit anywhere vertically.
   `crop_rect_containing` places the subject at a deliberate height
   (`DEFAULT_HEAD_Y`) whenever the crop is tighter than the full frame.

3. TWO SUBJECTS TOO FAR APART. Previously this became a wide shot with both
   people small, or a crop centred on the gap between them. It now becomes a
   split screen (vendored from pyautoflip), but only when the speaker evidence
   says both people are actually live in the exchange — see `decide_layout`.

WHAT IS DELIBERATELY NOT REUSED FROM pyautoflip
-----------------------------------------------
Its cropper was read and rejected; `vendor/pyautoflip/README.md` records why
per module. The short version: its STATIONARY mode averages key-frame
positions (which frames the gap between two speakers — the exact bug this
rebuild exists to fix), its crop window has no vertical component at all, and
its "padding" stretches the image rather than letterboxing it. Only its
saliency model and its split-screen geometry are used.

COORDINATES
-----------
All boxes and rects here are PIXELS in the source frame's own coordinate
space, matching `face_spine.py` (which stores raw InsightFace bboxes) and
`shot_planner.Shot.crop_rect`. Normalized coordinates appear only at the
boundary with the vendored split-screen helpers, which expect them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]        # (x, y, w, h) in pixels
Rect = Tuple[float, float, float, float]

# ---------------------------------------------------------------------------
# Attention weighting
# ---------------------------------------------------------------------------

ROLE_SPEAKER = "speaker"
ROLE_REACTOR = "reactor"
ROLE_BYSTANDER = "bystander"

# Boost applied to a face region in the attention map. The speaker sits above
# UNISAL's own maximum (1.0) so a confidently-bound speaker outranks any
# saliency peak; a reactor sits just above it so it can win against ordinary
# background but still lose to the speaker.
ROLE_WEIGHTS: Dict[str, float] = {
    ROLE_SPEAKER: 3.0,
    ROLE_REACTOR: 1.5,
}

# Bystanders are SUPPRESSED MULTIPLICATIVELY, not merely left unboosted. This
# is the one place we deliberately depart from upstream's approach: its
# `get_composite_mask` does `np.maximum(region, FACE_WEIGHT)`, which can only
# ever raise a value. A bystander standing on a high-saliency background would
# therefore keep their full saliency and still drag the centre of mass toward
# themselves — which is "biggest/brightest face wins", the bug this rebuild
# started from. Scaling down can actually remove them from contention.
BYSTANDER_SUPPRESSION = 0.25

# Attention below this fraction of the map maximum is treated as background
# when locating the subject. Guards against a broad, low, flat field of
# saliency (a busy background) pulling the centre of mass toward frame centre.
ATTENTION_FLOOR_FRAC = 0.35


@dataclass
class WeightedFace:
    """A face box plus what role it plays in this shot."""
    box: Box
    role: str = ROLE_BYSTANDER

    @property
    def weight(self) -> float:
        return ROLE_WEIGHTS.get(self.role, BYSTANDER_SUPPRESSION)


def build_attention_map(saliency: np.ndarray,
                        faces: Sequence[WeightedFace]) -> np.ndarray:
    """Combine a UNISAL saliency map with role-weighted face regions.

    `saliency` is (H, W) float in [0, 1] as returned by the vendored
    `SaliencyDetector`. Returns a map in the same shape where the eventual
    centre of mass reflects *who matters*, not merely what is bright.

    Boosting roles are applied before suppression so that a face which is both
    the speaker and sitting on a bystander's box cannot be suppressed by the
    latter — role precedence is decided by weight, never by list order.
    """
    composite = np.asarray(saliency, dtype=np.float32).copy()
    h, w = composite.shape[:2]

    def _clip(box: Box) -> Optional[Tuple[int, int, int, int]]:
        x, y, bw, bh = box
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    boosted = [f for f in faces if f.role in ROLE_WEIGHTS]
    suppressed = [f for f in faces if f.role not in ROLE_WEIGHTS]

    # Suppress first, then boost, so an overlapping boost always wins.
    for face in suppressed:
        region = _clip(face.box)
        if region:
            x0, y0, x1, y1 = region
            composite[y0:y1, x0:x1] *= BYSTANDER_SUPPRESSION

    for face in sorted(boosted, key=lambda f: f.weight):
        region = _clip(face.box)
        if region:
            x0, y0, x1, y1 = region
            composite[y0:y1, x0:x1] = np.maximum(
                composite[y0:y1, x0:x1], face.weight
            )

    return composite


def attention_center(composite: np.ndarray) -> Tuple[float, float]:
    """Weighted centre of mass of the attention map, normalized to 0-1.

    Only cells at or above `ATTENTION_FLOOR_FRAC` of the map maximum vote.
    Without that floor a large dim region outvotes a small bright one purely
    on area, which is how a busy background pulls the camera off the subject.
    Falls back to frame centre for an empty/degenerate map rather than raising
    — a missing subject should produce a neutral crop, not a failed render.
    """
    if composite.size == 0:
        return 0.5, 0.5

    peak = float(composite.max())
    if peak <= 0.0:
        return 0.5, 0.5

    mask = composite >= peak * ATTENTION_FLOOR_FRAC
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.5, 0.5

    weights = composite[ys, xs].astype(np.float64)
    total = weights.sum()
    if total <= 0:
        return 0.5, 0.5

    h, w = composite.shape[:2]
    return float(np.dot(xs, weights) / total / w), float(np.dot(ys, weights) / total / h)


# ---------------------------------------------------------------------------
# Crop geometry
# ---------------------------------------------------------------------------

# Fraction of the crop HEIGHT above the subject's centre. 0.36 matches
# main.CAMERA_HEAD_Y, the value already tuned on this footage: it leaves
# headroom above and body below rather than centring the face vertically,
# which reads as a snapshot.
DEFAULT_HEAD_Y = 0.36

# Breathing room around the subject box, as a fraction of its own size. The
# side band is wider than the vertical one because a face box is much narrower
# than a person's shoulders; cropping tight to a face box alone cuts ears and
# shoulders off.
DEFAULT_SIDE_MARGIN = 0.55
DEFAULT_VERT_MARGIN = 0.35

VERTICAL_9_16 = 9.0 / 16.0


def crop_rect_containing(subject: Box, frame_w: int, frame_h: int,
                         aspect: float = VERTICAL_9_16,
                         head_y: float = DEFAULT_HEAD_Y,
                         side_margin: float = DEFAULT_SIDE_MARGIN,
                         vert_margin: float = DEFAULT_VERT_MARGIN) -> Rect:
    """Smallest `aspect`-correct crop that CONTAINS `subject` with margins.

    Containment is the property that matters: the previous engine chose a crop
    from an aim point and only afterwards discovered the subject was half
    outside it. Here the subject box plus its margins is the input constraint,
    so a returned rect that fails `contains()` is a bug, not a tuning issue.

    Vertical placement uses `head_y` whenever the crop is shorter than the
    frame. When the crop is full-frame height (the usual 9:16-from-16:9 case)
    there is no vertical freedom left and the rect simply spans the frame —
    the same degenerate case upstream hard-codes, reached here as a
    consequence rather than as an assumption.
    """
    sx, sy, sw, sh = subject
    need_w = sw * (1.0 + 2.0 * side_margin)
    need_h = sh * (1.0 + 2.0 * vert_margin)

    # Grow to the target aspect, whichever dimension is binding.
    crop_h = max(need_h, need_w / aspect)
    crop_w = crop_h * aspect

    # Clamp to the frame, preserving aspect (the frame may be too small).
    if crop_w > frame_w:
        crop_w = float(frame_w)
        crop_h = crop_w / aspect
    if crop_h > frame_h:
        crop_h = float(frame_h)
        crop_w = crop_h * aspect
    crop_w = min(crop_w, float(frame_w))

    subject_cx = sx + sw / 2.0
    subject_cy = sy + sh / 2.0

    crop_x = subject_cx - crop_w / 2.0
    crop_y = subject_cy - crop_h * head_y

    crop_x = max(0.0, min(crop_x, frame_w - crop_w))
    crop_y = max(0.0, min(crop_y, frame_h - crop_h))

    return crop_x, crop_y, crop_w, crop_h


def contains(crop: Rect, subject: Box, tolerance: float = 0.5) -> bool:
    """Is `subject` fully inside `crop`? The containment check, used as an
    assertion by the validation stage rather than as a soft score."""
    cx, cy, cw, ch = crop
    sx, sy, sw, sh = subject
    return (sx >= cx - tolerance and sy >= cy - tolerance
            and sx + sw <= cx + cw + tolerance
            and sy + sh <= cy + ch + tolerance)


def union_box(*boxes: Box) -> Optional[Box]:
    """Smallest box covering all inputs; None if none were given."""
    real = [b for b in boxes if b is not None]
    if not real:
        return None
    x0 = min(b[0] for b in real)
    y0 = min(b[1] for b in real)
    x1 = max(b[0] + b[2] for b in real)
    y1 = max(b[1] + b[3] for b in real)
    return x0, y0, x1 - x0, y1 - y0


# ---------------------------------------------------------------------------
# Layout decision: one crop, a two-shot, or a split screen
# ---------------------------------------------------------------------------

LAYOUT_SINGLE = "single"
LAYOUT_TWO_SHOT = "two_shot"
LAYOUT_SPLIT = "split"

# A two-shot is only worth keeping while both people still read at a usable
# size. Once the union of both subjects needs a crop wider than this fraction
# of the frame, holding them in one window shrinks both faces to nothing —
# that is the point where a split screen wins.
MAX_TWO_SHOT_WIDTH_FRAC = 0.72

# Below this share of the shot, a second speaker is an interjection rather
# than a genuine exchange, and cutting to whoever holds the floor reads
# better than committing the whole shot to a split screen.
MIN_SECOND_SPEAKER_SHARE = 0.25


def decide_layout(subjects: Sequence[Box], speaker_shares: Sequence[float],
                  frame_w: int, frame_h: int,
                  aspect: float = VERTICAL_9_16) -> str:
    """Choose the composition for a shot with one or two subjects.

    `speaker_shares[i]` is the fraction of the shot during which subject i was
    the bound active speaker. This is what makes the split decision *editorial*
    rather than purely geometric: upstream's `find_split_faces` splits whenever
    two faces cannot share a crop, but if one person is holding the floor the
    better cut is a single on them — a split screen dedicating half the frame
    to a silent listener wastes the format. Splitting is reserved for a real
    exchange, which is exactly the overlapping-speech case a single crop
    handles worst.
    """
    real = [s for s in subjects if s is not None]
    if len(real) < 2:
        return LAYOUT_SINGLE

    both_live = (len(speaker_shares) >= 2
                 and sorted(speaker_shares, reverse=True)[1] >= MIN_SECOND_SPEAKER_SHARE)
    if not both_live:
        return LAYOUT_SINGLE

    combined = union_box(*real)
    crop = crop_rect_containing(combined, frame_w, frame_h, aspect)

    # Does one window still hold both at a usable size?
    if combined[2] <= frame_w * MAX_TWO_SHOT_WIDTH_FRAC and contains(crop, combined):
        return LAYOUT_TWO_SHOT
    return LAYOUT_SPLIT


def split_centers(subjects: Sequence[Box], frame_w: int, frame_h: int,
                  aspect_tuple: Tuple[int, int] = (9, 16)):
    """Normalized panel centres for a split screen, via the vendored helper.

    Delegates the trigger geometry to `find_split_faces` so the vendored
    behaviour stays the single source of truth, and falls back to the two
    outermost subjects when the vendored rule declines but `decide_layout`
    has already committed to a split (it can, when the shot is too wide to
    hold both at a usable size even though they would technically fit).
    """
    from vendor.pyautoflip import find_split_faces

    rects = [tuple(s) for s in subjects if s is not None]
    found = find_split_faces(rects, frame_w, frame_h, aspect_tuple)
    if found is not None:
        return found

    ordered = sorted(rects, key=lambda b: b[0] + b[2] / 2.0)
    if len(ordered) < 2:
        return None
    return [
        ((b[0] + b[2] / 2.0) / frame_w, (b[1] + b[3] / 2.0) / frame_h)
        for b in (ordered[0], ordered[-1])
    ]
