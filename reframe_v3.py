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
   `crop_rect_containing` places the subject's EYELINE at a measured height
   (`framing_contract.SINGLE_EYE_Y`) whenever the crop is tighter than the
   full frame, and sizes the crop from the same ratio pair.

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
import os

import numpy as np

import framing_contract

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

# Crop geometry now lives in framing_contract.py. The constants that used to
# sit here — DEFAULT_HEAD_Y = 0.36, DEFAULT_SIDE_MARGIN = 0.55,
# DEFAULT_VERT_MARGIN = 0.35 — are DELETED, not retuned.
#
# They were the head-cut. Sizing a crop from margins around the face box and
# then placing the face CENTRE at a fixed fraction of it works by luck: for a
# 9:16 single the width term binds and leaves ~7.5% headroom, but for a 9:8
# split panel it stops binding, the crop collapses to 1.70x the face height,
# and the hair (0.35x the face height above the detector box, which the
# detector never sees) ends up 14% of the panel height ABOVE the crop top.
# Every split panel this engine rendered cut the head, by construction.
#
# framing_contract derives size AND position from the same two measured
# ratios, so that failure mode cannot recur. See PLAN_FRAMING_CONTRACT.md §3.1.
VERTICAL_9_16 = 9.0 / 16.0

# What can be a SUBJECT at all. A detection outside these bounds is not a face
# we can compose on, and letting one through is how clip 1 framed a placard and
# clip 4 framed the back of a head (plan §3.4).
MIN_FACE_ASPECT = 0.55          # narrower than this is not a head, even in profile
MAX_FACE_ASPECT = 1.35          # wider than this is a placard held at chest height
MAX_FACE_AREA_FRAC = 0.10       # of the frame; above this it is a torso or a lens-filler
MIN_FACE_HEIGHT_FRAC = 0.02     # below this no crop can frame it above the blur floor


def _enforce_min_crop(crop: Rect, frame_w: int, frame_h: int,
                      aspect: float, min_frac: float = 0.45) -> Rect:
    """Never let a crop zoom tighter than `min_frac` of the source height.
    A tiny crop (a small face in a wide group shot) upscaled to 1080x1920 is
    exactly the "zoomed in and blurry, can't see what's going on" failure —
    this only ever WIDENS the crop (never tightens), preserves the aspect
    and the vertical centre so the head stays put."""
    x, y, w, h = crop
    floor_h = min_frac * frame_h
    if h >= floor_h or h >= frame_h:
        return crop
    new_h = min(floor_h, float(frame_h))
    new_w = new_h * aspect
    if new_w > frame_w:
        new_w = float(frame_w)
        new_h = new_w / aspect
    cy = y + h / 2.0
    new_y = max(0.0, min(cy - new_h / 2.0, frame_h - new_h))
    cx = x + w / 2.0
    new_x = max(0.0, min(cx - new_w / 2.0, frame_w - new_w))
    return (new_x, new_y, new_w, new_h)


def crop_rect_containing(subject: Box, frame_w: int, frame_h: int,
                         aspect: float = VERTICAL_9_16,
                         layout: str = framing_contract.SINGLE,
                         look_dir: float = 0.0) -> Rect:
    """The `aspect`-correct crop that FRAMES `subject` to the contract.

    Kept as the module's crop entry point so every call site inherits the fix,
    but the body is now `framing_contract.frame_subject`: size comes from the
    target face fraction and position from the target eyeline, both measured
    off reference footage. Margins are a consequence, not a control.

    `layout` selects the ratio pair — SINGLE/TWO_SHOT frame the face at 0.155
    of the output height with the eyeline at 0.22, a PANEL frames it at 0.30 of
    the PANEL height (= 0.15 of the output, deliberately the same subject size)
    with the eyeline at 0.34. Passing the panel aspect without passing
    `layout=PANEL` is the bug that cut every split panel's head; the two must
    agree, so a caller that widens the aspect must say why.

    `look_dir` in [-1, 1] biases the crop horizontally toward the person the
    speaker is facing, which is what reproduces the reference's
    over-the-shoulder composition from a single source frame. The shift is
    capped inside `frame_subject` so the crop centre can never leave the head.
    """
    if layout == framing_contract.PANEL:
        return framing_contract.frame_panel(
            subject, frame_w, frame_h, aspect, look_dir=look_dir)
    return framing_contract.frame_single(
        subject, frame_w, frame_h, aspect, look_dir=look_dir)


def _wide43_rect(frame_w: int, frame_h: int,
                 faces: Optional[Sequence[Box]] = None,
                 subject: Optional[Box] = None) -> Rect:
    """The 4:3 'show everyone' crop — anchored on a PERSON, never on the gap.

    This used to centre on `union_box(*faces)`. For two people with space
    between them the union's centre IS the empty space between them, which is
    how clip 4 shipped a crop of the aisle with a shoulder at each edge (I6,
    plan §3.3). Now the anchor is the bound `subject` when one is known, and the
    crop only slides far enough to pull the others in — it never slides so far
    that the subject leaves the middle of the frame.
    """
    crop_h = min(float(frame_h), float(frame_w) / WIDE_ASPECT)
    crop_w = crop_h * WIDE_ASPECT
    real = [f for f in (faces or ()) if f is not None]
    if subject is None and real:
        # No bound speaker: the biggest face is the best available stand-in.
        subject = max(real, key=lambda b: b[2] * b[3])
    if subject is None:
        return ((frame_w - crop_w) / 2.0, (frame_h - crop_h) / 2.0,
                crop_w, crop_h)

    head = framing_contract.head_box(subject, frame_w, frame_h)
    anchor_x = head[0] + head[2] / 2.0
    anchor_y = head[1] + head[3] / 2.0
    x = anchor_x - crop_w / 2.0
    y = anchor_y - crop_h / 2.0

    if real:
        heads = [framing_contract.head_box(f, frame_w, frame_h) for f in real]
        ux0 = min(h[0] for h in heads)
        ux1 = max(h[0] + h[2] for h in heads)
        uy0 = min(h[1] for h in heads)
        uy1 = max(h[1] + h[3] for h in heads)
        if (ux1 - ux0) <= crop_w and (uy1 - uy0) <= crop_h:
            # Everyone fits: showing everyone IS the job of the 4:3 wide, so
            # containment wins and the crop centres on the group.
            x = (ux0 + ux1) / 2.0 - crop_w / 2.0
            y = (uy0 + uy1) / 2.0 - crop_h / 2.0
        else:
            # They do NOT all fit. This is the case that produced clip 4's
            # crop of the aisle: centring the union here centres the gap. Stay
            # anchored on the subject and only slide within the slack that
            # keeps their head in the middle band (I6).
            band = crop_w * framing_contract.CENTRE_BAND / 2.0
            want = (ux0 + ux1) / 2.0
            x = max(anchor_x - band, min(want, anchor_x + band)) - crop_w / 2.0

    x = max(0.0, min(x, frame_w - crop_w))
    y = max(0.0, min(y, frame_h - crop_h))
    return (x, y, crop_w, crop_h)


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


def attention_shifted_crop(crop: Rect, subject: Box, attention_x: float,
                           frame_w: int, frame_h: int) -> Rect:
    """Shift a crop horizontally toward the attention centre, never losing
    containment or the target aspect.

    `crop` is the base composition (subject contained, aspect-correct).
    Containment fixes the crop's x to [sx + sw - cw, sx] (the subject's left
    edge must not move past the crop's left edge, and its right edge must not
    move past the crop's right edge); the frame fixes it to [0, frame_w - cw].
    The crop centre is moved toward `attention_x` (normalized 0-1) within the
    overlap of those two ranges, so a reaction beside the speaker pulls the
    frame toward where a human eye actually looks while the speaker stays
    fully in shot. No feasible shift (e.g. the crop already spans the frame)
    returns the base crop unchanged.

    Vertical placement is deliberately left untouched: the eyeline is a
    measured contract invariant (I3), and the failure mode this exists for —
    a reaction happening beside the speaker — is horizontal.

    The shift is ALSO bounded by the look-room budget, not just by
    containment. Two things now move the crop horizontally — the look-room
    bias applied in `frame_subject`, and this — and containment alone is a
    loose enough constraint that saliency could drag the crop until its centre
    sat off the subject's head entirely (invariant I6), or simply cancel the
    look room. Capping both adjustments with the same budget keeps them
    composable instead of competing.
    """
    cx, cy, cw, ch = crop
    sx, sy, sw, sh = subject
    lo = max(0.0, sx + sw - cw)                  # subject right edge inside
    hi = min(float(frame_w - cw), sx)            # subject left edge inside
    if hi <= lo:
        return crop
    budget = framing_contract.LOOK_ROOM * cw
    lo = max(lo, cx - budget)
    hi = min(hi, cx + budget)
    if hi <= lo:
        return crop
    target = attention_x * frame_w - cw / 2.0
    new_x = min(max(target, lo), hi)
    return new_x, cy, cw, ch


# ---------------------------------------------------------------------------
# Layout decision: one crop, a two-shot, or a split screen
# ---------------------------------------------------------------------------

LAYOUT_SINGLE = "single"
LAYOUT_TWO_SHOT = "two_shot"
LAYOUT_SPLIT = "split"
LAYOUT_VSPLIT = "vsplit"
LAYOUT_WIDE = "wide"

# The owner-approved "show everyone" wide: a 4:3 crop of the source,
# letterboxed into the 9:16 frame (fit-width, black bars top/bottom). Used
# for no-subject / reaction / "both people are relevant" moments — never as
# the default framing.
WIDE_ASPECT = 4.0 / 3.0

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


# ---------------------------------------------------------------------------
# Validation: fail loud rather than render something wrong
# ---------------------------------------------------------------------------

class CompositionError(ValueError):
    """A planned composition violates a hard guarantee.

    Raised rather than repaired on purpose. The framing bugs this rebuild
    exists to fix all shipped silently — a crop that cut a person in half
    still produced a playable file, so nothing surfaced until somebody
    watched it. Consistent with the Phase 5 no-silent-fallback rule: a
    violation here should stop the job, not quietly degrade it.
    """


@dataclass
class ComposedShot:
    """A shot with its composition resolved, ready for the render stage."""
    start: float
    end: float
    layout: str
    crop: Optional[Rect] = None
    subjects: List[Box] = None
    # Which spine track each entry of `subjects` belongs to (planner order).
    # The vsplit tracker uses this to fetch per-frame face boxes at render
    # time; single/two shots leave it None (their crops are static).
    track_ids: List[int] = None
    # VSPLIT only: (top_panel_crop, bottom_panel_crop), each a contained,
    # aspect-correct crop of its participant for the whole segment.
    panels: Optional[Tuple[Rect, Rect]] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def _soft(bucket: List[str], where: str, found: Sequence[str]) -> None:
    """Collect contract violations that depend on the source, not on us."""
    for f in found:
        if f.startswith("I2") or f.startswith("I0"):
            continue      # those are hard failures, reported separately
        bucket.append(f"{where}: {f}")


def validate_composition(shots: Sequence[ComposedShot], frame_w: int, frame_h: int,
                         aspect: float = VERTICAL_9_16,
                         min_shot_seconds: float = 1.8,
                         aspect_tolerance: float = 0.02) -> None:
    """Assert every hard guarantee, or raise listing ALL violations.

    Reports every problem at once rather than the first — a plan with four
    bad shots should take one run to diagnose, not four.

    Checked per shot:
      * every subject is fully inside the crop, at every shot (containment
        is the "half a person" property; average-case is not enough)
      * the crop lies inside the frame
      * the crop matches the target aspect
      * the shot is long enough to be worth a cut

    SPLIT layouts are exempt from crop containment: their subjects live in
    separate panels by construction, so a single containing rect is not a
    meaningful constraint on them.
    """
    problems: List[str] = []
    problems_soft: List[str] = []

    for i, shot in enumerate(shots):
        where = f"shot {i} [{shot.start:.2f}-{shot.end:.2f}s]"

        if shot.duration < min_shot_seconds - 1e-6:
            problems.append(
                f"{where}: duration {shot.duration:.2f}s < min {min_shot_seconds}s"
            )

        if shot.layout in (LAYOUT_SPLIT, LAYOUT_VSPLIT):
            # NO LONGER EXEMPT. `LAYOUT_SPLIT` used to `continue` outright and
            # `LAYOUT_VSPLIT` only checked contains(panel, face_box) — which a
            # panel whose top edge bisects the subject's hair satisfies, because
            # the detector box excludes hair. That exemption is why every clip
            # in the failing batch rendered without an error (plan §3.2).
            if not shot.panels or len(shot.panels) != 2:
                problems.append(f"{where}: split has no two panels")
                continue
            for j, (panel, subject) in enumerate(zip(shot.panels, shot.subjects or [])):
                if subject is None:
                    continue
                if not contains(panel, subject):
                    problems.append(
                        f"{where}: split panel {j} does not contain its subject "
                        f"{tuple(round(v) for v in subject)} in "
                        f"{tuple(round(v) for v in panel)}")
                room = framing_contract.headroom_frac(panel, subject,
                                                      frame_w, frame_h)
                if room < framing_contract.HEADROOM_MIN - 1e-6 and panel[1] > 0.5:
                    problems.append(
                        f"{where}: split panel {j} cuts the head — headroom "
                        f"{room:+.3f} of panel height, needs "
                        f"{framing_contract.HEADROOM_MIN}")
            _soft(problems_soft, where, framing_contract.check_panels(
                list(shot.panels), list(shot.subjects or []), frame_w, frame_h))
            continue

        if shot.crop is None:
            problems.append(f"{where}: layout {shot.layout!r} has no crop rect")
            continue

        x, y, w, h = shot.crop
        if w <= 0 or h <= 0:
            problems.append(f"{where}: degenerate crop {shot.crop}")
            continue

        if x < -0.5 or y < -0.5 or x + w > frame_w + 0.5 or y + h > frame_h + 0.5:
            problems.append(
                f"{where}: crop {tuple(round(v) for v in shot.crop)} "
                f"escapes frame {frame_w}x{frame_h}"
            )

        if shot.layout == LAYOUT_WIDE:
            # WIDE's 4:3 crop is deliberately a different aspect than the
            # 9:16 output (it letterboxes), so the aspect check does not
            # apply — but every subject must still fit inside it.
            _union = union_box(*(shot.subjects or []))
            if _union is not None and not contains(shot.crop, _union):
                problems.append(
                    f"{where}: wide crop "
                    f"{tuple(round(v) for v in shot.crop)} does not contain "
                    f"the subjects {tuple(round(v) for v in _union)}")
            continue

        actual = w / h
        if abs(actual - aspect) > aspect_tolerance:
            problems.append(
                f"{where}: aspect {actual:.4f} != target {aspect:.4f}"
            )

        for j, subject in enumerate(shot.subjects or []):
            if subject is None:
                continue
            if not contains(shot.crop, subject):
                problems.append(
                    f"{where}: subject {j} {tuple(round(v) for v in subject)} "
                    f"not contained in crop {tuple(round(v) for v in shot.crop)}"
                )

        # The head-cut check, on the layout that actually holds a person.
        _bound = next((s for s in (shot.subjects or []) if s is not None), None)
        if _bound is not None and shot.layout in (LAYOUT_SINGLE, LAYOUT_TWO_SHOT):
            room = framing_contract.headroom_frac(shot.crop, _bound,
                                                  frame_w, frame_h)
            if room < framing_contract.HEADROOM_MIN - 1e-6 and shot.crop[1] > 0.5:
                problems.append(
                    f"{where}: crop cuts the head — headroom {room:+.3f} of "
                    f"crop height, needs {framing_contract.HEADROOM_MIN}")
            _soft(problems_soft, where, framing_contract.check(
                shot.crop, list(shot.subjects or []), framing_contract.SINGLE,
                frame_w, frame_h))

    if problems_soft:
        # I3/I4/I5/I6 depend on what the SOURCE can deliver — a group wide with
        # 0.09-height faces breaches I4 no matter how the crop is placed (the
        # resolution budget, plan §5.7). Killing the job over that would punish
        # the clip for the camera's position, so these are loud by default and
        # fatal only on request.
        _report = "\n  ".join(problems_soft)
        if os.environ.get("FRAMING_STRICT", "0") == "1":
            raise CompositionError(
                f"{len(problems_soft)} framing-contract violation(s) "
                f"(FRAMING_STRICT=1):\n  {_report}")
        print(f"⚠️  {len(problems_soft)} framing-contract violation(s) — "
              f"rendering anyway (set FRAMING_STRICT=1 to fail):\n  {_report}")

    if problems:
        raise CompositionError(
            f"{len(problems)} composition violation(s):\n  "
            + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------
# Phase 5: clip analysis, static composition, and rendering
# ---------------------------------------------------------------------------

def _duration_and_size(video_path: str) -> Tuple[float, float, int, int]:
    """Read the source facts needed by the planner without importing main."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for v3 reframing: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if width <= 0 or height <= 0 or frames <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    return frames / fps, fps, width, height


def _directive_dicts(directives) -> List[dict]:
    """Accept both FocusDirective models and the planner's plain dict contract."""
    result = []
    for directive in directives or []:
        if isinstance(directive, dict):
            value = dict(directive)
        elif hasattr(directive, "model_dump"):
            value = directive.model_dump()
        else:
            value = {name: getattr(directive, name) for name in
                     ("start", "end", "x_position", "reason")
                     if hasattr(directive, name)}
        if {"start", "end", "x_position", "reason"}.issubset(value):
            result.append(value)
    return result


def _nearest_box(track: dict, timestamp: float) -> Optional[Box]:
    frames, boxes = track.get("frames") or [], track.get("boxes") or []
    if not frames or not boxes:
        return None
    index = min(range(min(len(frames), len(boxes))),
                key=lambda i: abs(float(frames[i]) - timestamp))
    return tuple(float(v) for v in boxes[index])


def _track_nearest_x(tracks: Dict[int, dict], x_norm: Optional[float],
                     frame_w: int, tolerance: float = 0.15) -> Optional[int]:
    """Track whose median box centre is closest to a normalized x position.

    Used when the scene-context layer names the clip's key subject
    (`primary_subject_x`) but speaker fusion produced no confident binding —
    the camera holds whoever the director said the clip is about, instead of
    whoever happened to be detected the most. Returns None when nothing is
    within `tolerance` (or no x was given), so callers keep their existing
    fallback.
    """
    if x_norm is None:
        return None
    best_id, best_dist = None, tolerance
    for track_id, track in tracks.items():
        boxes = track.get("boxes") or []
        if not boxes:
            continue
        med = np.median(np.asarray(boxes, dtype=np.float64), axis=0)
        cx = (med[0] + med[2] / 2.0) / float(frame_w)
        dist = abs(cx - x_norm)
        if dist < best_dist:
            best_dist, best_id = dist, track_id
    return best_id


def _clip_and_filter_box(box: Optional[Box], frame_w: int,
                         frame_h: int) -> Optional[Box]:
    """Clip a face box to the frame and drop detections that aren't a usable
    face. The ASR-first identity path can hand a shot a track whose box at
    that moment is half off-frame, a crowd false positive, or a body-sized
    detection — a union of those is uncontainable and fails composition
    ('subject (-48, 81, 2030, 741) not contained in crop'). Only the
    visible, plausible part is kept, so the crop can always contain it."""
    if box is None:
        return None
    x, y, w, h = (float(v) for v in box)
    if w <= 0 or h <= 0:
        return None
    vx0, vy0 = max(0.0, x), max(0.0, y)
    vx1, vy1 = min(float(frame_w), x + w), min(float(frame_h), y + h)
    vis_w, vis_h = vx1 - vx0, vy1 - vy0
    if vis_w <= 0 or vis_h <= 0:
        return None
    # Mostly off-frame (a face 70%+ outside the shot) is noise, not a face.
    if vis_w * vis_h < 0.3 * w * h:
        return None
    # Implausibly large "faces" (body/false detections in crowd shots).
    if w > 0.8 * frame_w or h > 0.95 * frame_h:
        return None
    # A real face is roughly as tall as it is wide, or taller. The show's
    # placards ("CHEATER", "Red Flag") are wide rectangles held at chest
    # height, and they repeatedly won the frame in clip 1.
    # NB: plausibility is judged on the ORIGINAL box, not the clipped one —
    # clipping a real face at the frame edge legitimately changes its aspect,
    # and rejecting it for that would throw away exactly the edge-of-frame
    # detections this function exists to rescue.
    ratio = w / max(h, 1e-6)
    if not (MIN_FACE_ASPECT <= ratio <= MAX_FACE_ASPECT):
        return None
    # Area: a "face" covering more than a tenth of the frame is a torso, a
    # placard, or the back of somebody's head filling the lens — clip 4 held
    # one of those for four seconds as if it were the subject.
    if w * h > MAX_FACE_AREA_FRAC * frame_w * frame_h:
        return None
    # ...and one too small to frame is not a subject either. Composing on it
    # would demand a crop far below the blur floor, so it can only ever
    # produce a mushy upscale (plan §5.7).
    if h < MIN_FACE_HEIGHT_FRAC * frame_h:
        return None
    return (vx0, vy0, vis_w, vis_h)


#: A track detected in fewer than this fraction of a shot's sampled instants
#: is not present enough to hold a tight single (plan §5.2).
MIN_TRACK_PRESENCE = 0.70


def _track_boxes_for_shot(shot, spine_tracks: Dict[int, dict],
                          frame_w: int, frame_h: int) -> List[Box]:
    """Per-subject box for the whole shot, as a ROBUST envelope of that
    subject's face boxes over [start, end].

    Two deliberate departures from the previous versions:

    * Not the planner's MEDIAN box (one instant). A subject who leans, turns or
      steps within a shot leaves a crop built from their midpoint, and gets
      their head cut — the original "only half his head" failure.
    * Not the raw UNION either. A union is a max over every sample, so one bad
      detection (a placard, a passer-by picked up by the same track) inflates
      the box permanently, and an inflated box drags the crop off the person
      and pushes the composition toward the 4:3 wide. This takes the ~p90
      envelope instead: wide enough to hold normal movement, immune to a
      single outlier.

    A track present in fewer than ``MIN_TRACK_PRESENCE`` of the sampled
    instants returns nothing. Holding a tight crop on somebody who is only
    on screen half the shot is how clip 4 ended up parked on the back of a
    head — the composition falls back to a wider layout instead.
    """
    from shot_planner import crop_rect_for_track

    boxes = []
    for track_id in shot.track_ids:
        track = spine_tracks.get(track_id)
        seen, sampled = [], 0
        if track:
            t = float(shot.start)
            while t <= float(shot.end) + 1e-6:
                sampled += 1
                b = _nearest_box(track, t)
                if b is not None:
                    b = _clip_and_filter_box(b, frame_w, frame_h)
                    if b is not None:
                        seen.append(b)
                t += 0.5

        envelope = None
        if seen and (sampled <= 1 or len(seen) / sampled >= MIN_TRACK_PRESENCE):
            envelope = _robust_envelope(seen)
        elif seen:
            # Present, but not enough to compose on. Say so by omission.
            continue

        if envelope is None:
            # No per-frame boxes sampled (sparse track) — fall back to the
            # planner's median, the previous behaviour.
            box = crop_rect_for_track(spine_tracks, track_id, shot.start, shot.end)
            if box is not None:
                box = _clip_and_filter_box(box, frame_w, frame_h)
                envelope = tuple(float(v) for v in box) if box is not None else None
        if envelope is not None:
            boxes.append(tuple(float(v) for v in envelope))
    return boxes


def _robust_envelope(boxes: Sequence[Box], quantile: float = 0.90) -> Box:
    """The box covering ``quantile`` of the samples on each edge.

    Same intent as a union — hold the subject for the whole shot — without
    letting one bad frame define the crop for all of it.
    """
    if len(boxes) <= 2:
        return union_box(*boxes)

    def q(values, frac):
        v = sorted(values)
        return v[min(len(v) - 1, max(0, int(round(frac * (len(v) - 1)))))]

    x0 = q([b[0] for b in boxes], 1.0 - quantile)
    y0 = q([b[1] for b in boxes], 1.0 - quantile)
    x1 = q([b[0] + b[2] for b in boxes], quantile)
    y1 = q([b[1] + b[3] for b in boxes], quantile)
    # Never smaller than the median box — the envelope is a floor, not a crop.
    mid = boxes[len(boxes) // 2]
    x0 = min(x0, mid[0])
    y0 = min(y0, mid[1])
    x1 = max(x1, mid[0] + mid[2])
    y1 = max(y1, mid[1] + mid[3])
    return (x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))


def _speaker_shares(active_tracks: Sequence[Optional[int]], shot) -> List[float]:
    seconds = range(max(0, int(shot.start)), min(len(active_tracks), int(np.ceil(shot.end))))
    total = max(1, len(list(seconds)))
    return [sum(active_tracks[i] == track_id for i in range(max(0, int(shot.start)),
                                                              min(len(active_tracks), int(np.ceil(shot.end))))) / total
            for track_id in shot.track_ids]


def _frame_at(cap, timestamp: float):
    import cv2

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not sample source frame at {timestamp:.2f}s")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _wide_fallback_composed(shot, subjects, frame_w, frame_h,
                            track_ids=None) -> ComposedShot:
    """A LAYOUT_WIDE composition that CONTAINS the subjects: 4:3 centred on
    them when they fit, else the whole frame (fit-rendered, letterboxed).
    Used when a tight single or a split's panels cannot contain the
    subjects' movement ranges (e.g. a full-frame-wide union in a fast show)
    — never fails containment."""
    _union = union_box(*[s for s in subjects if s is not None])
    if _union is not None:
        _wide = _wide43_rect(frame_w, frame_h, subjects)
        if contains(_wide, _union):
            return ComposedShot(shot.start, shot.end, LAYOUT_WIDE, _wide,
                                subjects, track_ids=track_ids)
    return ComposedShot(shot.start, shot.end, LAYOUT_WIDE,
                        (0.0, 0.0, float(frame_w), float(frame_h)),
                        subjects, track_ids=track_ids)


def _compose_shot(shot, spine_tracks: Dict[int, dict], active_tracks,
                  saliency_map: np.ndarray, frame_w: int, frame_h: int,
                  aspect: float) -> ComposedShot:
    """Resolve ONE planned shot into an immutable composition (pure numpy).

    `saliency_map` is the UNISAL map sampled at the shot's midpoint. The
    attention decision is made here, once, and baked into the crop — the
    render loop only holds the resulting rect, so saliency can never cause
    mid-shot drift (the jitter this engine exists to remove).
    """
    from shot_planner import SHOT_REACTION, SHOT_VSPLIT

    subjects = _track_boxes_for_shot(shot, spine_tracks, frame_w, frame_h)
    if not subjects:
        # WIDE shot (4:3, letterboxed into the 9:16 frame): the planner emits
        # these on purpose for moments with no confident subject (no
        # speaker binding, a reaction, "show everyone"). The owner-approved
        # wide shows the whole scene instead of a wrong tight shot — and
        # only in those moments, never as the default framing.
        if shot.crop_rect is not None:
            # An explicitly-provided wide rect (legacy) keeps the old
            # aspect-correct fill behaviour.
            crop = tuple(float(v) for v in shot.crop_rect)
            return ComposedShot(shot.start, shot.end, LAYOUT_SINGLE, crop, [])
        return ComposedShot(shot.start, shot.end, LAYOUT_WIDE,
                            _wide43_rect(frame_w, frame_h), [])

    midpoint = (shot.start + shot.end) / 2.0
    faces = []
    # The union boxes from _track_boxes_for_shot (subject's full movement
    # range) double as the saliency-weighted face regions — consistent with
    # the crop, so the attention map agrees with what the crop contains.
    subject_by_track = dict(zip(shot.track_ids, subjects))
    for track_id, track in spine_tracks.items():
        box = subject_by_track.get(track_id)
        if box is None:
            box = _nearest_box(track, midpoint)
        if box is None:
            continue
        if track_id in shot.track_ids:
            role = ROLE_REACTOR if shot.shot_type == SHOT_REACTION else ROLE_SPEAKER
        else:
            role = ROLE_BYSTANDER
        faces.append(WeightedFace(box, role))
    attention = build_attention_map(saliency_map, faces)
    attention_x = attention_center(attention)[0]

    if shot.shot_type == SHOT_VSPLIT and len(subjects) >= 2:
        # Vertical split: two contained panels (top = subject A, bottom =
        # subject B), each crop holding that participant's full movement
        # range over the segment — face never cut. The middle band is where
        # captions sit; the renderer draws it. Each panel FRAMES the person
        # (head and shoulders), sized to fill its half of the output frame —
        # the crop aspect matches the panel box, so there is no letterboxing
        # or distortion.
        _band = float(os.environ.get("VSPLIT_BAND_FRAC", "0"))
        _panel_aspect = aspect * 2.0 / (1.0 - _band)
        panel_a = crop_rect_containing(subjects[0], frame_w, frame_h,
                                       _panel_aspect,
                                       layout=framing_contract.PANEL)
        panel_b = crop_rect_containing(subjects[1], frame_w, frame_h,
                                       _panel_aspect,
                                       layout=framing_contract.PANEL)
        if not (contains(panel_a, subjects[0])
                and contains(panel_b, subjects[1])):
            # The subjects' movement ranges span more than the panels can
            # hold (the full-frame-wide union that failed clip 3) — show
            # the wide instead of failing composition.
            return _wide_fallback_composed(
                shot, subjects, frame_w, frame_h,
                track_ids=list(shot.track_ids))
        return ComposedShot(shot.start, shot.end, LAYOUT_VSPLIT, None, subjects,
                            track_ids=list(shot.track_ids),
                            panels=(panel_a, panel_b))

    layout = decide_layout(subjects, _speaker_shares(active_tracks, shot),
                           frame_w, frame_h, aspect)
    if layout == LAYOUT_SPLIT:
        # When the camera can actually capture BOTH people in one 4:3 crop
        # (the host standing next to the guest), show the wide instead of
        # splitting — the owner's "show them together" rule. Only far-apart
        # people who can't share a crop become the vertical split.
        _union = union_box(*subjects)
        _wide = _wide43_rect(frame_w, frame_h, subjects)
        if (_union is not None
                and _wide[2] >= _union[2] - 0.5
                and _wide[3] >= _union[3] - 0.5):
            return ComposedShot(shot.start, shot.end, LAYOUT_WIDE, _wide,
                                subjects, track_ids=list(shot.track_ids))
        # The owner's split is the VERTICAL stack — the legacy side-by-side
        # (vendored pyautoflip) rendered as a bordered two-up, and captions
        # on it followed the user's normal position instead of the middle
        # split rule. A far-apart exchange therefore becomes the SAME
        # edge-to-edge VSPLIT the beat planner produces: two stacked panels
        # (9:8 each for 9:16), middle captions over the seam.
        _band = float(os.environ.get("VSPLIT_BAND_FRAC", "0"))
        _panel_aspect = aspect * 2.0 / (1.0 - _band)
        panel_a = crop_rect_containing(subjects[0], frame_w, frame_h,
                                       _panel_aspect,
                                       layout=framing_contract.PANEL)
        panel_b = crop_rect_containing(subjects[1], frame_w, frame_h,
                                       _panel_aspect,
                                       layout=framing_contract.PANEL)
        if not (contains(panel_a, subjects[0])
                and contains(panel_b, subjects[1])):
            return _wide_fallback_composed(
                shot, subjects, frame_w, frame_h,
                track_ids=list(shot.track_ids or []))
        return ComposedShot(shot.start, shot.end, LAYOUT_VSPLIT, None,
                            subjects, track_ids=list(shot.track_ids or []),
                            panels=(panel_a, panel_b))
    else:
        subject = union_box(*subjects)
        # LOOK ROOM. We cannot select an over-the-shoulder shot — the source
        # gives us one frame per moment, already picked — but biasing the crop
        # toward the person the speaker faces pulls that listener into the near
        # edge, which is the composition the reference gets from real OTS
        # coverage (plan §1.3). The shift is capped inside frame_subject so the
        # crop centre can never leave the subject's head.
        _bound = subjects[0] if subjects else subject
        _others = [b for b in subjects[1:] if b is not None]
        if not _others:
            _others = [f.box for f in faces
                       if f.role == ROLE_BYSTANDER and f.box is not subject]
        _probe = crop_rect_containing(_bound, frame_w, frame_h, aspect)
        _look = framing_contract.look_room_dir(_bound, _others, _probe[2])
        crop = crop_rect_containing(subject, frame_w, frame_h, aspect,
                                    look_dir=_look)
        # Saliency is a nudge within containment slack, never a free aim:
        # a reaction beside the speaker pulls the frame toward it, but the
        # subject containment constraint (the "half a person" fix) still
        # bounds the result, and validate_composition below re-checks it.
        crop = attention_shifted_crop(crop, subject, attention_x, frame_w, frame_h)
        crop = _enforce_min_crop(
            crop, frame_w, frame_h, aspect,
            float(os.environ.get("MIN_CROP_FRAC", "0.45")))
        # A subject whose movement union is too wide for the 9:16 crop (a
        # speaker pacing across the shot, or a loose detection — the ASR-first
        # identity path hands us these) cannot be contained by a tight shot.
        # Emit the 4:3 WIDE centred on them instead of failing composition —
        # the owner-approved "show more context" output.
        if subject is not None and not contains(crop, subject):
            _wide = _wide43_rect(frame_w, frame_h, subjects, subject=_bound)
            if contains(_wide, subject):
                return ComposedShot(shot.start, shot.end, LAYOUT_WIDE, _wide,
                                    subjects, track_ids=list(shot.track_ids))
            # Even the wide can't hold the full movement range — show the
            # whole frame (fit-rendered, letterboxed).
            return ComposedShot(
                shot.start, shot.end, LAYOUT_WIDE,
                (0.0, 0.0, float(frame_w), float(frame_h)),
                subjects, track_ids=list(shot.track_ids))
        return ComposedShot(shot.start, shot.end, layout, crop, subjects)


def compose_shots(shots, spine_tracks: Dict[int, dict], active_tracks,
                  video_path: str, frame_w: int, frame_h: int,
                  aspect: float) -> List[ComposedShot]:
    """Sample each shot once and resolve its immutable composition.

    Saliency is sampled at the midpoint of every shot and is deliberately
    part of composition rather than the render loop: changing a crop while a
    shot is running would reintroduce the jitter this engine exists to remove.
    """
    import cv2
    from vendor.pyautoflip.saliency_detector import SaliencyDetector

    detector = SaliencyDetector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot sample video for v3 composition: {video_path}")

    composed: List[ComposedShot] = []
    try:
        for shot in shots:
            midpoint = (shot.start + shot.end) / 2.0
            frame_rgb = _frame_at(cap, midpoint)
            saliency = detector.detect(frame_rgb)["saliency_map"]
            composed.append(_compose_shot(
                shot, spine_tracks, active_tracks, saliency, frame_w, frame_h, aspect))
    finally:
        cap.release()
    return composed


def _integer_crop(rect: Rect, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    """Make an ffmpeg/OpenCV-safe even crop while retaining the validated rect."""
    x, y, w, h = rect
    x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x1, y1 = min(frame_w, int(np.ceil(x + w))), min(frame_h, int(np.ceil(y + h)))
    width, height = max(2, x1 - x0), max(2, y1 - y0)
    width -= width % 2
    height -= height % 2
    x0 = min(x0, frame_w - width)
    y0 = min(y0, frame_h - height)
    return x0, y0, width, height


def _crop_resize(frame, crop: Optional[Rect], frame_w: int, frame_h: int,
                 out_w: int, out_h: int):
    """Crop a pixel rect from ``frame`` and resize to ``(out_w, out_h)``."""
    import cv2
    if crop is None:
        crop = (0.0, 0.0, float(frame_w), float(frame_h))
    x, y, w, h = _integer_crop(crop, frame_w, frame_h)
    return cv2.resize(frame[y:y + h, x:x + w], (out_w, out_h),
                      interpolation=cv2.INTER_LANCZOS4)


def _crop_resize_panel(frame, crop: Optional[Rect], frame_w: int, frame_h: int,
                       out_w: int, out_h: int, aspect: float):
    """Crop + resize one vsplit panel. Aspect-correct crops fill the panel;
    a degenerate crop (tracking fallback, or a source smaller than the panel
    aspect) is letterboxed instead of stretched, so faces never distort."""
    import cv2
    if crop is None:
        crop = (0.0, 0.0, float(frame_w), float(frame_h))
    x, y, w, h = _integer_crop(crop, frame_w, frame_h)
    if w <= 0 or h <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    actual = w / h
    # Letterbox ONLY on the genuine fallback (a full-frame crop dropped into
    # a panel — a 16:9 frame into a 9:8 panel would distort faces badly).
    # Small deviations (a smart-crop containment nudge, the min-size floor)
    # fill the panel instead: the 1-5% stretch is imperceptible, and it
    # keeps the stacked panels edge-to-edge with no visible bars/border.
    if actual > aspect * 1.10 or actual < aspect * 0.90:
        scale = min(out_w / w, out_h / h)
        resized = cv2.resize(
            frame[y:y + h, x:x + w],
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_LANCZOS4)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        ox = max(0, (out_w - resized.shape[1]) // 2)
        oy = max(0, (out_h - resized.shape[0]) // 2)
        canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        return canvas
    return cv2.resize(frame[y:y + h, x:x + w], (out_w, out_h),
                      interpolation=cv2.INTER_LANCZOS4)


def _crop_resize_fit(frame, crop: Optional[Rect], frame_w: int, frame_h: int,
                     out_w: int, out_h: int):
    """Crop + FIT (letterboxed) into the output frame — the 4:3 wide shot:
    content keeps its 4:3 shape, centered, black bars top/bottom. Never
    stretches, so faces stay undistorted."""
    import cv2
    if crop is None:
        crop = (0.0, 0.0, float(frame_w), float(frame_h))
    x, y, w, h = _integer_crop(crop, frame_w, frame_h)
    if w <= 0 or h <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    scale = min(out_w / w, out_h / h)
    cw = max(1, int(round(w * scale)))
    ch = max(1, int(round(h * scale)))
    resized = cv2.resize(frame[y:y + h, x:x + w], (cw, ch),
                         interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    ox = max(0, (out_w - cw) // 2)
    oy = max(0, (out_h - ch) // 2)
    canvas[oy:oy + ch, ox:ox + cw] = resized
    return canvas


def _aspect_tuple(aspect: float) -> Tuple[int, int]:
    """Integer (w, h) ratio for the vendored split renderer."""
    if abs(aspect - 1.0) < 1e-9:
        return (1, 1)
    if abs(aspect - VERTICAL_9_16) < 1e-9:
        return (9, 16)
    return (round(aspect * 1000), 1000)


def _regular_filtergraph(composed: Sequence[ComposedShot], frame_w: int, frame_h: int,
                         out_w: int, out_h: int) -> str:
    parts, labels = [], []
    for i, shot in enumerate(composed):
        if shot.layout == LAYOUT_SPLIT or shot.crop is None:
            raise ValueError("split shots require the Python split renderer")
        x, y, w, h = _integer_crop(shot.crop, frame_w, frame_h)
        label = f"s{i}"
        if shot.layout == LAYOUT_WIDE:
            # Wide: scale to FIT within the 9:16 frame (preserving the
            # content's own aspect — 4:3 normally, 16:9 for the whole-frame
            # fallback) and pad the rest black. Never stretches.
            parts.append(
                f"[0:v]trim=start={shot.start:.6f}:end={shot.end:.6f},"
                f"setpts=PTS-STARTPTS,crop={w}:{h}:{x}:{y},"
                f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:"
                f"flags=lanczos,"
                f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[{label}]"
            )
        else:
            parts.append(
                f"[0:v]trim=start={shot.start:.6f}:end={shot.end:.6f},"
                f"setpts=PTS-STARTPTS,crop={w}:{h}:{x}:{y},"
                f"scale={out_w}:{out_h}:flags=lanczos,"
                f"setsar=1[{label}]"
            )
        labels.append(f"[{label}]")
    if not labels:
        raise CompositionError("v3 generated an empty shot plan")
    return ";".join(parts + ["".join(labels) + f"concat=n={len(labels)}:v=1:a=0[v]"])



def _vsplit_caption_ass(transcript, clip_start, clip_end, vsplit_ranges,
                        output_path):
    """Generate ONE ASS for a clip that contains vertical splits: words that
    fall inside a split window render MIDDLE (in the split's band) while
    every other word keeps the user's chosen caption position (usually
    bottom) — "middle when it moves to split, bottom otherwise".
    Returns (ass_path, ass_filter) or None when there is nothing to caption."""
    try:
        import subtitles as _subs
        import time
        import uuid
        style = dict(_subs.AUTO_CAPTION_STYLE)
        output_dir = os.path.dirname(os.path.abspath(output_path))
        ass_path = os.path.join(
            output_dir,
            f"autosubs_vsplit_{int(time.time())}_{uuid.uuid4().hex[:8]}.ass")
        ok = _subs.generate_ass(
            transcript, clip_start, clip_end, ass_path,
            max_chars=style["max_chars"], max_duration=style["max_duration"],
            alignment=style["alignment"], fontsize=style["font_size"],
            font_name=style["font_name"], font_color=style["font_color"],
            border_color=style["border_color"],
            border_width=_subs.auto_stroke_width(style["font_size"]),
            highlight_color=style["highlight_color"],
            effect=style["effect"], base_opacity=style["base_opacity"],
            uppercase=style["uppercase"],
            margin_v=style.get("margin_v", _subs.SAFE_MARGIN_V),
            general_ranges=None,
            middle_ranges=vsplit_ranges,
            speaker_colors=style.get("speaker_colors", False),
            letter_spacing_ratio=_subs.CAPTION_LETTER_SPACING_RATIO)
        if not ok:
            return None
        return ass_path, _subs.ass_filter_string(ass_path)
    except Exception as e:
        print(f"   ⚠️ vsplit middle-caption generation failed "
              f"({type(e).__name__}: {e})")
        return None


def _render_regular(input_video: str, output_video: str, composed: Sequence[ComposedShot],
                    frame_w: int, frame_h: int, out_w: int, out_h: int,
                    ass_filter=None, captioned_output=None) -> None:
    """Render static single/two shots in one native ffmpeg pass."""
    import subprocess
    import gpu_affinity
    from ffmpeg_utils import (METADATA_SCRUB, QUALITY_FAST, gpu_decode_args,
                              video_encode_args)

    device = gpu_affinity.current_device()
    graph = _regular_filtergraph(composed, frame_w, frame_h, out_w, out_h)
    main_label = "[v]"
    cap_label = None
    if ass_filter and captioned_output:
        # The filter graph emits [v] once — mapping it to TWO outputs is the
        # "Output with label 'v' ... was already used elsewhere" failure that
        # killed clips 1 & 4 in the 04:06 run. Split it into [vmain] (clean
        # clip) and [vcap], and apply the ASS captions INSIDE the graph (a
        # -vf on a complex-graph stream is rejected: "simple and complex
        # filtering cannot be used together").
        cap_label = "[vcapass]"
        graph += f"; [v]split=2[vmain][vcap]; [vcap]{ass_filter}[vcapass]"
        main_label = "[vmain]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           *gpu_decode_args(device=device), "-i", input_video,
           "-filter_complex", graph, "-map", main_label, "-map", "0:a?",
           *video_encode_args(QUALITY_FAST, device=device),
           "-c:a", "copy", *METADATA_SCRUB,
           "-movflags", "+faststart", output_video]
    if ass_filter and captioned_output:
        cmd += caption_output_args(ass_filter, captioned_output, device=device,
                                   label=cap_label)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=1800)
    except subprocess.CalledProcessError as e:
        _err = (e.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"ffmpeg render failed for {os.path.basename(output_video)}"
            + (f": {_err[-400:]}" if _err else "")) from e


DELIVERY_MIN_WIDTH = 1080


def delivery_size(orig_w: int, orig_h: int, aspect_ratio: float):
    """Output (width, height) for a reframe of this source.

    Picks the largest crop the source allows, then upscales to
    ``DELIVERY_MIN_WIDTH`` if that crop is narrower. Both dimensions come back
    even (x264/NVENC reject odd ones). This is the one place the delivery
    size is decided for every render.
    """
    out_h = orig_h
    out_w = int(out_h * aspect_ratio)
    if out_w > orig_w:
        out_w = orig_w
        out_h = int(out_w / aspect_ratio)

    if out_w < DELIVERY_MIN_WIDTH:
        out_w = DELIVERY_MIN_WIDTH
        out_h = int(round(out_w / aspect_ratio))

    return out_w + (out_w % 2), out_h + (out_h % 2)


def caption_output_args(ass_filter: str, captioned_output: str,
                        encode_tier: str = "quality", device=None,
                        label="[v]"):
    """Second-output args for the one-pass clean+captioned render.

    With these args the same ffmpeg invocation encodes BOTH files from the
    same reframed frames: the clean clip exactly as before, and the captioned
    clip with the ass filter applied. The captioned file therefore loses one
    generation instead of two.
    """
    from ffmpeg_utils import METADATA_SCRUB, video_encode_args

    return [
        "-map", label, "-map", "0:a?",
        *video_encode_args(encode_tier, device=device),
        "-c:a", "copy", *METADATA_SCRUB,
        "-movflags", "+faststart", captioned_output,
    ]


def _burn_captions_on(source: str, output: str, ass_filter: str, device):
    """Burn ASS captions onto an already-rendered clip (the vsplit path:
    captions land per the chosen position; set CAPTION_POSITION=middle to
    place them in the vsplit band)."""
    import subprocess
    from ffmpeg_utils import METADATA_SCRUB, QUALITY_FAST, video_encode_args

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", source,
           "-vf", ass_filter,
           *video_encode_args(QUALITY_FAST, device=device),
           "-c:a", "copy", *METADATA_SCRUB,
           "-movflags", "+faststart", output]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=1800)
    except subprocess.CalledProcessError as e:
        _err = (e.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"vsplit caption burn failed: {_err[-300:]}") from e


def _render_with_splits(input_video: str, output_video: str, composed: Sequence[ComposedShot],
                        frame_w: int, frame_h: int, out_w: int, out_h: int,
                        aspect: float, tracking: Optional[Dict[int, dict]] = None,
                        spine_tracks: Optional[Dict[int, dict]] = None) -> None:
    """Render split shots frame-by-frame; regular shots remain static crops.

    This intentionally handles only the split case in Python. The normal path
    stays an ffmpeg filtergraph, while the vendored split renderer receives the
    BGR arrays it was designed for. When `tracking` maps a shot index to
    ``{"track_ids": [...], "trackers": [...]}``, that shot's vsplit panels
    are per-frame tracked crops (smart_crop.PanelTracker) instead of the
    static union-box panels; spine_tracks supplies the per-frame face boxes.
    """
    import cv2
    import subprocess
    import tempfile
    import gpu_affinity
    from ffmpeg_utils import METADATA_SCRUB, QUALITY_FAST, video_encode_args
    from vendor.pyautoflip import render_split_screen_from_centers

    _panel_aspect = aspect * 2.0 / (
        1.0 - float(os.environ.get("VSPLIT_BAND_FRAC", "0")))
    _cut_threshold = float(os.environ.get("VSPLIT_CUT_THRESHOLD", "25.0"))
    duration, fps, _, _ = _duration_and_size(input_video)
    cap = cv2.VideoCapture(input_video)
    fd, silent = tempfile.mkstemp(prefix="openshorts_v3_", suffix=".mp4")
    os.close(fd)
    writer = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open temporary v3 split renderer")
    try:
        index = 0
        aspect_tuple = _aspect_tuple(aspect)
        shot_index = 0
        prev_small = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = index / fps
            while (shot_index < len(composed) - 1
                   and timestamp >= composed[shot_index + 1].start):
                shot_index += 1
            shot = composed[shot_index]
            # Cheap scene-cut signal for the tracked panels: mean abs diff of
            # a 64x36 gray thumbnail. A hard cut lights this up far above
            # talking-head motion, so a cut snaps the crop instead of
            # gliding over it. Computed for every frame so the comparison is
            # always against the true previous frame.
            if tracking:
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                   (64, 36))
                scene_cut = (prev_small is not None and float(np.mean(
                    cv2.absdiff(prev_small, small))) > _cut_threshold)
                prev_small = small
            else:
                scene_cut = False
            if shot.layout == LAYOUT_VSPLIT and shot.panels:
                # Panels stack edge-to-edge (no separating bar) by default;
                # VSPLIT_BAND_FRAC>0 re-adds a band for captions. With no
                # band the middle captions simply overlay the seam.
                _band = float(os.environ.get("VSPLIT_BAND_FRAC", "0"))
                band_h = max(0, int(round(out_h * _band)))
                panel_h = max(1, int(round((out_h - band_h) / 2)))
                total_h = 2 * panel_h + band_h
                top_y = max(0, (out_h - total_h) // 2)
                entry = (tracking or {}).get(shot_index)
                if entry:
                    boxes = []
                    for tid in entry["track_ids"]:
                        track = (spine_tracks or {}).get(tid)
                        boxes.append(_nearest_box(track, timestamp) if track else None)
                    top_crop = entry["trackers"][0].step(boxes[0], scene_cut)
                    bottom_crop = entry["trackers"][1].step(boxes[1], scene_cut)
                else:
                    top_crop, bottom_crop = shot.panels
                top = _crop_resize_panel(frame, top_crop, frame_w, frame_h,
                                         out_w, panel_h, _panel_aspect)
                bottom = _crop_resize_panel(frame, bottom_crop, frame_w, frame_h,
                                            out_w, panel_h, _panel_aspect)
                canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                canvas[top_y:top_y + panel_h] = top
                if band_h > 0:
                    canvas[top_y + panel_h:top_y + panel_h + band_h] = (22, 22, 22)
                canvas[top_y + panel_h + band_h:top_y + total_h] = bottom
                rendered = canvas
            elif shot.layout == LAYOUT_SPLIT:
                centers = split_centers(shot.subjects, frame_w, frame_h,
                                        aspect_tuple=aspect_tuple)
                canvas = render_split_screen_from_centers(frame, centers, aspect_tuple)
                rendered = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            elif shot.layout == LAYOUT_WIDE:
                rendered = _crop_resize_fit(frame, shot.crop, frame_w, frame_h,
                                            out_w, out_h)
            else:
                rendered = _crop_resize(frame, shot.crop, frame_w, frame_h,
                                        out_w, out_h)
            writer.write(rendered)
            index += 1
    finally:
        cap.release()
        writer.release()
    # The Python split renderer writes an mp4v intermediate (cv2.VideoWriter
    # has no nvenc backend), so the final mux re-encodes it to the normal
    # h264/nvenc delivery codec on the assigned worker GPU. One extra
    # generation vs the regular path, which encodes once from the source.
    device = gpu_affinity.current_device()
    try:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", input_video,
                 "-map", "0:v:0", "-map", "1:a?",
                 *video_encode_args(QUALITY_FAST, device=device),
                 "-c:a", "copy", *METADATA_SCRUB,
                 "-movflags", "+faststart", output_video], check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1800)
        except subprocess.CalledProcessError as e:
            _err = (e.stderr or b"").decode("utf-8", "replace").strip()
            raise RuntimeError(
                f"ffmpeg split mux failed for {os.path.basename(output_video)}"
                + (f": {_err[-400:]}" if _err else "")) from e
    finally:
        if os.path.exists(silent):
            os.remove(silent)


def _clip_relative_words(segments, clip_start: float,
                         duration: float) -> List[dict]:
    """ASR words for this clip, re-based to clip-relative seconds.

    The planner works in 0..duration; `segments` are absolute source times.
    Words outside the clip are dropped rather than clamped — a boundary must
    never snap to a gap that is not in the clip.
    """
    out: List[dict] = []
    for seg in segments or []:
        for w in (seg.get("words") or []):
            s, e = w.get("start"), w.get("end")
            if s is None or e is None:
                continue
            rs, re_ = float(s) - clip_start, float(e) - clip_start
            if re_ <= 0 or rs >= duration:
                continue
            out.append({"word": w.get("word", ""), "start": rs, "end": re_})
    out.sort(key=lambda w: w["start"])
    return out


def _sentence_end_times(words: Sequence[dict]) -> List[float]:
    """End times of words that close a sentence, by punctuation."""
    return [float(w["end"]) for w in words or []
            if str(w.get("word", "")).strip().endswith((".", "!", "?"))]


def render(input_video, final_output_video, aspect_ratio,
           transcript=None, clip_start=0.0, clip_end=None,
           focus_directives=None, primary_subject_x=None,
           ass_filter=None, captioned_output=None):
    """Full v3 reframe. Errors intentionally propagate; there is no v3→v2 fallback."""
    import face_spine
    import gpu_affinity
    import speaker_fusion
    import shot_planner

    print("   🚀 Reframe engine v3 (planned static shots)")
    import time as _time
    _t0 = _time.time()
    duration, fps, frame_w, frame_h = _duration_and_size(input_video)
    effective_end = clip_end if clip_end is not None else clip_start + duration
    # Thread the worker's assigned GPU (gpu_affinity.current_device()) into
    # every device-capable stage. On hosts/threads with no assignment this is
    # None and every stage keeps its existing default behaviour.
    worker_device = gpu_affinity.current_device()
    tracks = face_spine.build_face_spine(input_video, device=worker_device)
    print(f"   ↳ face spine: {_time.time() - _t0:.0f}s ({len(tracks)} track(s))")
    if not tracks:
        raise RuntimeError("Reframe v3 found no face tracks; refusing to silently fall back")

    segments = (transcript or {}).get("segments", [])
    # Director v2 (ASR-first): when Gemini confirms the speaker->face map for
    # this clip, the diarized transcript decides WHO and the map decides
    # WHICH FACE — no per-second LR-ASD voting. Fail-open: on any error the
    # LR-ASD fusion below runs exactly as before.
    _identity_map = None
    if any(seg.get("speaker") for seg in segments):
        try:
            import identity_confirm
            _identity_map = identity_confirm.confirm_clip_identities(
                input_video, tracks, segments, clip_start, effective_end)
        except Exception as exc:
            print(f"   ⚠️ Identity confirmation failed "
                  f"({type(exc).__name__}: {exc})")
    if _identity_map:
        _speaker_ps = speaker_fusion.per_second_speaker_label(
            segments, clip_start, effective_end)
        active = speaker_fusion.active_from_identity_map(
            _speaker_ps, _identity_map)
        print(f"   ↳ ASR-first binding ({len(_identity_map)} confirmed "
              "speaker(s)) — LR-ASD skipped")
    else:
        # LR-ASD runs only when the identity map did NOT land: it is the
        # fallback fusion's evidence. Director v2 trusts the Gemini map alone
        # when it lands — the owner's chosen tradeoff (speed over the
        # cross-check). The cross-check (option 1) can be built later on top
        # of this same branch.
        asd_boxes, asd_margins = [], None
        try:
            import asd_worker
            if asd_worker.available():
                # LR-ASD's per-frame face-candidate pass defaults to ctx_id=0
                # (GPU 0), which piled every worker's detection onto one card.
                # Pin it to THIS worker's GPU so both cards carry the load.
                _ctx = face_spine._resolve_ctx_id(worker_device)
                _detect = lambda frame, _ctx=_ctx: face_spine.detect_faces_per_frame(
                    frame, ctx_id=_ctx)
                _asd = asd_worker.score_clip(
                    input_video, _detect, device=worker_device)
                asd_boxes = _asd.get("per_second_box") or []
                # The model already computes how far the winning face led the
                # runner-up each second; carrying it through is free and lets
                # the fusion throw out the seconds where it was a coin flip.
                asd_margins = _asd.get("per_second_margin")
                print(f"   ↳ LR-ASD: {_time.time() - _t0:.0f}s")
        except Exception as exc:
            print(f"   ⚠️ LR-ASD unavailable for v3 "
                  f"({type(exc).__name__}: {exc})")
        _, active = speaker_fusion.fuse_speaker_tracks(
            asd_boxes, tracks, segments, clip_start, effective_end,
            smooth_window=int(os.environ.get("SPEAKER_SMOOTH_WINDOW", "3")),
            # `decisive_margin` stays at 0 (every second votes) — the
            # experiment's 0.10 gate degraded a session and nothing since has
            # re-tested it. Re-binding is now ON by default (4s of sustained,
            # decisive contradiction inside an 8s window): one binding per clip
            # killed the per-frame flip-flop, but it also meant a binding that
            # came out WRONG held the camera on the wrong person for the entire
            # clip, with no way out. That trade is worse.
            asd_per_second_margin=asd_margins,
            decisive_margin=float(os.environ.get("ASD_DECISIVE_MARGIN", "0")),
            rebind_seconds=int(os.environ.get("SPEAKER_REBIND_SECONDS", "4")),
            rebind_window=int(os.environ.get("SPEAKER_REBIND_WINDOW", "8")),
            # The no-guess rule: a diarized speaker we cannot map to a face
            # goes WIDE rather than being pointed at the nearest torso.
            unmapped_policy=os.environ.get(
                "SPEAKER_UNMAPPED_POLICY",
                speaker_fusion.UNMAPPED_POLICY_WIDE).strip().lower(),
            # ...but never to the point of an all-wide clip: past this share
            # of labelled seconds the bindings have failed, and a guess beats
            # nothing while the fusion gets fixed.
            max_wide=float(os.environ.get("SPEAKER_UNMAPPED_MAX_WIDE", "0.5")))
    if _identity_map is None and not any(track is not None for track in active):
        # NOTHING bound for the whole clip. This used to pick a track anyway —
        # the scene director's key subject, else whichever face was on screen
        # longest — and hold it for the entire clip. When that guess was wrong
        # it was wrong for every second, which is the "wrong person the whole
        # way through" failure.
        #
        # Under the no-guess rule we only take that gamble when the director
        # actually NAMED someone. Otherwise the honest output is the 4:3 wide:
        # we do not know who is talking, so show the room.
        _policy = os.environ.get(
            "SPEAKER_UNMAPPED_POLICY",
            speaker_fusion.UNMAPPED_POLICY_WIDE).strip().lower()
        fallback = _track_nearest_x(tracks, primary_subject_x, frame_w)
        if fallback is None and _policy != speaker_fusion.UNMAPPED_POLICY_WIDE:
            fallback = max(tracks,
                           key=lambda track_id: len(tracks[track_id].get("frames") or []))
        if fallback is not None:
            active = [fallback] * max(1, int(np.ceil(duration)))
            print("   ⚠️ No confident active-speaker binding; "
                  "holding the director's named subject")
        else:
            print("   ⚠️ No confident active-speaker binding and no named "
                  "subject; showing the wide rather than guessing")

    planned = shot_planner.plan_shots(
        active, tracks, total_duration=duration,
        max_shot_seconds=float(os.environ.get("MAX_SHOT_SECONDS", "14")))
    planned = shot_planner.insert_reaction_shots(
        planned, _directive_dicts(focus_directives), tracks, frame_w)
    # Conversation framing (owner-approved, fully local): vertical split on
    # genuine two-person exchanges — the "single -> split -> single" edit.
    # CONVERSATION_FRAMING=0 reverts to the pre-conversation pipeline;
    # SPLIT=0 keeps the speaker-binding fix but skips the beat planner.
    _conv = os.environ.get("CONVERSATION_FRAMING", "1").strip().lower()
    _split = os.environ.get("SPLIT", "1").strip().lower()
    # Speech timing, in CLIP-relative seconds (segments are absolute).
    _clip_words = _clip_relative_words(segments, clip_start, duration)
    _gaps = shot_planner.word_gaps(_clip_words)
    _sentence_ends = _sentence_end_times(_clip_words)
    if _conv not in ("0", "false", "no", "off") and _split not in ("0", "false", "no", "off"):
        planned = shot_planner.plan_conversation_beats(
            planned, active, tracks,
            min_exchange_s=float(os.environ.get("SPLIT_MIN_EXCHANGE_S", "2.5")),
            min_span_s=float(os.environ.get("SPLIT_MIN_SPAN_S", "2.5")),
            max_split_span=float(os.environ.get("SPLIT_MAX_SPAN_S", "10")),
            frame_w=frame_w, frame_h=frame_h, aspect=aspect_ratio,
            sentence_ends=_sentence_ends)
    # The cut follows the sentence (I9). Every failing clip opened mid-sentence
    # and two died mid-thought; the clip's own in/out are sentence-anchored
    # upstream, but the boundaries BETWEEN shots landed wherever the speaker
    # track happened to change. Nudge each onto the nearest silence.
    if _gaps:
        _before = [s.start for s in planned]
        planned = shot_planner.snap_shots_to_speech(planned, _gaps)
        _moved = sum(1 for a, s in zip(_before, planned) if abs(a - s.start) > 1e-6)
        if _moved:
            print(f"   ↳ snapped {_moved} shot boundary(ies) onto word gaps")
    print(f"   ↳ shot planning: {_time.time() - _t0:.0f}s")
    composed = compose_shots(planned, tracks, active, input_video, frame_w, frame_h, aspect_ratio)
    print(f"   ↳ saliency + composition: {_time.time() - _t0:.0f}s")
    validate_composition(composed, frame_w, frame_h, aspect_ratio)

    # Path instrumentation (same contract as the old v2 engine): when
    # REFRAME_DUMP_PATH is set, write the per-frame crop rects (source
    # pixels) so eval/ can measure jitter numerically. v3's shots are
    # static, so the motion metric on this dump is the jitter measurement.
    dump_dir = os.environ.get("REFRAME_DUMP_PATH", "").strip()
    if dump_dir:
        try:
            n_frames = max(1, int(round(duration * fps)))
            rects = []
            for i in range(n_frames):
                t = i / fps
                shot = next((s for s in composed if s.start <= t < s.end), composed[-1])
                rects.append(list(shot.crop) if shot.crop is not None
                             else [0.0, 0.0, float(frame_w), float(frame_h)])
            os.makedirs(dump_dir, exist_ok=True)
            tag = os.path.splitext(os.path.basename(final_output_video))[0]
            np.save(os.path.join(dump_dir, f"{tag}_rects.npy"),
                    np.asarray(rects, dtype=float))
            print(f"   📐 REFRAME_DUMP_PATH: wrote {tag}_rects.npy ({n_frames} rects)")
        except Exception as exc:
            print(f"   ⚠️ REFRAME_DUMP_PATH failed ({exc}) — continuing")

    out_w, out_h = delivery_size(frame_w, frame_h, aspect_ratio)
    print(f"   ↳ encoding to {out_w}x{out_h} (GPU NVENC)...")
    if any(shot.layout in (LAYOUT_SPLIT, LAYOUT_VSPLIT) for shot in composed):
        # The Python renderer handles regular + split + vsplit frames. For a
        # vsplit the middle band is reserved for captions; burn them in a
        # second pass onto the captioned output (the old hard refusal meant
        # captioned clips could never contain a split at all).
        # Per-panel smart-crop tracking (AutoFlip-style glide) is on by
        # default; VSPLIT_TRACK=0 keeps the static union-box panels.
        _tracking: Dict[int, dict] = {}
        _track = os.environ.get("VSPLIT_TRACK", "1").strip().lower()
        if _track not in ("0", "false", "no", "off"):
            from smart_crop import PanelTracker
            _panel_aspect = aspect_ratio * 2.0 / (
                1.0 - float(os.environ.get("VSPLIT_BAND_FRAC", "0")))
            _track_knobs = dict(
                # VSPLIT_HEADROOM / VSPLIT_SIDE_MARGIN / VSPLIT_VERT_MARGIN are
                # GONE. They described a framing method that cut heads by
                # construction (see smart_crop's module docstring); the tracker
                # now derives its crop from framing_contract, exactly like the
                # static panels, so a split really is "normal framing, then
                # stacked". Four fewer knobs to be wrong.
                min_height_frac=float(os.environ.get(
                    "VSPLIT_PANEL_MIN_FRAC",
                    str(framing_contract.DEFAULT_MIN_CROP_FRAC))),
                dead_zone=float(os.environ.get("VSPLIT_DEADZONE", "0.02")),
                smooth=float(os.environ.get("VSPLIT_SMOOTH", "0.12")),
                smooth_zoom=float(os.environ.get("VSPLIT_SMOOTH_ZOOM", "0.06")),
                full_width_area_frac=float(
                    os.environ.get("VSPLIT_FULLWIDTH_FRAC", "0.60")),
                lost_hold_frames=int(
                    os.environ.get("VSPLIT_LOST_HOLD_FRAMES", "30")),
            )
            for _idx, _s in enumerate(composed):
                if _s.layout == LAYOUT_VSPLIT and _s.panels and _s.track_ids:
                    _tracking[_idx] = {
                        "track_ids": list(_s.track_ids),
                        "trackers": [
                            PanelTracker(frame_w, frame_h, _panel_aspect,
                                         **_track_knobs).reset(_s.panels[0]),
                            PanelTracker(frame_w, frame_h, _panel_aspect,
                                         **_track_knobs).reset(_s.panels[1]),
                        ],
                    }
        _render_with_splits(input_video, final_output_video, composed, frame_w, frame_h,
                            out_w, out_h, aspect_ratio,
                            tracking=_tracking or None, spine_tracks=tracks)
        if ass_filter and captioned_output:
            # Vertical splits ALWAYS burn captions in the middle band, no
            # matter which caption position the user picked for normal clips;
            # the rest of this clip keeps their chosen position. One ASS
            # mixes both via middle_ranges (see _vsplit_caption_ass).
            _split_captions = None
            if any(shot.layout in (LAYOUT_SPLIT, LAYOUT_VSPLIT)
                   for shot in composed):
                _split_captions = _vsplit_caption_ass(
                    transcript, clip_start, effective_end,
                    [(s.start, s.end) for s in composed
                     if s.layout in (LAYOUT_SPLIT, LAYOUT_VSPLIT)],
                    final_output_video)
            _burn_captions_on(final_output_video, captioned_output,
                              _split_captions[1] if _split_captions else ass_filter,
                              gpu_affinity.current_device())
    else:
        _render_regular(input_video, final_output_video, composed, frame_w, frame_h, out_w, out_h,
                        ass_filter=ass_filter, captioned_output=captioned_output)
    print(f"   ✅ v3 clip saved to {final_output_video} ({len(composed)} static shot(s))")
    return True, [(shot.start, shot.end) for shot in composed]
