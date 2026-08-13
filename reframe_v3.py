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
import os

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

# Fraction of the crop HEIGHT above the subject's centre. 0.36 is the value
# tuned on this footage during the v1/v2 era (CAMERA_HEAD_Y) and kept here:
# it leaves headroom above and body below rather than centring the face
# vertically, which reads as a snapshot.
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

    Vertical placement is deliberately left untouched: `DEFAULT_HEAD_Y` is a
    footage-tuned aesthetic, and the failure mode this exists for — a
    reaction happening beside the speaker — is horizontal.
    """
    cx, cy, cw, ch = crop
    sx, sy, sw, sh = subject
    lo = max(0.0, sx + sw - cw)                  # subject right edge inside
    hi = min(float(frame_w - cw), sx)            # subject left edge inside
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


def validate_composition(shots: Sequence[ComposedShot], frame_w: int, frame_h: int,
                         aspect: float = VERTICAL_9_16,
                         min_shot_seconds: float = 1.2,
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

    for i, shot in enumerate(shots):
        where = f"shot {i} [{shot.start:.2f}-{shot.end:.2f}s]"

        if shot.duration < min_shot_seconds - 1e-6:
            problems.append(
                f"{where}: duration {shot.duration:.2f}s < min {min_shot_seconds}s"
            )

        if shot.layout == LAYOUT_SPLIT:
            continue

        if shot.layout == LAYOUT_VSPLIT:
            # Each participant lives in its own contained panel — verify the
            # panels exist and contain their subjects (the face-never-cut
            # guarantee for the split), then move on (no single containing
            # rect applies to a split by construction).
            if not shot.panels or len(shot.panels) != 2:
                problems.append(f"{where}: vsplit has no two panels")
                continue
            for j, (panel, subject) in enumerate(zip(shot.panels, shot.subjects or [])):
                if subject is not None and not contains(panel, subject):
                    problems.append(
                        f"{where}: vsplit panel {j} does not contain its subject "
                        f"{tuple(round(v) for v in subject)} in "
                        f"{tuple(round(v) for v in panel)}")
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


def _track_boxes_for_shot(shot, spine_tracks: Dict[int, dict]) -> List[Box]:
    """Per-subject box = the UNION of that subject's face boxes across the
    WHOLE shot (AutoFlip-style), so the static crop contains the subject for
    the entire duration.

    The old version used the planner's MEDIAN box — one instant. A subject who
    moves (leans, turns, steps) within a shot then exits the static crop and
    gets their head cut (the "only half his head" failure from the 04:06
    run). Unioning the sampled boxes over [start, end] makes containment a
    property of the whole shot, not of its midpoint.
    """
    from shot_planner import crop_rect_for_track

    boxes = []
    for track_id in shot.track_ids:
        track = spine_tracks.get(track_id)
        union = None
        if track:
            t = float(shot.start)
            while t <= float(shot.end) + 1e-6:
                b = _nearest_box(track, t)
                if b is not None:
                    union = b if union is None else union_box(union, b)
                t += 0.5
        if union is None:
            # No per-frame boxes sampled (sparse track) — fall back to the
            # planner's median, the previous behavior.
            box = crop_rect_for_track(spine_tracks, track_id, shot.start, shot.end)
            if box is not None:
                union = tuple(float(v) for v in box)
        if union is not None:
            boxes.append(tuple(float(v) for v in union))
    return boxes


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

    subjects = _track_boxes_for_shot(shot, spine_tracks)
    if not subjects:
        # WIDE shot: the planner emits these on purpose for leading seconds
        # with no speaker binding (hold_fill's "nothing to hold onto yet"
        # case) or when a default wide rect is supplied. Hold the full frame
        # as a neutral composition rather than failing the whole clip on a
        # shot type the planner is designed to produce.
        if shot.crop_rect is not None:
            crop = tuple(float(v) for v in shot.crop_rect)
        else:
            crop = crop_rect_containing(
                (0.0, 0.0, float(frame_w), float(frame_h)),
                frame_w, frame_h, aspect)
        return ComposedShot(shot.start, shot.end, LAYOUT_SINGLE, crop, [])

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
        panel_a = crop_rect_containing(subjects[0], frame_w, frame_h, _panel_aspect)
        panel_b = crop_rect_containing(subjects[1], frame_w, frame_h, _panel_aspect)
        return ComposedShot(shot.start, shot.end, LAYOUT_VSPLIT, None, subjects,
                            track_ids=list(shot.track_ids),
                            panels=(panel_a, panel_b))

    layout = decide_layout(subjects, _speaker_shares(active_tracks, shot),
                           frame_w, frame_h, aspect)
    if layout == LAYOUT_SPLIT:
        crop = None
    else:
        subject = union_box(*subjects)
        crop = crop_rect_containing(subject, frame_w, frame_h, aspect)
        # Saliency is a nudge within containment slack, never a free aim:
        # a reaction beside the speaker pulls the frame toward it, but the
        # subject containment constraint (the "half a person" fix) still
        # bounds the result, and validate_composition below re-checks it.
        crop = attention_shifted_crop(crop, subject, attention_x, frame_w, frame_h)
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
        parts.append(
            f"[0:v]trim=start={shot.start:.6f}:end={shot.end:.6f},setpts=PTS-STARTPTS,"
            f"crop={w}:{h}:{x}:{y},scale={out_w}:{out_h}:flags=lanczos,"
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
            # runner-up each second; carrying it through is free and lets the
            # fusion throw out the seconds where it was a coin flip.
            asd_margins = _asd.get("per_second_margin")
            print(f"   ↳ LR-ASD: {_time.time() - _t0:.0f}s")
    except Exception as exc:
        print(f"   ⚠️ LR-ASD unavailable for v3 ({type(exc).__name__}: {exc})")
    segments = (transcript or {}).get("segments", [])
    _, active = speaker_fusion.fuse_speaker_tracks(
        asd_boxes, tracks, segments, clip_start, effective_end,
        smooth_window=int(os.environ.get("SPEAKER_SMOOTH_WINDOW", "3")),
        asd_per_second_margin=asd_margins,
        decisive_margin=float(os.environ.get("ASD_DECISIVE_MARGIN", "0.10")),
        rebind_seconds=int(os.environ.get("SPEAKER_REBIND_SECONDS", "4")),
        rebind_window=int(os.environ.get("SPEAKER_REBIND_WINDOW", "8")))
    if not any(track is not None for track in active):
        # A transcript/ASD gap must not turn a known person into an untracked
        # full-frame crop. Prefer the scene context's key subject when it
        # names one; otherwise hold the most continuously observed identity.
        fallback = _track_nearest_x(tracks, primary_subject_x, frame_w)
        if fallback is None:
            fallback = max(tracks, key=lambda track_id: len(tracks[track_id].get("frames") or []))
        active = [fallback] * max(1, int(np.ceil(duration)))
        print("   ⚠️ No confident active-speaker binding; holding a single fallback track")

    planned = shot_planner.plan_shots(active, tracks, total_duration=duration)
    planned = shot_planner.insert_reaction_shots(
        planned, _directive_dicts(focus_directives), tracks, frame_w)
    # Conversation framing (owner-approved, fully local): vertical split on
    # genuine two-person exchanges — the "single -> split -> single" edit.
    # CONVERSATION_FRAMING=0 reverts to the pre-conversation pipeline;
    # SPLIT=0 keeps the speaker-binding fix but skips the beat planner.
    _conv = os.environ.get("CONVERSATION_FRAMING", "1").strip().lower()
    _split = os.environ.get("SPLIT", "1").strip().lower()
    if _conv not in ("0", "false", "no", "off") and _split not in ("0", "false", "no", "off"):
        planned = shot_planner.plan_conversation_beats(
            planned, active, tracks,
            min_exchange_s=float(os.environ.get("SPLIT_MIN_EXCHANGE_S", "2.5")),
            min_span_s=float(os.environ.get("SPLIT_MIN_SPAN_S", "2.0")),
            frame_w=frame_w, frame_h=frame_h, aspect=aspect_ratio)
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
                # Panel framing parity: the tracked panels use the SAME
                # breathing room as a static single shot (crop_rect_containing
                # above), so a split is "normal framing, then stacked" rather
                # than two zoomed-in faces.
                headroom=float(os.environ.get("VSPLIT_HEADROOM", "0.18")),
                side_margin=float(os.environ.get(
                    "VSPLIT_SIDE_MARGIN", str(DEFAULT_SIDE_MARGIN))),
                vert_margin=float(os.environ.get(
                    "VSPLIT_VERT_MARGIN", str(DEFAULT_VERT_MARGIN))),
                min_height_frac=float(os.environ.get(
                    "VSPLIT_PANEL_MIN_FRAC", "0.25")),
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
            if any(shot.layout == LAYOUT_VSPLIT for shot in composed):
                _split_captions = _vsplit_caption_ass(
                    transcript, clip_start, effective_end,
                    [(s.start, s.end) for s in composed
                     if s.layout == LAYOUT_VSPLIT],
                    final_output_video)
            _burn_captions_on(final_output_video, captioned_output,
                              _split_captions[1] if _split_captions else ass_filter,
                              gpu_affinity.current_device())
    else:
        _render_regular(input_video, final_output_video, composed, frame_w, frame_h, out_w, out_h,
                        ass_filter=ass_filter, captioned_output=captioned_output)
    print(f"   ✅ v3 clip saved to {final_output_video} ({len(composed)} static shot(s))")
    return True, [(shot.start, shot.end) for shot in composed]
