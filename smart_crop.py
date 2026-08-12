"""Per-panel smart crop for vertical splits (AutoFlip-style tracked glide).

The static v3 engine bakes ONE crop per shot — that stays the default for
single/two shots. A vertical split shows one person per panel for seconds at
a time, and the approved framing method wants a subtle camera that glides to
keep that person framed as they move. This module implements that per-panel
tracker with the exact guardrails that make it safe:

- head-anchored crops (15-20% headroom above the hair — hair never clips)
- dead zone (hysteresis): micro head-bobs do not move the camera at all
- exponential smoothing (lerp): the camera glides instead of teleporting
- containment safety valve: if the face ever leaves the crop despite the
  smoothing (fast move / cut), the crop snaps back — the face is never cut
- boundary clamping: the crop can never pan past the source frame
- hard-cut reset: a scene change snaps instantly instead of gliding
- fit-to-width fallback: if tracking loses the face or the box fills the
  frame (wide shot), the panel relaxes to a full-width view

`VSPLIT_TRACK=0` disables this and keeps the static union-box panels.
"""

from __future__ import annotations

from typing import Optional, Tuple

Box = Tuple[float, float, float, float]        # (x, y, w, h) in pixels
Rect = Tuple[float, float, float, float]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _contains(crop: Rect, subject: Box, tolerance: float = 0.5) -> bool:
    cx, cy, cw, ch = crop
    sx, sy, sw, sh = subject
    return (sx >= cx - tolerance and sy >= cy - tolerance
            and sx + sw <= cx + cw + tolerance
            and sy + sh <= cy + ch + tolerance)


def _recontain(rect: Rect, box: Box, frame_w: int, frame_h: int) -> Rect:
    """Project a crop onto the set of crops that fully contain `box`.

    Gliding toward a tighter head-anchored crop passes through intermediate
    crops that would clip the face (hair briefly above the crop top). Instead
    of a hard snap, nudge and (boundedly) enlarge the crop so the face is
    ALWAYS fully inside — the never-cut guarantee, without breaking the
    smoothness of the camera move.
    """
    x, y, w, h = rect
    bx, by, bw, bh = box
    if w < bw:
        w = min(bw, float(frame_w))
        x = _clamp(x, 0.0, frame_w - w)
    if h < bh:
        h = min(bh, float(frame_h))
        y = _clamp(y, 0.0, frame_h - h)
    if bx < x:
        x = bx
    if bx + bw > x + w:
        x = bx + bw - w
    if by < y:
        y = by
    if by + bh > y + h:
        y = by + bh - h
    x = _clamp(x, 0.0, max(0.0, frame_w - w))
    y = _clamp(y, 0.0, max(0.0, frame_h - h))
    return (x, y, w, h)


class PanelTracker:
    """Tracks ONE panel's crop window frame by frame.

    The crop is a rect in source-pixel space with the panel's aspect.
    `step()` consumes the subject's face box for the current frame (None when
    the face was not detected) and a scene-cut flag, and returns the crop
    rect to render for that frame.
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        aspect: float,
        headroom: float = 0.18,
        side_margin: float = 0.15,
        vert_margin: float = 0.10,
        dead_zone: float = 0.02,
        smooth: float = 0.12,
        smooth_zoom: float = 0.06,
        full_width_area_frac: float = 0.60,
        lost_hold_frames: int = 30,
        crop: Optional[Rect] = None,
    ):
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        self.aspect = float(aspect)
        self.headroom = float(headroom)
        self.side_margin = float(side_margin)
        self.vert_margin = float(vert_margin)
        self.dead_zone = float(dead_zone)
        self.smooth = float(smooth)
        self.smooth_zoom = float(smooth_zoom)
        self.full_width_area_frac = float(full_width_area_frac)
        self.lost_hold_frames = int(lost_hold_frames)
        self.crop = tuple(float(v) for v in crop) if crop else None
        self.lost_frames = 0

    def reset(self, initial: Rect) -> "PanelTracker":
        """Seed the tracker from a composition rect (the static panel crop),
        so the glide starts exactly where the static plan would."""
        self.crop = tuple(float(v) for v in initial)
        self.lost_frames = 0
        return self

    def _full_frame_crop(self) -> Rect:
        return (0.0, 0.0, float(self.frame_w), float(self.frame_h))

    def _target_crop(self, box: Optional[Box]) -> Optional[Rect]:
        if box is None:
            return None
        sx, sy, sw, sh = box
        # Wide-shot fallback: a box that fills the frame (or a person too
        # close to contain) relaxes to a full-width view instead of fighting
        # it — the panel renderer letterboxes the result.
        if sw * sh > self.full_width_area_frac * self.frame_w * self.frame_h:
            return self._full_frame_crop()
        # Head-anchored size: headroom above the hair + side/vertical
        # margins keep the face (and chin) clear of the panel edges and the
        # split boundary.
        need_h = sh * (1.0 + 2.0 * self.vert_margin + self.headroom)
        need_w = sw * (1.0 + 2.0 * self.side_margin)
        crop_h = max(need_h, need_w / self.aspect)
        crop_w = crop_h * self.aspect
        # Clamp to the frame, preserving aspect.
        if crop_w > self.frame_w:
            crop_w = float(self.frame_w)
            crop_h = crop_w / self.aspect
        if crop_h > self.frame_h:
            crop_h = float(self.frame_h)
            crop_w = crop_h * self.aspect
        x = sx + sw / 2.0 - crop_w / 2.0
        y = sy - sh * self.headroom
        x = _clamp(x, 0.0, self.frame_w - crop_w)
        y = _clamp(y, 0.0, self.frame_h - crop_h)
        return (x, y, crop_w, crop_h)

    def step(self, box: Optional[Box], scene_cut: bool = False) -> Rect:
        """Advance one frame and return the crop rect to render."""
        if self.crop is None:
            target = self._target_crop(box)
            self.crop = target if target is not None else self._full_frame_crop()
        target = self._target_crop(box)
        if target is None:
            # Face lost: hold the current framing (dead camera). If it has
            # been gone for a while, relax to the full-width fallback so the
            # viewer still sees the context.
            self.lost_frames += 1
            if self.lost_frames > self.lost_hold_frames:
                self.crop = self._full_frame_crop()
            return self.crop
        self.lost_frames = 0

        # Hard cut: snap instantly — gliding over a cut looks like a botched
        # transition, not a camera move.
        if scene_cut:
            self.crop = target
            return self.crop

        # Containment safety valve: smoothing must never let the face leave
        # the crop (the "never cut the head" guarantee). This fires only if
        # the invariant was broken upstream (seeded crop not containing its
        # subject); the per-step _recontain projection below keeps it intact.
        if box is not None and not _contains(self.crop, box):
            self.crop = target
            return self.crop

        cur_x, cur_y, cur_w, cur_h = self.crop
        tgt_x, tgt_y, tgt_w, tgt_h = target
        cur_cx, cur_cy = cur_x + cur_w / 2.0, cur_y + cur_h / 2.0
        tgt_cx, tgt_cy = tgt_x + tgt_w / 2.0, tgt_y + tgt_h / 2.0

        # Dead zone (hysteresis): tiny movements inside the zone do not PAN
        # the camera at all — kills micro-jitter from head bobs. Zoom is a
        # separate control (dynamic zoom scaling) and keeps easing toward
        # the target size, so a genuinely smaller box still zooms.
        dz_x = self.dead_zone * cur_w
        dz_y = self.dead_zone * cur_h
        pan_held = (abs(tgt_cx - cur_cx) < dz_x
                    and abs(tgt_cy - cur_cy) < dz_y)

        # Exponential smoothing (low-pass): a cinematic, delayed glide that
        # catches up gracefully. Size eases slower than position.
        new_w = _clamp(cur_w + (tgt_w - cur_w) * self.smooth_zoom,
                       1.0, float(self.frame_w))
        new_h = _clamp(cur_h + (tgt_h - cur_h) * self.smooth_zoom,
                       1.0, float(self.frame_h))
        if pan_held:
            new_cx, new_cy = cur_cx, cur_cy
        else:
            new_cx = cur_cx + (tgt_cx - cur_cx) * self.smooth
            new_cy = cur_cy + (tgt_cy - cur_cy) * self.smooth
        # Boundary clamping: never pan past the source frame edge.
        new_x = _clamp(new_cx - new_w / 2.0, 0.0, self.frame_w - new_w)
        new_y = _clamp(new_cy - new_h / 2.0, 0.0, self.frame_h - new_h)
        # Containment projection: keep the face fully inside while gliding,
        # so shrinking toward the head-anchored target never clips hair.
        new_x, new_y, new_w, new_h = _recontain(
            (new_x, new_y, new_w, new_h), box, self.frame_w, self.frame_h)
        self.crop = (new_x, new_y, new_w, new_h)
        return self.crop
