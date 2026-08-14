"""Per-panel smart crop for vertical splits (AutoFlip-style tracked glide).

The static v3 engine bakes ONE crop per shot — that stays the default for
single/two shots. A vertical split shows one person per panel for seconds at
a time, and the approved framing method wants a subtle camera that glides to
keep that person framed as they move. This module implements that per-panel
tracker with the exact guardrails that make it safe:

- contract-framed crops (`framing_contract.frame_panel`): size from the target
  face fraction, position from the target eyeline, so hair never clips
- dead zone (hysteresis): micro head-bobs do not move the camera at all
- exponential smoothing (lerp): the camera glides instead of teleporting
- containment safety valve: if the head ever leaves the crop despite the
  smoothing (fast move / cut), the crop snaps back
- boundary clamping: the crop can never pan past the source frame
- hard-cut reset: a scene change snaps instantly instead of gliding
- fit-to-width fallback: if tracking loses the face or the box fills the
  frame (wide shot), the panel relaxes to a full-width view

`VSPLIT_TRACK=0` disables this and keeps the static contract panels.

WHAT CHANGED, AND WHY
---------------------
This tracker used to compute its crop the same broken way the static engine
did, only spelled differently::

    need_h = sh * (1 + 2*vert_margin + headroom)     # size, from margins
    y      = sy - sh * headroom                      # position, from the FACE BOX

Two problems, both fatal. First, the size and the anchor were derived from
different references — `crop_h` could be floored to 45% of the source height
while `y` stayed pinned `0.18 * face_height` above the box, so the actual
headroom depended on which term happened to win. Second, `headroom = 0.18` is
measured against the DETECTOR box, and the detector box stops at the eyebrows:
real hair sits about `0.35 * face_height` higher. `0.18 < 0.35`, so the anchor
was inside the subject's hair every single time.

Now both come from `framing_contract`, which derives size and position from the
same two measured ratios. See PLAN_FRAMING_CONTRACT.md §3.1 and §5.3.
"""

from __future__ import annotations

from typing import Optional, Tuple

import framing_contract

Box = Tuple[float, float, float, float]        # (x, y, w, h) in pixels
Rect = Tuple[float, float, float, float]

#: Floor on a panel crop's height, as a fraction of the source frame. A small
#: or distant face box would otherwise produce a crop only a couple of hundred
#: pixels tall — technically "correctly framed" and unwatchably blurry once
#: upscaled. The floor only ever pulls the camera WIDER, never tighter.
DEFAULT_MIN_HEIGHT_FRAC = framing_contract.DEFAULT_MIN_CROP_FRAC


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _contains(crop: Rect, subject: Box, tolerance: float = 0.5) -> bool:
    cx, cy, cw, ch = crop
    sx, sy, sw, sh = subject
    return (sx >= cx - tolerance and sy >= cy - tolerance
            and sx + sw <= cx + cw + tolerance
            and sy + sh <= cy + ch + tolerance)


def _recontain(rect: Rect, box: Box, frame_w: int, frame_h: int,
               aspect: Optional[float] = None) -> Rect:
    """Project a crop onto the set of ASPECT-CORRECT crops containing `box`.

    Gliding toward a tighter crop passes through intermediate crops that would
    clip the head. Instead of a hard snap, nudge and (boundedly) enlarge the
    crop so the head is ALWAYS fully inside — the never-cut guarantee, without
    breaking the smoothness of the camera move.

    **The aspect argument is not optional in spirit.** This function used to
    grow `w` and `h` independently, which silently drifted the panel away from
    its target ratio. `reframe_v3._crop_resize_panel` letterboxes anything more
    than 10% off-aspect, so that drift is exactly the black bar between panels
    reported in clip 2. Growing to the aspect-correct size instead keeps the
    stacked panels edge to edge.
    """
    x, y, w, h = rect
    bx, by, bw, bh = box

    need_w, need_h = max(w, bw), max(h, bh)
    if aspect:
        need_h = max(need_h, need_w / aspect)
        need_w = need_h * aspect
        if need_w > frame_w:
            need_w = float(frame_w)
            need_h = need_w / aspect
        if need_h > frame_h:
            need_h = float(frame_h)
            need_w = need_h * aspect
    else:
        need_w, need_h = min(need_w, float(frame_w)), min(need_h, float(frame_h))

    # Grow around the current centre, then slide the minimum distance that
    # brings the box fully inside.
    cx, cy = x + w / 2.0, y + h / 2.0
    x, y, w, h = cx - need_w / 2.0, cy - need_h / 2.0, need_w, need_h
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
        dead_zone: float = 0.02,
        smooth: float = 0.12,
        smooth_zoom: float = 0.06,
        full_width_area_frac: float = 0.60,
        lost_hold_frames: int = 30,
        min_height_frac: float = DEFAULT_MIN_HEIGHT_FRAC,
        crop: Optional[Rect] = None,
    ):
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        self.aspect = float(aspect)
        self.dead_zone = float(dead_zone)
        self.smooth = float(smooth)
        self.smooth_zoom = float(smooth_zoom)
        self.full_width_area_frac = float(full_width_area_frac)
        self.lost_hold_frames = int(lost_hold_frames)
        self.min_height_frac = float(min_height_frac)
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
        if sw <= 0 or sh <= 0:
            return None
        # Wide-shot fallback: a box that fills the frame (or a person too
        # close to contain) relaxes to a full-width view instead of fighting
        # it — the panel renderer letterboxes the result.
        if sw * sh > self.full_width_area_frac * self.frame_w * self.frame_h:
            return self._full_frame_crop()
        # THE crop. Size and position from the same two contract ratios, so
        # the headroom is a property of the geometry rather than of whichever
        # margin term happened to bind.
        return framing_contract.frame_panel(
            box, self.frame_w, self.frame_h, self.aspect,
            min_frac=self.min_height_frac)

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

        # Containment safety valve, now measured on the HEAD box rather than
        # the detector box. A crop that contains the face rectangle can still
        # shear the hair, which is the whole defect this rebuild removes.
        head = framing_contract.head_box(box, self.frame_w, self.frame_h)
        if not _contains(self.crop, head):
            self.crop = target
            return self.crop

        cur_x, cur_y, cur_w, cur_h = self.crop
        tgt_x, tgt_y, tgt_w, tgt_h = target
        cur_cx, cur_cy = cur_x + cur_w / 2.0, cur_y + cur_h / 2.0
        tgt_cx, tgt_cy = tgt_x + tgt_w / 2.0, tgt_y + tgt_h / 2.0

        # Dead zone (hysteresis): tiny movements inside the zone do not PAN
        # the camera at all — kills micro-jitter from head bobs. Zoom is a
        # separate control and keeps easing toward the target size, so a
        # genuinely smaller box still zooms.
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
        # Containment projection: keep the HEAD fully inside while gliding,
        # aspect preserved so the panel never letterboxes mid-move.
        self.crop = _recontain(
            (new_x, new_y, new_w, new_h), head,
            self.frame_w, self.frame_h, self.aspect)
        return self.crop
