"""Reframe engine v2: analyze in Python, render natively in ffmpeg.

v1 decodes every frame at full resolution in OpenCV, crops/resizes in numpy
and pipes raw frames back into ffmpeg. v2 splits that into:

  1. ANALYSIS — one ffmpeg-decoded pass at <=640px feeding the same detectors
     and the same SmoothedCameraman/SpeakerTracker state machines as v1, so
     the resulting camera trajectory (crop x per frame) is equivalent.
  2. RENDER — ONE ffmpeg process for the whole clip: decode -> dynamic crop
     (sendcmd) -> scale -> overlay onto a blurred background -> encode
     natively. A single consistent crop SHAPE (see UNIFIED_CROP_RATIO) is
     used for every frame, repositioned to whoever's relevant, rather than
     switching between a narrow single-subject crop and a separate wide
     group layout — no per-scene segment/concat step anymore.

No raw-frame piping, no second full-res decode, one less intermediate encode.
Callers must treat any exception as "fall back to the v1 loop".

Pure helpers (sendcmd generation, scene slicing) have no heavy imports so
they stay unit-testable in CI.
"""
import os
import copy
import math
import subprocess
import tempfile
import time
import threading
import face_id

from ffmpeg_utils import (video_encode_args, gpu_decode_args, QUALITY_FAST,
                          METADATA_SCRUB)

ANALYSIS_MAX_WIDTH = 640


# Short-form platforms (TikTok / Reels / Shorts) expect a 1080-wide vertical
# upload; anything smaller is treated as low quality and re-encoded from the
# already-soft source. The crop region is whatever the source height allows, so
# a 720p input yields a 406x720 crop — we scale that up to the delivery floor
# rather than shipping sub-HD. Sources that already exceed it are left alone
# (never downscale quality the user supplied).
DELIVERY_MIN_WIDTH = 1080


# --- pure helpers (CI-testable) --------------------------------------------

def delivery_size(orig_w, orig_h, aspect_ratio):
    """Output (width, height) for a reframe of this source.

    Picks the largest crop the source allows, then upscales to
    ``DELIVERY_MIN_WIDTH`` if that crop is narrower. Both dimensions come back
    even (x264/NVENC reject odd ones).
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


def _sendcmd_timestamp(frame_index, fps):
    """sendcmd time for a crop change meant to land ON ``frame_index``.

    Empirically verified against ffmpeg (1-aug-2026): a command stamped at
    the 4-decimal rounded value of ``frame_index/fps`` (e.g. "0.0667" for
    frame 2 of a 30fps clip, whose exact pts is 0.0666667) lands ONE FRAME
    LATE — sendcmd's time comparison misses the intended frame and fires on
    the next one, which is exactly the sub-second wrong-crop flash around
    hard cuts. The fix floors to 6 decimals and backs off a further
    microsecond so the command time is always strictly before the intended
    frame's pts (and always after the previous frame's, since 1/fps is
    orders of magnitude larger than the 1us back-off) — the change can
    never leak onto the frame after the cut.
    """
    t = max(0.0, frame_index / fps - 1e-6)
    return f"{t:.6f}"


def dedupe_sendcmd_lines(xs, fps, target="crop@c", scale=1):
    """sendcmd lines driving crop@c, deduped to change-points.

    Entries are either a bare x value (pan-only, emits ``crop@c x {x}`` —
    kept for compatibility with the old x-only renderer) or a
    ``(x, y, w, h)`` tuple (the unified 3:4 crop's full rect — emits w/h/x/y
    commands together so a zoom-in changes crop size AND position at the
    same timestamp, while a pure pan still collapses to one line per change
    point since w/h/y are unchanged). Timestamps are relative to the clip.

    ``scale`` (supersampling, 6-aug-2026): the renderer may upscale the
    source by CROP_SUPERSAMPLE and crop in that 2x space, so crop positions
    that were quantized to a 1/supersample source grid become whole pixels
    again here. scale=1 reproduces the historical behavior exactly.
    """
    lines = []
    prev = None
    for i, entry in enumerate(xs):
        if entry == prev:
            continue
        stamp = _sendcmd_timestamp(i, fps)
        if isinstance(entry, (tuple, list)):
            x, y, w, h = entry
            if scale != 1:
                x, y, w, h = (int(round(v * scale)) for v in (x, y, w, h))
            # sendcmd grammar: each ';'-separated command carries its own
            # timestamp ("time target param value;") — repeating the stamp is
            # how multiple crop@c params change at the SAME time.
            lines.append(
                f"{stamp} {target} w {w}; {stamp} {target} h {h}; "
                f"{stamp} {target} x {x}; {stamp} {target} y {y};")
        else:
            value = entry if scale == 1 else int(round(entry * scale))
            lines.append(f"{stamp} {target} x {value};")
        prev = entry
    return lines


# Split cells use a FIXED-size crop window; the zoom-in comes from a
# downstream scale, never from resizing the crop. ffmpeg trac #10984: the
# crop filter silently stops providing frames when its HEIGHT changes via
# sendcmd (position changes are fine), which froze the reaction cells
# mid-split — the overlay kept running with the last live cell frame held,
# so the output read as a hard content stutter while the audio continued.
#
# 990x880 (>half the 1920-wide source) was wide enough to cram in a whole
# group standing near each other — ground-truthed 31-jul-2026: a t≈22.5s
# split crammed host+contestant into one cell instead of showing one clear
# person per cell, and a t≈5.5s split's top cell was empty because the
# window was wide enough to include the gap next to the actual subject.
# 720x640 keeps the fixed-per-render-pass shape (still 9:8, so the trac
# #10984 freeze fix — the window must never resize during render — is
# untouched) but tight enough that a cell reads as ONE person, not a group.
SPLIT_CELL_MAX_W = 720
SPLIT_CELL_MAX_H = 640


def split_cell_aspect(out_w, out_h):
    """Aspect one split half must have: the output width over HALF its height.

    This is derived, never constant. It used to be hardcoded 9:8, which is
    right only for a 9:16 canvas (half of 1080x1920 is 1080x960 = 1.125). On
    the 3:4 output actually in use, half the canvas is 1080x720 = 1.5, so a
    9:8 cell was scaled to 1080x960 and then centre-cropped to 1080x720 —
    silently throwing away 25% of each person's height. That is why split
    screen showed only a middle band and defeated its own purpose.
    """
    half_h = max(1, int(out_h) // 2)
    return float(out_w) / float(half_h)


def split_cell_size(orig_w, orig_h, aspect=None):
    """Fixed cell crop size for one split-screen half, shaped to ``aspect``.

    Sized as large as the source allows within the caps, then trimmed to the
    exact target aspect so the downstream scale fills the half-canvas with no
    cover-crop. The window never changes size during a render (ffmpeg trac
    #10984 — a crop whose height changes via sendcmd stops producing frames),
    so this is computed once per clip.
    """
    if aspect is None:
        aspect = SPLIT_CELL_ASPECT
    w = min(orig_w, SPLIT_CELL_MAX_W)
    h = min(orig_h, SPLIT_CELL_MAX_H)
    # Trim whichever dimension overshoots the target aspect.
    if w / h > aspect:
        w = h * aspect
    else:
        h = w / aspect
    w = min(int(w), orig_w) & ~1
    h = min(int(h), orig_h) & ~1
    return max(w, 2), max(h, 2)


def cell_initial_xy(rect, cell_w, cell_h, orig_w, orig_h):
    """Clamped (x, y) placing the fixed-size cell window on the rect center.

    The window is centered on the tracked subject's box center and clamped
    so it stays inside the source frame (a box near the edge would otherwise
    push the window off-frame).
    """
    x, y, w, h = rect
    cx = x + w / 2.0
    cy = y + h / 2.0
    max_x = max(0, orig_w - cell_w)
    max_y = max(0, orig_h - cell_h)
    return (min(max(int(cx - cell_w / 2.0), 0), max_x),
            min(max(int(cy - cell_h / 2.0), 0), max_y))


def cell_scale_target(rect, cell_w, cell_h, out_w, half_h):
    """(sw, sh) the zoom scale should target for this tracked rect.

    A rect that fills the whole cell maps to the cell output size exactly
    (no zoom); a smaller rect targets a LARGER scale, so the fixed window is
    magnified and the trailing centered crop shows a tighter view of the
    subject. Targets are floored at the output size (the crop needs enough
    input) and rounded even (x264/NVENC reject odd dimensions).
    """
    _x, _y, w, h = rect
    sw = int(round(out_w * cell_w / max(w, 1)))
    sh = int(round(half_h * cell_h / max(h, 1)))
    return max(out_w, sw + (sw % 2)), max(half_h, sh + (sh % 2))


# Where the subject's head sits vertically inside the rendered split cell.
# 0.36 leaves natural headroom above and shoulders below at typical zooms.
SPLIT_CELL_HEAD_Y = float(os.environ.get("SPLIT_CELL_HEAD_Y", "0.36"))


def cell_frame_xy(rect, cell_xy, cell_w, cell_h, scale_wh, out_w, half_h):
    """(x, y) for the FINAL crop of a split cell, anchored on the subject.

    This exists because a centered final crop cuts heads off. The chain is
    fixed-size window -> magnify -> crop back to the cell's output size, and
    that last crop used to default to the centre of the magnified image. The
    window is centred on the subject only when it CAN be: cell_initial_xy
    clamps it to stay inside the source frame, and in a 16:9 frame heads sit
    high, so the clamp regularly pins the window at y=0. The subject then sits
    ABOVE the window's centre, and a centred final crop removes exactly the
    part with the head in it — the more it zooms, the more it takes.

    So the final crop is positioned explicitly: convert the subject's head
    point into magnified-cell coordinates and place the crop so the head lands
    SPLIT_CELL_HEAD_Y down the output cell, clamped to the magnified bounds.
    """
    wx, wy = cell_xy
    sw, sh = scale_wh
    rx, ry, rw, rh = rect
    # _subject_cell_rect returns a rect already CENTRED on the head anchor,
    # so the rect centre is the head point.
    hx = rx + rw / 2.0
    hy = ry + rh / 2.0
    sx = (hx - wx) * (sw / float(max(cell_w, 1)))
    sy = (hy - wy) * (sh / float(max(cell_h, 1)))
    x = int(round(sx - out_w / 2.0))
    y = int(round(sy - half_h * SPLIT_CELL_HEAD_Y))
    x = min(max(x, 0), max(0, sw - out_w))
    y = min(max(y, 0), max(0, sh - half_h))
    return x & ~1, y & ~1


def cell_frame_sendcmd_lines(rects, fps, target, cell_w, cell_h,
                             orig_w, orig_h, out_w, half_h):
    """sendcmd x/y for a cell's final crop so the head stays put.

    Only x/y are emitted — the crop's w/h stay fixed at the output cell size,
    which is what keeps this clear of ffmpeg trac #10984 (a crop whose HEIGHT
    changes via sendcmd silently stops producing frames).
    """
    lines = []
    prev = None
    for i, rect in enumerate(rects):
        cell_xy = cell_initial_xy(rect, cell_w, cell_h, orig_w, orig_h)
        scale_wh = cell_scale_target(rect, cell_w, cell_h, out_w, half_h)
        pos = cell_frame_xy(rect, cell_xy, cell_w, cell_h, scale_wh, out_w, half_h)
        if pos == prev:
            continue
        stamp = _sendcmd_timestamp(i, fps)
        lines.append(f"{stamp} {target} x {pos[0]}; {stamp} {target} y {pos[1]};")
        prev = pos
    return lines


def cell_xy_sendcmd_lines(rects, fps, target, cell_w, cell_h, orig_w, orig_h):
    """sendcmd lines that only REPOSITION a fixed-size cell crop.

    Emits crop@ct/crop@cb x/y change-points (deduped) so the window follows
    the subject. Never emits w/h — resizing crop via sendcmd is the #10984
    freeze bug this whole split design works around.
    """
    lines = []
    prev = None
    for i, rect in enumerate(rects):
        pos = cell_initial_xy(rect, cell_w, cell_h, orig_w, orig_h)
        if pos == prev:
            continue
        stamp = _sendcmd_timestamp(i, fps)
        lines.append(f"{stamp} {target} x {pos[0]}; {stamp} {target} y {pos[1]};")
        prev = pos
    return lines


def cell_zoom_sendcmd_lines(rects, fps, target, cell_w, cell_h, out_w, half_h):
    """sendcmd lines for the zoom stage of a split cell.

    Drives scale@st/scale@sb w/h change-points (deduped) — the push-in. The
    crop window itself stays fixed; only this downstream scale resizes, which
    ffmpeg handles fine (only crop height changes freeze, trac #10984).
    """
    lines = []
    prev = None
    for i, rect in enumerate(rects):
        val = cell_scale_target(rect, cell_w, cell_h, out_w, half_h)
        if val == prev:
            continue
        stamp = _sendcmd_timestamp(i, fps)
        lines.append(f"{stamp} {target} w {val[0]}; {stamp} {target} h {val[1]};")
        prev = val
    return lines


# A single crop SHAPE used consistently for every frame — single-subject or
# group, regardless of scene — repositioned to whoever's relevant instead of
# switching between a narrow single-subject crop and a wide letterboxed
# layout. 3:4 is wide enough to hold 2-3 people without cramming (unlike the
# old TRACK crop, which was 9:16-shaped and only ever fit one), while still
# being narrower than the 9:16 canvas so it letterboxes into a blurred
# background — explicit user direction (31-jul-2026): "the 3:4 they use...
# make sure it's consistent... just switch it over to who is talking...
# that is cleaner than mixing aspect ratios."
UNIFIED_CROP_RATIO = 3 / 4

# Fraction of the OUTPUT frame height the letterboxed content actually fills
# for the standard 9:16 delivery — derived directly from the fixed crop
# shape above (fg_h/out_h = delivery_aspect / UNIFIED_CROP_RATIO). Unlike
# the old GENERAL_CONTENT_HEIGHT_RATIO this isn't independently tunable —
# there's only one layout now, so it falls straight out of the crop math.
# subtitles.py uses this to keep captions inside the actual content box
# instead of floating in the blur band above/below it.
UNIFIED_CONTENT_HEIGHT_RATIO = (9 / 16) / UNIFIED_CROP_RATIO


# "Polished" grade applied to every delivered frame (user request,
# 1-aug-2026: "add a little bit of color grading to make it look 4k").
# Phone-shot/consumer-camera source on a plain backdrop (this pipeline's
# typical content) reads flat and a bit soft next to professionally graded
# reference footage. First pass (contrast=1.06/saturation=1.15) was
# imperceptible on real footage — user directly reported "i cant see the
# color grading" — so this is pushed noticeably further: firmer contrast,
# a real saturation lift, and a stronger luma-only sharpen. Still not a
# stylized LUT/tint — the goal is "obviously more polished," not a color
# cast. unsharp's chroma amount is 0 so skin tones don't posterize.
COLOR_GRADE_FILTER = (
    "eq=contrast=1.15:brightness=0.02:saturation=1.35:gamma=1.05,"
    "unsharp=5:5:1.2:5:5:0.0"
)


def unified_filtergraph(out_w, out_h, crop_w, crop_h, cmd_path, initial_x,
                         initial_y=0, supersample=1):
    """Consistent 3:4-shaped crop window, letterboxed into the output canvas
    over a blurred full-frame background — repositioned per frame via sendcmd
    (x/y/zoom all drive crop@c parameters at change-points), so a tight
    single subject, a wider group, and deliberate push-ins all render through
    this ONE composition style rather than cutting between visual layouts.

    crop_w/crop_h are the BASE (zoom=1.0) crop sizes; the sendcmd file
    overrides w/h/x/y whenever the zoom level changes. The final scale stays
    at the fixed output size regardless of zoom — a smaller crop scaled up to
    that same size is what produces the zoomed-in look.

    The blurred background is derived from the SAME moving crop, not from the
    untouched source. Splitting before the crop (as this did originally) left
    the background a frozen centre-cut of the full frame while the foreground
    panned across it — two different motions in one picture, which reads as
    the whole panel sliding around and was the single most distracting thing
    on playback (user, 31-jul-2026). Cropping first and splitting after means
    background and foreground travel together, which is what the blurred-
    backdrop look is supposed to do.

    ``supersample`` (>1, 6-aug-2026): upscale the source before cropping so
    the eased crop path can step sub-pixel. The caller passes crop
    dims/positions ALREADY scaled into 2x space (the camera emits positions on
    a 1/supersample source grid; scaling by ss and rounding makes them whole
    2x pixels, which ffmpeg's integer crop filter can represent). Without it,
    every eased tail below 1px stalls and then jumps — the follow-shot judder.
    The final scale to the delivery size is unchanged, so output dimensions
    are identical; only the internal crop resolution differs.
    """
    fg_w = out_w
    fg_h = int(round(out_w * crop_h / crop_w))
    fg_h += fg_h % 2
    ss = max(1, int(supersample))
    in_prefix = (f"scale=iw*{ss}:ih*{ss}:flags=bicubic,"
                 if ss > 1 else "")
    return (
        f"[0:v]{in_prefix}sendcmd=f='{cmd_path}',"
        f"crop@c=w={crop_w}:h={crop_h}:x={initial_x}:y={initial_y},"
        f"split=2[fga][bga];"
        # Same content, blown up to cover the full canvas and blurred hard
        # enough that the upscale never reads as softness.
        f"[bga]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},gblur=sigma=24[bg];"
        f"[fga]scale={fg_w}:{fg_h},setsar=1[fg];"
        f"[bg][fg]overlay=x=0:y=(H-h)/2,setsar=1[vraw];"
        f"[vraw]{COLOR_GRADE_FILTER}[v]"
    )


def unified_split_filtergraph(out_w, out_h, crop_w, crop_h, main_cmd_path,
                              initial_x, initial_y,
                              top_xy_cmd, top_zoom_cmd,
                              bottom_xy_cmd, bottom_zoom_cmd,
                              cell_w, cell_h,
                              initial_top, initial_bottom,
                              enable_expr,
                              top_frame_cmd, bottom_frame_cmd,
                              initial_top_frame, initial_bottom_frame,
                              supersample=1):
    """Stacked two-cell reaction-cam variant of unified_filtergraph.

    The normal single-crop letterboxed path renders the whole clip (so the
    switch back to single coverage after a beat is a clean cut, not a
    rebuild); during enable_expr windows (hook + reaction beats) two tight
    subject cells — top = primary subject, bottom = the other prominent
    person — are overlaid on top, each filling exactly half the canvas.

    Each cell is a FIXED-SIZE crop (``cell_w`` x ``cell_h``) repositioned by
    x/y from its own sendcmd file (crop@ct / crop@cb), followed by a second
    sendcmd-driven scale (scale@st / scale@sb) that produces the push-in.
    The crop NEVER resizes: ffmpeg trac #10984 has the crop filter silently
    stop providing frames when its height changes via sendcmd, which froze
    the cells mid-split — the overlay held the last live cell frame and the
    output read as a hard content stutter while audio kept playing. Only
    x/y go to crop; w/h (zoom) go to the downstream scale, which handles
    runtime resizes fine.

    ``initial_top`` / ``initial_bottom`` are ``(x, y, scale_w, scale_h)`` —
    the crop window's start position and the zoom scale's start size.

    ``supersample`` (>1): same 2x-space treatment as unified_filtergraph. The
    caller scales the main/cell rects, cell dims and source bounds into 2x
    space; the zoom-scale targets and the final cell crops stay in output
    pixels, so the whole filtergraph is consistent at 2x source resolution.
    """
    fg_h = int(round(out_w * crop_h / crop_w))
    fg_h += fg_h % 2
    half_h = out_h // 2
    top_x, top_y, top_sw, top_sh = initial_top
    bot_x, bot_y, bot_sw, bot_sh = initial_bottom
    top_fx, top_fy = initial_top_frame
    bot_fx, bot_fy = initial_bottom_frame
    ss = max(1, int(supersample))
    in_prefix = (f"scale=iw*{ss}:ih*{ss}:flags=bicubic,"
                 if ss > 1 else "")
    return (
        f"[0:v]{in_prefix}split=3[vm][va][vb];"
        f"[vm]sendcmd=f='{main_cmd_path}',"
        f"crop@c=w={crop_w}:h={crop_h}:x={initial_x}:y={initial_y},"
        f"split=2[fga][bga];"
        f"[bga]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h},gblur=sigma=24[bg];"
        f"[fga]scale={out_w}:{fg_h},setsar=1[fg];"
        f"[bg][fg]overlay=x=0:y=(H-h)/2[mainv];"
        # Fixed-size cell crop repositioned by x/y only (never resized —
        # trac #10984), then a sendcmd-driven scale supplies the zoom. The
        # FINAL crop is explicitly positioned (crop@ft / crop@fb) rather than
        # centred: the cell window gets clamped at the frame edge for any
        # subject near the top, which puts the head above the window centre,
        # and a centred final crop then cut the head off.
        f"[va]sendcmd=f='{top_xy_cmd}',"
        f"crop@ct=w={cell_w}:h={cell_h}:x={top_x}:y={top_y},"
        f"sendcmd=f='{top_zoom_cmd}',"
        f"scale@st={top_sw}:{top_sh},"
        f"sendcmd=f='{top_frame_cmd}',"
        f"crop@ft=w={out_w}:h={half_h}:x={top_fx}:y={top_fy},setsar=1[top];"
        f"[vb]sendcmd=f='{bottom_xy_cmd}',"
        f"crop@cb=w={cell_w}:h={cell_h}:x={bot_x}:y={bot_y},"
        f"sendcmd=f='{bottom_zoom_cmd}',"
        f"scale@sb={bot_sw}:{bot_sh},"
        f"sendcmd=f='{bottom_frame_cmd}',"
        f"crop@fb=w={out_w}:h={half_h}:x={bot_fx}:y={bot_fy},setsar=1[bottom];"
        f"[mainv][top]overlay=x=0:y=0:enable='{enable_expr}'[t1];"
        f"[t1][bottom]overlay=x=0:y={half_h}:enable='{enable_expr}',setsar=1[vraw];"
        f"[vraw]{COLOR_GRADE_FILTER}[v]"
    )



# --- analysis ---------------------------------------------------------------


# How far a face's center has to move (as a fraction of its own box width)
# between detection samples to count as a "reaction" rather than normal
# head movement while talking — confirmed via reference analysis that real
# edits cut to a non-speaker specifically for a strong visible reaction
# (laughing, an exaggerated gesture, a big head turn), not routine motion.
REACTION_MOTION_THRESHOLD = 0.6
# Multiplies a reacting candidate's score (face area) before tracker
# selection, so a strongly-reacting non-speaker can win over a passively
# framed current speaker — reuses SpeakerTracker's existing size-based
# selection/hysteresis rather than a parallel decision system.
REACTION_SCORE_BOOST = 2.5

# Floor for reaction/speech boosts: a candidate must already be at least this
# fraction of the frame's LARGEST raw score before its score can be boosted.
# Stops a small background person's motion/mouth-jitter from out-boosting a
# large foreground subject's baseline (ground-truthed 31-jul-2026: exactly
# that hijack was reported as "focusing on irrelevant things").
BOOST_MIN_RELATIVE_SCORE = float(os.environ.get("BOOST_MIN_RELATIVE_SCORE", "0.10"))

# --- Hand-gesture evidence (round-5 spec 3, OFF by default) ----------------
# The pipeline sees faces (MediaPipe), lips (LR-ASD) and YOLO bodies — never
# hands. USE_GESTURE=1 adds a MediaPipe pose pass (shoulders/elbows/wrists)
# inside candidate boxes, mirroring the mouth-activity scale-invariant
# movement window. With the flag off (default) nothing changes: the function
# returns before touching scores, so behaviour is byte-identical.
USE_GESTURE = os.environ.get("USE_GESTURE", "0").strip().lower() not in ("0", "false", "no")
GESTURE_BOOST = float(os.environ.get("GESTURE_BOOST", "2.0"))
GESTURE_MOTION_THRESHOLD = float(os.environ.get("GESTURE_MOTION_THRESHOLD", "0.06"))
GESTURE_HISTORY_LEN = 4
_GESTURE_LANDMARKS = (11, 12, 13, 14, 15, 16)  # shoulders, elbows, wrists
_pose_graph = None
_pose_lock = threading.Lock()


def _gesture_pose_graph():
    global _pose_graph
    if _pose_graph is None:
        import mediapipe as mp
        _pose_graph = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=0,
            min_detection_confidence=0.4, min_tracking_confidence=0.4)
    return _pose_graph


def _apply_gesture_boost(candidates, frame, orig_w, boosted=None):
    """Boost visibly-gesturing candidates (wrist/elbow movement magnitude).

    Follows the exact boost contract of _apply_reaction_boost / the ASD boost:
    BOOST_MIN_RELATIVE_SCORE floor against raw_score (never a background
    sliver), score multiplied in place, ``boosted.add(id(cand['box']))`` (BOX
    id, not dict id — see spec 1.1), fail open when pose finds no landmarks.
    Movement is scale-invariant (normalized landmark coords, windowed range),
    mirroring _update_mouth_activity's approach.
    """
    if not USE_GESTURE or not candidates or frame is None:
        return
    try:
        import cv2
    except Exception:
        return
    try:
        pose = _gesture_pose_graph()
    except Exception as e:
        # Fail open: a missing/un-downloadable pose model must never break a
        # render — the gesture signal is additive and OFF by default.
        print(f"⚠️ [Gesture] pose init failed ({type(e).__name__}: {e}) — "
              f"skipping gesture boost for this run")
        return
    small_w = max(frame.shape[1], 1)
    scale = orig_w / small_w
    max_raw = max((c.get('raw_score', c['score']) for c in candidates), default=0)
    for cand in candidates:
        if max_raw and cand.get('raw_score', cand['score']) < max_raw * BOOST_MIN_RELATIVE_SCORE:
            continue
        sx, sy, sw, sh = cand['box']
        x = int(sx / scale)
        y = int(sy / scale)
        w = max(int(sw / scale), 16)
        h = max(int(sh / scale), 32)
        cx, cy = x + w // 2, y + h // 2
        cw, ch = max(w * 3 // 2, 32), max(h * 2, 64)
        x0 = max(0, cx - cw // 2)
        y0 = max(0, cy - ch // 2)
        x1 = min(frame.shape[1], x0 + cw)
        y1 = min(frame.shape[0], y0 + ch)
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        crop = frame[y0:y1, x0:x1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with _pose_lock:
            results = pose.process(rgb)
        if not results or not results.pose_landmarks:
            cand['gesture_history'] = []  # fail open
            continue
        pts = [(results.pose_landmarks.landmark[i].x,
                results.pose_landmarks.landmark[i].y)
               for i in _GESTURE_LANDMARKS]
        prev_pts = cand.get('gesture_positions')
        if prev_pts:
            movement = sum(
                ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                for a, b in zip(pts, prev_pts)) / max(len(pts), 1)
        else:
            movement = 0.0
        cand['gesture_positions'] = pts
        history = ((cand.get('gesture_history') or []) + [movement])[-GESTURE_HISTORY_LEN:]
        # Average movement magnitude over the window — a sustained gesture
        # keeps the mean high (a range would read 0 for steady motion).
        cand['gesture_history'] = history
        activity = sum(history) / len(history) if history else 0.0
        cand['gesture_activity'] = activity
        if activity >= GESTURE_MOTION_THRESHOLD:
            cand['score'] = cand.get('score', 0) * GESTURE_BOOST
            if boosted is not None:
                boosted.add(id(cand['box']))


def _apply_reaction_boost(candidates, prev_candidates, boosted=None):
    """Boost the score of any candidate whose position moved sharply since
    the last detection sample — a proxy for "is this person doing something
    visually notable right now" (research finding: ~30% of real cuts in
    studied reference clips are reaction cuts to a non-speaker, something
    this pipeline previously had no signal for at all). Matches each
    current candidate to its nearest previous-frame candidate by center
    proximity; unmatched candidates (just entered frame) get no boost —
    one sample isn't enough to call it motion. Mutates candidates in place.

    ``boosted`` (optional set): if given, the id() of each boosted candidate's
    box is added, so callers can tell which candidate actually won a boost.
    """
    if not prev_candidates:
        return
    max_raw = max((c.get('raw_score', c['score']) for c in candidates), default=0)
    for cand in candidates:
        if max_raw and cand.get('raw_score', cand['score']) < max_raw * BOOST_MIN_RELATIVE_SCORE:
            continue  # too small to be the show — never boost background noise
        x, y, w, h = cand['box']
        cx = x + w / 2
        best_prev, best_dist = None, None
        for prev in prev_candidates:
            px, py, pw, ph = prev['box']
            pcx = px + pw / 2
            dist = abs(cx - pcx)
            if best_dist is None or dist < best_dist:
                best_dist, best_prev = dist, prev
        if best_prev is None:
            continue
        px, py, pw, ph = best_prev['box']
        pcx = px + pw / 2
        # Only match to a plausibly-the-same-person previous box — a huge
        # jump is more likely a different person than fast motion.
        if best_dist > max(w, pw) * 3:
            continue
        motion = (abs(cx - pcx) + abs(h - ph)) / max(w, 1)
        if motion >= REACTION_MOTION_THRESHOLD:
            cand['score'] = cand['score'] * REACTION_SCORE_BOOST
            if boosted is not None:
                boosted.add(id(cand['box']))


# Minimum mouth-position range (as a fraction of face-box height) over the
# rolling window to count as "this face is moving its mouth," vs. a static
# face or detection jitter. Multiplies the winning candidate's score so
# SpeakerTracker's existing size-based selection picks the actual talker
# instead of whoever's biggest/stickiest.
MOUTH_ACTIVITY_THRESHOLD = 0.04
SPEECH_ACTIVITY_SCORE_BOOST = 3.0
MOUTH_HISTORY_LEN = 6

# J-cut pre-roll (owner editing spec, 4-aug-2026): hard cut to the NEXT
# speaker 0.5s before their audio starts, so the viewer sees their
# micro-expression before they speak — human anticipation, not reaction.
J_CUT_PRE_ROLL_SECONDS = float(os.environ.get("J_CUT_PRE_ROLL", "0.5"))
# Short-utterance suppression: a turn shorter than this ("yeah", "right", a
# laugh) must NOT drag the camera away from the current speaker.
SHORT_UTTERANCE_SECONDS = float(os.environ.get("SHORT_UTTERANCE", "1.5"))


def _update_mouth_activity(candidates, prev_candidates):
    """Carries each candidate's recent mouth-position samples forward by
    matching to the nearest previous-frame candidate (same center-x
    heuristic as _apply_reaction_boost), building a short rolling window.
    Mutates candidates in place, adding 'mouth_activity' (the window's
    range — 0 until there's enough history to judge movement)."""
    for cand in candidates:
        x, y, w, h = cand['box']
        cx = x + w / 2
        history = [cand.get('mouth_frac', 0.5)]
        if prev_candidates:
            best_prev, best_dist = None, None
            for prev in prev_candidates:
                px, py, pw, ph = prev['box']
                pcx = px + pw / 2
                dist = abs(cx - pcx)
                if best_dist is None or dist < best_dist:
                    best_dist, best_prev = dist, prev
            if best_prev is not None and best_dist <= max(w, best_prev['box'][2]) * 0.6:
                history = (best_prev.get('mouth_history', []) + history)[-MOUTH_HISTORY_LEN:]
        cand['mouth_history'] = history
        cand['mouth_activity'] = (max(history) - min(history)) if len(history) >= 3 else 0.0


# How far (fraction of frame width) an ASD speaker box may sit from a
# candidate and still be considered the same person. Generous: the ASD pass
# samples at 25fps while the renderer samples at its own stride, so the two
# observations are up to a couple of frames apart and the head will have moved.
ASD_MATCH_TOLERANCE = float(os.environ.get("ASD_MATCH_TOLERANCE", "0.10"))
# Score multiplier for the face LR-ASD says is speaking.
ASD_SPEAKER_BOOST = float(os.environ.get("ASD_SPEAKER_BOOST", "4.0"))
# How much closer the nearest face must be than the runner-up for an LR-ASD
# position match to count as an identification rather than a coin toss.
ASD_MATCH_MARGIN = float(os.environ.get("ASD_MATCH_MARGIN", "1.6"))
# How far LR-ASD may move within ONE diarized turn before it is judged to have
# jumped to a different person (fraction of source width).
ASD_TURN_LOCK_TOL = float(os.environ.get("ASD_TURN_LOCK_TOL", "0.08"))


def _apply_asd_speaker_boost(candidates, asd_box, orig_w, boosted=None):
    """Boost whichever visible face LR-ASD says is producing the audio.

    This is the same job as _apply_speech_activity_boost but with a much
    shorter evidence chain. That path goes: diarization label -> scene anchor
    or live binding -> candidate id, and every link can fail (a label that
    never binds, an anchor learned in a different scene, a diarization pass
    that collapsed every speaker into one). LR-ASD skips all of it — it names
    a POSITION on screen, matched here by proximity, so it works even when
    diarization produced nothing usable at all.

    Matching is positional on purpose: the ASD pass runs its own tracker at
    25fps, so its track ids mean nothing in the renderer's id space.
    """
    if not asd_box or not candidates:
        return None
    ax, ay, aw, ah = asd_box
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    tol = ASD_MATCH_TOLERANCE * orig_w
    # HORIZONTAL distance decides, with a vertical-overlap sanity check.
    #
    # Euclidean centre distance was wrong here and it was the binding bug
    # (traced 4-aug-2026 on Pop The Balloon): LR-ASD emits a FACE box, while
    # candidates are a mix of MediaPipe face boxes and YOLO head-and-chest
    # boxes whose centres sit far lower. So dy dominated dx, and a
    # vertically-closer WRONG person beat the horizontally-correct right one.
    # Measured at second 14: ASD box at x=1572, candidates at 450 / 1197 /
    # 1566, and the match returned 1197 — 375px away — over 1566, 6px away.
    # That single mismatch is what put the camera on the co-host all clip.
    #
    # People in this format sit side by side, so x separates them cleanly and
    # y does not. Vertical overlap is kept only as a sanity check, so a face
    # in a picture-in-picture strip above/below cannot claim the match.
    dists = []
    for c in candidates:
        bx, by, bw, bh = c["box"]
        if by > ay + ah or by + bh < ay:
            continue  # no vertical overlap at all — different band of frame
        dists.append((abs((bx + bw / 2.0) - acx), c))
    if not dists:
        return None
    dists.sort(key=lambda t: t[0])
    best_d, best = dists[0]
    if best is None or best_d > tol:
        return None
    # The match must be DECISIVE. In a lineup the ASD box sits roughly
    # equidistant from several faces, and ordinary detection jitter then
    # flips which one is nearest from sample to sample. Traced 3-aug-2026 on
    # a 7-person shot: the ASD box was constant for a whole second while the
    # matched face hopped across five different people at 0.17s intervals,
    # and because lip-sync is the policy's strongest evidence, every hop
    # became a cut. An ambiguous match is not evidence — reporting nothing
    # lets the policy hold the current subject instead.
    if len(dists) > 1:
        runner_up = dists[1][0]
        if runner_up < best_d * ASD_MATCH_MARGIN:
            return None
    best["score"] = best.get("score", 0) * ASD_SPEAKER_BOOST
    if boosted is not None:
        boosted.add(id(best["box"]))
    return best.get("id")


def _apply_speech_activity_boost(candidates, active_speaker, boosted=None,
                                 bound_id=None, current_target_id=None):
    """The only direct audio-to-visual speaker-IDENTITY link in the system:
    when diarization says someone is actively talking, prefer whichever
    visible face is actually moving its mouth over whoever's biggest/
    stickiest. Without this, TRACK mode has no way to tell WHICH of several
    on-screen faces is producing the current audio — ground-truthed as a
    real failure (31-jul-2026): the crop landed between two people, closer
    to a silent non-speaker, at the exact moment the actual speaker
    delivered the clip's key line. Mutates candidates' score in place.

    ``boosted`` (optional set): receives the id() of the boosted box, so
    callers can bind an audio speaker label to the tracker's candidate id.

    Identity gate (only engages when candidates carry stamped ``'id'``s, as
    the live pipeline does): a mouth-mover may only be boosted when it is
    the verified identity of the active speaker (``bound_id``) — OR, for an
    as-yet-unbound speaker, when it is the candidate the camera is already
    framing (``current_target_id``). This is what stops a laughing reactor
    in a group shot from dragging the camera off the actual speaker
    (ground-truthed 31-jul-2026: the host laughed through the primary's
    line, got speech-boosted 3 samples running, and the camera parked on
    him for 2+ seconds while she talked).
    """
    if active_speaker is None or len(candidates) < 2:
        return
    active = [c for c in candidates if c.get('mouth_activity', 0) >= MOUTH_ACTIVITY_THRESHOLD]
    if not active:
        return
    max_raw = max((c.get('raw_score', c['score']) for c in candidates), default=0)
    best = max(active, key=lambda c: c['mouth_activity'])
    if max_raw and best.get('raw_score', best['score']) < max_raw * BOOST_MIN_RELATIVE_SCORE:
        return  # the mouth-mover is background-small — don't let it hijack focus
    others = [c['mouth_activity'] for c in active if c is not best]
    # Require a clear enough margin over the next-most-active face — a
    # marginal win is as likely to be noise as a real signal, and a wrong
    # forced pick is worse than falling back to the existing size/sticky
    # selection.
    if others and best['mouth_activity'] < max(others) * 1.3:
        return
    stamped = any(c.get('id') is not None for c in candidates)
    if stamped:
        if bound_id is not None:
            if best.get('id') != bound_id:
                return  # the mouth-mover isn't the speaker we identified -> a reactor
        elif current_target_id is None or best.get('id') != current_target_id:
            # Unverified speaker: the mouth signal may only reinforce who
            # we're already framing — it can never drag the camera to a
            # stranger who happens to be moving their mouth.
            return
    best['score'] = best['score'] * SPEECH_ACTIVITY_SCORE_BOOST
    if boosted is not None:
        boosted.add(id(best['box']))


def _speaker_at_frame(speaker_turns, frame_number):
    """speaker_turns: list of (start_frame, end_frame, speaker_or_None),
    sorted, no gaps (see main.speaker_turn_frame_ranges). Linear scan is
    fine — called once per analyzed frame, ranges are few per clip."""
    if not speaker_turns:
        return None
    for sf, ef, speaker in speaker_turns:
        if sf <= frame_number < ef:
            return speaker
    return None


def _effective_speaker_label(speaker_turns, frame_number, fps):
    """(label, is_jcut) for this frame, applying the J-cut pre-roll and the
    short-utterance suppression.

    When the next speaker's turn starts within J_CUT_PRE_ROLL_SECONDS AND
    that turn is at least SHORT_UTTERANCE_SECONDS long, propose the next
    speaker NOW (is_jcut=True) — the cut lands before their audio begins. A
    short turn ("yeah", "right", a laugh) is never proposed early.
    """
    label = _speaker_at_frame(speaker_turns, frame_number)
    if not speaker_turns:
        return label, False
    pre = max(1, int(J_CUT_PRE_ROLL_SECONDS * fps))
    for s, e, nxt in speaker_turns:
        if s > frame_number:
            if s - frame_number <= pre:
                dur = (e - s) / fps if fps else 0.0
                if dur >= SHORT_UTTERANCE_SECONDS:
                    return nxt, True
            break
    return label, False


def _by_id_present(candidates, cid):
    """Is this id among the current candidates?"""
    return cid is not None and any(c.get("id") == cid for c in candidates)


def _banter_two_shot_eligible(decision_tier, evidence, target_id):
    """Owner tip 1.3 (4-aug-2026): during rapid banter, when the minimum-shot
    floor BLOCKS a strong speaker switch, the camera should show BOTH
    speakers in the 3:4 crop instead of lagging on the old one until the
    floor expires. True when the policy just HELD (did not switch) while a
    strong proposal (lip-sync / diarized / J-cut) names a DIFFERENT
    candidate — the incoming speaker is visible and the union fits."""
    import subject_policy
    if decision_tier != subject_policy.TIER_HOLD or evidence is None:
        return False
    w_id, w_tier = evidence.proposal()
    return (w_tier in subject_policy.STRONG_TIERS and w_id is not None
            and w_id != target_id)


def _merge_person_candidates(face_candidates, yolo_candidates):
    """Adds a YOLO body-box candidate for anyone MediaPipe's face detector
    missed entirely — a face mask or a head turned away from camera defeats
    MediaPipe outright (it needs a visible mouth/nose/eyes), which
    previously meant that person had NO candidate at all, however
    trackable their body position was (ground-truthed 31-jul-2026: the
    clip's actual speaker was masked/turned at points, vanishing from
    tracking completely). Only adds a YOLO candidate when it doesn't
    already sit on top of an existing face candidate, so a person with a
    good face detection isn't double-counted against themselves.

    Dedup requires BOTH horizontal and vertical proximity (box centers
    relative to the wider/taller box of the pair) — an x-only check merged
    two different people standing at the same x (one behind the other),
    silently dropping one as a candidate."""
    merged = list(face_candidates)
    for yc in yolo_candidates:
        yx, yy, yw, yh = yc['box']
        ycx = yx + yw / 2
        ycy = yy + yh / 2
        dup = False
        for fc in face_candidates:
            fx, fy, fw, fh = fc['box']
            fcx = fx + fw / 2
            fcy = fy + fh / 2
            if (abs(ycx - fcx) < max(yw, fw) * 0.6
                    and abs(ycy - fcy) < max(yh, fh) * 0.8):
                dup = True
                break
        if dup:
            continue
        merged.append(yc)
    return merged


# Minimum frames between shot changes. The reference edits' shortest shots
# are ~1s, so 30 frames (1.25s) was itself a floor above the real cadence.
# 20 -> 30 (round-5 spec 2.3, measured 3-aug-2026 on a 60s segment of the
# blind-dating source, 21 source cuts): SWITCH_COOLDOWN_FRAMES=30 reduced
# rapid pairs (<0.7s apart) from 4 to 1 with added cuts unchanged (15=15).
# The other candidates measured worse or flat on both metrics
# (MIN_SHOT_HOLD_SECONDS=2.0: 15/4; RETARGET_DEAD_ZONE_FRAC=0.10: 16/4), so
# the cooldown was the only clean keeper. Env-overridable as always.
SWITCH_COOLDOWN_FRAMES = max(int(os.environ.get("SWITCH_COOLDOWN_FRAMES", "30")), 1)

# Identity backend. "botsort" adds camera-motion compensation over "bytetrack",
# which matters on handheld/panning sources; both come from the ultralytics
# package already used for YOLO. Set USE_TRACKER_IDENTITY=0 to fall back to the
# legacy x-position matcher.
USE_TRACKER_IDENTITY = os.environ.get("USE_TRACKER_IDENTITY", "1").strip() not in ("0", "false", "no")
# LR-ASD active-speaker detection — ON by default since 3-aug-2026.
#
# It is the single most valuable signal in the pipeline and it was shipping
# switched off. Measured on the Blind Dating source: LR-ASD locates the
# speaking face in 19 of 20 seconds (81% of detection frames), where the
# next-best chain — diarization label -> scene anchor -> candidate id —
# resolved on far fewer and mis-resolved often. Selecting by face size
# instead agrees with it only 44.9% of the time, which is the "it's
# focusing on the host and the host is the one holding the mic" complaint.
#
# Costs roughly 1.3x realtime per clip on CPU and far less on a GPU (batched
# on CUDA, 48.5x realtime). Set USE_ASD=0 to trade framing accuracy for that
# time on a CPU-only deployment; the policy degrades to diarization cleanly.
USE_ASD = os.environ.get("USE_ASD", "1").strip() not in ("0", "false", "no")
TRACKER_IDENTITY_BACKEND = os.environ.get("TRACKER_IDENTITY_BACKEND", "botsort").strip()

# A directive-driven target switch (see _apply_directive_boost's use in
# _analyze_trajectory) bypasses SWITCH_COOLDOWN_FRAMES entirely — it sets
# tracker.active_speaker_id/last_switch_frame directly, since a directive is
# authoritative, not a vote. That's correct for WHO to frame, but it means
# back-to-back directive windows (Gemini emits one roughly every 2-3s) each
# force an instant hard cut on top of whatever the tracker's own cadence
# already produced, compounding with the source's own already-fast multi-cam
# cuts into a noticeably more aggressive pace than intended (ground-truthed
# 1-aug-2026: 7 cuts in the first 15s, ~1.9s average shot). This is a SECOND,
# lower floor that applies uniformly regardless of what triggered the switch
# (directive or tracker) — a genuine SOURCE scene cut always overrides it
# (that's a real cut in the footage, never suppressed).
MIN_SHOT_HOLD_SECONDS = float(os.environ.get("MIN_SHOT_HOLD_SECONDS", "1.5"))

# --- coverage / zoom policy ------------------------------------------------
# Directorial "coverage" discipline (universal rule, user 31-jul-2026): the
# clip has a PRIMARY subject (whichever diarized speaker talks most); a
# secondary subject is a brief reaction cutaway, and the camera must return
# to the primary instead of parking on the reactor. MAX_CUTAWAY_SECONDS is
# how long a non-boosted cutaway may hold before the return bias kicks in;
# a still-strong reaction extends it (the cap only accumulates on frames
# where the secondary is NOT actively boosted).
MAX_CUTAWAY_SECONDS = float(os.environ.get("MAX_CUTAWAY_SECONDS", "2.5"))
# Multiplies the primary subject's score when the return bias fires.
PRIMARY_RETURN_BOOST = float(os.environ.get("PRIMARY_RETURN_BOOST", "3.0"))

# --- primary-anchor (audio-assert) policy -----------------------------------
# The speech-identity boost can't identify a primary whose face is hidden
# (embarrassed, turned away, hand over mouth) — and in a group shot the
# visible mouth-mover is usually a REACTOR, not the speaker (ground-truthed
# 31-jul-2026: on real footage, both audio labels ended up bound to the same
# guest id because the guests laughed through every turn). The fix is a
# boost-free pre-pass that learns, per scene, which candidate the natural
# size/sticky selection holds during the primary's turns — the "anchor".
# During the primary's turns the camera then ASSERTS that candidate even
# with zero mouth activity, so a hidden-face primary is still framed (this
# is what kills both the park-on-the-reactor freeze and the off-focus crop).
# All env-overridable so it can be dialled without a deploy.
PRIMARY_ANCHOR_LEARNING = os.environ.get("PRIMARY_ANCHOR_LEARNING", "1").strip().lower() not in ("0", "false", "no", "off")
# Minimum fraction of the primary's active detection samples (in a scene)
# the anchor id must be the natural-selection target of.
PRIMARY_ANCHOR_MIN_PRESENCE = float(os.environ.get("PRIMARY_ANCHOR_MIN_PRESENCE", "0.35"))
PRIMARY_ANCHOR_MIN_SAMPLES = max(int(os.environ.get("PRIMARY_ANCHOR_MIN_SAMPLES", "3")), 1)
# The anchor's center must stay put (std as a fraction of source width) — a
# jumping box is a different person churning through the scene, not a stable
# subject worth asserting.
PRIMARY_ANCHOR_MAX_STD_FRACTION = float(os.environ.get("PRIMARY_ANCHOR_MAX_STD_FRACTION", "0.08"))
# Score multiplier for the audio-assert boost (primary talking, face hidden).
PRIMARY_ASSERT_BOOST = float(os.environ.get("PRIMARY_ASSERT_BOOST", "4.0"))
# Tolerance (fraction of source width) for matching a speaker's learned
# position anchor (see _learn_speaker_anchors' per-scene 'cx') to a candidate
# whose id has since drifted — ground-truthed 31-jul-2026: the host's scene-4
# binding stayed on a stale scene-0 id because the calm camera held the girl
# during his short lines, so scene 4 never learned a fresh id anchor for him.
# Position survives an id change; a candidate near the learned position is
# still trustworthy even when its id doesn't match.
SPEAKER_ANCHOR_TOLERANCE = float(os.environ.get("SPEAKER_ANCHOR_TOLERANCE", "0.15"))
# Zoom push-in scale for an emphasis beat (1.0 = base crop, <1 = tighter).
ZOOM_EMPHASIS = float(os.environ.get("ZOOM_EMPHASIS", "0.85"))
# A boosted target must be this many times the NEXT-largest candidate's raw
# score to earn a push-in — i.e. the camera must be locked tightly on a
# single dominant subject. Never zoom into a crowd/group shot or onto a
# background figure (ground-truthed 31-jul-2026: the old 40%-of-max floor
# pushed in on an 8-person lineup shot).
ZOOM_DOMINANCE_RATIO = float(os.environ.get("ZOOM_DOMINANCE_RATIO", "1.25"))
# A push-in also has to be SUSTAINED: the boost/dominance decision must hold
# for this many consecutive detection cycles before target_zoom moves at all,
# so a one-sample reaction blip doesn't cause a quick push-in/push-out wobble.
ZOOM_CONFIRM_CYCLES = max(int(os.environ.get("ZOOM_CONFIRM_CYCLES", "3")), 1)
# Once a push-in IS confirmed, keep pushing (ignore relax decisions) for this
# many cycles so a pause in the boost doesn't bounce the zoom back out
# immediately — a completed push-in reads deliberate, a push-pull reads broken.
ZOOM_HOLD_CYCLES = max(int(os.environ.get("ZOOM_HOLD_CYCLES", "3")), 1)

# --- split-screen reaction layout (user direction, 31-jul-2026) -------------
# Stacked two-cell "reaction cam" view, used ONLY when two distinct things
# are happening at once — someone performing/clowning/pointing while others
# react — so a split is punctuation, never the default. The single 3:4 crop
# is the base layout (both people usually fit in it; splitting the hook or a
# plain two-shot would defeat the purpose). Split intervals come from
# scene-context "causing_reaction"/"reacting" beats and strong motion
# reactions detected against the non-target subject.
SPLIT_SCREEN = os.environ.get("SPLIT_SCREEN", "0").strip().lower() not in (
    "0", "false", "no", "off")
SPLIT_MIN_SECONDS = float(os.environ.get("SPLIT_MIN_SECONDS", "0.35"))
SPLIT_MERGE_GAP_SECONDS = float(os.environ.get("SPLIT_MERGE_GAP_SECONDS", "0.5"))
SPLIT_MAX_SECONDS = float(os.environ.get("SPLIT_MAX_SECONDS", "12.0"))
# Detections an identity must have persisted for before it may occupy a split
# cell. Filters transient non-person faces (held-up photos, posters, faces on
# screens) that a raw size ranking would otherwise promote to a full half of
# the frame.
SPLIT_MIN_TRACK_DETECTIONS = int(os.environ.get("SPLIT_MIN_TRACK_DETECTIONS", "4"))
SPLIT_FRESH_SECONDS = float(os.environ.get("SPLIT_FRESH_SECONDS", "0.6"))
SPLIT_PRIMARY_TOP = os.environ.get("SPLIT_PRIMARY_TOP", "1").strip().lower() not in (
    "0", "false", "no", "off")
# Each cell is shaped to exactly fill half the 9:16 canvas (1080x960), so the
# crop-to-scale-to-crop chain below needs no letterbox bars inside a cell.
SPLIT_CELL_ASPECT = 9 / 8
# Vertical focus point inside the subject's detected box: 0.16 = just
# below the box top. MediaPipe candidates' boxes ARE face boxes (head-sized),
# so this centers the cell on the head/eyes; a YOLO body-box fallback
# (masked/turned face) gets the same head allowance near the box top. Either
# way the head lands fully inside the cell instead of a torso crop (the
# 0.42 upper-body centering cut heads off in tight cells — ground-truthed
# 31-jul-2026).
SPLIT_HEAD_FRACTION = 0.16
# Cell crop width = max(subject-box width * mult, subject-box height * floor)
# — enough to frame head + shoulders but still tight enough to read as a
# reaction cam rather than a wide group shot. 2.4x was wide enough to pull
# in a whole standing group next to the subject (ground-truthed 31-jul-2026:
# a 3-man group filled the bottom cell instead of one clear reactor); 1.4x
# plus the SPLIT_RECT_MAX_*_FRACTION caps below keep a cell to one person.
SPLIT_WIDTH_MULT = 1.4
SPLIT_HEIGHT_FLOOR = 0.62
# Hard caps on a cell's tracked-rect size, as a fraction of the source frame
# — belt-and-suspenders alongside SPLIT_WIDTH_MULT/SPLIT_HEIGHT_FLOOR so a
# large subject box (someone standing close to camera) can't blow the rect
# back out to group-shot width even with the tighter multiplier above.
SPLIT_RECT_MAX_WIDTH_FRACTION = 0.32
SPLIT_RECT_MAX_HEIGHT_FRACTION = 0.45
# The hook (first ~2s) frames BOTH people in the single 3:4 crop — a two-shot
# centered between the key subject and the other prominent person — instead
# of a split screen: both fit, and splitting the open read as carelessness
# (user, 31-jul-2026). No zoom during the two-shot.
HOOK_TWO_SHOT_SECONDS = float(os.environ.get("HOOK_TWO_SHOT_SECONDS", "1.8"))
# Opening window in which a j-cut pre-roll is forbidden. The hook is the one
# moment the viewer has no context, so cutting to whoever speaks NEXT — before
# they have said a word — lands on a silent face and reads as a mistake.
HOOK_NO_JCUT_SECONDS = float(os.environ.get("HOOK_NO_JCUT_SECONDS", "3.0"))
# Horizontal tolerance (fraction of frame width) for matching the scene
# context's key-subject position to a detected candidate.
CONTEXT_PRIMARY_TOLERANCE = float(os.environ.get("CONTEXT_PRIMARY_TOLERANCE", "0.30"))

# --- split trigger gating (user 31-jul-2026: live Gemini run hallucinated a
# "causing_reaction" directive at 3.5-6.6s with no real reaction anywhere in
# frame) ---------------------------------------------------------------------
# A directive-driven split (causing_reaction/reacting) only renders when it's
# corroborated by an actual motion-boosted non-primary candidate somewhere
# near the directive's window — Gemini's shot-direction call is advisory,
# not self-sufficient, for the specific "should this become a split" decision
# (unlike ordinary framing, where it outranks the heuristics outright).
# Motion-only reaction beats (no directive at all) are unaffected — they
# already require a real motion boost by construction and split on their own.
SPLIT_REACTOR_CORROBORATION_SECONDS = float(
    os.environ.get("SPLIT_REACTOR_CORROBORATION_SECONDS", "0.8"))
# The two cells must also be spatially distinct people, not the same person
# (or two people standing shoulder-to-shoulder) counted twice into both
# halves — minimum separation between cell centers, as a fraction of source
# width.
SPLIT_DISTINCT_MIN_FRACTION = float(
    os.environ.get("SPLIT_DISTINCT_MIN_FRACTION", "0.15"))


def _primary_speaker_label(speaker_turns):
    """Whichever diarized speaker has the most cumulative talking time across
    the clip is the clip's primary subject — reused from
    main.speaker_turn_frame_ranges, no new inference. None when there's no
    diarization or nobody talks."""
    durations = {}
    for sf, ef, speaker in (speaker_turns or []):
        if speaker is None:
            continue
        durations[speaker] = durations.get(speaker, 0) + (ef - sf)
    if not durations:
        return None
    return max(durations, key=durations.get)


OPENING_TURN_MIN_SECONDS = float(
    os.environ.get("OPENING_TURN_MIN_SECONDS", "1.0"))


def _resolve_opening_target(speaker_turns, speaker_anchors, fps,
                            asd_speaking_boxes=None):
    """Who the OPENING shot should commit to, resolved from the FIRST TURN's
    evidence in AGGREGATE (owner spec, 4-aug-2026 — see §2h(1)).

    The old open committed frame 0's instantaneous evidence, and on Pop The
    Balloon that was unanimously wrong (lip-sync AND the diarized binding both
    landed on a silent listener, and the 1.5s absolute-min floor then held
    the mistake for the whole open). The transcript is deterministic where
    the per-second signals are not, so the open is chosen here, before any
    frame is rendered:

      * the FIRST diarized turn long enough to anchor the open decides who
        the opening speaker is (a 0.2s host tail before the guest speaks is
        not the open — the guest is);
      * that label's FACE is bound from the AGGREGATE of their LR-ASD
        speaking positions across ALL their turns (the clip's whole ASD
        evidence, not just the first second — measured 4-aug-2026: LR-ASD
        was steadily wrong for the first ~2s, and the diarized anchor chain
        collapsed both speakers onto the biggest face). Majority by 50px
        band; the diarized scene anchor is the fail-open fallback when ASD
        is absent or the majority is not decisive.

    Returns (candidate_id, centre_x, lock_until_frame) or (None, None, None)
    when there is no transcript, no anchor, or the open cannot be resolved —
    the policy then falls back to per-frame evidence exactly as before (fail
    open). The candidate id is scene-local, so the ASD path returns
    (None, centre_x, ...) and the policy matches the position at frame 0.
    ``lock_until_frame`` is the end of the opening turn: the ASD continuity
    gate holds the aggregate face for the whole opening turn, not just the
    floor.
    """
    if not speaker_turns:
        return None, None, None
    min_frames = max(1, int(OPENING_TURN_MIN_SECONDS * fps))
    opening_label = None
    lock_until = None
    for sf, ef, label in speaker_turns:
        if label is None or ef - sf < min_frames:
            continue
        opening_label = label
        lock_until = ef
        break
    if opening_label is None:
        return None, None, None
    # The opening speaker's face, from the aggregate of their LR-ASD
    # positions across every one of their turns in the clip.
    if asd_speaking_boxes:
        positions = []
        for sf, ef, label in speaker_turns:
            if label != opening_label:
                continue
            for sec in range(int(sf / fps), int((ef - 1) / fps) + 1):
                if 0 <= sec < len(asd_speaking_boxes):
                    b = asd_speaking_boxes[sec]
                    if b:
                        positions.append(b[0] + b[2] / 2.0)
        if positions:
            bands = {}
            for x in positions:
                bands.setdefault(int(x // 50), []).append(x)
            top = max(bands, key=lambda k: len(bands[k]))
            n_top = len(bands[top])
            second = max((len(v) for k, v in bands.items() if k != top),
                         default=0)
            # A decisive plurality: the top band holds a strict majority of
            # the named samples, or doubles the runner-up while holding at
            # least 40%. Anything weaker (a tie) is ambiguous — fall back to
            # the diarized anchor.
            if (n_top > len(positions) * 0.5
                    or (n_top >= 2 * second
                        and n_top >= len(positions) * 0.4)):
                xs = bands[top]
                return None, (sum(xs) / len(xs) if xs else None), lock_until
    anchor = ((speaker_anchors or {}).get(0) or {}).get(opening_label)
    if anchor is None:
        return None, None, None
    return anchor.get("id"), anchor.get("cx"), lock_until


def _apply_primary_return_bias(candidates, primary_id):
    """Boost the primary subject's score so the tracker's next decision
    returns to them. No-op if the primary isn't on screen. Returns True if
    the bias was applied (caller resets its armed flag)."""
    for cand in candidates:
        if cand.get('id') == primary_id:
            cand['score'] = cand['score'] * PRIMARY_RETURN_BOOST
            return True
    return False


def _apply_primary_assert_boost(candidates, primary_id, boosted=None):
    """Audio-assert fallback for the PRIMARY subject: diarization says they
    are talking, so boost their known candidate regardless of visible mouth
    motion — a primary who is embarrassed/turned away has a hand over their
    mouth and MediaPipe never sees a mouth to move, which is exactly when
    the camera used to drift off them (ground-truthed 31-jul-2026: the crop
    parked on the laughing host for 2+ seconds while the primary delivered
    her line with her face hidden). The size floor keeps a barely-visible
    sliver at the frame edge from winning. Mutates candidates in place;
    returns True if the boost was applied."""
    if primary_id is None:
        return False
    max_raw = max((c.get('raw_score', c['score']) for c in candidates), default=0)
    for cand in candidates:
        if cand.get('id') != primary_id:
            continue
        if boosted is not None and id(cand['box']) in boosted:
            return True  # the speech boost already asserted them
        if max_raw and cand.get('raw_score', cand['score']) < max_raw * BOOST_MIN_RELATIVE_SCORE:
            return False  # background-sliver — not worth asserting
        cand['score'] = cand['score'] * PRIMARY_ASSERT_BOOST
        if boosted is not None:
            boosted.add(id(cand['box']))
        return True
    return False


def _pick_primary_anchor(scene_id_stats, primary_label, width,
                         min_presence=PRIMARY_ANCHOR_MIN_PRESENCE,
                         min_samples=PRIMARY_ANCHOR_MIN_SAMPLES,
                         max_std_fraction=PRIMARY_ANCHOR_MAX_STD_FRACTION):
    """Resolve one scene's per-label/per-id stats into the primary label's
    anchor id, or None. ``scene_id_stats`` maps label -> id -> dict with
    'n' (samples where that id was the natural-selection target during the
    label's turns), 'total' (samples where the label was active) and 'cx'
    (list of target centers). A stable, dominant target during the
    primary's turns is the candidate worth asserting; anything that jumps
    around (a different person churning through the shot) is rejected."""
    ids = scene_id_stats.get(primary_label) or {}
    if not ids:
        return None
    best_id, best = None, None
    for pid, e in ids.items():
        total = e.get('total', 0) or 1
        if e['n'] < min_samples or (e['n'] / total) < min_presence:
            continue
        xs = e.get('cx') or []
        if len(xs) < 2:
            continue
        mean = sum(xs) / len(xs)
        std = (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5
        if std > max_std_fraction * width:
            continue
        if best is None or e['n'] > best['n']:
            best_id, best = pid, e
    return best_id


def _learn_speaker_anchors(input_video, scenes_boundaries, fps, orig_w, orig_h,
                            speaker_turns):
    """Boost-free pre-pass that learns, per scene, which tracker candidate
    id the natural size/sticky selection holds while EACH diarized speaker
    is active. Returns ``(anchors, frame_detections)`` where anchors is
    {scene_index: {label: {'id': id, 'cx': position}}} and frame_detections
    is the raw pre-boost detection cache the camera pass replays instead of
    decoding + detecting the clip a second time (plan item 9):
    {frame_number: [merged, scaled candidate dicts]} plus a reserved
    ``_total_frames`` key.

    ``cx`` is the mean face-center position (x, in source pixels) sampled
    while the label's anchor id was the natural-selection target — a
    position anchor alongside the id anchor, since a stale id from an
    earlier scene is exactly what let a speaker's identity binding go wrong
    (the host's scene-4 lines had no fresh id anchor because the calm
    camera held the girl through his short lines, so the stale scene-0 id
    kept winning the id fast-path). Position survives an id reassignment
    that a bare id anchor can't.

    Runs BEFORE the camera pass with a fresh tracker over the SAME
    detection pipeline, so ids are identical to the camera pass's (id
    assignment is position-based and boost-independent). This is the
    identity source for the audio-assert boost: a speaker's face may be
    hidden during their key line (no mouth signal ever fires), but the
    candidate the camera naturally holds during their turns is still who
    they are. ({}, {}) when there's no diarization or learning is off.
    """
    import numpy as np
    import main as m

    if not speaker_turns or not PRIMARY_ANCHOR_LEARNING:
        return {}, {}

    small_w = min(ANALYSIS_MAX_WIDTH, orig_w)
    if small_w % 2:
        small_w -= 1
    small_h = max(int(orig_h * small_w / orig_w), 2)
    if small_h % 2:
        small_h += 1
    scale = orig_w / small_w
    frame_bytes = small_w * small_h * 3
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-i", input_video,
         "-vf", f"scale={small_w}:{small_h}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 4)

    # Same identity backend as the render pass. Both passes walk the SAME
    # detections at the SAME stride, so two independently-constructed trackers
    # produce the same id sequence — which is what lets the scene anchors
    # learned here resolve by id later instead of falling back to position
    # matching every time.
    anchor_identity = None
    if USE_TRACKER_IDENTITY:
        try:
            import identity_tracker
            if identity_tracker.available():
                anchor_identity = identity_tracker.IdentityTracker(
                    detection_fps=max(1.0, fps / float(m.DETECT_STRIDE)),
                    tracker_type=TRACKER_IDENTITY_BACKEND)
        except Exception:
            anchor_identity = None
    tracker = m.SpeakerTracker(cooldown_frames=30, identity=anchor_identity)
    prev_candidates = None
    cached_yolo_candidates = []
    cached_yolo_frame = -10 ** 9
    max_yolo_stale = m.YOLO_FALLBACK_STRIDE * 2
    frame_detections = {}
    scene_stats = {}  # scene_index -> label -> id -> {'n', 'total', 'cx'}
    active_totals = {}  # scene_index -> label -> active-sample count
    current_scene_index = 0
    frame_number = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((small_h, small_w, 3))

            if (current_scene_index < len(scenes_boundaries) - 1
                    and frame_number >= scenes_boundaries[current_scene_index][1]):
                current_scene_index += 1

            active = _speaker_at_frame(speaker_turns, frame_number)
            if frame_number % m.DETECT_STRIDE != 0:
                frame_number += 1
                continue
            candidates = m.detect_face_candidates(frame)
            for cand in candidates:
                cand['box'] = [int(v * scale) for v in cand['box']]
                cand['score'] = cand['box'][2] * cand['box'][3]
                cand['raw_score'] = cand['score']
            if frame_number % m.YOLO_FALLBACK_STRIDE == 0:
                yolo_candidates = m.detect_person_candidates_yolo(frame)
                for cand in yolo_candidates:
                    cand['box'] = [int(v * scale) for v in cand['box']]
                    cand['score'] = cand['box'][2] * cand['box'][3]
                    cand['raw_score'] = cand['score']
                cached_yolo_candidates = yolo_candidates
                cached_yolo_frame = frame_number
            else:
                yolo_candidates = _discount_stale_yolo(
                    cached_yolo_candidates, frame_number - cached_yolo_frame,
                    max_yolo_stale)
            candidates = _merge_person_candidates(candidates, yolo_candidates)
            # Cache RAW (pre-boost, pre-id) candidates — deep copies so the
            # boosts/assign_ids below (and the tracker's in-place mutations)
            # never leak into what the camera pass replays.
            frame_detections[frame_number] = copy.deepcopy(candidates)
            # Reaction signal only — the mouth/speech boost is deliberately
            # absent so the selection mirrors what a calm camera would hold.
            _apply_reaction_boost(candidates, prev_candidates)
            prev_candidates = candidates
            tracker.set_identity_frame(frame)
            tracker.assign_ids(candidates, frame_number, orig_w)
            target_box, target_id = tracker.get_target_id(
                candidates, frame_number, orig_w, continuity_bias=True)
            if active is not None and target_id is not None:
                sc = scene_stats.setdefault(current_scene_index, {})
                label_stats = sc.setdefault(active, {})
                entry = label_stats.setdefault(target_id, {'n': 0, 'total': 0, 'cx': []})
                entry['n'] += 1
                cand = next((c for c in candidates if c.get('id') == target_id), None)
                if cand is not None:
                    entry['cx'].append(cand['box'][0] + cand['box'][2] / 2)
            # Track 'total' per (scene, label): every sample the label is
            # active, whether or not a target existed.
            if active is not None:
                totals = active_totals.setdefault(current_scene_index, {})
                totals[active] = totals.get(active, 0) + 1
            frame_number += 1
    finally:
        if proc is not None:
            proc.stdout.close()
            proc.wait()

    frame_detections['_total_frames'] = frame_number
    anchors = {}
    for scene_index, label_stats in scene_stats.items():
        totals = active_totals.get(scene_index, {})
        for label, e in label_stats.items():
            for pid in e:
                e[pid]['total'] = totals.get(label, 0)
        scene_anchors = {}
        for label in label_stats:
            anchor_id = _pick_primary_anchor(label_stats, label, orig_w)
            if anchor_id is None:
                continue
            xs = label_stats[label].get(anchor_id, {}).get('cx') or []
            cx = sum(xs) / len(xs) if xs else None
            scene_anchors[label] = {'id': anchor_id, 'cx': cx}
        if scene_anchors:
            anchors[scene_index] = scene_anchors
    return anchors, frame_detections


def _resolve_speaker_binding(candidates, label, scene, anchors, speaker_to_id,
                              orig_w):
    """Resolve which currently-detected candidate id is `label` right now.

    id fast-path: the scene's learned anchor id, if it's among today's
    candidates -> that id (cheapest, most confident case: the exact same
    tracked identity is still present). Otherwise fall back to POSITION:
    the candidate nearest the anchor's learned cx, within
    SPEAKER_ANCHOR_TOLERANCE of the source width -> that candidate's id.
    This is what survives an id reassignment across scenes (see
    _learn_speaker_anchors) — a stale id from scene 0 no longer wins just
    because nothing fresher was ever learned for the label in scene 4.
    Falls back to the live speaker_to_id binding (built incrementally by
    the speech-activity boost) when there's no usable scene anchor at all,
    and to None when nothing resolves.
    """
    if label is None:
        return None
    anchor = ((anchors or {}).get(scene) or {}).get(label)
    if anchor is not None:
        aid = anchor.get('id')
        if aid is not None and any(c.get('id') == aid for c in candidates):
            return aid
        cx = anchor.get('cx')
        if cx is not None:
            tol = SPEAKER_ANCHOR_TOLERANCE * orig_w
            best, best_dist = None, None
            for c in candidates:
                c_cx = c['box'][0] + c['box'][2] / 2
                d = abs(c_cx - cx)
                if best_dist is None or d < best_dist:
                    best_dist, best = d, c
            if best is not None and best_dist <= tol:
                return best.get('id')
    return (speaker_to_id or {}).get(label)

def _discount_stale_yolo(cached_yolo_candidates, stale_frames, max_stale):
    """Stale YOLO boxes (positions from an older shot/pose) are discounted
    by staleness so a slow camera move doesn't merge a person at their OLD
    location as if they were there right now; dropped entirely past
    ``max_stale`` frames."""
    if not cached_yolo_candidates:
        return []
    if stale_frames > max_stale:
        return []
    discount = max(1.0 - (stale_frames / max_stale), 0.5)
    if discount >= 1.0:
        return cached_yolo_candidates
    return [{**c, 'score': c['score'] * discount} for c in cached_yolo_candidates]


def _zoom_for_target(candidates, target_box, boosted_ids):
    """Zoom policy: push in only when the CURRENT target is the one that won
    a boost this frame AND it's tightly dominant — its raw score must be
    ZOOM_DOMINANCE_RATIO x the next-largest candidate's. That's a deliberate
    emphasis move on a single locked subject, not constant zooming into
    crowd/group shots. Returns the zoom scale to ease toward (1.0 = no zoom)."""
    if id(target_box) not in boosted_ids:
        return 1.0
    target = next((c for c in candidates if c['box'] is target_box), None)
    if target is None:
        return 1.0
    raw = target.get('raw_score', target['score'])
    others = [c.get('raw_score', c['score']) for c in candidates
              if c['box'] is not target_box]
    if others and raw < max(others) * ZOOM_DOMINANCE_RATIO:
        return 1.0
    return ZOOM_EMPHASIS


def _is_reaction_beat(candidates, boosted, speech_boosted, target_box,
                      effective_primary_id, directed_box):
    """True when a non-primary, non-target candidate won a motion boost.

    Round-5 spec 1.1: the old inline check compared ``id(c)`` (the candidate
    DICT's id) against sets that store ``id(c['box'])`` (the box LIST's id) —
    dict ids never match box ids, so reaction_beat was permanently False and
    BOTH split paths (motion splits + directive corroboration) were dead code.
    Box identity is the contract every producer already uses (including
    _zoom_for_target and the directive add), so compare like with like.
    """
    if directed_box is not None:
        return False
    for c in candidates:
        if (id(c['box']) in boosted
                and id(c['box']) not in speech_boosted
                and c['box'] is not target_box
                and (effective_primary_id is None
                     or c.get("id") != effective_primary_id)):
            return True
    return False


def _has_second_subject(candidates, directed_box):
    """True when a candidate OTHER than the directed subject exists.

    Round-5 spec 2.1: a causing_reaction/referenced directive becomes a SPLIT
    only when two distinct subjects are available; with a single subject it
    must fall back to an override cut (show the cause) instead of being
    dropped. Distinctness is by tracker id — the same person detected twice
    never counts as two subjects.
    """
    directed_cand = next(
        (c for c in candidates if c['box'] is directed_box), None)
    if directed_cand is None:
        return False
    return any(
        c is not directed_cand and c.get("id") != directed_cand.get("id")
        for c in candidates)


def _zoom_confirm(decision, confirm_frames, confirm_cycles):
    """Hysteresis gate on top of _zoom_for_target: a push-in decision must
    persist for ``confirm_cycles`` consecutive detection cycles before the
    zoom target moves. Returns (zoom_target, next_confirm_frames) where
    zoom_target is None while confirming (caller holds the current zoom),
    1.0 on the first non-boosted cycle (relax immediately), or the confirmed
    decision once the streak clears the threshold."""
    if decision < 1.0:
        confirm_frames += 1
    else:
        confirm_frames = 0
    if confirm_frames >= confirm_cycles:
        return decision, confirm_frames
    if confirm_frames == 0:
        return 1.0, 0
    return None, confirm_frames


def _zoom_hold_decision(zoom_target, hold_frames, hold_cycles):
    """Hold gate on top of _zoom_confirm: once a push-in is active, a relax
    decision is ignored for ``hold_cycles`` cycles so a brief pause in the
    boost doesn't bounce the zoom back out. Returns (effective_zoom_target,
    next_hold_frames); None means "hold the current zoom target"."""
    if zoom_target is not None and zoom_target < 1.0:
        return zoom_target, hold_cycles  # (re)arming the hold
    if hold_frames > 0:
        return None, hold_frames - 1
    return zoom_target, 0


# --- scene-context direction ------------------------------------------------
# The heuristics above (size, motion, mouth movement, diarization) can tell
# WHO is talking and WHO is moving, but never WHY — so when a group laughs at
# someone clowning, they frame the laughers, not the clown. main.analyze_scene_
# context asks Gemini to watch the finished cut and return shot direction; the
# helpers below turn that into a per-frame target the tracker honours.
#
# A directive names its subject by horizontal position (0..1 of frame width),
# so matching is a nearest-centre lookup against whatever the detectors found.
# The match must be reasonably close or the directive is ignored — a confident
# instruction pointing at empty frame is worse than falling back to heuristics.
DIRECTIVE_MATCH_TOLERANCE = float(os.environ.get("DIRECTIVE_MATCH_TOLERANCE", "0.18"))
# Score multiplier for a directed subject. Far above the reaction/speech
# boosts (2.5/3.0) because this is semantic intent, not a proxy signal.
DIRECTIVE_SCORE_BOOST = float(os.environ.get("DIRECTIVE_SCORE_BOOST", "12.0"))
# A "show the cause" / "show who was referenced" beat is the payoff shot, so
# it earns a tighter push-in than an ordinary speaking beat.
DIRECTIVE_PAYOFF_REASONS = ("causing_reaction", "referenced")


def _directive_at_time(focus_directives, t):
    """The directive covering clip-relative second `t`, or None. Linear scan —
    a clip has tens of directives, and this runs once per analyzed frame."""
    for d in focus_directives or []:
        if d["start"] <= t < d["end"]:
            return d
    return None


def _apply_directive_boost(candidates, directive, orig_w):
    """Boost whichever detected candidate sits closest to the directive's
    x_position. Returns the boosted candidate's box (so the caller can zoom on
    it), or None when nothing was near enough to trust."""
    if not directive or not candidates:
        return None
    want_x = directive["x_position"] * orig_w
    tol = DIRECTIVE_MATCH_TOLERANCE * orig_w
    best, best_dist = None, None
    for cand in candidates:
        x, _y, w, _h = cand['box']
        dist = abs((x + w / 2) - want_x)
        if best_dist is None or dist < best_dist:
            best, best_dist = cand, dist
    if best is None or best_dist > tol:
        return None
    best['score'] = best['score'] * DIRECTIVE_SCORE_BOOST
    return best['box']


def _directive_matched_candidate(candidates, directive, orig_w):
    """The candidate a directive points at (by x_position), WITHOUT boosting
    anything. Same nearest-centre match and tolerance as
    _apply_directive_boost; returns the candidate dict or None.

    Kept separate because a referenced/causing_reaction directive can be
    demoted to split intent or dropped as an override further down while the
    semantic reaction trigger (see referenced_id in _analyze_trajectory)
    still needs to know WHO the dialogue pointed at."""
    if not directive or not candidates:
        return None
    want_x = directive["x_position"] * orig_w
    tol = DIRECTIVE_MATCH_TOLERANCE * orig_w
    best, best_dist = None, None
    for cand in candidates:
        x, _y, w, _h = cand['box']
        dist = abs((x + w / 2) - want_x)
        if best_dist is None or dist < best_dist:
            best, best_dist = cand, dist
    if best is None or best_dist > tol:
        return None
    return best


def _directive_zoom(directive):
    """Push-in level a directive asks for. Payoff beats (the cause of a
    reaction, or the person just referenced) commit harder than a plain
    speaking beat, and `intensity` scales between no zoom and ZOOM_EMPHASIS."""
    if not directive:
        return 1.0
    strength = directive.get("intensity", 0.5)
    if directive.get("reason") in DIRECTIVE_PAYOFF_REASONS:
        strength = min(1.0, strength + 0.25)
    if strength < 0.35:
        return 1.0
    return 1.0 - (1.0 - ZOOM_EMPHASIS) * strength


def _subject_cell_rect(box, orig_w, orig_h, aspect=None):
    """Tight head-and-shoulders crop for one split-screen cell.

    HEAD-anchored: the vertical center sits SPLIT_HEAD_FRACTION (0.16) down
    the detected box — the head region whether the box is a MediaPipe face
    box or a YOLO body fallback — so the head is fully inside the cell, never
    a torso crop (0.42 upper-body centering cut heads off — ground-truthed
    31-jul-2026). Shaped to SPLIT_CELL_ASPECT (9:8), so when scaled to half
    the 9:16 canvas it fills the cell exactly — no bars, no cover-crop
    surprise. Clamped to the source frame and rounded to even pixels for the
    crop filter.
    """
    if aspect is None:
        aspect = SPLIT_CELL_ASPECT
    bx, by, bw, bh = box
    cx = bx + bw / 2.0
    cy = by + bh * SPLIT_HEAD_FRACTION
    w = min(orig_w, max(bw * SPLIT_WIDTH_MULT, bh * SPLIT_HEIGHT_FLOOR))
    # Belt-and-suspenders caps (see SPLIT_RECT_MAX_*_FRACTION): a one-person
    # rect never grows past a one-person SIZE, even for a subject box large
    # enough that SPLIT_WIDTH_MULT alone would still balloon it out —
    # ground-truthed 31-jul-2026 as the root cause of group/two-person
    # split cells that showed no single clear subject.
    w = min(w, orig_w * SPLIT_RECT_MAX_WIDTH_FRACTION)
    h = w / aspect
    if h > orig_h * SPLIT_RECT_MAX_HEIGHT_FRACTION:
        h = orig_h * SPLIT_RECT_MAX_HEIGHT_FRACTION
        w = h * aspect
    if h > orig_h:
        h = orig_h
        w = h * aspect
    x = cx - w / 2.0
    y = cy - h / 2.0
    x = min(max(x, 0.0), orig_w - w)
    y = min(max(y, 0.0), orig_h - h)
    return (int(round(x)) & ~1, int(round(y)) & ~1,
            max(int(round(w)) & ~1, 2), max(int(round(h)) & ~1, 2))


def _directive_split_corroborated(frame_number, motion_reaction_frames,
                                  reactor_corroboration_frames,
                                  top_rect, bottom_rect, orig_w):
    """Whether a directive-driven split (causing_reaction/reacting) should
    actually render at this frame — see SPLIT_REACTOR_CORROBORATION_SECONDS
    / SPLIT_DISTINCT_MIN_FRACTION. A directive alone is advisory, not
    self-sufficient, for this specific decision (a live Gemini run
    hallucinated a causing_reaction beat at 3.5-6.6s with nobody actually
    reacting anywhere in frame): it needs a real motion-boosted reactor
    observed within ``reactor_corroboration_frames`` of now, AND the two
    cells must be spatially distinct people rather than the same person (or
    a tight pair) landing in both halves.
    """
    if top_rect is None or bottom_rect is None:
        return False
    corroborated = any(
        abs(frame_number - rf) <= reactor_corroboration_frames
        for rf in motion_reaction_frames)
    if not corroborated:
        return False
    tcx = top_rect[0] + top_rect[2] / 2.0
    bcx = bottom_rect[0] + bottom_rect[2] / 2.0
    return abs(tcx - bcx) >= SPLIT_DISTINCT_MIN_FRACTION * orig_w


def _union_box(a, b, orig_w, orig_h, pad=0.12):
    """Bounding box spanning two subject boxes, padded — the hook two-shot
    target, so the fixed 3:4 crop centered on it frames BOTH people."""
    x1 = min(a[0], b[0])
    y1 = min(a[1], b[1])
    x2 = max(a[0] + a[2], b[0] + b[2])
    y2 = max(a[1] + a[3], b[1] + b[3])
    px = (x2 - x1) * pad
    py = (y2 - y1) * pad
    x1 = max(0.0, x1 - px)
    y1 = max(0.0, y1 - py)
    x2 = min(float(orig_w), x2 + px)
    y2 = min(float(orig_h), y2 + py)
    return [x1, y1, x2 - x1, y2 - y1]


def _hook_secondary(primary, candidates, crop_w, orig_w, orig_h,
                    exclude_ids=None):
    """The person a hook two-shot should pair the star with.

    The naive biggest-other-candidate rule picks whoever's largest ANYWHERE
    in frame — on a wide set that is a distant third person, and the union
    with the star is wider than the fixed 3:4 crop, so the crop falls on the
    gap and shows neither subject (ground-truthed 31-jul-2026: the opening
    landed on the host with the girl out of frame). The hook's co-subject is
    whoever is substantial AND near the star — within the crop width of
    them — so both can share the crop. Falls back to the biggest overall
    when nobody's near (rare; keeps the old behavior).
    """
    if primary is None:
        return None
    px = primary["box"][0] + primary["box"][2] / 2.0
    best_fit, best_fit_score = None, -1.0
    best_any, best_any_score = None, -1.0
    for c in candidates:
        if c is primary or (c.get("id") in (exclude_ids or ())):
            continue
        score = c.get("score", 0)
        if score > best_any_score:
            best_any_score, best_any = score, c
        cx = c["box"][0] + c["box"][2] / 2.0
        if abs(cx - px) <= crop_w * 1.2 and score > best_fit_score:
            best_fit_score, best_fit = score, c
    return best_fit if best_fit is not None else best_any


def _hook_union(primary, secondary, crop_w, orig_w, orig_h, pad=0.12):
    """Union of the two hook subjects, padded only as far as the crop holds.

    A padded union wider than the fixed 3:4 crop makes the crop fall on the
    gap between the two people (showing neither) — the hook bug. Pad is
    capped at whatever keeps the union inside ``crop_w``; if the raw span
    already overflows, no padding is applied and the cameraman centers on
    the pair as-is.
    """
    x1 = min(primary[0], secondary[0])
    y1 = min(primary[1], secondary[1])
    x2 = max(primary[0] + primary[2], secondary[0] + secondary[2])
    y2 = max(primary[1] + primary[3], secondary[1] + secondary[3])
    span = x2 - x1
    if span > 0:
        pad = min(pad, max(0.0, (crop_w - span) / span))
    px = span * pad
    py = (y2 - y1) * pad
    x1 = max(0.0, x1 - px)
    y1 = max(0.0, y1 - py)
    x2 = min(float(orig_w), x2 + px)
    y2 = min(float(orig_h), y2 + py)
    return [x1, y1, x2 - x1, y2 - y1]


def _reaction_two_shot_boxes(primary_box, secondary_box, crop_width,
                             orig_w, orig_h, max_span_ratio=2.0):
    """Union box for an acted-upon two-shot (round-5 spec 2.2).

    When a reaction/acted-upon beat fires but no split is being rendered,
    prefer framing BOTH subjects over cropping to the actor alone — but only
    when they are close enough to share the fixed crop (union span within
    ``max_span_ratio`` of the crop width); otherwise the crop would land on
    the gap between them and show neither. Returns the union box or None.
    """
    if primary_box is None or secondary_box is None:
        return None
    union = _hook_union(primary_box, secondary_box, crop_width, orig_w, orig_h)
    if union[2] > crop_width * max_span_ratio:
        return None
    return union


def _split_ranges_from_flags(flags, fps, min_frames, merge_frames, max_seconds):
    """Collapse per-frame split booleans into clip-relative second ranges.

    Joins runs separated by <= merge_frames (detection-stride gaps between
    samples), drops runs shorter than min_frames, and caps total coverage at
    max_seconds (longest beats first) so split-screen stays punctuation, not
    the default layout.
    """
    ranges = []
    start = None
    for i, on in enumerate(flags):
        if on and start is None:
            start = i
        elif not on and start is not None:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(flags)))
    if not ranges:
        return []
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ps, pe = merged[-1]
        if s - pe <= merge_frames:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    sec_ranges = []
    for s, e in merged:
        if e - s >= min_frames:
            sec_ranges.append((round(s / fps, 3), round(e / fps, 3)))
    if not sec_ranges:
        return []
    picked = []
    total = 0.0
    for r in sorted(sec_ranges, key=lambda r: r[1] - r[0], reverse=True):
        if total + (r[1] - r[0]) <= max_seconds:
            picked.append(r)
            total += r[1] - r[0]
    picked.sort(key=lambda r: r[0])
    return picked


def _ident_bbox_xywh(bbox):
    """Normalize an InsightFace bbox [x1, y1, x2, y2] to (x, y, w, h)."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def _accumulate_face_id_votes(candidates, face_identifications, frame_number,
                              fps, votes):
    """Vote candidate ids -> names from clip-relative face-ID samples.

    Runs on every detection frame after ids are stamped. Each named sample
    within ±0.5s whose box overlaps a candidate's box (IoU >= 0.3) votes for
    that candidate id. The votes let _face_id_binding name a track only once
    multiple samples agree — a face that flickers between two people never
    gets committed.
    """
    if not face_identifications or not candidates:
        return votes
    ts = frame_number / fps if fps else 0.0
    for ident in face_identifications:
        if abs(float(ident["timestamp"]) - ts) > 0.5:
            continue
        ib = _ident_bbox_xywh(ident["bbox"])
        for c in candidates:
            cid = c.get("id")
            if cid is None or not c["box"]:
                continue
            if face_id._iou(c["box"], ib) >= 0.3:
                votes.setdefault(cid, {})
                votes[cid][ident["name"]] = \
                    votes[cid].get(ident["name"], 0) + 1
    return votes


def _face_id_binding(candidates, votes, active_speaker, frame_number, fps,
                     face_identifications):
    """Face-ID upgrade of the diarized speaker binding (PART 4, 6-aug-2026).

    Diarization produces a speaker LABEL; binding that label to a face is the
    chain's weak link (see _resolve_speaker_binding). When face ID has named
    a candidate (and the transcript's labels were renamed by
    face_id.enrich_if_configured), a candidate whose name matches the current
    speaker is returned even if the anchor chain failed — TIER_DIARIZED then
    frames them. Never forces a cut (the cooldown/hysteresis still apply) and
    never fires when face ID is off.

    Two paths: a fresh identification at this timestamp naming the speaker's
    box directly, or accumulated votes (>= 2 samples, >= 60% agreement).
    """
    if not active_speaker or not candidates:
        return None
    ts = frame_number / fps if fps else 0.0
    for ident in face_identifications or []:
        if (abs(float(ident["timestamp"]) - ts) > 0.5
                or ident["name"] != active_speaker):
            continue
        ib = _ident_bbox_xywh(ident["bbox"])
        for c in candidates:
            if c["box"] and face_id._iou(c["box"], ib) >= 0.3:
                return c.get("id")
    for cid, name_counts in (votes or {}).items():
        total = sum(name_counts.values())
        if total < 2:
            continue
        best = max(name_counts, key=name_counts.get)
        if best == active_speaker and name_counts[best] / total >= 0.6:
            return cid
    return None


def _analyze_trajectory(input_video, scenes_boundaries, fps, orig_w, orig_h,
                        cameraman, tracker, speaker_turns=None,
                        focus_directives=None, primary_subject_x=None,
                        asd_speaking_boxes=None, cell_aspect=None,
                        face_identifications=None):
    """Per-frame crop trajectory for the unified 3:4-consistent render (see
    UNIFIED_CROP_RATIO) — every frame gets a crop position from the SAME
    crop shape, repositioned to whoever's relevant, instead of switching
    between a narrow single-subject crop and a wide letterboxed layout.

    A detected SOURCE scene cut is treated as a deliberate cut to a new
    subject — not fought, not panned to. Ground-truthed 31-jul-2026: this
    source is pre-edited multi-cam footage where the creator already cuts
    to whoever's being talked about (e.g. a lineup of people a speaker is
    describing), not a single continuous shot of the speaker — our job at
    that cut is to frame whatever's now on screen well, not keep hunting
    for a speaker who may not even be in this shot. Bypassing the
    jump-confirm gate (force_next_update) and holding the snap for one
    detection cycle (cut_grace_frames) is what actually lands the crop on
    the real new target instead of wherever the previous shot left off.

    Detection merges MediaPipe face candidates with YOLO body candidates
    (see main.detect_person_candidates_yolo / _merge_person_candidates) so
    a masked or turned-away subject still has a trackable box.

    Returns rects: (x1, y1, w, h) per frame — the full crop window, so the
    render can drive zoom (w/h) and vertical position (y) as well as x
    (always populated — there's no more GENERAL fallback to skip).

    ``face_identifications`` (PART 4, 6-aug-2026): clip-relative named
    face-ID samples from face_id.identify_faces_in_video. When present, they
    upgrade the weakest link in speaker binding (diarized label -> face) so
    a named speaker whose anchor chain failed still gets framed.
    """
    import numpy as np
    import main as m

    small_w = min(ANALYSIS_MAX_WIDTH, orig_w)
    if small_w % 2:
        small_w -= 1
    small_h = max(int(orig_h * small_w / orig_w), 2)
    if small_h % 2:
        small_h += 1
    scale = orig_w / small_w
    frame_bytes = small_w * small_h * 3

    rects = []
    frame_number = 0
    current_scene_index = 0
    prev_candidates = None
    cached_yolo_candidates = []
    cached_yolo_frame = -10 ** 9
    # Coverage policy state (see the helpers above): the primary subject is
    # whichever diarized speaker talks most; speaker_to_id links an audio
    # label to a tracker candidate id once the talking face is seen; the
    # cutaway cap + return bias stop the camera parking on a reactor.
    primary_label = _primary_speaker_label(speaker_turns)
    # Boost-free pre-pass: per-scene identity of EVERY diarized speaker (the
    # candidate the natural selection holds during their turns). Seeded
    # into speaker_to_id so the audio-assert boost can frame whoever is
    # talking per the transcript, even with a hidden/turned face — the
    # transcript is the basis (user, 31-jul-2026): the host's lines were
    # landing on the girl because only the primary had an anchor.
    speaker_anchors, frame_detections = _learn_speaker_anchors(
        input_video, scenes_boundaries, fps, orig_w, orig_h, speaker_turns)
    # The opening shot's subject, resolved from the first turn's aggregate
    # evidence BEFORE the first frame commits (see _resolve_opening_target).
    # Seeded into the policy so frame 0 lands on the actual opening speaker
    # instead of whatever the instantaneous signals happened to say.
    opening_target_id, opening_target_cx, opening_lock_until = \
        _resolve_opening_target(
            speaker_turns, speaker_anchors, fps,
            asd_speaking_boxes=asd_speaking_boxes)
    # When the anchor pass ran (diarized speakers + learning on), its raw
    # pre-boost detection cache REPLACES this pass's own decode+detect —
    # the whole clip was already decoded and detected once at the same
    # stride (plan item 9; the assert below is the same-stride guarantee).
    # Without diarization/learning there is no cache, so the camera pass
    # falls back to its own decode+detect exactly as before.
    # USE_GESTURE needs actual pixels (pose runs on the detection frame), so
    # the cache-and-replay optimization is bypassed while the flag is on — an
    # acceptable cost for an opt-in feature that is off by default.
    use_detection_cache = bool(frame_detections) and not USE_GESTURE
    total_frames = frame_detections.get('_total_frames') if use_detection_cache else None
    proc = None
    if not use_detection_cache:
        proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-i", input_video,
             "-vf", f"scale={small_w}:{small_h}",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes * 4)
    speaker_to_id = {}
    for scene_anchors in (speaker_anchors or {}).values():
        for label, anchor in scene_anchors.items():
            if anchor.get('id') is not None:
                speaker_to_id.setdefault(label, anchor['id'])
    max_cutaway_frames = int(MAX_CUTAWAY_SECONDS * fps)
    cutaway_frames = 0
    return_bias_armed = False
    zoom_confirm_frames = 0
    zoom_hold_frames = 0
    max_yolo_stale = m.YOLO_FALLBACK_STRIDE * 2
    # Frames left to hold the crop snapped after a scene cut, long enough
    # for one detection sample to land with force_next_update set (see
    # SmoothedCameraman) so the snap lands on the REAL new target.
    cut_grace_frames = 0
    # Which diarized speaker the current ASD face-lock belongs to, and where
    # that face sat when the turn began (see the continuity gate below).
    asd_turn_label = None
    asd_turn_cx = None
    # Eased framing box for the current subject (see subject_policy.
    # stabilize_box). Reset on every real cut so a new shot lands exactly.
    stable_box = None
    # What detection shape (face/body) the eased box was last derived from —
    # lets stabilize_box aim at the FACE box when a subject alternates
    # between a MediaPipe face and a YOLO head-and-chest box (see
    # stabilize_box / §2h(2b)).
    last_target_kind = None
    id_seen_counts = {}
    face_votes = {}  # candidate id -> {name: vote_count}
    face_binding_uses = 0
    # Whoever the camera is currently on. A change here is a hard cut.
    last_target_id = None
    # Frame of the last hard cut, from ANY trigger (directive, tracker
    # switch, or scene change) — see MIN_SHOT_HOLD_SECONDS.
    last_hard_cut_frame = -10 ** 9
    min_shot_hold_frames = max(1, int(MIN_SHOT_HOLD_SECONDS * fps))
    # Who is on screen is decided here, from tiered evidence, instead of by
    # an accumulating size score (see subject_policy).
    import subject_policy
    policy = subject_policy.SubjectPolicy(
        fps, opening_target_id=opening_target_id,
        opening_target_cx=opening_target_cx)
    # Split-screen reaction-cam state (see SPLIT_* constants): per-frame
    # flags + the two subject cell rects, with carry-forward so a hidden or
    # undetected subject keeps their cell (never flickers out mid-beat).
    split_flags = []
    split_top_rects = []
    split_bottom_rects = []
    cur_top = None
    cur_bot = None
    # Raw (non-cell) boxes of the two people, with freshness timestamps —
    # used for the hook two-shot and the split cells. Carry forward so a
    # masked/turned subject keeps their position (never flickers out).
    primary_raw_box = None
    secondary_raw_box = None
    primary_raw_at = -10 ** 9
    secondary_raw_at = -10 ** 9
    top_fresh_at = -10 ** 9
    bot_fresh_at = -10 ** 9
    reaction_split_until = -1
    directive_split_until = -1
    # Frame numbers where a real motion-boosted non-primary candidate was
    # observed — corroboration evidence for a directive-driven split (see
    # SPLIT_REACTOR_CORROBORATION_SECONDS). A directive alone never renders
    # a split; it needs actual reactor motion somewhere near its window.
    motion_reaction_frames = []
    reactor_corroboration_frames = max(
        1, int(SPLIT_REACTOR_CORROBORATION_SECONDS * fps))
    hook_two_shot_active = False
    min_split_frames = max(1, int(SPLIT_MIN_SECONDS * fps))
    merge_split_frames = max(0, int(SPLIT_MERGE_GAP_SECONDS * fps))
    fresh_frames = max(1, int(SPLIT_FRESH_SECONDS * fps))
    two_shot_frames = max(1, int(HOOK_TWO_SHOT_SECONDS * fps))
    hook_jcut_frames = max(1, int(HOOK_NO_JCUT_SECONDS * fps))
    try:
        while True:
            if use_detection_cache:
                if frame_number >= total_frames:
                    break
                frame = None
            else:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape((small_h, small_w, 3))

            scene_changed = False
            if current_scene_index < len(scenes_boundaries):
                start_f, end_f = scenes_boundaries[current_scene_index]
                if frame_number >= end_f and current_scene_index < len(scenes_boundaries) - 1:
                    current_scene_index += 1
                    scene_changed = True

            active_speaker, _jcut = (
                _effective_speaker_label(speaker_turns, frame_number, fps)
                if speaker_turns else (None, False))

            # True only on frames where the detector actually ran, so the
            # cut-grace window below can end on real evidence rather than a
            # fixed frame count.
            fresh_detection_this_frame = (frame_number % m.DETECT_STRIDE == 0)
            if frame_number % m.DETECT_STRIDE == 0:
                if use_detection_cache:
                    # Same-stride guarantee: the anchor pass cached every
                    # DETECT_STRIDE frame (and only those); this pass's
                    # sampler is the same modulo, so every sample must be
                    # present. Any drift is a bug in the cache contract.
                    assert frame_number in frame_detections, (
                        f"anchor detection cache missing frame {frame_number}")
                    candidates = frame_detections[frame_number]
                else:
                    candidates = m.detect_face_candidates(frame)
                    for cand in candidates:
                        cand['box'] = [int(v * scale) for v in cand['box']]
                        cand['score'] = cand['box'][2] * cand['box'][3]
                        cand['raw_score'] = cand['score']

                    # YOLO is heavier than MediaPipe — throttled same as before,
                    # but now a continuous contributor (merged every sample it
                    # runs on) instead of only firing when face detection found
                    # nothing at all. Stale caches (positions from an older pose/
                    # shot) are discounted by age so a slow camera move doesn't
                    # merge a person at their OLD location (see _discount_stale_yolo).
                    if frame_number % m.YOLO_FALLBACK_STRIDE == 0:
                        yolo_candidates = m.detect_person_candidates_yolo(frame)
                        for cand in yolo_candidates:
                            cand['box'] = [int(v * scale) for v in cand['box']]
                            cand['score'] = cand['box'][2] * cand['box'][3]
                            cand['raw_score'] = cand['score']
                        cached_yolo_candidates = yolo_candidates
                        cached_yolo_frame = frame_number
                    candidates = _merge_person_candidates(
                        candidates,
                        _discount_stale_yolo(cached_yolo_candidates,
                                             frame_number - cached_yolo_frame,
                                             max_yolo_stale))

                # BoT-SORT's camera-motion compensation needs the actual pixels
                # to tell a pan apart from people moving. Handed over before the
                # first assign_ids of this frame.
                tracker.set_identity_frame(frame)

                # Reaction-priority: a strongly-reacting non-speaker (a big,
                # sudden head/body movement — laughing, gesturing, mugging
                # for camera) can outscore a passively-framed current
                # speaker, so the tracker's existing size-based selection
                # naturally cuts to them (real edits do this ~30% of the
                # time — RESEARCH_viral_clip_patterns.md, deep pass 31-jul-2026).
                # Stamp ids BEFORE the boosts so the speech boost's identity
                # gate can tell whose mouth is moving (the second assign_ids
                # below re-stamps and snapshots the BOOSTED scores into the
                # tracker's accumulation — idempotent within a frame).
                tracker.assign_ids(candidates, frame_number, orig_w)
                # PART 4 (6-aug-2026): vote named face-ID samples onto the
                # freshly stamped candidate ids. Only runs when FACE_ID_DB is
                # configured (face_identifications is None otherwise).
                _accumulate_face_id_votes(
                    candidates, face_identifications, frame_number, fps,
                    face_votes)
                # How many detections each identity has persisted for. A real
                # person accumulates a long track; a face that is not a person
                # -- a printed photo held up to camera, a poster, a face on a
                # screen -- appears briefly and never builds one. Split cells
                # gate on this (see SPLIT_MIN_TRACK_DETECTIONS) because a split
                # is only worth showing when BOTH halves are established
                # subjects. Ground-truthed: on "Blind Dating by Celebrity
                # Lookalikes" contestants hold up celebrity photos, MediaPipe
                # detects those printed faces, and an ungated split stacked a
                # photograph over a blank wall.
                for _c in candidates:
                    _cid = _c.get("id")
                    if _cid is not None:
                        id_seen_counts[_cid] = id_seen_counts.get(_cid, 0) + 1
                boosted = set()
                _apply_reaction_boost(candidates, prev_candidates, boosted=boosted)
                # Speech-identity: which of these faces is actually the one
                # talking right now (see _apply_speech_activity_boost) —
                # takes priority over the reaction boost above since it's
                # matching audio to a specific face, not just "something
                # moved." Needs prev_candidates BEFORE it's overwritten.
                _update_mouth_activity(candidates, prev_candidates)
                # Identity-gated: the mouth-mover may only be boosted when
                # it is the active speaker's verified candidate (the primary
                # anchor or a confirmed live binding) — or, with no binding
                # yet, only when it's the candidate the camera already
                # frames. A laughing reactor can no longer hijack the crop.
                # LR-ASD first when we have it: lip-sync evidence beats a
                # diarization label that still has to be bound to a face.
                asd_boosted = set()
                asd_id = None
                if asd_speaking_boxes:
                    sec = int(frame_number / fps) if fps else 0
                    if 0 <= sec < len(asd_speaking_boxes):
                        asd_id = _apply_asd_speaker_boost(
                            candidates, asd_speaking_boxes[sec], orig_w,
                            boosted=asd_boosted)
                # DIARIZATION CONTINUITY GATE (measured 4-aug-2026).
                #
                # Within one diarized turn the speaker CANNOT change, but
                # LR-ASD moved to a different face ~21% of the time during a
                # single continuous turn on this source (x~1565 for 17s, then
                # x~1180 for 5s, same speaker, no cut) — it calls a reacting
                # listener the speaker. That rate matched the wrong-person
                # framing in the render almost exactly, and it is what
                # survived removing every cutaway override.
                #
                # So the transcript arbitrates: the first confident ASD face
                # of a turn is held for the rest of it, and a mid-turn jump to
                # a different face is rejected rather than followed. ASD
                # re-decides freely at a turn boundary or a source cut, and
                # with no diarization this is a no-op (per-second behaviour).
                if active_speaker is not None:
                    if active_speaker != asd_turn_label:
                        asd_turn_label = active_speaker
                        # THE OPENING TURN'S LOCK IS THE AGGREGATE (owner
                        # spec, 4-aug-2026, §2h(1)): the first confident ASD
                        # face of the clip's opening turn was measured wrong
                        # (LR-ASD named a silent listener for the first ~2s),
                        # and the frame-0 aggregate commit was then overridden
                        # as soon as that wrong lock engaged. So during the
                        # opening turn, the gate holds the AGGREGATE face
                        # (resolved from the whole clip's evidence for that
                        # speaker) instead of the first confident sample —
                        # the same principle applied to clip start. Normal
                        # first-confident behaviour resumes after the turn.
                        if (opening_lock_until is not None
                                and frame_number < opening_lock_until
                                and opening_target_cx is not None):
                            asd_turn_cx = opening_target_cx
                        else:
                            asd_turn_cx = None
                    if asd_id is not None:
                        _m = next((c for c in candidates
                                   if c.get("id") == asd_id), None)
                        _cx = (_m["box"][0] + _m["box"][2] / 2.0) if _m else None
                        if _cx is not None:
                            if asd_turn_cx is None:
                                asd_turn_cx = _cx
                            elif abs(_cx - asd_turn_cx) > ASD_TURN_LOCK_TOL * orig_w:
                                # Same speaker, different face -> ASD is wrong.
                                # Re-resolve onto the face this turn started on.
                                _lock = min(
                                    candidates,
                                    key=lambda c: abs(
                                        c["box"][0] + c["box"][2] / 2.0 - asd_turn_cx))
                                _lcx = _lock["box"][0] + _lock["box"][2] / 2.0
                                asd_id = (_lock.get("id")
                                          if abs(_lcx - asd_turn_cx)
                                          <= ASD_TURN_LOCK_TOL * orig_w else None)
                elif asd_turn_label is not None:
                    asd_turn_label = None
                    asd_turn_cx = None
                boosted |= asd_boosted
                bound_id = _resolve_speaker_binding(
                    candidates, active_speaker, current_scene_index,
                    speaker_anchors, speaker_to_id, orig_w) if active_speaker else None
                # PART 4: if the anchor chain failed to bind the diarized
                # speaker to a face, let face ID do it — a named candidate
                # whose face-ID name matches the current speaker label. This
                # is the upgrade that makes FACE_ID_DB actually drive framing
                # instead of only renaming transcript labels.
                if bound_id is None and active_speaker:
                    _fid = _face_id_binding(
                        candidates, face_votes, active_speaker,
                        frame_number, fps, face_identifications)
                    if _fid is not None:
                        bound_id = _fid
                        face_binding_uses += 1
                        if face_binding_uses == 1:
                            print(f"   🪪 Face ID binding active — framing "
                                  f"speaker '{active_speaker}' by name")
                # Kept separate from bound_id for the policy: these are two
                # different tiers of evidence (lip-sync names a face on
                # screen; diarization names an audio label that still has to
                # be bound to one), and collapsing them hides which one
                # actually drove a decision. A J-CUT binding (next speaker,
                # pre-roll) goes to its own slot so it can outrank lip-sync
                # for the 0.5s before the new turn's audio starts.
                diarized_id = None if _jcut else bound_id
                jcut_id = bound_id if _jcut else None
                # A confident ASD identification is a better binding than the
                # anchor/position chain that would otherwise resolve it.
                if asd_id is not None:
                    bound_id = asd_id
                speech_boosted = set()
                _apply_speech_activity_boost(candidates, active_speaker,
                                             boosted=speech_boosted,
                                             bound_id=bound_id,
                                             current_target_id=tracker.active_speaker_id)
                boosted |= speech_boosted
                # Hand gestures (round-5 spec 3, OFF by default): evidence of
                # performing, ranked BELOW the LR-ASD/speech evidence above —
                # the audio-assert boost applied next can still override it.
                if USE_GESTURE and frame is not None:
                    gesture_boosted = set()
                    _apply_gesture_boost(candidates, frame, orig_w,
                                         boosted=gesture_boosted)
                    boosted |= gesture_boosted
                # Audio-assert (transcript is the basis, user 31-jul-2026):
                # whoever the diarization says is talking gets framed — even
                # with zero mouth activity (hidden/turned face). Applies to
                # EVERY speaker, not just the primary: the host's lines were
                # landing on the girl because only the primary had an anchor
                # and a mislabeled "speaking" directive could win.
                #
                # NOT gated on a motion reaction happening elsewhere: an
                # earlier version of this suspended the assert whenever
                # ANY other candidate had a motion boost, meant to let a
                # genuine off-transcript group reaction (e.g. someone
                # reacting to being "popped" while off-screen from the
                # bound speaker) win the frame. Ground-truthed regression
                # (1-aug-2026): it also fired whenever the bound speaker's
                # own scene partner made a big gesture (the girl pointing
                # sharply while the host delivered his line right next to
                # her), suppressing the assert and cutting away from the
                # host mid-line — the exact bug this boost exists to
                # prevent. It's unnecessary anyway: _apply_primary_assert_
                # boost already no-ops when the bound speaker isn't a
                # candidate this frame (or is a background sliver — see its
                # own size floor), which is precisely the off-screen-
                # reaction case; a genuine reactor elsewhere already wins
                # normally through the plain reaction boost with nothing
                # to override.
                if bound_id is not None:
                    _apply_primary_assert_boost(candidates, bound_id,
                                                boosted=boosted)
                # Scene-context direction — applied LAST so it outranks every
                # heuristic above. Those are all PROXIES for intent (size,
                # motion, mouth movement, who's diarized); this IS intent:
                # Gemini watched this exact cut and said who the shot is on
                # and why. It's the only signal that can say "frame the guy
                # clowning, he's why they're laughing" (see
                # main.analyze_scene_context). Marking it boosted also tells
                # the coverage rule below this cutaway is deliberate, so the
                # return-to-primary cap won't cut the beat short.
                directive = _directive_at_time(focus_directives, frame_number / fps)
                directed_box = _apply_directive_boost(candidates, directive, orig_w)
                # Transcript-first policy (user 31-jul-2026): diarized speaker
                # labels are deterministic; Gemini's shot direction is not —
                # it repeatedly pointed shots at the wrong person (the host
                # during the contestant's lines, and vice versa) AND
                # hallucinated "causing_reaction" splits with no
                # corroborating motion. While someone is actively talking,
                # ANY directive that would frame someone else is subordinate
                # to the talker: a plain speaking/referenced-style directive
                # is simply dropped (the audio-assert boost already frames
                # the real talker); a causing_reaction/reacting directive
                # means "show both" — dropped as an override too, but it
                # requests a split further down so the talker AND the
                # directed subject are both shown instead of losing the
                # talker to a wrong single-cell target.
                if directed_box is not None and directive is not None and active_speaker is not None:
                    reason = directive.get("reason")
                    if reason in ("causing_reaction", "reacting"):
                        # Split intent (spec 2.1): show cause + reactor when TWO
                        # distinct subjects are available — otherwise fall back
                        # to an override CUT to the cause rather than dropping
                        # the directive entirely ("drop never").
                        if _has_second_subject(candidates, directed_box):
                            directed_box = None  # split, not an override — see below
                    else:
                        # Position-aware for EVERYONE, including the primary:
                        # a directive is kept only when it demonstrably points
                        # at the active speaker's own position (scene anchor
                        # cx, id fast-path via bound_id) — anything farther
                        # than SPEAKER_ANCHOR_TOLERANCE is framing a different
                        # person and loses to the talker. This is the Gemini
                        # visual layer (owner spec: it must drive framing), and
                        # it is still outranked by lip-sync/diarization in the
                        # policy whenever the two disagree.
                        directed_cand = next(
                            (c for c in candidates if c['box'] is directed_box),
                            None)
                        anchor = ((speaker_anchors or {}).get(current_scene_index) or {}).get(active_speaker)
                        if directed_cand is not None:
                            if anchor is not None and anchor.get('cx') is not None:
                                dcx = directed_cand['box'][0] + directed_cand['box'][2] / 2
                                if abs(dcx - anchor['cx']) > SPEAKER_ANCHOR_TOLERANCE * orig_w:
                                    directed_box = None
                            elif bound_id is not None and directed_cand.get('id') != bound_id:
                                directed_box = None
                if directed_box is not None:
                    boosted.add(id(directed_box))
                # (The old `no_boost` flag fed SpeakerTracker's continuity
                # bias — "no signal this frame, so raise stickiness". The
                # policy expresses that directly as tier 5 "hold", so the
                # flag has no remaining consumer.)
                prev_candidates = candidates

                # Re-stamp + snapshot BOOSTED scores into the tracker's
                # accumulation. Must run AFTER the boosts — assign_ids
                # snapshots candidate scores at this moment.
                tracker.assign_ids(candidates, frame_number, orig_w)
                primary_id = speaker_to_id.get(primary_label) if primary_label else None
                # Key-subject override from scene context: the clip's ONE
                # person (by x position) replaces the talk-time primary as
                # the anchor the return-bias / split cells orbit. Tolerance-
                # gated so a wrong position degrades to talk-time fallback.
                effective_primary_id = primary_id
                if primary_subject_x is not None:
                    best_c, best_d = None, CONTEXT_PRIMARY_TOLERANCE
                    for c in candidates:
                        c_cx = (c["box"][0] + c["box"][2] / 2) / orig_w
                        d = abs(c_cx - primary_subject_x)
                        if d < best_d:
                            best_d, best_c = d, c
                    if best_c is not None:
                        effective_primary_id = best_c["id"]

                # Coverage return-bias (Problem 3): a non-boosted cutaway to
                # a secondary subject that overstays gets nudged back to the
                # primary BEFORE selection — one-shot, re-arms only if the
                # switch doesn't take.
                # THE SPEAKER OUTRANKS THE PRIMARY. The "primary" is whoever
                # has the most cumulative talk time, which on a hosted format
                # (game show, dating show, interview) is the HOST — the person
                # holding the mic, and usually the least interesting person to
                # frame. Measured on the Blind Dating source: the host has
                # 33.8% of all speech, so he became primary, and this 3x return
                # bias dragged the camera off whoever was actually answering
                # after only MAX_CUTAWAY_SECONDS (2.5s). That is the "it keeps
                # cutting back to the guy with the mic" failure.
                #
                # The rule was written for a single-protagonist clip (a podcast
                # guest, a monologue) where "return to the subject" is right. It
                # inverts on a conversation. So: never pull off a candidate we
                # have positively identified as the speaker — the whole point of
                # diarization + lip-sync ASD is that this signal is better than
                # a talk-time heuristic.
                # last_target_id is the subject currently on screen (set after
                # each committed switch). target_id is not assigned until the
                # selection block BELOW this point, so using it here would be a
                # NameError on the first detection frame.
                speaker_is_framed = (
                    bound_id is not None and last_target_id == bound_id)
                if speaker_is_framed:
                    cutaway_frames = 0
                    return_bias_armed = False
                if (return_bias_armed and effective_primary_id is not None
                        and not speaker_is_framed):
                    if _apply_primary_return_bias(candidates, effective_primary_id):
                        return_bias_armed = False

                # A directive is authoritative, not a vote, so it normally
                # bypasses the tracker's own SWITCH_COOLDOWN_FRAMES entirely —
                # but that let back-to-back directive windows (Gemini emits
                # one every ~2-3s) each force an instant hard cut regardless
                # of how recently the frame already cut, compounding with the
                # source's own fast multi-cam editing into a noticeably more
                # aggressive pace (ground-truthed 1-aug-2026: 7 cuts in the
                # first 15s). MIN_SHOT_HOLD_SECONDS is a second, lower floor
                # that applies no matter what's triggering the switch —
                # skipped for a genuine SOURCE scene cut (always honored),
                # AND skipped whenever the directive agrees with the
                # diarized active speaker (directive_target_id == bound_id).
                # That second exception is load-bearing, not an optimization:
                # gating a directive falls back to tracker.get_target_id's
                # accumulating score, which — per the comment below — can
                # take MANY frames to overcome a sticky incumbent even
                # against a 4x assert boost. Gating the one directive that
                # was correctly putting the audio-asserted speaker on screen
                # reintroduced exactly the "never accumulates enough to beat
                # a speaker who's held one stable id" bug that directives
                # exist to bypass (ground-truthed 1-aug-2026: gating this
                # unconditionally dropped the host from his own ~1.8s turn
                # entirely — worse than the aggressive-pacing complaint this
                # was meant to fix).
                directive_target_id = None
                if directed_box is not None:
                    directive_target_id = next(
                        (c['id'] for c in candidates if c['box'] is directed_box),
                        None)
                # SEMANTIC REACTION TRIGGER (owner spec, 4-aug-2026, §2h(3)):
                # a referenced/causing_reaction directive IS the dialogue
                # deliberately drawing attention to someone — a transcript
                # signal, not motion. The match is computed independently of
                # the directed_box gating above (a causing_reaction directive
                # may be demoted to split intent or dropped as an override
                # while the reaction beat still needs to know WHO was
                # pointed at). The policy gives this candidate a bounded
                # reaction shot, then the speaker reclaims. No directive ->
                # None -> no semantic reaction (fail open).
                referenced_id = None
                if directive is not None and directive.get("reason") in (
                        "referenced", "causing_reaction"):
                    _ref_cand = _directive_matched_candidate(
                        candidates, directive, orig_w)
                    if _ref_cand is not None:
                        referenced_id = _ref_cand.get("id")
                # Mouth motion: the weakest speaker signal we have. Recovered
                # from the speech boost's own pick so the policy can rank it
                # explicitly instead of it arriving as an anonymous multiplier.
                mouth_id = next((c.get('id') for c in candidates
                                 if id(c['box']) in speech_boosted), None)
                # THE DECISION (see subject_policy). Everything above this
                # point now PRODUCES evidence; this is the single place that
                # consumes it. The boosts still run because the split-screen,
                # zoom and coverage bookkeeping below read the `boosted` set —
                # but they no longer decide who is on screen, because a
                # multiplicative score could not: measured on this source,
                # raising the lip-sync boost 250x moved framing accuracy only
                # 61% -> 70%, while removing the hysteresis that was damping
                # it took the same clip to 97%.
                shot_before = policy.shot_started
                evidence = subject_policy.Evidence(
                    asd_id=asd_id,
                    diarized_id=diarized_id,
                    jcut_id=jcut_id,
                    # No j-cut pre-roll during the opening hook
                    # (owner spec, 4-aug-2026).
                    in_hook=(frame_number < hook_jcut_frames),
                    directive_id=directive_target_id,
                    directive_reason=(directive or {}).get('reason'),
                    referenced_id=referenced_id,
                    mouth_id=mouth_id,
                    scene_changed=scene_changed)
                target_box, target_id, decision_tier = policy.decide(
                    candidates, evidence, frame_number, orig_w)
                # Banter two-shot: a strong switch the floor blocked (the
                # policy HELD) means the incoming speaker is already visible —
                # frame both instead of lagging (see _banter_two_shot_eligible).
                banter_blocked = _banter_two_shot_eligible(
                    decision_tier, evidence, target_id)
                # A genuinely new shot, as judged by the policy — the signal
                # the hard cut below keys off.
                subject_changed = (policy.shot_started != shot_before
                                   and target_box is not None)
                # Keep tracker state coherent for the consumers that still
                # read it — the speech boost's identity gate
                # (current_target_id) and the cooldown bookkeeping.
                if target_id is not None and target_id != tracker.active_speaker_id:
                    tracker.active_speaker_id = target_id
                    tracker.last_switch_frame = frame_number
                # Hook two-shot: for the first ~2s frame BOTH people in the
                # single 3:4 crop (they fit — that's the point; no split
                # screen on the open). Centered between the key subject and
                # the other prominent person, no zoom, and treated as
                # on-primary so coverage bookkeeping stays quiet.
                # The opening two-shot must NEVER displace an identified
                # speaker. It anchors on effective_primary_id — the talk-time
                # primary — which on a hosted format IS the host, so on a clip
                # that opens with a guest answering, the guest was not on
                # screen at all for the first 1.8s (owner report, 4-aug-2026:
                # "he wasn't even on screen to begin with, the host was being
                # prioritized over him"). The clip was selected FOR what this
                # person says; the open is the least acceptable place to frame
                # someone else. Falls back to the two-shot only when nothing
                # has identified a speaker yet.
                _op_id, _op_tier = (evidence.proposal() if evidence is not None
                                    else (None, None))
                opening_speaker_known = (
                    _op_tier in subject_policy.STRONG_TIERS
                    and _by_id_present(candidates, _op_id))
                two_shot = ((frame_number < two_shot_frames
                             and not opening_speaker_known
                             and primary_raw_box is not None
                             and secondary_raw_box is not None
                             and frame_number - primary_raw_at < fresh_frames
                             and frame_number - secondary_raw_at < fresh_frames)
                            or (banter_blocked and len(candidates) >= 2))
                if two_shot:
                    # Compose the two-shot ONLY on entry; while it is active
                    # the rect is LOCKED (no recompute, no re-snap). The old
                    # per-detection recompute moved the crop every stride and
                    # read as jitter on playback (owner report, 4-aug-2026:
                    # 'the jittering used to be less').
                    if not hook_two_shot_active:
                        if banter_blocked:
                            # Banter two-shot (owner tip 1.3): the floor
                            # blocked a strong switch, so the incoming speaker
                            # is already visible. Pair the HELD subject with
                            # that incoming speaker in one 3:4 crop instead of
                            # lagging on one of them through the floor.
                            held_cand = None
                            if policy.target_box is not None:
                                held_cand = next(
                                    (c for c in candidates
                                     if subject_policy.same_subject(
                                         tuple(c["box"]), tuple(policy.target_box),
                                         orig_w)),
                                    None)
                            w_id, _wt = evidence.proposal()
                            incoming = next(
                                (c for c in candidates if c.get("id") == w_id),
                                None)
                            if (held_cand is not None and incoming is not None
                                    and held_cand is not incoming):
                                primary_raw_box = held_cand["box"]
                                secondary_raw_box = incoming["box"]
                                primary_raw_at = frame_number
                                secondary_raw_at = frame_number
                                target_box = _hook_union(
                                    primary_raw_box, secondary_raw_box,
                                    cameraman.crop_width, orig_w, orig_h)
                                target_id = held_cand.get("id")
                        else:
                            # Hook two-shot: frame the star + the OTHER person
                            # who can actually share the fixed 3:4 crop. The
                            # naive biggest-other rule pairs the star with a
                            # distant third person, whose union is wider than
                            # the crop — the crop then falls on the gap and
                            # shows neither subject (ground-truthed
                            # 31-jul-2026).
                            target_box = None
                            prim = None
                            if effective_primary_id is not None:
                                prim = next(
                                    (c for c in candidates
                                     if c.get("id") == effective_primary_id),
                                    None)
                            if prim is not None:
                                sec = _hook_secondary(
                                    prim, candidates,
                                    cameraman.crop_width, orig_w, orig_h,
                                    exclude_ids={effective_primary_id})
                                if sec is not None:
                                    primary_raw_box = prim["box"]
                                    secondary_raw_box = sec["box"]
                                    primary_raw_at = frame_number
                                    secondary_raw_at = frame_number
                                    top_fresh_at = frame_number
                                    bot_fresh_at = frame_number
                                    target_box = _hook_union(
                                        primary_raw_box, secondary_raw_box,
                                        cameraman.crop_width, orig_w, orig_h)
                                    target_id = effective_primary_id
                            if target_box is None:
                                target_box = _union_box(
                                    primary_raw_box, secondary_raw_box,
                                    orig_w, orig_h)
                                target_id = effective_primary_id
                        if target_box is not None:
                            # Land the two-shot NOW instead of waiting out the
                            # jump-confirm gate (the pair is stable — the gate
                            # exists for false-positive detector jumps, not a
                            # deliberate composition).
                            cameraman.force_next_update = True
                            hook_two_shot_active = True
                    # else: the two-shot is already locked — keep the existing
                    # target_box so the crop does not move until it ends.
                else:
                    hook_two_shot_active = False
                if target_box:
                    # Bind the audio label to the talking face (only the
                    # speech-identity boost ties audio to a specific face).
                    if active_speaker and id(target_box) in speech_boosted:
                        # Never let a later boost overwrite the primary
                        # anchor — the anchor is the pre-pass's verified
                        # identity; live binds only fill gaps (e.g. a
                        # non-primary speaker confirmed mouth-consistent).
                        if active_speaker not in speaker_to_id:
                            speaker_to_id[active_speaker] = target_id
                    primary_id = speaker_to_id.get(primary_label) if primary_label else None
                    boosted_now = id(target_box) in boosted
                    if two_shot:
                        cutaway_frames = 0
                        return_bias_armed = False
                    elif effective_primary_id is not None and target_id != effective_primary_id:
                        # Universal coverage rule: a secondary subject is a
                        # brief reaction cutaway. The cap only accumulates on
                        # NON-boosted frames, so a sustained reaction extends
                        # the hold instead of cutting back mid-beat.
                        if boosted_now:
                            cutaway_frames = 0
                        else:
                            cutaway_frames += 1
                            if cutaway_frames > max_cutaway_frames:
                                return_bias_armed = True
                    else:
                        cutaway_frames = 0
                        return_bias_armed = False
                    zoom_decision = 1.0 if two_shot else _zoom_for_target(
                        candidates, target_box, boosted)
                    # A directed shot carries its own push-in level: a payoff
                    # beat (the cause of a reaction, or whoever was just
                    # pointed at) commits tighter than a talking head.
                    if directed_box is not None and target_box is directed_box:
                        zoom_decision = min(zoom_decision, _directive_zoom(directive))
                    zoom_target, zoom_confirm_frames = _zoom_confirm(
                        zoom_decision, zoom_confirm_frames, ZOOM_CONFIRM_CYCLES)
                    zoom_target, zoom_hold_frames = _zoom_hold_decision(
                        zoom_target, zoom_hold_frames, ZOOM_HOLD_CYCLES)
                    # What the CAMERA is aimed at is an eased version of the
                    # detection, not the raw box (see stabilize_box): the same
                    # person alternates between a MediaPipe face box and a
                    # YOLO head-and-chest box, and snapping between the two
                    # moves the crop every detection. `target_box` itself is
                    # left untouched — the boosted set, split cells and the
                    # directive check all match on its identity.
                    # AIM AT THE FACE BOX (owner spec, 4-aug-2026, §2h(2b)):
                    # blending the whole box dragged the aim between the face
                    # centre and the body's lower centre on every alternation,
                    # so the crop wandered inside a held shot. The centre is
                    # the face's position; only the size is eased.
                    if hook_two_shot_active:
                        # A union box is neither a face nor a body — do not
                        # let a stale kind from before the two-shot pull the
                        # aim somewhere the policy never chose.
                        last_target_kind = None
                    target_kind = next(
                        (c.get("kind") for c in candidates
                         if c["box"] is target_box), None)
                    aim_box = subject_policy.stabilize_box(
                        target_box, None if subject_changed else stable_box,
                        new_kind=target_kind, prev_kind=last_target_kind)
                    stable_box = aim_box
                    if target_kind is not None:
                        last_target_kind = target_kind
                    # None = confirming/holding: keep the current zoom target.
                    if zoom_target is not None:
                        cameraman.update_target(aim_box,
                                                zoom_target=zoom_target)
                    else:
                        cameraman.update_target(aim_box)

                    # A change of subject is a CUT, not a journey. Reference
                    # edits in this genre are 100% hard cuts between people
                    # (camera-movement research, 31-jul-2026), so bypass the
                    # jump-confirm gate and land the new framing on this very
                    # frame instead of easing across the room to reach it.
                    #
                    # "Changed subject" is the POLICY's judgement, not an id
                    # comparison. Ids churn constantly (a face and the body it
                    # belongs to, a confirmed track and its provisional
                    # replacement), and the policy already recognises those as
                    # the same person. Keying the hard cut off the raw id
                    # re-snapped the camera on every relabel — the same
                    # sub-0.2s flicker the policy exists to prevent, injected
                    # after the decision was correctly made.
                    if subject_changed:
                        cameraman.force_next_update = True
                        cameraman.update_target(target_box)
                        cut_grace_frames = max(cut_grace_frames, 1)
                        last_target_id = target_id
                        last_hard_cut_frame = frame_number

                    # --- split-screen state (reaction beats only) ---
                    # Top cell = key subject (the clip's star); bottom cell =
                    # the other prominent person (the reactor). A reaction
                    # beat = a NON-primary candidate that took the motion
                    # boost while the camera is on someone else — the exact
                    # moment a real editor would cut to the reaction. The
                    # hook is deliberately NOT split (two-shot instead).
                    if SPLIT_SCREEN:
                        # Top cell = the primary when one is known, otherwise
                        # the strongest candidate. Requiring effective_primary_id
                        # meant cur_top was NEVER built on footage where no
                        # primary resolves (diarization collapsed, or the scene
                        # director returned no primary_subject_x) — measured on
                        # real footage: top-cell built 0x, bottom-cell 273x, so
                        # cells_ready was permanently False and split screen
                        # still could not render even after the id() fix. The
                        # bottom cell already falls back this way; the top cell
                        # now matches it.
                        prim_cand = None
                        if effective_primary_id is not None:
                            prim_cand = next(
                                (c for c in candidates
                                 if c.get("id") == effective_primary_id),
                                None)
                        if prim_cand is None and candidates:
                            # Established subjects only — a transient face
                            # (held-up photo) must never win a split cell.
                            established = [
                                c for c in candidates
                                if id_seen_counts.get(c.get("id"), 0)
                                >= SPLIT_MIN_TRACK_DETECTIONS]
                            if established:
                                prim_cand = max(established,
                                                key=lambda c: c.get("score", 0))
                        if prim_cand is not None:
                            primary_raw_box = prim_cand["box"]
                            primary_raw_at = frame_number
                            cur_top = _subject_cell_rect(
                                prim_cand["box"], orig_w, orig_h,
                                aspect=cell_aspect)
                            top_fresh_at = frame_number
                        sec_cand = None
                        sec_best = -1.0
                        for c in candidates:
                            # Exclude whoever is in the TOP cell, by identity
                            # when we have one and by box otherwise. Excluding
                            # only effective_primary_id let the fallback top
                            # cell reappear as the bottom cell — the same face
                            # stacked on itself.
                            if prim_cand is not None:
                                if (c is prim_cand
                                        or (c.get("id") is not None
                                            and c.get("id") == prim_cand.get("id"))):
                                    continue
                            elif (effective_primary_id is not None
                                    and c.get("id") == effective_primary_id):
                                continue
                            # Same establishment gate as the top cell.
                            if (id_seen_counts.get(c.get("id"), 0)
                                    < SPLIT_MIN_TRACK_DETECTIONS):
                                continue
                            if c["score"] > sec_best:
                                sec_best = c["score"]
                                sec_cand = c
                        if sec_cand is not None:
                            secondary_raw_box = sec_cand["box"]
                            secondary_raw_at = frame_number
                            cur_bot = _subject_cell_rect(
                                sec_cand["box"], orig_w, orig_h,
                                aspect=cell_aspect)
                            bot_fresh_at = frame_number
                        reaction_beat = _is_reaction_beat(
                            candidates, boosted, speech_boosted, target_box,
                            effective_primary_id, directed_box)
                        if reaction_beat:
                            # Hold the split through the detection-stride gap
                            # so consecutive samples read as one interval.
                            reaction_split_until = frame_number + m.DETECT_STRIDE
                            # Corroboration evidence for a nearby directive-
                            # driven split (see SPLIT_REACTOR_CORROBORATION_
                            # SECONDS) — a real motion-boosted reactor was
                            # actually observed here, not just claimed.
                            motion_reaction_frames.append(frame_number)
                        # Semantic reaction beats: when the scene-context
                        # director says someone is CAUSING a reaction (or is
                        # reacting hard enough to deserve its own shot), stack
                        # the two people — causer + reactor — for the whole
                        # beat. Motion-only detection misses these when a face
                        # is masked/occluded (ground-truthed 31-jul-2026).
                        if (directive is not None
                                and directive.get("reason") in ("causing_reaction", "reacting")):
                            directive_split_until = directive["end"] * fps

            if scene_changed or frame_number == 0:
                cameraman.force_next_update = True
                cut_grace_frames = m.DETECT_STRIDE + 1
                last_hard_cut_frame = frame_number
                scene_anchors = (speaker_anchors or {}).get(current_scene_index) or {}
                for label, anchor in scene_anchors.items():
                    if anchor.get('id') is not None:
                        speaker_to_id[label] = anchor['id']
            is_scene_start = (
                current_scene_index < len(scenes_boundaries)
                and frame_number == scenes_boundaries[current_scene_index][0])
            force_snap = is_scene_start or cut_grace_frames > 0
            if cut_grace_frames > 0:
                cut_grace_frames -= 1
                # End the grace window the moment we've snapped onto a FRESH
                # detection. The window exists to skip the stale-target crawl
                # right after a cut — the first detection may still describe
                # the OLD shot, so one frame isn't enough. But holding
                # force_snap for the whole window re-snaps the crop on every
                # frame, and a new detection landing mid-window (they arrive
                # every DETECT_STRIDE frames) moves the target and produces a
                # SECOND hard jump 1-4 frames after the first.
                #
                # Measured on the user's clip: 45 output cuts against 28 in the
                # source, only 13 of them inherited — 32 added, including pairs
                # 0.04s apart (a single-frame flicker). MIN_SHOT_HOLD_SECONDS
                # cannot prevent those because force_snap bypasses it entirely.
                if fresh_detection_this_frame:
                    cut_grace_frames = 0
            x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=force_snap)
            rects.append((x1, y1, x2 - x1, y2 - y1))

            if SPLIT_SCREEN:
                motion_active = frame_number < reaction_split_until
                directive_active = frame_number < directive_split_until
                both_fresh = (frame_number - top_fresh_at < fresh_frames
                              and frame_number - bot_fresh_at < fresh_frames)
                cells_ready = (both_fresh and cur_top is not None
                              and cur_bot is not None)
                directive_ok = (directive_active and cells_ready
                                and _directive_split_corroborated(
                                    frame_number, motion_reaction_frames,
                                    reactor_corroboration_frames,
                                    cur_top, cur_bot, orig_w))
                want = (motion_active and cells_ready) or directive_ok
                if want:
                    split_flags.append(True)
                    if SPLIT_PRIMARY_TOP:
                        split_top_rects.append(cur_top)
                        split_bottom_rects.append(cur_bot)
                    else:
                        split_top_rects.append(cur_bot)
                        split_bottom_rects.append(cur_top)
                else:
                    split_flags.append(False)
                    fallback_rect = (x1, y1, x2 - x1, y2 - y1)
                    split_top_rects.append(cur_top if cur_top is not None
                                           else fallback_rect)
                    split_bottom_rects.append(cur_bot if cur_bot is not None
                                              else fallback_rect)
                    # NOTE: a two-shot "widen instead of cut" pass used to run
                    # here (round-5 spec 2.2). REMOVED — measured as the source
                    # of severe camera shake. This block sits outside the
                    # detection-frame guard, and both `reaction_beat` and
                    # `directive_active` persist between detections, so it
                    # re-targeted the camera and set force_next_update on EVERY
                    # frame of a reaction/directive window. force_snap is an
                    # unconditional hard jump (main.SmoothedCameraman.
                    # get_crop_box), so this produced 23 forced snaps in a 20s
                    # clip — more than one per second, on top of the source's
                    # own cuts. It was harmless only while reaction_beat was
                    # permanently False; fixing that bug made it fire.
                    #
                    # If re-introduced: gate it to detection frames only, skip
                    # it when the two-shot target hasn't meaningfully moved,
                    # and never set force_next_update from it.

            frame_number += 1
    finally:
        if proc is not None:
            proc.stdout.close()
            proc.wait()

    # What actually decided the framing. A regression here (e.g. "size 60%")
    # is visible in the render log without bisecting anything.
    if policy.tier_counts:
        print(f"   🎯 Framing evidence: {policy.summary()}")

    split_info = None
    if SPLIT_SCREEN and any(split_flags):
        ranges = _split_ranges_from_flags(
            split_flags, fps, min_split_frames, merge_split_frames,
            SPLIT_MAX_SECONDS)
        if ranges:
            split_info = {
                "ranges": ranges,
                "top": split_top_rects,
                "bottom": split_bottom_rects,
            }
    return rects, split_info


# --- render -----------------------------------------------------------------

def _run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=1800)


def render(input_video, final_output_video, aspect_ratio,
          transcript=None, clip_start=0.0, clip_end=None,
          focus_directives=None, primary_subject_x=None):
    """Full v2 reframe of one clip. Raises on failure (caller falls back).

    Unified 3:4-consistent crop (see UNIFIED_CROP_RATIO): ONE composition
    style for the whole clip — a fixed-shape crop window, letterboxed into
    the output canvas, repositioned per scene/speaker — instead of cutting
    between a narrow single-subject crop and a separate wide group layout.

    transcript/clip_start/clip_end: optional. When the caller has the
    source video's full transcript plus this clip's absolute time range,
    AssemblyAI-diarized speaker-turn timing informs SpeakerTracker's
    active-speaker hysteresis (see main.speaker_change_frames) — the camera
    stops fighting a genuine switch in who's talking with artificial
    stickiness. Without them, tracking is pure-visual, exactly as before.

    focus_directives: optional shot direction from the scene-context layer
    (main.analyze_scene_context) — clip-relative spans naming who the camera
    should be on and why. Outranks every visual heuristic when present; when
    absent, framing is exactly as it was.
    """
    import main as m

    print("   🚀 Reframe engine v2 (ffmpeg-native render)")
    supersample = max(1, int(os.environ.get("CROP_SUPERSAMPLE", "1")))
    scenes, fps = m.detect_scenes(input_video)
    fps = float(fps)  # PySceneDetect can hand back a Fraction
    orig_w, orig_h = m.get_video_resolution(input_video)

    out_w, out_h = delivery_size(orig_w, orig_h, aspect_ratio)

    if not scenes:
        import cv2
        cap = cv2.VideoCapture(input_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        from scenedetect import FrameTimecode
        scenes = [(FrameTimecode(0, fps), FrameTimecode(total, fps))]

    scene_boundaries = [(s.get_frames(), e.get_frames()) for s, e in scenes]

    # The crop SHAPE is fixed (3:4, see UNIFIED_CROP_RATIO) regardless of the
    # delivery aspect_ratio — SmoothedCameraman derives crop_width/height
    # from video_width/height and this ratio only, never from out_w/out_h.
    cameraman = m.SmoothedCameraman(out_w, out_h, orig_w, orig_h,
                                    aspect_ratio=UNIFIED_CROP_RATIO, fps=fps,
                                    supersample=supersample)
    _clip_end = clip_end if clip_end is not None else clip_start + (scene_boundaries[-1][1] / fps)
    # LR-ASD: who is actually speaking, from lip movement synced to audio.
    # Clip-scoped on purpose — see asd_worker.score_clip. Entirely optional:
    # any failure just leaves the existing diarization path in charge.
    asd_speaking_boxes = None
    asd_per_second = None
    if USE_ASD:
        try:
            import asd_worker
            if asd_worker.available():
                t_asd = time.time()
                asd = asd_worker.score_clip(input_video, m.detect_face_candidates)
                asd_speaking_boxes = asd.get("per_second_box") or None
                asd_per_second = asd.get("per_second") or None
                if asd_speaking_boxes:
                    named = sum(1 for b in asd_speaking_boxes if b)
                    print(f"   🗣️  LR-ASD: speaker located in {named}/"
                          f"{len(asd_speaking_boxes)}s ({time.time() - t_asd:.1f}s)")
        except Exception as e:
            print(f"   ⚠️ LR-ASD unavailable ({type(e).__name__}: {e}); "
                  "falling back to diarization for speaker identity")

    unlock_frames = (m.speaker_change_frames(transcript, clip_start, _clip_end, fps)
                     if transcript else set())
    # LR-ASD turns unlock the camera too. The sticky hysteresis exists to stop
    # the camera chasing unreliable evidence — but a lip-sync identification is
    # not unreliable evidence, it is the best signal in the system. Suppressing
    # stickiness at ASD-detected turns is what lets the ASD work actually reach
    # the screen instead of being damped out by hysteresis tuned for a world
    # without it. This UNLOCKS (removes an artificial bonus); it never forces a
    # switch, and the jitter-tuned cooldown stays fully intact either way.
    if asd_speaking_boxes:
        prev = None
        for sec, tid in enumerate(asd_per_second or []):
            if tid is not None and prev is not None and tid != prev:
                turn_f = int(sec * fps)
                unlock_frames |= {turn_f + d for d in range(-2, 3)}
            if tid is not None:
                prev = tid
    # NOTE: deliberately NOT raising cooldown_frames to match
    # MIN_SHOT_HOLD_SECONDS here. Tried that (1-aug-2026) and it regressed
    # the audio-assert boost: when a directive or scene change had recently
    # forced the tracker onto the WRONG target, the tracker's own cooldown
    # then blocked the assert-boosted (correct, much higher-scored)
    # candidate from reclaiming the frame for the full cooldown window —
    # long enough to eat most of a short (~2s) speaker turn. This cooldown
    # was already deliberately tuned (see SWITCH_COOLDOWN_FRAMES comment)
    # against real jitter; MIN_SHOT_HOLD_SECONDS only needs to close the
    # DIRECTIVE-bypass loophole (directives set active_speaker_id/
    # last_switch_frame directly, skipping this cooldown entirely) — see
    # the gate in _analyze_trajectory.
    # Real multi-object tracking owns identity when available (see
    # identity_tracker). Its clock is the DETECTION rate, not the video frame
    # rate — the pipeline only detects every DETECT_STRIDE frames, and feeding
    # a Kalman tracker the wrong rate scales all its velocities and its
    # track_buffer by the stride. Falls back to the legacy x-position matcher
    # if the backend isn't importable, so tracking never disappears entirely.
    identity = None
    if USE_TRACKER_IDENTITY:
        try:
            import identity_tracker
            if identity_tracker.available():
                identity = identity_tracker.IdentityTracker(
                    detection_fps=max(1.0, fps / float(m.DETECT_STRIDE)),
                    tracker_type=TRACKER_IDENTITY_BACKEND)
                print(f"   🆔 Identity: {TRACKER_IDENTITY_BACKEND} "
                      f"@ {fps / float(m.DETECT_STRIDE):.1f} det/s")
        except Exception as e:
            print(f"   ⚠️ Identity tracker unavailable ({e}); using position matching")
    tracker = m.SpeakerTracker(cooldown_frames=SWITCH_COOLDOWN_FRAMES,
                                 unlock_frames=unlock_frames,
                                 identity=identity)
    # Dynamic reaction-camera signal: crop in on whoever audio diarization
    # shows is actively talking instead of staying static — see
    # _analyze_trajectory and RESEARCH_viral_clip_patterns.md §1/§6/§8.
    speaker_turns = (m.speaker_turn_frame_ranges(transcript, clip_start, _clip_end, fps,
                                                 scene_boundaries[-1][1])
                     if transcript else None)

    # PART 4 (6-aug-2026): named face-ID samples for THIS clip (clip-relative
    # timestamps, so they match the loop's frame clock). Opt-in and fail-open:
    # without FACE_ID_DB, face_identifications stays None and framing is
    # byte-identical to before. The DB/model are cached process-wide by
    # face_id.get_known_faces_db, so the per-clip cost is the 2fps scan only.
    face_identifications = None
    if face_id.available():
        try:
            _db = face_id.get_known_faces_db()
            if _db is not None and _db.embeddings:
                face_identifications = face_id.identify_faces_in_video(
                    input_video, _db,
                    sample_fps=float(
                        os.environ.get("FACE_ID_SAMPLE_FPS", "2")),
                    threshold=float(
                        os.environ.get("FACE_ID_THRESHOLD", "0.4")))
                if face_identifications:
                    _names = sorted({i["name"] for i in face_identifications})
                    print(f"   🪪 Face ID: {len(face_identifications)} named "
                          f"sample(s) [{', '.join(_names)}] — upgrading "
                          "speaker binding")
        except Exception as e:
            print(f"   ⚠️ Face ID clip pass failed ({e}) — "
                  "framing without names")

    rects, split_info = _analyze_trajectory(
        input_video, scene_boundaries, fps, orig_w, orig_h,
        cameraman, tracker, speaker_turns=speaker_turns,
        focus_directives=focus_directives,
        primary_subject_x=primary_subject_x,
        asd_speaking_boxes=asd_speaking_boxes,
        cell_aspect=split_cell_aspect(out_w, out_h),
        face_identifications=face_identifications)
    if not rects:
        raise RuntimeError("analysis produced no frames")

    # Path instrumentation (PART 1.1): REFRAME_DUMP_PATH=<dir> writes the
    # emitted per-frame crop rects (source pixels, pre-supersample) so camera
    # jitter can be diagnosed numerically instead of by eyeballing compressed
    # video — the plan's measurement rule (HANDOFF_FRAMING.md §2e).
    dump_dir = os.environ.get("REFRAME_DUMP_PATH", "").strip()
    if dump_dir:
        try:
            import numpy as np
            os.makedirs(dump_dir, exist_ok=True)
            tag = os.path.splitext(os.path.basename(final_output_video))[0]
            np.save(os.path.join(dump_dir, f"{tag}_rects.npy"),
                    np.asarray(rects, dtype=float))
            print(f"   📐 REFRAME_DUMP_PATH: wrote {tag}_rects.npy "
                  f"({len(rects)} rects)")
        except Exception as e:
            print(f"   ⚠️ REFRAME_DUMP_PATH failed ({e}) — continuing")

    # Base crop (zoom=1.0) sizes for the filtergraph's initial crop@c; the
    # sendcmd file overrides w/h/x/y at every change-point, so zoom levels
    # are driven entirely by the trajectory rects.
    crop_w, crop_h = cameraman.crop_width, cameraman.crop_height
    initial_x, initial_y = rects[0][0], rects[0][1]
    workdir = tempfile.mkdtemp(prefix="reframe_v2_")
    try:
        ss = supersample
        # Crop in 2x space so sub-pixel (1/ss source px) steps become whole
        # pixels for ffmpeg's integer crop filter. The camera already emitted
        # rects on the 1/ss grid (SmoothedCameraman._q); scaling by ss and
        # rounding makes them exact 2x-pixel coordinates.
        cmd_rects = [(int(round(x * ss)), int(round(y * ss)),
                      int(round(w * ss)), int(round(h * ss)))
                     for (x, y, w, h) in rects]
        cmd_path = os.path.join(workdir, "cmd.txt")
        with open(cmd_path, "w") as f:
            f.write("\n".join(dedupe_sendcmd_lines(cmd_rects, fps)) + "\n")
        graph = unified_filtergraph(out_w, out_h,
                                    crop_w * ss, crop_h * ss, cmd_path,
                                    int(round(initial_x * ss)),
                                    initial_y=int(round(initial_y * ss)),
                                    supersample=ss)
        if split_info:
            # Fixed-size cell crop (never resized — see split_cell_size), so
            # each cell gets TWO sendcmd files: x/y reposition the crop, w/h
            # drive the downstream zoom scale. Sending w/h to crop itself is
            # the ffmpeg trac #10984 freeze bug this split avoids.
            cell_w, cell_h = split_cell_size(
                orig_w, orig_h, aspect=split_cell_aspect(out_w, out_h))
            # Supersampling scales every SOURCE-space value (rects, cell
            # dims, source bounds) into 2x space; the zoom-scale targets and
            # the final cell crops stay in output pixels, so the scale ratios
            # inside cell_* helpers cancel out and the graph stays consistent.
            def _ss_rects(rs):
                return [(int(round(a * ss)), int(round(b * ss)),
                         int(round(c * ss)), int(round(d * ss)))
                        for (a, b, c, d) in rs]
            top_rects = _ss_rects(split_info["top"])
            bot_rects = _ss_rects(split_info["bottom"])
            cell_w, cell_h = cell_w * ss, cell_h * ss
            ss_orig = (orig_w * ss, orig_h * ss)
            half_h = out_h // 2
            top_xy_path = os.path.join(workdir, "top_xy.txt")
            top_zoom_path = os.path.join(workdir, "top_zoom.txt")
            bot_xy_path = os.path.join(workdir, "bot_xy.txt")
            bot_zoom_path = os.path.join(workdir, "bot_zoom.txt")
            top_frame_path = os.path.join(workdir, "top_frame.txt")
            bot_frame_path = os.path.join(workdir, "bot_frame.txt")
            with open(top_xy_path, "w") as f:
                f.write("\n".join(cell_xy_sendcmd_lines(
                    top_rects, fps, "crop@ct", cell_w, cell_h,
                    *ss_orig)) + "\n")
            with open(bot_xy_path, "w") as f:
                f.write("\n".join(cell_xy_sendcmd_lines(
                    bot_rects, fps, "crop@cb", cell_w, cell_h,
                    *ss_orig)) + "\n")
            with open(top_zoom_path, "w") as f:
                f.write("\n".join(cell_zoom_sendcmd_lines(
                    top_rects, fps, "scale@st", cell_w, cell_h,
                    out_w, half_h)) + "\n")
            with open(bot_zoom_path, "w") as f:
                f.write("\n".join(cell_zoom_sendcmd_lines(
                    bot_rects, fps, "scale@sb", cell_w, cell_h,
                    out_w, half_h)) + "\n")
            # Final-crop tracking so the head stays at a fixed height in the
            # cell instead of drifting out of a centred crop.
            with open(top_frame_path, "w") as f:
                f.write("\n".join(cell_frame_sendcmd_lines(
                    top_rects, fps, "crop@ft", cell_w, cell_h,
                    *ss_orig, out_w, half_h)) + "\n")
            with open(bot_frame_path, "w") as f:
                f.write("\n".join(cell_frame_sendcmd_lines(
                    bot_rects, fps, "crop@fb", cell_w, cell_h,
                    *ss_orig, out_w, half_h)) + "\n")
            enable_expr = "+".join(
                f"between(t,{s},{e})" for s, e in split_info["ranges"])
            init_top = cell_initial_xy(top_rects[0], cell_w, cell_h,
                                       *ss_orig) + cell_scale_target(
                top_rects[0], cell_w, cell_h, out_w, half_h)
            init_bottom = cell_initial_xy(
                bot_rects[0], cell_w, cell_h,
                *ss_orig) + cell_scale_target(
                bot_rects[0], cell_w, cell_h, out_w, half_h)
            init_top_frame = cell_frame_xy(
                top_rects[0], init_top[:2], cell_w, cell_h,
                init_top[2:], out_w, half_h)
            init_bottom_frame = cell_frame_xy(
                bot_rects[0], init_bottom[:2], cell_w, cell_h,
                init_bottom[2:], out_w, half_h)
            graph = unified_split_filtergraph(
                out_w, out_h, crop_w * ss, crop_h * ss, cmd_path,
                int(round(initial_x * ss)), int(round(initial_y * ss)),
                top_xy_path, top_zoom_path, bot_xy_path, bot_zoom_path,
                cell_w, cell_h, init_top, init_bottom, enable_expr,
                top_frame_path, bot_frame_path,
                init_top_frame, init_bottom_frame, supersample=ss)
            spans = ", ".join(f"{s:.1f}-{e:.1f}s"
                              for s, e in split_info["ranges"])
            total = sum(e - s for s, e in split_info["ranges"])
            print(f"   ➗ Split-screen reaction cam: {len(split_info['ranges'])} "
                  f"interval(s), {total:.1f}s total — {spans}")

        # One pass over the whole clip — no more per-scene segment/concat
        # step, since every frame now renders through the same filtergraph.
        _run([
            "ffmpeg", "-y", "-loglevel", "error",
            *gpu_decode_args(), "-i", input_video,
            "-filter_complex", graph, "-map", "[v]", "-map", "0:a?",
            *video_encode_args(QUALITY_FAST), "-c:a", "copy", *METADATA_SCRUB,
            # +faststart moves the moov atom to the front so the browser <video>
            # can start playing before the whole file downloads.
            "-movflags", "+faststart",
            final_output_video,
        ])
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)

    # Every frame is now letterboxed into the same consistent content box
    # (no more separate GENERAL-only positioning), so captions should
    # always use the content-box math — the whole clip range qualifies.
    total_duration = len(rects) / fps
    print(f"   ✅ Clip saved to {final_output_video}")
    return True, [(0.0, total_duration)]
