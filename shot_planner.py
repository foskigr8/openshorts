"""Phase 4 of the framing-engine rebuild: the shot planner.

WHY THIS EXISTS
---------------
The v1/v2 engines decided the crop PER FRAME, reactively: the camera eases
toward a target every frame and the subject-policy tier decision re-runs
every frame, and jitter is the visible result of that — a crop that is always
slightly re-aiming even when nothing about who should be on screen has
actually changed. The owner's own instruction was exact: "if you cut
something, remove that end, and the person is in a
stationary place, there is no need for you to move the person."

This module plans the WHOLE shot list up front instead: a small number of
(start, end, shot_type, track_ids, crop_rect) entries, each with ONE static
crop_rect for its entire duration. There is nothing left to re-aim within a
shot, so jitter is not reduced here, it is structurally impossible — the
render stage (Phase 5) just holds `crop_rect` constant until the next entry.

It reads the outputs of the two phases before it: `face_spine.py` (Phase 1,
who the people are and where their faces are, second by second) and
`speaker_fusion.py` (Phase 3, which track is the bound speaker at each
moment). It does not re-derive either.

RESOLUTION HONESTY
-------------------
`speaker_fusion.per_second_active_track` is 1-second granularity today (see
the rebuild plan's §1.2: the ASD collapse to whole seconds was flagged as a
real precision loss, not fixed by this rebuild's Phase 2/3 — LR-ASD's own
25fps scores are still there to use later, just not consumed at that rate
yet). Building this planner as a 1Hz-only algorithm would quietly inherit
that same flaw. Instead the core functions here work on arbitrary
`(timestamp, value)` samples — a value holds from its own timestamp to the
NEXT sample's timestamp — so today's 1Hz feed works via `plan_shots` (a thin
adapter), but nothing downstream has to change if the fusion layer is ever
made finer-grained.

WHAT IS NOT HERE
-----------------
TWO_SHOT needs an "who is X addressing" signal (head pose, per the rebuild
plan's tool proposal — 6DRepNet, not yet built). `apply_two_shot` takes that
signal as a plain argument rather than computing it, so it is ready to wire
in without redesigning this module once that signal exists; it is not
fabricated here.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


SHOT_SINGLE = "single"
SHOT_TWO_SHOT = "two_shot"
SHOT_REACTION = "reaction"
SHOT_WIDE = "wide"

# A directive-driven cutaway is punctuation, not the main shot — bounded
# short on purpose. Bounded so a reaction never outlasts a real shot, the
# same floor the old per-frame reaction hold enforced.
DEFAULT_MAX_REACTION_SECONDS = 2.0

# Below this, a directive-suggested window is too small to be worth a cut at
# all (the "one frame flicker" scale) -- skip rather than splice.
MIN_REACTION_WINDOW_SECONDS = 0.3

DIRECTIVE_PAYOFF_REASONS = ("causing_reaction", "referenced")

# Corroboration gate defaults (see is_directive_corroborated below). A real
# documented failure: Gemini hallucinated a causing_reaction beat with
# nobody actually reacting at that timestamp (HANDOFF_FRAMING.md). Kept low
# on purpose -- this is a cheap position-based proxy for "something is
# actually happening here," not a real gesture/expression detector, and a
# false rejection (blocking a genuine but visually subtle reaction) is worse
# than a false acceptance for this specific use, since a rejected directive
# just means the base shot list is used instead -- never a broken render.
MIN_CORROBORATION_DETECTIONS = 1
MIN_CORROBORATION_MOTION_FRAC = 0.03


@dataclass
class Shot:
    start: float
    end: float
    shot_type: str
    track_ids: List[int]
    crop_rect: Optional[Tuple[float, float, float, float]] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---------------------------------------------------------------------------
# Core: samples -> runs -> a stable shot list
# ---------------------------------------------------------------------------

def hold_fill(samples: List[Tuple[float, Optional[int]]]) -> List[Tuple[float, Optional[int]]]:
    """Forward-fill `None` samples with the last confident value.

    `samples` must be sorted by timestamp. A leading run of `None` (nobody
    resolvable yet, e.g. before the first confident speaker binding) stays
    `None` — there is nothing to hold onto yet, so this correctly produces a
    WIDE shot rather than inventing an early target.
    """
    filled = []
    last = None
    for t, v in samples:
        if v is not None:
            last = v
        filled.append((t, last))
    return filled


def raw_runs(filled_samples: List[Tuple[float, Optional[int]]],
            total_duration: float) -> List[Tuple[float, float, Optional[int]]]:
    """Collapse hold-filled samples into `(start, end, value)` runs.

    A sample's value holds from its own timestamp to the NEXT sample's
    timestamp (or `total_duration` for the last sample) — this definition
    works for any sampling rate/spacing, not just a fixed 1Hz grid.
    """
    if not filled_samples:
        return []
    runs: List[Tuple[float, float, Optional[int]]] = []
    for i, (t, v) in enumerate(filled_samples):
        end_t = filled_samples[i + 1][0] if i + 1 < len(filled_samples) else total_duration
        if end_t <= t:
            continue  # duplicate/out-of-order timestamp -- no duration to represent
        if runs and runs[-1][2] == v:
            runs[-1] = (runs[-1][0], end_t, v)
        else:
            runs.append((t, end_t, v))
    return runs


def split_at_forced_boundaries(runs: List[Tuple[float, float, Optional[int]]],
                               forced_boundaries: Optional[List[float]]
                               ) -> List[Tuple[float, float, Optional[int]]]:
    """Force a run edge at every forced-boundary timestamp inside a run.

    Forced boundaries are jump-cut splice points (main.py's keep_spans dead-
    air removal) — the underlying video is physically discontinuous there,
    so a shot boundary MUST exist regardless of whether the framing target
    happens to be the same on both sides. This is the direct fix for the
    "half a person after a cut" bug (rebuild plan §3.2): today those splice
    points are rediscovered by re-running scene detection on the spliced
    file, and a splice between near-identical frames can be missed, letting
    the camera ease across what should be a hard cut. Here the boundary is
    passed in, known, and cannot be missed.
    """
    if not forced_boundaries:
        return list(runs)
    forced = sorted(set(forced_boundaries))
    result = []
    for start, end, value in runs:
        cuts = [b for b in forced if start < b < end]
        prev = start
        for b in cuts:
            result.append((prev, b, value))
            prev = b
        result.append((prev, end, value))
    return result


def merge_short_runs(runs: List[Tuple[float, float, Optional[int]]],
                     min_shot_seconds: float,
                     forced_boundaries: Optional[List[float]]
                     ) -> List[Tuple[float, float, Optional[int]]]:
    """Reabsorb a run shorter than `min_shot_seconds` into the PRECEDING
    run, extending it and keeping the PRECEDING run's target.

    A brief flicker to a different track (one bad ASD second, a momentary
    fusion gap) is not enough evidence to justify a real shot change — the
    same floor-protection idea as the old per-frame minimum shot hold,
    applied once per shot instead of re-checked every frame.

    Never reabsorbs across a forced boundary (a run starting exactly at one
    is kept even if short — the video is physically spliced there, a cut
    must exist no matter how brief the resulting shot is) and never
    reabsorbs the very first run (nothing precedes it). Also coalesces
    immediately-adjacent runs left with an identical target after a
    reabsorption, so the output never contains two consecutive shots with
    the same track for no reason.
    """
    forced = set(forced_boundaries or [])
    output: List[Tuple[float, float, Optional[int]]] = []
    for start, end, value in runs:
        duration = end - start
        starts_at_forced = start in forced
        should_reabsorb = output and not starts_at_forced and (
            duration < min_shot_seconds or output[-1][2] == value)
        if should_reabsorb:
            prev_start, _prev_end, prev_value = output[-1]
            output[-1] = (prev_start, end, prev_value)
        else:
            output.append((start, end, value))
    return output


def _nearest_box_in_track(track: dict, timestamp: float, time_tolerance: float = 1.0):
    """The track's box nearest `timestamp`, or None beyond `time_tolerance`.

    Deliberately a LOCAL copy of speaker_fusion._nearest_box_at rather than
    an import: that function's tolerance (0.5s) is tuned for matching a
    same-instant ASD prediction to a track. This one is used to find a
    representative box across a whole SHOT span, where the nearest detection
    may legitimately be a bit further away (sparse sampling, brief
    occlusion) — a materially different tolerance for a materially different
    purpose, not a copy that should stay byte-identical to the original.
    """
    frames = track.get("frames") or []
    boxes = track.get("boxes") or []
    if not frames:
        return None
    best_i, best_dt = None, None
    for i, t in enumerate(frames):
        dt = abs(t - timestamp)
        if best_dt is None or dt < best_dt:
            best_i, best_dt = i, dt
    if best_i is None or best_dt > time_tolerance:
        return None
    return boxes[best_i]


def crop_rect_for_track(spine_tracks: Dict[int, dict], track_id: int,
                        start: float, end: float) -> Optional[Tuple[float, float, float, float]]:
    """ONE static box representing `track_id` for the whole [start, end) span.

    Uses the MEDIAN of every detection box within the span, not a min/max
    union — a union lets a single noisy/occluded-partial detection distort
    the whole shot's framing; the median is robust to exactly that outlier
    while still reflecting where the person actually was most of the time.
    If nothing was detected inside the span (the track's evidence is sparser
    than this shot, e.g. a brief look-away), falls back to the single
    nearest detection to the shot's midpoint anywhere in the track, so a
    shot is never left without a crop_rect entirely.
    """
    track = spine_tracks.get(track_id)
    if not track:
        return None
    frames = track.get("frames") or []
    boxes = track.get("boxes") or []
    in_span = [b for t, b in zip(frames, boxes) if start <= t < end]
    if in_span:
        xs, ys, ws, hs = zip(*in_span)
        return (statistics.median(xs), statistics.median(ys),
                statistics.median(ws), statistics.median(hs))
    fallback = _nearest_box_in_track(track, (start + end) / 2.0, time_tolerance=float("inf"))
    return fallback


def plan_shots_from_samples(samples: List[Tuple[float, Optional[int]]],
                            spine_tracks: Dict[int, dict],
                            total_duration: float,
                            forced_boundaries: Optional[List[float]] = None,
                            min_shot_seconds: float = 1.2,
                            default_wide_rect: Optional[tuple] = None) -> List[Shot]:
    """The general entry point: arbitrary-timestamp samples -> a stable shot
    list. See `plan_shots` for the 1Hz-feed convenience wrapper.
    """
    filled = hold_fill(sorted(samples, key=lambda s: s[0]))
    runs = raw_runs(filled, total_duration)
    runs = split_at_forced_boundaries(runs, forced_boundaries)
    runs = merge_short_runs(runs, min_shot_seconds, forced_boundaries)

    shots = []
    for start, end, target in runs:
        if target is None:
            shots.append(Shot(start, end, SHOT_WIDE, [], default_wide_rect))
        else:
            rect = crop_rect_for_track(spine_tracks, target, start, end)
            shots.append(Shot(start, end, SHOT_SINGLE, [target], rect))
    return shots


def plan_shots(per_second_active_track: List[Optional[int]],
               spine_tracks: Dict[int, dict],
               total_duration: Optional[float] = None,
               forced_boundaries: Optional[List[float]] = None,
               min_shot_seconds: float = 1.2,
               default_wide_rect: Optional[tuple] = None) -> List[Shot]:
    """Convenience entry point for TODAY's feed: `speaker_fusion.
    per_second_active_track`'s 1Hz list, index i = second i.
    """
    samples = [(float(i), v) for i, v in enumerate(per_second_active_track)]
    duration = (total_duration if total_duration is not None
               else float(len(per_second_active_track)))
    return plan_shots_from_samples(samples, spine_tracks, duration,
                                   forced_boundaries=forced_boundaries,
                                   min_shot_seconds=min_shot_seconds,
                                   default_wide_rect=default_wide_rect)


# ---------------------------------------------------------------------------
# Splicing a bounded, type-changed window into an existing shot list —
# shared by REACTION (below) and TWO_SHOT once its addressee signal exists
# ---------------------------------------------------------------------------

def _splice_window(shots: List[Shot], w_start: float, w_end: float,
                   shot_type: str, track_ids: List[int],
                   crop_rect: Optional[tuple],
                   min_shot_seconds: float) -> List[Shot]:
    """Insert one bounded, differently-typed shot into `shots`, trimming the
    single existing shot that fully contains [w_start, w_end).

    Conservative on purpose: if the window straddles an existing boundary
    (no single shot fully contains it), or trimming would leave either
    remaining fragment shorter than `min_shot_seconds`, the insertion is
    skipped and `shots` is returned unchanged. A director's suggested
    cutaway is not worth breaking the no-sub-minimum-shot guarantee for —
    the base shot list's stability matters more than honoring every
    directive.
    """
    for i, shot in enumerate(shots):
        if not (shot.start <= w_start and w_end <= shot.end):
            continue
        if shot.track_ids == track_ids:
            return shots  # already framing this -- nothing to insert
        left_len = w_start - shot.start
        right_len = shot.end - w_end
        if 0 < left_len < min_shot_seconds or 0 < right_len < min_shot_seconds:
            return shots  # would create a sub-minimum sliver -- skip

        replacement = []
        if left_len > 0:
            replacement.append(Shot(shot.start, w_start, shot.shot_type,
                                    shot.track_ids, shot.crop_rect))
        replacement.append(Shot(w_start, w_end, shot_type, track_ids, crop_rect))
        if right_len > 0:
            replacement.append(Shot(w_end, shot.end, shot.shot_type,
                                    shot.track_ids, shot.crop_rect))

        new_shots = list(shots)
        new_shots[i:i + 1] = replacement
        return new_shots
    return shots  # window does not sit cleanly inside one shot -- skip


# ---------------------------------------------------------------------------
# REACTION shots — from gemini_worker.FocusDirective, the scene-context
# signal that ALREADY exists and already carries causing_reaction/referenced
# ---------------------------------------------------------------------------

def resolve_directive_track(x_position: float, spine_tracks: Dict[int, dict],
                            frame_width: float, at_time: float,
                            x_tolerance_frac: float = 0.15) -> Optional[int]:
    """Which face_spine track a directive's `x_position` (0..1 fraction of
    frame width, gemini_worker.FocusDirective's own convention) points at,
    by nearest horizontal centre at `at_time`. None if nothing is close
    enough to trust (`x_tolerance_frac` of the frame width).
    """
    target_x = x_position * frame_width
    best_id, best_dist = None, x_tolerance_frac * frame_width
    for track_id, track in spine_tracks.items():
        box = _nearest_box_in_track(track, at_time)
        if box is None:
            continue
        cx = box[0] + box[2] / 2.0
        dist = abs(cx - target_x)
        if dist <= best_dist:
            best_id, best_dist = track_id, dist
    return best_id


def _track_positional_motion(track: dict, start: float, end: float) -> Optional[float]:
    """How much `track`'s box centre moved within [start, end), normalized
    by its own average box width (so the number means the same thing for a
    tight close-up and a wide shot). None if there are no detections in the
    window at all.
    """
    frames = track.get("frames") or []
    boxes = track.get("boxes") or []
    in_span = [b for t, b in zip(frames, boxes) if start <= t < end]
    if not in_span:
        return None
    centres_x = [b[0] + b[2] / 2.0 for b in in_span]
    centres_y = [b[1] + b[3] / 2.0 for b in in_span]
    avg_w = sum(b[2] for b in in_span) / len(in_span)
    if avg_w <= 0:
        return 0.0
    dx = max(centres_x) - min(centres_x)
    dy = max(centres_y) - min(centres_y)
    return ((dx ** 2 + dy ** 2) ** 0.5) / avg_w


def is_directive_corroborated(spine_tracks: Dict[int, dict], track_id: int,
                              start: float, end: float,
                              min_detections: int = MIN_CORROBORATION_DETECTIONS,
                              min_motion_frac: float = MIN_CORROBORATION_MOTION_FRAC
                              ) -> bool:
    """Does independent evidence — the face spine's OWN detections, already
    computed for other reasons, not a second Gemini call — support a
    causing_reaction/referenced directive pointing at `track_id` during
    [start, end)?

    Two checks, both cheap and both already-available data:
      1. PRESENCE: the track must actually have detections in the window at
         all. This alone catches the documented failure mode directly — a
         directive pointing at a moment/person where nothing was really
         there (HANDOFF_FRAMING.md's hallucinated causing_reaction beat).
      2. MOTION (soft, skipped when too few samples to judge): some minimum
         positional movement, since a "causing_reaction" moment is
         definitionally someone DOING something, not sitting still. Skipped
         rather than enforced when there is only one detection in the
         window — a single sample cannot prove or disprove movement, and
         penalizing sparse-but-real evidence would reject genuine reactions
         on tracks the spine only sampled lightly.

    This never calls any model — it is a read of data Phase 1/3 already
    computed, so it adds no latency and no API cost to the pipeline.
    """
    track = spine_tracks.get(track_id)
    if not track:
        return False
    frames = track.get("frames") or []
    n_in_span = sum(1 for t in frames if start <= t < end)
    if n_in_span < min_detections:
        return False
    motion = _track_positional_motion(track, start, end)
    if motion is None:
        return False
    if n_in_span >= 2 and motion < min_motion_frac:
        return False
    return True


def insert_reaction_shots(shots: List[Shot], directives: List[dict],
                          spine_tracks: Dict[int, dict], frame_width: float,
                          max_reaction_seconds: float = DEFAULT_MAX_REACTION_SECONDS,
                          min_shot_seconds: float = 1.2,
                          require_corroboration: bool = True) -> List[Shot]:
    """For each `causing_reaction`/`referenced` directive (gemini_worker's
    "show the cause, not the reaction" / "cut to who was referenced" rule),
    resolve which track it points at and — if that differs from what the
    base shot list already frames there, AND independent evidence backs it
    up (`is_directive_corroborated`) — splice in a bounded REACTION shot,
    then resume the original target. Directives with any other `reason`
    (`speaking`, `reacting`) are not payoff moments and are ignored here;
    they already inform who the SPEAKING track is via the transcript/ASD
    fusion that built the base shot list.

    The corroboration check exists because a director call CAN be wrong (a
    real, documented case: HANDOFF_FRAMING.md's hallucinated causing_reaction
    beat with nobody actually reacting) — this is what stops that failure
    from ever reaching the render, regardless of which model produced the
    directive. `require_corroboration=False` is an escape hatch for
    debugging/comparison only, not meant to run in production.

    `directives` are plain dicts matching gemini_worker.FocusDirective's
    fields (`start`, `end`, `x_position`, `reason`) — accepted as dicts
    rather than the pydantic model so this module has no dependency on
    gemini_worker or the google-genai stack.
    """
    result = list(shots)
    for d in directives:
        if d.get("reason") not in DIRECTIVE_PAYOFF_REASONS:
            continue
        mid = (d["start"] + d["end"]) / 2.0
        track = resolve_directive_track(d["x_position"], spine_tracks, frame_width, mid)
        if track is None:
            continue
        half = max_reaction_seconds / 2.0
        w_start = max(d["start"], mid - half)
        w_end = min(d["end"], mid + half)
        if w_end - w_start < MIN_REACTION_WINDOW_SECONDS:
            continue
        if require_corroboration and not is_directive_corroborated(
                spine_tracks, track, w_start, w_end):
            continue
        rect = crop_rect_for_track(spine_tracks, track, w_start, w_end)
        result = _splice_window(result, w_start, w_end, SHOT_REACTION, [track],
                                rect, min_shot_seconds)
    return result


# ---------------------------------------------------------------------------
# TWO_SHOT — composition only; addressee detection is a separate, not-yet-
# built signal (head pose) per the rebuild plan's tool proposal
# ---------------------------------------------------------------------------

def two_shot_crop_rect(spine_tracks: Dict[int, dict], track_a: int, track_b: int,
                       start: float, end: float) -> Optional[tuple]:
    """A single static box covering BOTH tracks for [start, end) — the union
    of each track's own crop_rect_for_track box.
    """
    rect_a = crop_rect_for_track(spine_tracks, track_a, start, end)
    rect_b = crop_rect_for_track(spine_tracks, track_b, start, end)
    if rect_a is None:
        return rect_b
    if rect_b is None:
        return rect_a
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    x0, y0 = min(ax, bx), min(ay, by)
    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return (x0, y0, x1 - x0, y1 - y0)


def apply_two_shot(shots: List[Shot], addressee_per_second: List[Optional[int]],
                   spine_tracks: Dict[int, dict],
                   min_shot_seconds: float = 1.2) -> List[Shot]:
    """Widen a SINGLE shot to a TWO_SHOT wherever its speaking track has a
    known, DIFFERENT addressee for the whole shot span.

    `addressee_per_second[i]` = the track index i's speaker is addressing at
    second i, or None — supplied by the caller. This function does not
    detect addressees itself (that needs head pose, not yet built per the
    rebuild plan's tool proposal); it only turns an already-known addressee
    into a composition change, through the same min-duration-safe splice
    (`_splice_window`) the reaction shots use, so wiring in a real addressee
    signal later needs no change here.
    """
    result = list(shots)
    for shot in shots:
        if shot.shot_type != SHOT_SINGLE or len(shot.track_ids) != 1:
            continue
        speaker = shot.track_ids[0]
        start_i, end_i = int(shot.start), int(shot.end)
        addressees = {addressee_per_second[i] for i in range(start_i, min(end_i, len(addressee_per_second)))
                     if 0 <= i < len(addressee_per_second) and addressee_per_second[i] is not None
                     and addressee_per_second[i] != speaker}
        if len(addressees) != 1:
            continue  # no addressee, or it changed mid-shot -- not confident enough
        addressee = next(iter(addressees))
        rect = two_shot_crop_rect(spine_tracks, speaker, addressee, shot.start, shot.end)
        result = _splice_window(result, shot.start, shot.end, SHOT_TWO_SHOT,
                                [speaker, addressee], rect, min_shot_seconds)
    return result
