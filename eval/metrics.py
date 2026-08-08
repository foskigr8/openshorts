"""Pure-Python metrics for the framing-engine rebuild (see the plan's
Phase 0). No video/cv2/torch dependency — everything here consumes data
already produced by the pipeline (REFRAME_DUMP_PATH rects, transcripts,
ground-truth labels), so it runs anywhere, including outside the
Kaggle/Docker environment that actually renders.

Ground rule from the plan: every metric here must be checkable against a
number, not a description. `run.py` wires these to real pipeline output;
this module only computes.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# crop motion — "if the subject is stationary, the frame does not move"
# ---------------------------------------------------------------------------

@dataclass
class ShotMotion:
    shot_index: int
    start_frame: int
    end_frame: int
    max_delta_px: float
    mean_delta_px: float
    frame_count: int


def crop_motion_within_shot(rects, shot_boundaries):
    """Per-shot crop-position stability.

    rects: list of (x, y, w, h) in source pixels, one per output frame, in
        frame order — exactly what `REFRAME_DUMP_PATH` writes
        (reframe_v3.render's dump block).
    shot_boundaries: list of (start_frame, end_frame) exclusive-end ranges
        partitioning `rects` into shots (the v3 render's returned shot list,
        converted to frame indices — see eval/run.py).

    Returns a list of ShotMotion, one per shot. A shot's crop is meant to be
    pixel-static: `max_delta_px` must be 0 for a v3 shot's rows (the whole
    point of planned static shots). Non-zero values quantify jitter, in
    pixels of (x, y) centre movement between consecutive
    frames — not a proxy metric, the actual crop position delta.
    """
    results = []
    for i, (start, end) in enumerate(shot_boundaries):
        shot_rects = rects[start:end]
        if len(shot_rects) < 2:
            results.append(ShotMotion(i, start, end, 0.0, 0.0, len(shot_rects)))
            continue
        deltas = []
        for (x0, y0, w0, h0), (x1, y1, w1, h1) in zip(shot_rects, shot_rects[1:]):
            cx0, cy0 = x0 + w0 / 2.0, y0 + h0 / 2.0
            cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
            deltas.append(((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5)
        results.append(ShotMotion(
            shot_index=i, start_frame=start, end_frame=end,
            max_delta_px=max(deltas), mean_delta_px=sum(deltas) / len(deltas),
            frame_count=len(shot_rects),
        ))
    return results


def shot_count_and_durations(shot_boundaries, fps):
    """(shot_count, [duration_seconds, ...]) — the raw material for a
    duration histogram. Kept separate from any binning choice so callers
    (eval/run.py today, a style-profile comparison in Phase 7 later) can bin
    however they need without recomputing durations.
    """
    durations = [(end - start) / float(fps) for start, end in shot_boundaries]
    return len(shot_boundaries), durations


# ---------------------------------------------------------------------------
# speaker-on-screen accuracy — the headline number
# ---------------------------------------------------------------------------

def speaker_on_screen_pct(per_second_framed, per_second_ground_truth):
    """% of seconds where the framed track/identity matches ground truth.

    per_second_framed: list, index = second offset into the clip, value =
        whatever identifies who was framed that second (track id, name,
        whatever the two sides agree on).
    per_second_ground_truth: same shape, hand-labeled (see
        eval/ground_truth.py) — the actual speaker for that second.

    Seconds where ground truth is None (no clear speaker, e.g. silence) are
    excluded from the denominator — they are not evidence either way.
    Returns (pct, matched, total_scored).
    """
    n = min(len(per_second_framed), len(per_second_ground_truth))
    matched = 0
    scored = 0
    for i in range(n):
        gt = per_second_ground_truth[i]
        if gt is None:
            continue
        scored += 1
        if per_second_framed[i] == gt:
            matched += 1
    pct = (100.0 * matched / scored) if scored else 0.0
    return pct, matched, scored


# ---------------------------------------------------------------------------
# boundary quality — "no starting or ending mid-sentence"
# ---------------------------------------------------------------------------

def boundary_sentence_completeness(clip_start, clip_end, words, tolerance=0.35):
    """Does this clip open and close on a sentence boundary?

    words: the same {'w','s','e'} word-dict list clip_selection.py and
        main.py already use (main._build_word_list). Reuses
        clip_selection.sentence_boundaries so this measures the exact same
        boundaries the snapper is supposed to land on — not a re-derivation
        that could silently drift from what the pipeline actually uses.
    tolerance: seconds of slack (a snapper is allowed to lead/trail into
        silence around the sentence edge; see clip_selection.snap_clip_to_words
        max_lead/max_tail, currently 0.35/0.45).

    Returns (starts_on_sentence: bool, ends_on_sentence: bool).
    """
    from clip_selection import sentence_boundaries
    starts, ends = sentence_boundaries(words)
    starts_ok = any(abs(clip_start - s) <= tolerance for s in starts)
    ends_ok = any(abs(clip_end - e) <= tolerance for e in ends)
    return starts_ok, ends_ok


# ---------------------------------------------------------------------------
# report formatting — one block, comparable run to run
# ---------------------------------------------------------------------------

def format_report(label, shot_motions, speaker_pct_result, boundary_result=None):
    """One printable block. Every field here is a number from the functions
    above, not a claim — this is the thing that goes in a PR description or
    a conversation instead of a prose "it works now."
    """
    pct, matched, scored = speaker_pct_result
    jittery = [m for m in shot_motions if m.max_delta_px > 0.0]
    lines = [
        f"=== {label} ===",
        f"speaker_on_screen_pct: {pct:.1f}%  ({matched}/{scored} scored seconds)",
        f"shots: {len(shot_motions)}  |  shots with any crop motion: {len(jittery)}",
    ]
    if jittery:
        worst = max(jittery, key=lambda m: m.max_delta_px)
        lines.append(
            f"worst shot motion: shot {worst.shot_index} "
            f"max={worst.max_delta_px:.1f}px mean={worst.mean_delta_px:.1f}px "
            f"over {worst.frame_count} frames"
        )
    if boundary_result is not None:
        starts_ok, ends_ok = boundary_result
        lines.append(f"starts_on_sentence: {starts_ok}  ends_on_sentence: {ends_ok}")
    return "\n".join(lines)
