"""Pure-Python metrics for the Stage 3 (clip selection) rebuild.

No LLM/network/video dependency — everything here consumes plain (start, end)
spans and word-level transcript data already produced by the pipeline, so it
runs anywhere and is fast to test. Mirrors eval/metrics.py's ground rule:
every metric here must be checkable against a number, not a description.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# span overlap — "did selection find the moments a human would pick"
# ---------------------------------------------------------------------------

def _iou(a, b):
    """Intersection-over-union of two (start, end) spans, in seconds. 0.0
    for non-overlapping spans (never negative — clamped)."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter <= 0.0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


@dataclass
class ExpectedClipMatch:
    expected_start: float
    expected_end: float
    note: str
    best_iou: float
    matched_start: Optional[float]
    matched_end: Optional[float]


@dataclass
class SpanMatchResult:
    expected_count: int
    matched_count: int  # best_iou >= threshold
    match_rate: float  # matched_count / expected_count, 1.0 if expected_count == 0
    mean_best_iou: float
    matches: List[ExpectedClipMatch] = field(default_factory=list)


def span_overlap_score(selected_clips, expected_clips, iou_threshold=0.5):
    """For every expected (ground-truth) clip, find the best-overlapping
    selected clip and score it by IoU.

    A clip is "matched" when its best IoU against any selected clip meets
    `iou_threshold` (default 0.5 — substantial overlap, not just touching).
    This is deliberately asymmetric (scored per EXPECTED clip, not per
    selected one): the question this answers is "did selection find the
    moments that matter," not "is everything selection picked in the
    ground truth" — extra reasonable picks beyond the hand-labeled set are
    not a failure, since the ground truth is intentionally not exhaustive.

    `selected_clips`/`expected_clips`: iterables of dicts with "start"/"end"
    (seconds). Returns a SpanMatchResult; expected_count == 0 reports
    match_rate 1.0 (vacuously true — nothing to have missed) rather than
    dividing by zero or reporting 0.0, which would misread as total failure.
    """
    selected = [(float(c["start"]), float(c["end"])) for c in selected_clips]
    expected = list(expected_clips)

    if not expected:
        return SpanMatchResult(0, 0, 1.0, 1.0, [])

    matches = []
    matched_count = 0
    iou_sum = 0.0
    for exp in expected:
        exp_span = (float(exp["start"]), float(exp["end"]))
        best_iou = 0.0
        best_sel = None
        for sel in selected:
            score = _iou(exp_span, sel)
            if score > best_iou:
                best_iou, best_sel = score, sel
        iou_sum += best_iou
        if best_iou >= iou_threshold:
            matched_count += 1
        matches.append(ExpectedClipMatch(
            expected_start=exp_span[0], expected_end=exp_span[1],
            note=str(exp.get("note", "")), best_iou=best_iou,
            matched_start=best_sel[0] if best_sel else None,
            matched_end=best_sel[1] if best_sel else None,
        ))

    return SpanMatchResult(
        expected_count=len(expected), matched_count=matched_count,
        match_rate=matched_count / len(expected),
        mean_best_iou=iou_sum / len(expected), matches=matches,
    )


@dataclass
class MustNotIncludeViolation:
    selected_start: float
    selected_end: float
    forbidden_start: float
    forbidden_end: float
    note: str
    overlap_seconds: float


def must_not_include_violations(selected_clips, must_not_include):
    """Selected clips that overlap a `must_not_include` span at all (any
    overlap, not IoU-thresholded — a clip is either clean of a known-bad
    span or it is not). Returns the list of violations; empty means clean.
    """
    violations = []
    for sel in selected_clips:
        s_start, s_end = float(sel["start"]), float(sel["end"])
        for bad in must_not_include:
            b_start, b_end = float(bad["start"]), float(bad["end"])
            overlap = min(s_end, b_end) - max(s_start, b_start)
            if overlap > 0:
                violations.append(MustNotIncludeViolation(
                    selected_start=s_start, selected_end=s_end,
                    forbidden_start=b_start, forbidden_end=b_end,
                    note=str(bad.get("note", "")), overlap_seconds=overlap,
                ))
    return violations


# ---------------------------------------------------------------------------
# sentence-boundary hit rate — "does this still cut sentences sometimes"
# ---------------------------------------------------------------------------

@dataclass
class BoundaryCheck:
    clip_start: float
    clip_end: float
    start_on_sentence: bool
    end_on_sentence: bool
    start_delta: float  # distance to nearest sentence start, seconds
    end_delta: float  # distance to nearest sentence end, seconds


@dataclass
class SentenceBoundaryResult:
    clip_count: int
    both_clean_count: int  # start AND end both on a sentence boundary
    hit_rate: float  # both_clean_count / clip_count, 1.0 if clip_count == 0
    checks: List[BoundaryCheck] = field(default_factory=list)


def _nearest_delta(values, target):
    if not values:
        return float("inf")
    return min(abs(v - target) for v in values)


def sentence_boundary_hit_rate(selected_clips, sentence_starts, sentence_ends,
                               tolerance=0.05):
    """Does every selected clip open and close exactly on a sentence edge?

    This is the direct, numeric answer to "still cuts sentence sometimes":
    `sentence_starts`/`sentence_ends` come from
    clip_selection.sentence_boundaries(words) — the same discrete boundary
    lists the rebuilt selector is meant to choose FROM (see the Stage 3 plan)
    rather than snap to after the fact. `tolerance` (default 50ms) absorbs
    float rounding, not real slack — a genuine mid-sentence cut will miss by
    much more than that.
    """
    checks = []
    both_clean = 0
    for clip in selected_clips:
        start, end = float(clip["start"]), float(clip["end"])
        start_delta = _nearest_delta(sentence_starts, start)
        end_delta = _nearest_delta(sentence_ends, end)
        start_ok = start_delta <= tolerance
        end_ok = end_delta <= tolerance
        if start_ok and end_ok:
            both_clean += 1
        checks.append(BoundaryCheck(
            clip_start=start, clip_end=end,
            start_on_sentence=start_ok, end_on_sentence=end_ok,
            start_delta=start_delta, end_delta=end_delta,
        ))
    count = len(checks)
    return SentenceBoundaryResult(
        clip_count=count, both_clean_count=both_clean,
        hit_rate=(both_clean / count) if count else 1.0, checks=checks,
    )


# ---------------------------------------------------------------------------
# latency / cost — "is it still slow"
# ---------------------------------------------------------------------------

@dataclass
class SelectionRunStats:
    engine: str
    wall_seconds: float
    call_count: int
    total_cost_usd: Optional[float] = None
    timed_out: bool = False


def summarize_cost_analysis(cost_analysis_list):
    """Sum a list of `cost_analysis` dicts (the shape
    deepseek_worker._calculate_deepseek_cost_analysis / main.py's inline
    equivalent already produce — reused as-is, not reinvented) into a total.
    None entries are skipped (a call that didn't report usage). Returns None
    (not 0.0) if nothing reported cost, so "no data" and "confirmed free"
    stay distinguishable.
    """
    entries = [c for c in (cost_analysis_list or []) if c]
    if not entries:
        return None
    return sum(float(c.get("total_cost") or 0.0) for c in entries)
