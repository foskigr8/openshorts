"""Stage 3 rebuild, Phase 1: Stage A (map) — per-chunk candidate proposal.

Replaces the "one giant single-shot prompt" (viral_clip_finder.py) and the
"sequential 5-minute windows with a 200-char carry-over" (deepseek_worker's
chunked narrative path) that were both diagnosed as the source of "still
slow" and inconsistent picks (see the Stage 3 rebuild plan). Every chunk here
is INDEPENDENT — no chunk's prompt depends on another chunk's output — so
callers can run them concurrently (Phase 3 wires that through
gemini_pool.GeminiKeyPool; this module itself is concurrency-agnostic, one
chunk in, one candidate list out).

BOUNDARY-LEGAL BY CONSTRUCTION, not by snapping afterward
-----------------------------------------------------------
The old pipeline let the model propose ANY float timestamp, then ran three
separate stacked correction passes (word-snapping, question-extension,
scene-clamping) that could only nudge a bad guess to the nearest legal point
— never ask the model to reconsider. That is the direct, diagnosed source of
"cuts sentence sometimes" surviving this far.

Here, every sentence boundary in a chunk is assigned a short numeric ID and
inlined into the transcript text shown to the model (e.g. "...you don't say.
[7] Well I'll tell you..." — [7] marks a legal sentence-edge). The model is
asked to return BOUNDARY IDS, not floats. A response citing an ID that does
not exist is a real parse failure (raises), not a candidate to silently
tolerate — there is no float to be "close enough" to. This makes an
off-sentence cut a schema violation instead of a rounding error, which is a
structurally stronger guarantee than validating a float against a tolerance
window ever gives.

Scene boundaries are exposed to the model as a second, informational list
(nearest scene cut before/after each sentence boundary) so it can avoid
proposing an end that would land mid-scene, but scene alignment is NOT
mechanically enforced the way sentence alignment is — a viral moment does not
reliably end exactly on a shot cut the way it reliably ends on a sentence,
so forcing that would be a false constraint. Scene awareness stays informative
here; `main.py`'s existing `_clamp_candidate_end_to_scene` remains downstream
as a safety net for the rare case a chunk's pick does drift past a cut.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from pydantic import BaseModel, Field, model_validator

from clip_selection import build_transcript_windows

# NOTE: deepseek_worker.build_compact_segments is imported lazily, inside
# propose_candidates_for_chunk, not at module level. deepseek_worker pulls in
# gemini_worker -> google.genai -> a heavy transitive chain; keeping that out
# of this module's top level means the pure parts here (schemas, boundary
# construction, marker insertion, resolution) stay importable and testable
# without the full provider stack, the same reasoning vendor/pyautoflip's
# saliency_detector import laziness already follows in this codebase.

# ---------------------------------------------------------------------------
# Boundary marker construction
# ---------------------------------------------------------------------------


@dataclass
class ChunkBoundaries:
    """Every legal sentence-start/end in one chunk, ID-indexed both ways."""
    start_ids: Dict[int, float] = field(default_factory=dict)  # id -> timestamp
    end_ids: Dict[int, float] = field(default_factory=dict)
    id_by_start_ts: Dict[float, int] = field(default_factory=dict)
    id_by_end_ts: Dict[float, int] = field(default_factory=dict)


def build_chunk_boundaries(chunk_start: float, chunk_end: float,
                           all_sentence_starts: List[float],
                           all_sentence_ends: List[float]) -> ChunkBoundaries:
    """Slice the FULL video's sentence-boundary lists down to what falls
    inside one chunk's [start, end) window, and assign each a stable local
    ID (0, 1, 2, ...) — starts and ends share one ID sequence so a prompt
    reader/writer only has to track one namespace, not two.
    """
    starts_in = sorted(t for t in all_sentence_starts if chunk_start <= t < chunk_end)
    ends_in = sorted(t for t in all_sentence_ends if chunk_start <= t < chunk_end)

    # Interleave by timestamp so IDs read in the same order the transcript
    # text does — a human (or model) skimming the prompt sees markers count
    # up left to right, not jump around.
    tagged = sorted(
        [(t, "s") for t in starts_in] + [(t, "e") for t in ends_in],
        key=lambda pair: pair[0],
    )
    result = ChunkBoundaries()
    for i, (ts, kind) in enumerate(tagged):
        if kind == "s":
            result.start_ids[i] = ts
            result.id_by_start_ts[ts] = i
        else:
            result.end_ids[i] = ts
            result.id_by_end_ts[ts] = i
    return result


def _insert_markers(compact_segments: List[dict], boundaries: ChunkBoundaries) -> str:
    """Render a chunk's compact segments to text with [N] markers spliced in
    at every legal sentence boundary that falls inside a segment's span.

    Markers are placed by nearest-word-boundary approximation: since compact
    segments carry only segment-level start/end (no word timestamps — see
    build_compact_segments' own docstring on why), a marker whose timestamp
    falls within a segment is appended at that segment's end if it's an END
    marker, or prepended to the FOLLOWING segment if it's a START marker.
    This is coarser than word-level placement but exactly matches the
    granularity the model is already reasoning over (segment text, not raw
    words) — segments here are already tight (30-90 char merged units per
    build_compact_segments), so a marker is never more than one short segment
    away from its true timestamp.
    """
    all_markers = sorted(
        [(ts, i, "start") for i, ts in boundaries.start_ids.items()]
        + [(ts, i, "end") for i, ts in boundaries.end_ids.items()]
    )
    lines = []
    marker_idx = 0
    for seg in compact_segments:
        prefix_markers = []
        while marker_idx < len(all_markers) and all_markers[marker_idx][0] <= seg["s"]:
            ts, mid, kind = all_markers[marker_idx]
            if kind == "start":
                prefix_markers.append(f"[{mid}]")
            marker_idx += 1
        speaker = f"{seg['sp']}: " if seg.get("sp") else ""
        line = f"{''.join(prefix_markers)}{speaker}{seg['t']}"
        suffix_markers = []
        while marker_idx < len(all_markers) and all_markers[marker_idx][0] <= seg["e"]:
            ts, mid, kind = all_markers[marker_idx]
            if kind == "end":
                suffix_markers.append(f"[{mid}]")
            marker_idx += 1
        lines.append(line + "".join(suffix_markers))
    # Any markers past the last segment's end (rounding at the chunk edge)
    # still need to be resolvable if cited — append them, unattached to text.
    while marker_idx < len(all_markers):
        ts, mid, kind = all_markers[marker_idx]
        lines.append(f"[{mid}] (boundary at {ts:.1f}s)")
        marker_idx += 1
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class _NullTolerantModel(BaseModel):
    """Drop explicit JSON nulls before validation so an optional field sent
    as null doesn't fail the whole response over a field nobody requires.

    This is the exact lesson from the documented incident that killed the
    old skill engine silently for an unknown stretch of time (a `null` on
    one optional field rejected every candidate, every retry, with the
    fallback masking it) — carried forward as a design rule even though none
    of that engine's code is being reused. Required fields (start/end
    boundary IDs) are unaffected: dropping a null key here means the field's
    own default applies, and a REQUIRED field with no default still fails
    loudly, exactly as before.
    """

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data):
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


class ChunkCandidate(_NullTolerantModel):
    start_boundary_id: int = Field(description="ID of the legal sentence-start marker to open on")
    end_boundary_id: int = Field(description="ID of the legal sentence-end marker to close on")
    score: int = Field(default=0, ge=0, le=100)
    clip_type: str = Field(default="short", description='"short" or "long_context"')
    rationale: str = Field(default="", description="why this moment, in one sentence")
    hook_payoff_summary: str = Field(
        default="", description="1-2 sentences: what's set up, what pays it off — "
                               "Stage B's ONLY view of this chunk's content")


class ChunkCandidatesResponse(_NullTolerantModel):
    candidates: List[ChunkCandidate] = []


# ---------------------------------------------------------------------------
# Chunk construction (thin wrapper over the existing primitive)
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SECONDS = 210  # ~3.5 min — open decision #1 in the plan; tune
                             # against eval/selection once real numbers exist
DEFAULT_OVERLAP_SECONDS = 30


def build_chunks(transcript_result, video_duration,
                 window_seconds=DEFAULT_CHUNK_SECONDS,
                 overlap_seconds=DEFAULT_OVERLAP_SECONDS):
    """Chunk windows for Stage A — reuses clip_selection.build_transcript_
    windows exactly (segment-aligned, already handles the "don't cut a
    sentence across a window edge" concern for the WINDOW boundary itself;
    within-window sentence boundaries are handled separately, above)."""
    return build_transcript_windows(
        transcript_result, video_duration,
        window_seconds=window_seconds, overlap_seconds=overlap_seconds)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CHUNK_PROMPT_TEMPLATE = """You are reviewing a {duration:.0f}-second slice of a longer video's transcript \
(from {chunk_start:.0f}s to {chunk_end:.0f}s in the source) to find moments worth cutting into a \
short-form vertical clip.

Every [N] marker in the transcript below is a LEGAL cut boundary — a real sentence edge, not an \
arbitrary point. You MUST choose start_boundary_id and end_boundary_id from these markers only. \
A boundary you invent that isn't listed will be rejected.

TRANSCRIPT (with boundary markers):
{marked_transcript}

Propose up to {max_candidates} candidate moments from this slice. For each: pick the opening \
marker that starts the clip cleanly (not mid-thought), the closing marker that lands after the \
payoff actually resolves, a 0-100 score for how well it works as a standalone short, and a ONE-TO-TWO \
SENTENCE hook_payoff_summary — this is the ONLY thing another reviewer will see of this chunk when \
deciding what makes the final cut across the whole video, so it must capture what's set up and what \
pays it off, not just "funny moment."

If nothing in this slice is worth cutting, return an empty candidates list — do not force a weak pick.

Respond with JSON matching this exact shape:
{{"candidates": [{{"start_boundary_id": <int>, "end_boundary_id": <int>, "score": <int 0-100>, \
"clip_type": "short"|"long_context", "rationale": "<one sentence>", \
"hook_payoff_summary": "<1-2 sentences>"}}]}}
"""


def build_chunk_prompt(chunk, boundaries: ChunkBoundaries, compact_segments,
                       max_candidates=3):
    marked = _insert_markers(compact_segments, boundaries)
    return CHUNK_PROMPT_TEMPLATE.format(
        duration=chunk["end"] - chunk["start"],
        chunk_start=chunk["start"], chunk_end=chunk["end"],
        marked_transcript=marked, max_candidates=max_candidates,
    )


# ---------------------------------------------------------------------------
# Resolution: boundary IDs -> real (start, end) seconds
# ---------------------------------------------------------------------------


class BoundaryResolutionError(ValueError):
    """A response cited a boundary ID that doesn't exist in this chunk, or a
    start/end pair that doesn't form a valid (start < end) span. Raised, not
    silently dropped — an invented ID means the model didn't respect the
    hard constraint the whole marker scheme exists to enforce, which is
    exactly the class of failure that must be visible, not smoothed over."""


@dataclass
class ResolvedCandidate:
    start: float
    end: float
    score: int
    clip_type: str
    rationale: str
    hook_payoff_summary: str


def resolve_candidates(response: ChunkCandidatesResponse,
                       boundaries: ChunkBoundaries) -> List[ResolvedCandidate]:
    resolved = []
    for c in response.candidates:
        if c.start_boundary_id not in boundaries.start_ids:
            raise BoundaryResolutionError(
                f"start_boundary_id {c.start_boundary_id} is not a legal "
                f"marker in this chunk (valid: {sorted(boundaries.start_ids)})")
        if c.end_boundary_id not in boundaries.end_ids:
            raise BoundaryResolutionError(
                f"end_boundary_id {c.end_boundary_id} is not a legal "
                f"marker in this chunk (valid: {sorted(boundaries.end_ids)})")
        start = boundaries.start_ids[c.start_boundary_id]
        end = boundaries.end_ids[c.end_boundary_id]
        if end <= start:
            raise BoundaryResolutionError(
                f"end boundary {c.end_boundary_id} ({end:.1f}s) is not after "
                f"start boundary {c.start_boundary_id} ({start:.1f}s)")
        resolved.append(ResolvedCandidate(
            start=start, end=end, score=c.score, clip_type=c.clip_type,
            rationale=c.rationale, hook_payoff_summary=c.hook_payoff_summary,
        ))
    return resolved


# ---------------------------------------------------------------------------
# The actual call (thin wrapper over the existing transport)
# ---------------------------------------------------------------------------


def propose_candidates_for_chunk(chunk, all_sentence_starts, all_sentence_ends,
                                 transcript_result, api_key, model_name,
                                 base_url, max_candidates=3):
    """One Stage A call for one chunk. Reuses deepseek_worker._run_deepseek_
    stage for the actual transport (retry/backoff, cost accounting) rather
    than reimplementing an HTTP client — this module owns chunking, boundary
    legality, and prompt/schema; the wire protocol is deepseek_worker's job,
    same as every other Stage 3 caller.

    Returns (resolved_candidates, cost_analysis). Raises
    BoundaryResolutionError if the model cited an illegal boundary —
    deliberately not caught here; the caller (Phase 3's concurrency wiring)
    decides whether one chunk's failure sinks the whole job or is logged and
    skipped, per the plan's "partial chunk failure" policy.
    """
    # Lazy import: see the module-level NOTE above build_transcript_windows'
    # import for why deepseek_worker's transport/compaction helpers aren't
    # imported at module level here.
    from deepseek_worker import _run_deepseek_stage, build_compact_segments

    boundaries = build_chunk_boundaries(
        chunk["start"], chunk["end"], all_sentence_starts, all_sentence_ends)

    compact_segments = [
        seg for seg in build_compact_segments(transcript_result)
        if seg["s"] < chunk["end"] and seg["e"] > chunk["start"]
    ]
    prompt = build_chunk_prompt(chunk, boundaries, compact_segments, max_candidates)

    parsed, cost = _run_deepseek_stage(
        api_key, model_name, prompt, ChunkCandidatesResponse, base_url=base_url)
    response = ChunkCandidatesResponse.model_validate(parsed)
    resolved = resolve_candidates(response, boundaries)
    return resolved, cost
