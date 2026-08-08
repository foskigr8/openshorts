"""Phase 3 of the framing-engine rebuild: global speaker<->track fusion.

WHY THIS EXISTS
---------------
Today's binding (`reframe_v2._apply_asd_speaker_boost`) matches LR-ASD's
speaker box to a candidate box EVERY FRAME, with a decisiveness gate
(`ASD_MATCH_MARGIN`) that discards the match outright when two candidates
look similarly close. Discarded evidence falls through to "hold whoever the
camera already had" — and because the match is re-attempted independently
every single frame, a momentary ambiguity (a reacting listener leaning in,
two faces briefly at similar distance) can flip the binding mid-turn even
though the SAME diarized speaker is still talking. That per-frame flip is
the actual mechanism behind "long stares at the wrong person" and the
"audio matches but the frame doesn't" symptom (see the rebuild plan, §1.1).

This module replaces per-frame matching with ONE decision per clip: for each
diarized speaker label, look at ALL the evidence across the whole clip (every
second where ASD pointed at a track while that label was talking), and bind
the label to whichever track the evidence most consistently supports. Once
bound, a speaker's track cannot flip mid-turn, because there is no more
per-frame re-decision left to make — the shot planner (Phase 4) just looks
the binding up.

A speaker that never accumulates confident evidence is left UNBOUND rather
than guessed at (see resolve_speaker_bindings) — Phase 4 owns the fallback
tier for that case (hold / size / whatever the shot-planning rules decide),
matching subject_policy's existing tier philosophy of "no strong evidence
means don't invent a decision."
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# ASD box -> face_spine track (position lookup against KNOWN tracks, not a
# live disambiguation between two unrelated id spaces)
# ---------------------------------------------------------------------------

def _box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _nearest_box_at(track: dict, timestamp: float, time_tolerance: float = 0.5):
    """The track's box from the detection nearest `timestamp`, or None if
    the nearest detection is further than `time_tolerance` away (the track
    was not on screen at this moment).
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


def match_box_to_track(spine_tracks: Dict[int, dict], timestamp: float, box,
                       iou_threshold: float = 0.3,
                       time_tolerance: float = 0.5) -> Optional[int]:
    """Which face_spine track a predicted (timestamp, box) belongs to.

    This is a LOOKUP against known tracks (Phase 1's already-resolved
    identities), not the live disambiguation between two unrelated id spaces
    that caused the original binding bug — the tracks here are the ground
    truth for "who is on screen," so there is nothing ambiguous left to
    reconcile, only "which known box is this."

    Returns the best-IoU track id at or above `iou_threshold`, or None if no
    track's box overlaps enough (predicted box does not correspond to any
    known person — e.g. a false positive on background).
    """
    best_id, best_iou = None, iou_threshold
    for track_id, track in spine_tracks.items():
        candidate = _nearest_box_at(track, timestamp, time_tolerance)
        if candidate is None:
            continue
        iou = _box_iou(box, candidate)
        if iou >= best_iou:
            best_id, best_iou = track_id, iou
    return best_id


def asd_predicted_track_per_second(asd_per_second_boxes: List[Optional[tuple]],
                                   spine_tracks: Dict[int, dict],
                                   iou_threshold: float = 0.3) -> List[Optional[int]]:
    """An ASD model's raw per-second output -> per-second track id, via the
    face spine. `asd_per_second_boxes[i]` is the model's predicted
    active-speaker box at second i, or None where the model made no call.
    """
    result = []
    for t, box in enumerate(asd_per_second_boxes):
        if box is None:
            result.append(None)
            continue
        result.append(match_box_to_track(spine_tracks, float(t), box, iou_threshold))
    return result


# ---------------------------------------------------------------------------
# Diarized transcript -> per-second speaker label
# ---------------------------------------------------------------------------

def per_second_speaker_label(segments, clip_start: float, clip_end: float,
                             speaker_names: Optional[Dict[str, str]] = None
                             ) -> List[Optional[str]]:
    """Per-second diarized speaker label for one clip span.

    segments: transcript_result["segments"], each with 'start', 'end',
        'speaker' (assemblyai backend; see transcribe_backends.py) — reads
        the SAME transcript the rest of the pipeline uses, not a parallel one.
    speaker_names: optional {"A": "host", ...} to resolve diarization letters
        to names; unresolved labels pass through as their raw letter.

    Returns a list, index = second offset from clip_start, value = the
    speaker label active that second, or None where no segment covers it
    (silence, cross-talk gap) — those seconds contribute no fusion evidence.
    """
    speaker_names = speaker_names or {}
    duration = int(clip_end - clip_start)
    result = [None] * max(0, duration)
    for seg in segments or []:
        s, e = float(seg.get("start", 0)), float(seg.get("end", 0))
        speaker = seg.get("speaker")
        if speaker is None or e <= clip_start or s >= clip_end:
            continue
        label = speaker_names.get(speaker, speaker)
        lo = max(0, int(s - clip_start))
        hi = min(duration, math.ceil(e - clip_start))
        for i in range(lo, hi):
            result[i] = label
    return result


# ---------------------------------------------------------------------------
# The fusion itself — one binding table per clip
# ---------------------------------------------------------------------------

def resolve_speaker_bindings(per_second_speaker: List[Optional[str]],
                             predicted_track_ps: List[Optional[int]],
                             min_agreement: float = 0.6,
                             min_seconds: float = 1.0) -> Dict[str, int]:
    """The core Phase 3 fusion: ONE speaker_label -> track_id binding per
    label, decided from ALL the evidence across the whole clip at once.

    For every second where BOTH a diarized speaker is talking AND ASD
    pointed at a resolvable track, that track gets one vote toward that
    speaker's binding. A label binds to its top-voted track only when:
      - that track has at least `min_agreement` of the label's total votes
        (a speaker whose evidence is split roughly evenly between two
        tracks — e.g. ASD flickering between the speaker and a reacting
        listener — should NOT bind confidently to either), and
      - the label accumulated at least `min_seconds` of evidence at all
        (a label that barely appears has too little signal to trust).

    Labels that do not clear both bars are left OUT of the returned dict
    entirely — never bound to a guess. The caller (Phase 4's shot planner)
    is responsible for the fallback behavior when a speaker has no binding.

    This function's whole reason to exist: once computed, a binding cannot
    flip mid-turn, because nothing re-evaluates it per frame anymore.
    """
    n = min(len(per_second_speaker), len(predicted_track_ps))
    votes: Dict[str, Dict[int, int]] = {}
    for i in range(n):
        label = per_second_speaker[i]
        track = predicted_track_ps[i]
        if label is None or track is None:
            continue
        votes.setdefault(label, {})
        votes[label][track] = votes[label].get(track, 0) + 1

    bindings: Dict[str, int] = {}
    for label, track_votes in votes.items():
        total = sum(track_votes.values())
        if total < min_seconds:
            continue
        best_track = max(track_votes, key=track_votes.get)
        agreement = track_votes[best_track] / total
        if agreement >= min_agreement:
            bindings[label] = best_track
    return bindings


def per_second_active_track(per_second_speaker: List[Optional[str]],
                            bindings: Dict[str, int]) -> List[Optional[int]]:
    """Expand the binding table across the clip: for each second, whichever
    track is bound to the speaker active that second.

    None where no speaker is active that second, OR where the active
    speaker never got a confident binding (see resolve_speaker_bindings) —
    both cases mean "no strong evidence," never a guess.
    """
    return [bindings.get(label) if label is not None else None
            for label in per_second_speaker]


def fuse_speaker_tracks(asd_per_second_boxes: List[Optional[tuple]],
                        spine_tracks: Dict[int, dict],
                        segments, clip_start: float, clip_end: float,
                        speaker_names: Optional[Dict[str, str]] = None,
                        iou_threshold: float = 0.3,
                        min_agreement: float = 0.6,
                        min_seconds: float = 1.0
                        ) -> tuple:
    """Full Phase 3 pipeline for one clip: ASD boxes + the Phase 1 face
    spine + the diarized transcript -> (bindings, per_second_active_track).

    This is the plan's Pass 2 deliverable: "who is speaking, and which
    track is them" — the input Phase 4's shot planner reads, and the
    thing that replaces reframe_v2._apply_asd_speaker_boost's per-frame
    matching once wired in (Phase 5).
    """
    predicted_track_ps = asd_predicted_track_per_second(
        asd_per_second_boxes, spine_tracks, iou_threshold)
    speaker_ps = per_second_speaker_label(
        segments, clip_start, clip_end, speaker_names)
    bindings = resolve_speaker_bindings(
        speaker_ps, predicted_track_ps, min_agreement, min_seconds)
    active = per_second_active_track(speaker_ps, bindings)
    return bindings, active
