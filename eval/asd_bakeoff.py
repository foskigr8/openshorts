"""Phase 2: score any active-speaker-detection model against the Phase 1
face spine, so LR-ASD / LoCoNet-on-AVA / LoCoNet-on-UniTalk (or anything
else) can be compared on the SAME identity system and the SAME ground truth.

Deliberately does not know which model produced its input. Every ASD model
speaks the same language at its output boundary — "at second t, the active
speaker's face was roughly here" — so scoring only needs that box, plus the
face_spine tracks (Phase 1) to say WHICH person that box belongs to, plus a
one-time human label of which track is which person (the same ~8-click cost
as eval/ground_truth.py's speaker_names).

This module does NOT run any model. Each model's own inference code (LR-ASD:
asd_worker.py; LoCoNet: not yet wrapped, see the module docstring in
face_spine.py's sibling — no ready single-video inference script ships
upstream, so wrapping it is separate, real engineering, not done here)
produces the per-second predicted box; this module only judges the boxes.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from eval import metrics


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

    Pure position match against KNOWN tracks (Phase 1's output), not the
    ambiguous same-frame candidate matching that caused the original binding
    bug (reframe_v2._apply_asd_speaker_boost) — there the two id spaces were
    unrelated and had to be reconciled live; here the tracks are already the
    ground truth for "who is on screen," so this is a lookup, not a decision.

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
    """One model's raw per-second output -> per-second track id, via the
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


def expected_track_per_second(track_names: Dict[int, str],
                               per_second_speaker_name: List[Optional[str]]
                               ) -> List[Optional[int]]:
    """Invert a one-time {track_id: name} label against ground truth's
    per-second speaker NAME (eval.ground_truth.per_second_speaker_from_transcript)
    to get per-second EXPECTED track id — the number every model's
    predicted-track list is actually scored against.

    A name with more than one track (shouldn't happen for a correctly-merged
    Phase 1 spine, but a badly-merged spine could split one person into two
    tracks) resolves to whichever track_id is lowest, deterministically,
    rather than raising — this is a scoring convenience, not silent
    correctness; a name mapping to 2+ tracks is itself worth noticing
    separately as a Phase 1 quality signal.
    """
    name_to_track: Dict[str, int] = {}
    for track_id, name in sorted(track_names.items()):
        name_to_track.setdefault(name, track_id)
    return [name_to_track.get(name) if name is not None else None
            for name in per_second_speaker_name]


def score_model(name: str, asd_per_second_boxes: List[Optional[tuple]],
                spine_tracks: Dict[int, dict],
                expected_track_ps: List[Optional[int]],
                iou_threshold: float = 0.3) -> dict:
    """Full pipeline for one model: boxes -> predicted tracks -> accuracy
    against the expected tracks. Returns a dict ready for compare_models'
    ranking and for direct printing.
    """
    predicted = asd_predicted_track_per_second(
        asd_per_second_boxes, spine_tracks, iou_threshold)
    pct, matched, scored = metrics.speaker_on_screen_pct(predicted, expected_track_ps)
    no_call = sum(1 for b in asd_per_second_boxes if b is None)
    return {
        "name": name, "pct": pct, "matched": matched, "scored": scored,
        "no_call_seconds": no_call, "predicted_track_per_second": predicted,
    }


def compare_models(models: Dict[str, List[Optional[tuple]]],
                   spine_tracks: Dict[int, dict],
                   expected_track_ps: List[Optional[int]],
                   iou_threshold: float = 0.3) -> List[dict]:
    """Score every model and rank best-first by accuracy. The whole point of
    the bake-off: pick the winner by this number, not by a published
    benchmark that may not reflect this footage (see the rebuild plan's
    Phase 2 gate).
    """
    results = [score_model(name, boxes, spine_tracks, expected_track_ps, iou_threshold)
              for name, boxes in models.items()]
    results.sort(key=lambda r: r["pct"], reverse=True)
    return results


def format_comparison(results: List[dict]) -> str:
    lines = ["=== ASD bake-off ==="]
    for r in results:
        lines.append(f"{r['name']:<24} {r['pct']:5.1f}%  "
                     f"({r['matched']}/{r['scored']} scored, "
                     f"{r['no_call_seconds']} no-call seconds)")
    if results:
        lines.append(f"\nwinner: {results[0]['name']}")
    return "\n".join(lines)
