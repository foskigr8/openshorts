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
# Position-matching (ASD box -> face_spine track) is a production concern —
# Phase 3's fusion (speaker_fusion.py) needs the exact same lookup, so it
# lives there and this eval module imports it, not the other way around.
from speaker_fusion import (  # noqa: F401 (re-exported for existing callers)
    _box_iou, _nearest_box_at, match_box_to_track, asd_predicted_track_per_second,
)


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
