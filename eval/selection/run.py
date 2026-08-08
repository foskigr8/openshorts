#!/usr/bin/env python3
"""Phase 0 harness entry point for the Stage 3 rebuild: runs a real
transcript fixture through a real selection entry point end to end and
prints the metrics block from eval/selection/metrics.py.

Per HANDOFF_FRAMING.md's rule (carried over from the framing rebuild): a
measurement not calling the real entry point is not a measurement. This
calls main.get_viral_clips directly — the same function process_video_to_
vertical's caller uses — not a reimplementation of its logic.

Must run inside an environment with main.py's full dependency set (google-
genai, scenedetect, etc — see requirements.txt); the metrics/ground_truth
modules themselves are pure Python and importable anywhere.

Usage:
    python3 eval/selection/run.py \\
        --transcript eval/selection/fixtures/some_video_metadata.json \\
        --ground-truth eval/selection/fixtures/some_video_ground_truth.json \\
        --duration 3600 \\
        --engine current   # or: skill | narrative (VIRAL_ENGINE override)

No --source-video is accepted on purpose for Phase 0: passing one would
route through _vision_confirm_candidates (a real Gemini video-upload call
per candidate), which is vision confirmation, already fixed this session and
explicitly OUT OF SCOPE for the selection rebuild this harness measures.
Selection-only scoring needs source_video_path=None, which main.py's own
_vision_confirm_candidates already treats as "skip confirmation, return
shorts unchanged" — so this harness measures exactly the thing Phase 0-5 of
the Stage 3 plan are changing, nothing else.
"""
import argparse
import json
import sys
import time

from eval.selection import ground_truth, metrics


def _run_selection(transcript, video_duration, engine):
    """Calls the real Stage 3 entry point. Imported lazily so this module's
    pure parts (metrics/ground_truth) stay importable without main.py's full
    dependency chain (cv2/scenedetect/google-genai)."""
    import os

    import main as m

    if engine and engine != "current":
        os.environ["VIRAL_ENGINE"] = engine
    result = m.get_viral_clips(transcript, video_duration, source_video_path=None)
    return (result or {}).get("shorts") or []


def _build_words(transcript):
    words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []) or []:
            words.append({"w": w.get("word", ""), "s": w.get("start"), "e": w.get("end")})
    return words


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcript", required=True,
                    help="Job metadata.json or a standalone transcript fixture")
    ap.add_argument("--ground-truth", required=True,
                    help="Ground-truth JSON, see eval/selection/ground_truth.py")
    ap.add_argument("--duration", type=float, required=True,
                    help="Source video duration in seconds")
    ap.add_argument("--engine", default="current",
                    help="current (VIRAL_ENGINE unset) | skill | narrative")
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    args = ap.parse_args()

    transcript = ground_truth.load_transcript_fixture(args.transcript)
    truth = ground_truth.load_ground_truth(args.ground_truth)
    words = _build_words(transcript)

    from clip_selection import sentence_boundaries
    sentence_starts, sentence_ends = sentence_boundaries(words)

    t0 = time.time()
    try:
        shorts = _run_selection(transcript, args.duration, args.engine)
        timed_out = False
    except Exception as e:
        print(f"❌ Selection failed: {type(e).__name__}: {e}", file=sys.stderr)
        shorts, timed_out = [], False
    wall = time.time() - t0

    span_result = metrics.span_overlap_score(
        shorts, truth["expected_clips"], iou_threshold=args.iou_threshold)
    violations = metrics.must_not_include_violations(shorts, truth["must_not_include"])
    boundary_result = metrics.sentence_boundary_hit_rate(
        shorts, sentence_starts, sentence_ends)
    stats = metrics.SelectionRunStats(
        engine=args.engine, wall_seconds=wall, call_count=len(shorts),
        timed_out=timed_out)

    print(f"\n=== Stage 3 selection: {args.engine} on {truth.get('video') or args.transcript} ===")
    print(f"clips selected:        {len(shorts)}")
    print(f"wall time:             {wall:.1f}s")
    print(f"span match rate:       {span_result.match_rate:.0%} "
          f"({span_result.matched_count}/{span_result.expected_count} expected clips found, "
          f"IoU>={args.iou_threshold})")
    print(f"mean best IoU:         {span_result.mean_best_iou:.2f}")
    print(f"must-not-include hits: {len(violations)}")
    print(f"sentence-cut clean:    {boundary_result.hit_rate:.0%} "
          f"({boundary_result.both_clean_count}/{boundary_result.clip_count} clips)")

    if span_result.matches:
        print("\nper-expected-clip:")
        for m in span_result.matches:
            status = "OK " if m.best_iou >= args.iou_threshold else "MISS"
            print(f"  [{status}] {m.expected_start:.1f}-{m.expected_end:.1f}s "
                  f"(best IoU {m.best_iou:.2f}) — {m.note}")
    if violations:
        print("\nmust-not-include violations:")
        for v in violations:
            print(f"  {v.selected_start:.1f}-{v.selected_end:.1f}s overlaps "
                  f"forbidden {v.forbidden_start:.1f}-{v.forbidden_end:.1f}s — {v.note}")

    print(json.dumps({
        "engine": stats.engine, "wall_seconds": stats.wall_seconds,
        "clip_count": stats.call_count, "span_match_rate": span_result.match_rate,
        "mean_best_iou": span_result.mean_best_iou,
        "must_not_include_violations": len(violations),
        "sentence_cut_hit_rate": boundary_result.hit_rate,
    }))


if __name__ == "__main__":
    main()
