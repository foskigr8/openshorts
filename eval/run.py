#!/usr/bin/env python3
"""Phase 0 harness entry point: renders a span with the current engine,
loads its REFRAME_DUMP_PATH rects + the job's transcript, and prints the
metrics block from eval/metrics.py.

Must run inside the Docker/Kaggle environment (needs cv2, torch, ffmpeg,
the LR-ASD vendor weights) — it is a thin wrapper around reframe_v2.render(),
the same call main.py:1603 makes, per HANDOFF_FRAMING.md's hard rule that a
measurement not calling render() the real way is not a measurement.

Usage:
    REFRAME_DUMP_PATH=/tmp/dump python3 eval/run.py \\
        /path/to/source.mp4 --span 481-513 \\
        --metadata /app/output/<job>/<base>_metadata.json \\
        --ground-truth eval/ptb_ground_truth.json \\
        --engine v2

Ground truth JSON: see eval/ground_truth.py for the format.
"""
import argparse
import json
import os
import sys
import tempfile


def _load_transcript(metadata_path):
    with open(metadata_path) as f:
        meta = json.load(f)
    t = meta.get("transcript")
    if t is None:
        raise SystemExit(f"{metadata_path} has no 'transcript' key")
    return t if isinstance(t, dict) else {"segments": t}


def _build_word_list(transcript):
    words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []) or []:
            words.append({"w": w.get("word", ""), "s": w.get("start"), "e": w.get("end")})
    return words


def _shot_boundaries_from_rects(rects):
    """Fallback shot segmentation for the CURRENT (v2) engine, which has no
    explicit shot list: a "shot" is a maximal run of frames with IDENTICAL
    crop rects (v2 is meant to hold a rect steady between cuts, easing
    aside). This is deliberately naive — it is a baseline measurement of the
    engine AS IT EXISTS, not a stand-in for the real shot list Phase 4 will
    produce, which is why crop_motion_within_shot on v2 output is expected
    to be non-zero even within one of these "shots": v2 eases continuously,
    it doesn't hold a truly static rect.
    """
    if not rects:
        return []
    boundaries = []
    start = 0
    for i in range(1, len(rects)):
        if rects[i] != rects[i - 1]:
            pass  # v2 rarely repeats a rect exactly frame-to-frame; see below
    # v2's easing means adjacent rects almost never match exactly, so the
    # "identical rect" definition above degenerates to one shot per frame.
    # Segment on SUBJECT CHANGE instead, approximated here by a crop-centre
    # jump bigger than 15% of the crop width in one frame (a hard cut, per
    # reframe_v2's own "force_next_update" re-target-then-cut behaviour) —
    # small continuous drift within a shot stays grouped, a real cut splits.
    boundaries = []
    start = 0
    for i in range(1, len(rects)):
        x0, y0, w0, h0 = rects[i - 1]
        x1, y1, w1, h1 = rects[i]
        cx0, cx1 = x0 + w0 / 2.0, x1 + w1 / 2.0
        if w0 <= 0:
            continue
        if abs(cx1 - cx0) > 0.15 * w0:
            boundaries.append((start, i))
            start = i
    boundaries.append((start, len(rects)))
    return boundaries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="Path to the source video")
    ap.add_argument("--span", required=True, help="start-end in seconds, e.g. 481-513")
    ap.add_argument("--metadata", required=True, help="Job metadata.json with the full transcript")
    ap.add_argument("--ground-truth", help="Ground-truth JSON (see eval/ground_truth.py)")
    ap.add_argument("--engine", default="v2", choices=["v2"], help="v3 lands in Phase 5")
    ap.add_argument("--aspect", type=float, default=9 / 16)
    args = ap.parse_args()

    start_s, end_s = (float(x) for x in args.span.split("-"))

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import reframe_v2
    from eval import metrics, ground_truth

    dump_dir = os.environ.get("REFRAME_DUMP_PATH") or tempfile.mkdtemp(prefix="eval_dump_")
    os.environ["REFRAME_DUMP_PATH"] = dump_dir

    transcript = _load_transcript(args.metadata)
    out_path = os.path.join(tempfile.mkdtemp(prefix="eval_render_"), "eval_clip.mp4")

    print(f"Rendering [{start_s}, {end_s}] with engine={args.engine} ...")
    reframe_v2.render(args.source, out_path, args.aspect,
                       transcript=transcript, clip_start=start_s, clip_end=end_s)

    tag = os.path.splitext(os.path.basename(out_path))[0]
    rects_path = os.path.join(dump_dir, f"{tag}_rects.npy")
    if not os.path.exists(rects_path):
        raise SystemExit(f"expected rects dump at {rects_path} — did render() run to completion?")

    import numpy as np
    rects = [tuple(r) for r in np.load(rects_path)]
    shot_boundaries = _shot_boundaries_from_rects(rects)
    shot_motions = metrics.crop_motion_within_shot(rects, shot_boundaries)

    speaker_pct_result = (0.0, 0, 0)
    if args.ground_truth:
        gt_names = ground_truth.load_speaker_names(args.ground_truth)
        gt_per_second = ground_truth.per_second_speaker_from_transcript(
            transcript.get("segments", []), start_s, end_s, speaker_names=gt_names)
        # NOTE: v2 has no per-second "who did we frame" output today — that
        # is added in Phase 3 (global fusion) / Phase 4 (shot list), so this
        # is wired to an empty list until then. Kept here, not deferred to a
        # later script, so the CLI shape doesn't change once it's real.
        framed_per_second = []
        speaker_pct_result = metrics.speaker_on_screen_pct(framed_per_second, gt_per_second)

    boundary_result = None
    words = _build_word_list(transcript)
    if words:
        boundary_result = metrics.boundary_sentence_completeness(start_s, end_s, words)

    print()
    print(metrics.format_report(f"{args.engine} {args.span}", shot_motions,
                                 speaker_pct_result, boundary_result))
    print(f"\n(rects dump kept at {rects_path} for further inspection)")


if __name__ == "__main__":
    main()
