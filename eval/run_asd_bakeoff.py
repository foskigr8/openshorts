#!/usr/bin/env python3
"""Phase 2 harness entry point: runs LR-ASD against the Phase 1 face spine
on one already-cut clip and scores it with eval/asd_bakeoff.py.

Decision recorded (see the rebuild plan, Phase 2): LR-ASD stays the ASD
model for now. Its accuracy ceiling is not the confirmed root cause of the
wrong-person framing bug (that was the two-unrelated-trackers wiring issue
Phase 1 fixes); a model swap (LoCoNet) is real, unverified integration work
with an uncertain payoff and, for the UniTalk-trained checkpoint
specifically, no listed license. This script exists so that decision is
CHECKED against real numbers, not just reasoned about — if LR-ASD's measured
accuracy here is bad in a way Phase 1/3 cannot explain, that is the signal
to revisit the model, not a guess.

Must run inside the Docker/Kaggle environment (needs cv2, torch, insightface
+ onnxruntime, the LR-ASD vendor weights).

Usage:
    python3 eval/run_asd_bakeoff.py /path/to/already_cut_clip.mp4 \\
        --metadata /app/output/<job>/<base>_metadata.json \\
        --ground-truth eval/ptb_ground_truth.json \\
        --clip-start 481.0 --clip-end 513.0

The clip must already cover just the span you want scored (e.g. the output
of the notebook's optional render-test cell, or any short clip cut with
ffmpeg) — this script does not trim a longer source itself, to keep
face_spine's whole-video pass and asd_worker's clip-scoped pass looking at
the exact same frames.

--clip-start/--clip-end are the clip's OWN absolute position in the source
video, only needed to align it against the transcript for ground truth;
they are 0/clip-duration if the clip file itself starts at zero.
"""
import argparse
import json
import os
import sys


def _load_transcript(metadata_path):
    with open(metadata_path) as f:
        meta = json.load(f)
    t = meta.get("transcript")
    if t is None:
        raise SystemExit(f"{metadata_path} has no 'transcript' key")
    return t if isinstance(t, dict) else {"segments": t}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", help="Path to an already-cut clip covering the span to score")
    ap.add_argument("--metadata", required=True, help="Job metadata.json with the full transcript")
    ap.add_argument("--ground-truth", required=True,
                    help="Ground-truth JSON with speaker_names + track_names (see eval/ground_truth.py)")
    ap.add_argument("--clip-start", type=float, default=0.0,
                    help="The clip's own start time in the SOURCE video (for transcript alignment)")
    ap.add_argument("--clip-end", type=float, default=None,
                    help="The clip's own end time in the SOURCE video (default: clip-start + duration)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import cv2
    import asd_worker
    import face_spine
    from eval import asd_bakeoff, ground_truth

    cap = cv2.VideoCapture(args.clip)
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0))
    cap.release()
    clip_end = args.clip_end if args.clip_end is not None else args.clip_start + duration

    print(f"Building face spine over {args.clip} ...")
    spine_tracks = face_spine.build_face_spine(args.clip)
    print(f"  {len(spine_tracks)} track(s) after identity merge")

    # Feed LR-ASD the SAME detector face_spine used (SCRFD), so both passes
    # see identical boxes for identical faces — the point of Phase 1 was one
    # shared spine, and that only holds if every consumer detects the same way.
    ctx_id = face_spine._resolve_ctx_id(None)
    analyzer = face_spine._get_analyzer(ctx_id=ctx_id)

    def detect_faces(frame):
        try:
            faces = analyzer.get(frame)
        except Exception:
            return []
        return [{"box": face_spine._xywh_from_bbox(f.bbox),
                 "score": float(getattr(f, "det_score", 1.0))}
                for f in faces]

    print("Running LR-ASD ...")
    asd_result = asd_worker.score_clip(args.clip, detect_faces)
    per_second_box = asd_result.get("per_second_box", [])
    print(f"  {len(per_second_box)} second(s) scored, "
          f"{sum(1 for b in per_second_box if b is None)} with no confident speaker")

    transcript = _load_transcript(args.metadata)
    speaker_names = ground_truth.load_speaker_names(args.ground_truth)
    track_names = ground_truth.load_track_names(args.ground_truth)
    if not track_names:
        raise SystemExit(
            "No 'track_names' in the ground-truth file — watch the spine's "
            "tracks once and add e.g. {\"track_names\": {\"0\": \"host\"}} "
            "before scoring (see eval/ground_truth.py's module docstring).")

    per_second_speaker_name = ground_truth.per_second_speaker_from_transcript(
        transcript.get("segments", []), args.clip_start, clip_end,
        speaker_names=speaker_names)
    expected_track_ps = asd_bakeoff.expected_track_per_second(
        track_names, per_second_speaker_name)

    result = asd_bakeoff.score_model("LR-ASD", per_second_box, spine_tracks, expected_track_ps)
    print()
    print(asd_bakeoff.format_comparison([result]))


if __name__ == "__main__":
    main()
