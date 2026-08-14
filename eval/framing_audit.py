#!/usr/bin/env python3
"""Framing audit — the gate that makes PLAN_FRAMING_CONTRACT.md enforceable.

Measures a RENDERED clip against the invariants and reports pass/fail. It reads
only the output file, so it audits what the viewer actually sees rather than
what the planner believed it was doing — the three shipped failures all had a
self-consistent plan.

    python3 eval/framing_audit.py clip.mp4 [more.mp4 ...] [--json out.json]
    python3 eval/framing_audit.py --strict clip.mp4      # exit 1 on any FAIL

Thresholds are CALIBRATED, not chosen: they are set so both reference shorts in
`video-refrence/` pass and all three known-bad renders in `videosflop/` fail.
A gate that passes the known-bad clips is not a gate.

Calibration baseline (2026-08-14, the numbers the thresholds are drawn around;
`tests/test_framing_audit.py` locks them in so a future tweak cannot quietly
let the bad clips through):

    clip                    eyeline  face_h  empty  split  seam  med.shot
    reference A               0.218   0.149    10%     2%    0%     2.90s  PASS
    reference B               0.220   0.163     0%     0%    0%     3.27s  PASS
    flop clip 1               0.562   0.219     0%   100%   68%     1.96s  FAIL
    flop clip 2               0.412   0.092    21%     4%    0%     1.58s  FAIL
    flop clip 4               0.385   0.100    27%     8%    0%     1.23s  FAIL

`seam` is the fraction of SPLIT frames whose seam runs through a face — the
single most visible defect and the one clip 1 is made of.

CAPTIONS ARE OUT OF SCOPE. Invariant I8 is deliberately not audited — caption
placement is owned by subtitles.py and is not part of this rebuild.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import framing_contract as fc   # noqa: E402


MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "models", "face_detection_yunet_2023mar.onnx")

#: Sampling rate. 2fps is enough to catch a held bad crop and cheap enough to
#: run on every clip; a 1-frame glitch is not what this gate is for.
SAMPLE_FPS = 2.0

#: Detector confidence. Deliberately lower than the 0.7 used for the one-off
#: measurements: the reference is full of hard profiles and over-the-shoulder
#: angles, and treating a missed profile as "nobody on screen" would fail good
#: footage for a detector limitation.
DETECT_CONF = 0.5

#: Scene-cut sensitivity for the shot-rhythm metrics.
SCENE_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Thresholds — see module docstring on calibration
# ---------------------------------------------------------------------------

@dataclass
class Thresholds:
    # I2 — a head this close to the frame top is cut or about to be.
    head_top_min: float = 0.010
    head_top_max_frac: float = 0.15      # fail if >15% of frames breach it
    # I3 — where the eyeline sits (median over the clip).
    eye_band: Tuple[float, float] = (0.16, 0.34)
    # I4 — how big the subject reads (median over the clip).
    face_band: Tuple[float, float] = (0.10, 0.225)
    # I5 — frames showing nobody.
    empty_max_frac: float = 0.15
    # I5 — a face bisected by the split seam. This one is near-zero tolerance:
    # the reference never does it and it is the single most visible defect.
    seam_max_frac: float = 0.05
    # I6 — frames whose centre column lands on no one.
    gap_max_frac: float = 0.35
    # I9 — cut rhythm.
    median_shot_min: float = 2.0
    short_shot_max_frac: float = 0.15
    # Informational only (never fails a clip): see plan §5.4.
    split_share_warn: float = 0.60


DEFAULT = Thresholds()


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    t: float
    w: int
    h: int
    faces: List[Tuple[float, float, float, float]] = field(default_factory=list)
    seam_edge: float = 0.0     # horizontal discontinuity strength at y = H/2


def _detect(path: str) -> List[Sample]:
    import cv2

    if not os.path.exists(MODEL):
        raise SystemExit(
            f"YuNet model missing at {MODEL}\n"
            "  curl -L -o eval/models/face_detection_yunet_2023mar.onnx \\\n"
            "    https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / SAMPLE_FPS)))
    det = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), DETECT_CONF, 0.3, 5000)

    out: List[Sample] = []
    index = 0
    while True:
        # grab() advances without converting to an array — the frames we skip
        # cost a fraction of a full read(), which is the difference between
        # this gate taking seconds and taking minutes.
        if not cap.grab():
            break
        if index % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            det.setInputSize((w, h))
            _, raw = det.detect(frame)
            faces = ([] if raw is None
                     else [tuple(float(v) for v in b[:4]) for b in raw])
            # Split-seam detector: a stacked split puts two unrelated images
            # edge to edge, so the row pair straddling H/2 differs far more than
            # neighbouring row pairs do. Cheap, and it needs no plan file.
            mid = h // 2
            import numpy as np
            band = frame[max(0, mid - 6):min(h, mid + 6)].astype("int16")
            diffs = [float(abs(band[i + 1] - band[i]).mean())
                     for i in range(len(band) - 1)]
            centre = diffs[len(diffs) // 2] if diffs else 0.0
            others = sorted(diffs)[:-1]
            baseline = (sum(others) / len(others)) if others else 1.0
            out.append(Sample(index / fps, w, h, faces,
                              centre / max(baseline, 1e-3)))
        index += 1
    cap.release()
    return out


def _cuts(path: str, duration: float) -> List[float]:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-vf",
             f"select='gt(scene,{SCENE_THRESHOLD})',metadata=print:file=-",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return []
    times = []
    for line in (proc.stdout + proc.stderr).splitlines():
        if "pts_time:" in line:
            try:
                times.append(float(line.split("pts_time:")[1].split()[0]))
            except (IndexError, ValueError):
                pass
    return sorted(t for t in times if 0.0 < t < duration)


def _median(values: Sequence[float]) -> float:
    v = sorted(values)
    if not v:
        return float("nan")
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


# ---------------------------------------------------------------------------
# The invariants, measured in OUTPUT space
# ---------------------------------------------------------------------------

def measure(path: str) -> Dict[str, object]:
    samples = _detect(path)
    if not samples:
        raise SystemExit(f"no frames decoded from {path}")
    duration = samples[-1].t + 1.0 / SAMPLE_FPS
    cuts = _cuts(path, duration)

    bounds = [0.0] + cuts + [duration]
    shots = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]

    eyes, sizes = [], []
    empty = head_high = gap = seam = split_frames = 0

    for s in samples:
        H, W = s.h, s.w
        heads = [fc.head_box(f, W, H) for f in s.faces]
        big = [h for h in heads if h[3] / H >= fc.MIN_HEAD_FRAC]
        if not big:
            empty += 1
        is_split = s.seam_edge >= 3.0
        if is_split:
            split_frames += 1
        if not s.faces:
            continue

        # I5 — a face bisected by the split seam. Only meaningful on frames
        # that ARE a split: in a normal single, y = H/2 is just a line, and a
        # low foreground face crossing it is fine (the reference does it in 8%
        # of frames). Gating on the seam detector is what stops this metric
        # from failing good footage.
        if is_split and any(f[1] < H / 2.0 < f[1] + f[3] for f in s.faces):
            seam += 1

        # The dominant (largest) face stands in for the bound speaker: the
        # audit has no plan, and in a correct render the subject IS the biggest
        # face on screen.
        dom = max(s.faces, key=lambda f: f[2] * f[3])
        head = fc.head_box(dom, W, H)
        eyes.append(fc.eyeline_y(dom) / H)
        sizes.append(dom[3] / H)

        # I2 — in output space the crop top IS the frame top.
        if head[1] / H < DEFAULT.head_top_min:
            head_high += 1

        # I6 — does the centre column land on anybody?
        lo, hi = W * (1 - fc.CENTRE_BAND) / 2, W * (1 + fc.CENTRE_BAND) / 2
        if not any(lo <= (h[0] + h[2] / 2.0) <= hi for h in heads):
            gap += 1

    n = len(samples)
    n_faces = max(1, len(eyes))
    return {
        "clip": os.path.basename(path),
        "duration_s": round(duration, 2),
        "frames_sampled": n,
        "cuts": len(cuts),
        "median_shot_s": round(_median(shots), 2),
        "short_shot_frac": round(
            sum(1 for d in shots if d < 1.0) / max(1, len(shots)), 3),
        "eyeline_median": round(_median(eyes), 3),
        "face_h_median": round(_median(sizes), 3),
        "empty_frac": round(empty / n, 3),
        "head_top_frac": round(head_high / n_faces, 3),
        # Denominator is SPLIT frames: "of the frames that are a split, how
        # many put the seam through a face". On a clip with no split at all
        # this is 0 by definition.
        "seam_frac": round(seam / max(1, split_frames), 3),
        "gap_frac": round(gap / n_faces, 3),
        "split_share": round(split_frames / n, 3),
    }


def verdict(m: Dict[str, object], t: Thresholds = DEFAULT
            ) -> Tuple[List[str], List[str]]:
    """(failures, warnings). Empty failures == the clip passes the gate."""
    fail, warn = [], []

    if m["head_top_frac"] > t.head_top_max_frac:
        fail.append(
            f"I2 head touching the frame top in {m['head_top_frac']:.0%} of "
            f"frames (max {t.head_top_max_frac:.0%})")
    lo, hi = t.eye_band
    if not (lo <= m["eyeline_median"] <= hi):
        fail.append(
            f"I3 median eyeline {m['eyeline_median']} outside [{lo}, {hi}] "
            f"(reference: 0.217)")
    lo, hi = t.face_band
    if not (lo <= m["face_h_median"] <= hi):
        fail.append(
            f"I4 median face height {m['face_h_median']} outside [{lo}, {hi}] "
            f"(reference: 0.155)")
    if m["empty_frac"] > t.empty_max_frac:
        fail.append(
            f"I5 {m['empty_frac']:.0%} of frames show nobody "
            f"(max {t.empty_max_frac:.0%})")
    if m["seam_frac"] > t.seam_max_frac:
        fail.append(
            f"I5 {m['seam_frac']:.0%} of SPLIT frames put the seam through a "
            f"face (max {t.seam_max_frac:.0%})")
    if m["gap_frac"] > t.gap_max_frac:
        fail.append(
            f"I6 {m['gap_frac']:.0%} of frames centre on nobody "
            f"(max {t.gap_max_frac:.0%})")
    if m["median_shot_s"] < t.median_shot_min:
        fail.append(
            f"I9 median shot {m['median_shot_s']}s < {t.median_shot_min}s "
            f"(reference: 2.9-3.3s)")
    if m["short_shot_frac"] > t.short_shot_max_frac:
        fail.append(
            f"I9 {m['short_shot_frac']:.0%} of shots are under 1s "
            f"(max {t.short_shot_max_frac:.0%})")

    if m["split_share"] > t.split_share_warn:
        warn.append(
            f"split screen on {m['split_share']:.0%} of frames — not a failure, "
            f"but the two-shot test is probably not firing (plan §5.4)")
    return fail, warn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _report(path: str, t: Thresholds) -> Tuple[Dict[str, object], bool]:
    m = measure(path)
    fail, warn = verdict(m, t)
    ok = not fail
    print(f"\n{'PASS' if ok else 'FAIL'}  {m['clip']}")
    print(f"      {m['duration_s']}s, {m['cuts']} cuts, "
          f"median shot {m['median_shot_s']}s")
    print(f"      eyeline {m['eyeline_median']}   face height "
          f"{m['face_h_median']}   empty {m['empty_frac']:.0%}   "
          f"gap {m['gap_frac']:.0%}")
    print(f"      split {m['split_share']:.0%} of frames, "
          f"seam-through-face {m['seam_frac']:.0%} of those")
    for f in fail:
        print(f"      ✗ {f}")
    for w in warn:
        print(f"      ! {w}")
    m["pass"] = ok
    m["failures"] = fail
    m["warnings"] = warn
    return m, ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--json", help="write the full metric table here")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any clip fails")
    args = ap.parse_args(argv)

    results, all_ok = [], True
    for path in args.clips:
        m, ok = _report(path, DEFAULT)
        results.append(m)
        all_ok = all_ok and ok

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")

    print(f"\n{sum(1 for r in results if r['pass'])}/{len(results)} passed")
    return 0 if (all_ok or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
