"""Machine-readable pipeline progress + per-clip ready markers.

Standalone on purpose: main.py (the producer) and app.py (the consumer) both
need it, and it must stay importable in tests WITHOUT dragging in main.py's
heavy stack (torch/ultralytics/cv2/mediapipe). Everything fails silent — a
progress write or marker write must never break the pipeline that is actually
producing clips.
"""
import json
import os
import time

# Stage weights map onto a 0-100 overall percentage. Render interpolates
# between "analyze" (45) and "render" (90) using clips_done/clips_total.
PROGRESS_WEIGHTS = {
    "download": 8,
    "transcribe": 25,
    "analyze": 45,
    "render": 90,
    "finalize": 100,
}

# Canonical stage order, used for ETA math (app.py): remaining time is the
# rolling average of the current stage's unfinished share plus every later
# stage's full average.
STAGE_ORDER = ("download", "transcribe", "analyze", "render", "finalize")


def write_progress(output_dir, stage, clips_done=0, clips_total=0, note=None,
                   duration_seconds=None):
    """Atomically persist a progress snapshot for app.py's poll loop."""
    try:
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        pct = PROGRESS_WEIGHTS.get(stage, 0)
        stage_pct = None
        if stage == "render" and clips_total > 0:
            base = PROGRESS_WEIGHTS["analyze"]
            span = PROGRESS_WEIGHTS["render"] - base
            pct = base + int(span * max(0, clips_done) / clips_total)
            stage_pct = int(round(100 * max(0, clips_done) / clips_total))
        elif stage == "finalize":
            stage_pct = 100
        payload = {
            "stage": stage,
            "overall_pct": min(100, max(0, pct)),
            "stage_pct": stage_pct,
            "clips_done": clips_done,
            "clips_total": clips_total,
            "updated_at": time.time(),
        }
        if note:
            payload["note"] = note
        if duration_seconds is not None:
            payload["duration_seconds"] = duration_seconds
        tmp = os.path.join(output_dir, ".progress.json.tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, os.path.join(output_dir, "progress.json"))
    except Exception:
        pass


def clip_ready_marker(output_dir, clean_filename):
    """Path of the .ready marker for a clean clip filename."""
    return os.path.join(output_dir, f"{clean_filename}.ready")


def mark_clip_ready(output_dir, clean_filename):
    """Write the .ready marker app.py gates on before serving a clip.

    The marker is written only AFTER the clip's final file (including any
    burned captions) is fully on disk, so the dashboard never points a
    preview at a file that is mid-write or still awaiting captions.
    """
    try:
        marker = clip_ready_marker(output_dir, clean_filename)
        tmp = f"{marker}.tmp"
        with open(tmp, "w") as f:
            f.write(str(time.time()))
        os.replace(tmp, marker)
    except Exception:
        pass


def record_stage_durations(output_dir, durations):
    """Merge per-stage wall-clock seconds into a rolling average.

    Written to ``stage_durations.json`` beside progress.json — app.py reads it
    to estimate time remaining for the CURRENT job (plan round 2, item 3).
    Simple running mean (n + mean per stage); absent stages stay absent.
    """
    try:
        if not output_dir or not durations:
            return
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "stage_durations.json")
        current = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                current = json.load(f)
        for stage, seconds in durations.items():
            if seconds is None or seconds < 0:
                continue
            entry = current.get(stage) or {"n": 0, "mean": 0.0}
            n = int(entry.get("n") or 0)
            mean = float(entry.get("mean") or 0.0)
            new_n = n + 1
            new_mean = mean + (seconds - mean) / new_n
            current[stage] = {"n": new_n, "mean": new_mean}
        tmp = os.path.join(output_dir, ".stage_durations.json.tmp")
        with open(tmp, "w") as f:
            json.dump(current, f)
        os.replace(tmp, path)
    except Exception:
        pass


def estimate_eta_seconds(output_dir, progress):
    """Estimated seconds remaining for the stage in ``progress``, else None.

    Uses the rolling per-stage averages from stage_durations.json. The current
    stage contributes its average scaled by its unfinished share (stage_pct),
    every later stage its full average. First-ever run (no data) -> None.
    """
    try:
        if not output_dir or not progress:
            return None
        path = os.path.join(output_dir, "stage_durations.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            durations = json.load(f)
        current_stage = progress.get("stage")
        if current_stage not in STAGE_ORDER:
            return None
        idx = STAGE_ORDER.index(current_stage)
        stage_pct = progress.get("stage_pct")
        total = 0.0
        for i, stage in enumerate(STAGE_ORDER):
            entry = durations.get(stage)
            if not entry:
                continue
            mean = float(entry.get("mean") or 0.0)
            if i == idx:
                remaining = 1.0
                if isinstance(stage_pct, (int, float)) and 0 <= stage_pct < 100:
                    remaining = (100.0 - stage_pct) / 100.0
                total += mean * remaining
            elif i > idx:
                total += mean
        return round(total) if total > 0 else None
    except Exception:
        return None
