"""Shared hardware-aware defaults for main.py (clip workers) and app.py
(concurrent jobs). CPU-bound encode/transcribe is far more likely to thrash
a small host than a GPU one — the concurrency defaults tuned for a T4 (5/5)
oversubscribe a 4-core CPU box badly, so they must differ, not just "not
crash".
"""
import os


def gpu_configured() -> bool:
    """True when docker-compose.gpu.yml was applied (start_studio.sh's
    nvidia-runtime autodetect sets these env vars) — the same signal
    main.py's YOLO_DEVICE startup guard already trusts, reused here so every
    hardware-dependent default agrees on what's actually available."""
    if os.environ.get("YOLO_DEVICE"):
        return True
    return os.environ.get("WHISPER_DEVICE", "").strip().lower() == "cuda"


def default_clip_workers() -> int:
    """Per-job thread pool size for parallel clip rendering (main.py)."""
    if gpu_configured():
        return 5
    cores = os.cpu_count() or 2
    return max(1, min(cores - 1, 3))


def default_max_concurrent_jobs() -> int:
    """Cross-job semaphore size (app.py). CPU hosts run one job at a time by
    default — each job already uses several cores via CLIP_WORKERS, and
    letting two jobs' whisper/ffmpeg passes fight over the same few cores
    slows both down instead of adding throughput."""
    if gpu_configured():
        return 5
    return 1
