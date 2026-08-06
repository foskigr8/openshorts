"""Per-clip-worker GPU assignment, so a second GPU is not left idle.

Measured on Kaggle 5-aug-2026 (job b86b8c5a, 2×T4): three clips rendered
concurrently and all three ran on GPU 0. GPU 1 was idle for the whole job.

**Why not CUDA_VISIBLE_DEVICES.** The optimization plan proposed setting it per
worker. That works for worker *processes*; OpenShorts renders clips on a
ThreadPoolExecutor inside one process (main.py's `_process_one_clip`), and
CUDA reads that variable once at initialization — setting it from a thread
changes nothing. What IS per-thread is PyTorch's current device, so affinity is
established with ``torch.cuda.set_device`` on the worker thread plus an
explicit device string handed to the torch code that accepts one (LR-ASD).

**What this does and does not parallelize.** LR-ASD is the torch workload
inside a clip render (20-34s of a 94-110s clip in the measured run) and it does
shard. FFmpeg decode/encode now follows the worker's device too (the reframe,
clip-cut and caption-burn call sites pass gpu_affinity.current_device() into
`-hwaccel_device`), so the second GPU is used for the ffmpeg stage as well.
MediaPipe and YOLO detection stay on the default device: they run under
main.py's global DETECT_LOCK because the graph and the model are not
thread-safe, so they are serialized regardless of how many GPUs exist —
sharding them would mean a model instance per device, which is a bigger change
than this. So expect a partial improvement on multi-clip jobs, not 2×, and
nothing at all on single-clip jobs.

Environment
-----------
CLIP_GPUS   comma-separated device indices to spread across. UNSET = no
            affinity at all (every worker uses the default device), which is
            the default because sharding broke TransNetV2 scene detection on
            the first real 2xT4 run — see available_devices(). Set
            CLIP_GPUS=0,1 to opt in. Ignored when CUDA is unavailable.
"""

import os
import threading

_local = threading.local()
_devices_cache = None
_devices_lock = threading.Lock()


def available_devices():
    """Device indices clip workers may use. Empty list on CPU hosts.

    Cached: torch.cuda.device_count() is cheap but this is consulted per clip,
    and the answer cannot change within a process.
    """
    global _devices_cache
    if _devices_cache is not None:
        return _devices_cache
    with _devices_lock:
        if _devices_cache is not None:
            return _devices_cache
        devices = []
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                configured = os.environ.get("CLIP_GPUS", "").strip()
                if configured:
                    for part in configured.split(","):
                        part = part.strip()
                        if part.isdigit() and int(part) < count:
                            devices.append(int(part))
                # OPT-IN ONLY. Spreading workers across GPUs by default broke
                # scene detection on the first real 2xT4 run (6-aug-2026):
                #
                #   TransNetV2 scene detection failed (RuntimeError: Expected
                #   all tensors to be on the same device, but got weight is on
                #   cuda:0, different from other tensors on cuda:1)
                #
                # The model is loaded once on the default device; a worker
                # thread whose current device is cuda:1 then feeds it tensors
                # from the wrong device. It failed open to PySceneDetect, so
                # the job still finished — with WORSE scene boundaries, which
                # is exactly the kind of silent quality regression this
                # codebase keeps getting bitten by.
                #
                # The measured upside was never large (LR-ASD is 20-34s of a
                # 94-110s clip, and detection stays serialized under
                # DETECT_LOCK either way), so it is not worth shipping on by
                # default until every model load is device-aware. Set
                # CLIP_GPUS=0,1 to opt back in.
        except Exception:
            devices = []
        _devices_cache = devices
        return devices


def assign_worker(index):
    """Bind the CALLING thread to one GPU, round-robin by clip index.

    Returns the device string ("cuda:1") or None when there is nothing to
    assign. Safe to call on CPU hosts and single-GPU hosts — the latter still
    returns "cuda:0", which is what the code did implicitly anyway.
    """
    devices = available_devices()
    if not devices:
        _local.device = None
        return None
    device_index = devices[index % len(devices)]
    try:
        import torch
        # Thread-local in PyTorch: this sets the current device for THIS
        # worker thread only, which is exactly the granularity needed.
        torch.cuda.set_device(device_index)
    except Exception as e:
        print(f"⚠️ GPU affinity: could not select cuda:{device_index} "
              f"({type(e).__name__}: {e}) — using the default device")
        _local.device = None
        return None
    _local.device = f"cuda:{device_index}"
    return _local.device


def current_device():
    """The device assigned to this thread, or None if it was never assigned.

    Consumers pass this to torch code that takes an explicit device; None means
    "decide for yourself", preserving the previous behaviour exactly.
    """
    return getattr(_local, "device", None)


def describe():
    """One log line for job start. Empty string when there is nothing to say."""
    devices = available_devices()
    if len(devices) > 1:
        return (f"🎛️  Clip workers will spread across {len(devices)} GPUs "
                f"({', '.join('cuda:%d' % d for d in devices)}) — LR-ASD and "
                "ffmpeg decode/encode follow the worker device; detection "
                "stays serialized on the default device.")
    return ""
