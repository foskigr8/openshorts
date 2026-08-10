"""Scene detection engines for the reframe pipeline.

Primary engine: TransNetV2, a small neural shot-boundary detector that is
markedly more accurate than threshold-based detection (handles fast camera
motion, flashes and gradual transitions such as fades/dissolves). Runs on
48x27 frames, faster than realtime even on CPU.

Fallback/legacy engine: PySceneDetect ContentDetector, byte-for-byte the
pre-existing behavior. Any TransNetV2 failure (missing package, corrupt
weights, decode error) falls back automatically so a scene-detection edge
case can never kill a job.

Environment variables:
  SCENE_ENGINE          "transnetv2" (default) | "pyscenedetect" (legacy)
  SCENE_GPU_ONLY        "1" (default) — if the GPU cannot decode the source,
                        scene detection SKIPS (no CPU decode, no
                        PySceneDetect); the end-clamp polish is dropped with
                        a clear log line. "0" allows the cheap 48x27 CPU
                        retry and the PySceneDetect fallback for hosts that
                        accept CPU decode.
  SCENE_DETECT_TIMEOUT  max seconds for the TransNetV2 whole-video decode
                        (default 300; a 103-min AV1 source on CPU took 900s+
                        — the cap exists so scene detection can never hold a
                        job hostage for a quarter hour; the scene clamp fails
                        open when detection times out)
  SCENE_MIN_SEC         minimum scene length in seconds; shorter scenes are
                        merged into a neighbor (default 0.4, TransNetV2 path
                        only — the legacy path stays untouched)
  TRANSNETV2_THRESHOLD  shot-boundary probability threshold (default 0.5)
  TRANSNETV2_DEVICE     torch device: "auto" (default) | "cpu" | "cuda" | "mps"
"""

import os
import subprocess
import threading

import cv2
import numpy as np
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector

import ffmpeg_utils

# TransNetV2 input size (width x height), fixed by the trained model.
_TN2_W, _TN2_H = 48, 27
_SCENE_DETECT_TIMEOUT = int(os.environ.get("SCENE_DETECT_TIMEOUT", "300"))

# One shared model instance; clips can process in parallel (CLIP_WORKERS) so
# inference is serialized like the other detectors in main.py.
_TN2_LOCK = threading.Lock()
_tn2_model = None


class _model_device_guard:
    """Run torch inference on the MODEL's device, then restore the thread's
    previous current device (6-aug-2026).

    Clip workers get a thread-local torch device via gpu_affinity (cuda:1 for
    half of them), but the shared TransNetV2 model lives on the device it was
    loaded on. Ops inside the forward pass that create tensors without an
    explicit device use the thread's CURRENT device — so a cuda:1 worker
    feeding the cuda:0 model died with "Expected all tensors to be on the
    same device" and silently degraded every 2×T4 run to PySceneDetect. This
    guard pins the current device to the model's for the duration of the
    call and restores it afterwards, so scene detection is correct no matter
    which GPU the worker was assigned.
    """

    def __init__(self, model_device):
        self.model_device = model_device
        self.prev = None

    def __enter__(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.prev = torch.cuda.current_device()
                dev = torch.device(self.model_device)
                if dev.type == "cuda":
                    torch.cuda.set_device(dev)
        except Exception:
            self.prev = None
        return self

    def __exit__(self, *exc):
        if self.prev is not None:
            try:
                import torch
                torch.cuda.set_device(self.prev)
            except Exception:
                pass
        return False


def detect_scenes(video_path):
    """Detect scenes. Returns (scene_list, fps) where scene_list is a list of
    (FrameTimecode, FrameTimecode) pairs — the same contract PySceneDetect's
    SceneManager.get_scene_list() has always given callers."""
    engine = os.environ.get("SCENE_ENGINE", "transnetv2").strip().lower()
    if engine != "pyscenedetect":
        try:
            return _detect_transnetv2(video_path)
        except Exception as e:
            gpu_only = os.environ.get("SCENE_GPU_ONLY", "1").strip().lower()
            if gpu_only in ("1", "true", "yes"):
                print(f"   ⚠️ TransNetV2 scene detection skipped: GPU decode "
                      f"unavailable for this file ({type(e).__name__}: "
                      f"{str(e)[:200]}). SCENE_GPU_ONLY=1 — no CPU decode, "
                      f"scene clamp will no-op (clips still render).")
                return [], 30.0
            print(f"   ⚠️ DEGRADED OUTPUT: TransNetV2 scene detection failed "
                  f"({type(e).__name__}: {e}) — falling back to PySceneDetect")
    return _detect_pyscenedetect(video_path)


# --- legacy engine ----------------------------------------------------------

def _detect_pyscenedetect(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()
    fps = video.frame_rate
    return scene_list, fps


# --- TransNetV2 engine ------------------------------------------------------

def _get_tn2_model():
    """Lazily build the shared model — LOCKED (7-aug-2026).

    Clip workers each get their own GPU via gpu_affinity, and scene detection
    for concurrent clips used to race here: two worker threads (cuda:0 and
    cuda:1) could both see ``_tn2_model is None`` before either finished
    constructing, "auto" resolving to whichever device that thread currently
    had — producing the exact "weight is on cuda:0, different from other
    tensors on cuda:1" crash seen in production (confirmed 7-aug-2026: 2xT4,
    CLIP_GPUS=0,1). _model_device_guard only pins INFERENCE to the model's
    device; it can't fix a model that was torn across two devices during
    construction. The lock makes construction atomic, at the one-time cost
    of the second worker briefly waiting instead of racing.
    """
    global _tn2_model
    if _tn2_model is None:
        with _TN2_LOCK:
            if _tn2_model is None:
                from transnetv2_pytorch import TransNetV2
                device = os.environ.get("TRANSNETV2_DEVICE", "auto")
                model = TransNetV2(device=device)
                model.eval()
                _tn2_model = model
    return _tn2_model


def _extract_frames_small(video_path):
    """Decode the whole clip as 48x27 RGB frames via ffmpeg (~4KB/frame).

    Uses CUDA decode when the ffmpeg on PATH supports it (the nvenc build) —
    decoding a 103-min episode frame-by-frame is exactly the stage that
    used to eat 15 CPU-minutes.

    If the GPU decode command itself fails (hwdownload/pixel-format mismatch,
    NVDEC rejecting the stream, all decode surfaces busy because five clip
    workers are hammering both cards), retry the SAME decode on the CPU
    before giving up (10-aug-2026). The alternative is not "CPU ffmpeg vs GPU
    ffmpeg" — it is falling through to PySceneDetect, which walks the full
    2560x1440 video on the CPU and took 26 minutes on a 103-min episode.
    A 48x27 rawvideo decode is a fraction of that, so the retry is strictly
    the cheaper failure mode, and the ffmpeg stderr is printed either way
    (it used to go to DEVNULL, which is why the first failure was
    undiagnosable).
    """
    attempts = []
    decode_args = ffmpeg_utils.gpu_decode_args(output_format=True)
    if decode_args:
        # Frames arrive in CUDA memory; pull them back to system memory for
        # the tiny CPU resize + rawvideo pipe. hwdownload WITHOUT a format pin:
        # pinning nv12 broke VP9 streams whose decoded sw format is p010
        # (10-bit VP9) — 'hwdownload,format=nv12' then fails with "Invalid
        # output format nv12 for hwframe download" and scene detection
        # degraded even though the GPU can decode the file.
        attempts.append(("gpu", decode_args,
                         f"hwdownload,scale={_TN2_W}:{_TN2_H}"))
    # SCENE_GPU_ONLY=1 (default): never decode on the CPU — if the GPU cannot
    # handle this file, the failure propagates and detect_scenes skips the
    # stage entirely. "0" keeps the cheap 48x27 CPU retry as a safety net.
    gpu_only = os.environ.get("SCENE_GPU_ONLY", "1").strip().lower()
    if gpu_only not in ("1", "true", "yes"):
        attempts.append(("cpu", [], f"scale={_TN2_W}:{_TN2_H}"))
    if not attempts:
        raise RuntimeError(
            "GPU decode unavailable and SCENE_GPU_ONLY=1 — scene detection "
            "skipped (no CPU decode).")

    last_err = None
    for label, args, vf in attempts:
        cmd = (["ffmpeg", "-nostdin"] + args
               + ["-i", video_path, "-vf", vf,
                  "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, check=True,
                                  timeout=_SCENE_DETECT_TIMEOUT)
            frame_bytes = _TN2_H * _TN2_W * 3
            n = len(proc.stdout) // frame_bytes
            if n == 0:
                raise RuntimeError("ffmpeg produced no frames")
            return np.frombuffer(proc.stdout[:n * frame_bytes],
                                 dtype=np.uint8).reshape(n, _TN2_H, _TN2_W, 3)
        except Exception as e:
            detail = ""
            if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                detail = e.stderr.decode("utf-8", "replace").strip()
                detail = " | ".join(detail.splitlines()[-3:])
            if label == "gpu":
                try:
                    _probe = subprocess.run(
                        ["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=codec_name,profile,pix_fmt",
                         "-of", "csv=p=0", video_path],
                        capture_output=True, text=True, timeout=20)
                    _stream = (_probe.stdout.strip()
                               if _probe.returncode == 0 else "")
                    if _stream:
                        detail = f"{detail} | source: {_stream}".strip(" |")
                except Exception:
                    pass
            last_err = RuntimeError(
                f"{label} decode failed ({type(e).__name__}: {e})"
                + (f" — ffmpeg: {detail}" if detail else ""))
            if label != attempts[-1][0]:
                print(f"   ⚠️ TransNetV2 {last_err} — retrying on CPU decode")
    raise last_err


def _detect_transnetv2(video_path):
    import torch

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frames = _extract_frames_small(video_path)
    model = _get_tn2_model()
    threshold = float(os.environ.get("TRANSNETV2_THRESHOLD", "0.5"))

    with _TN2_LOCK, torch.no_grad(), _model_device_guard(model.device):
        tensor = torch.from_numpy(np.ascontiguousarray(frames)).to(model.device)
        single_frame_pred, _ = model.predict_frames(tensor, quiet=True)

    # predictions_to_scenes returns [[start, end], ...] with INCLUSIVE ends;
    # downstream expects PySceneDetect's exclusive ends.
    raw = model.predictions_to_scenes(single_frame_pred.numpy(), threshold=threshold)
    bounds = [(int(s), int(e) + 1) for s, e in raw]

    # cv2's frame count can differ by a few frames from what ffmpeg decodes;
    # downstream loops run to the decoder's count, so cover the gap.
    if total_frames > bounds[-1][1]:
        bounds[-1] = (bounds[-1][0], total_frames)

    min_sec = float(os.environ.get("SCENE_MIN_SEC", "0.4"))
    bounds = _merge_short_scenes(bounds, fps, min_sec)

    scene_list = [(FrameTimecode(s, fps), FrameTimecode(e, fps))
                  for s, e in bounds]
    print(f"   🎬 Scene engine: TransNetV2 — {len(scene_list)} scenes")
    return scene_list, fps


def _merge_short_scenes(bounds, fps, min_sec):
    """Absorb scenes shorter than min_sec into a neighbor. Ultra-short scenes
    cause camera-snap bursts and starve the per-scene strategy sampling."""
    if len(bounds) <= 1 or min_sec <= 0:
        return bounds
    min_frames = max(1, int(round(min_sec * float(fps))))

    merged = []
    for s, e in bounds:
        if merged and (e - s) < min_frames:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    # The first scene can still be short — fold it into the one after it.
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_frames:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return merged
