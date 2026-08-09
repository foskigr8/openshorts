"""Central video-encoder selection for every ffmpeg encode call site.

FFMPEG_ENCODER env values:
  x264  (default) — CPU libx264, exact pre-GPU behavior
  nvenc           — force h264_nvenc; probed once and falls back to x264
                    (with a warning) if the GPU/driver is unavailable
  auto            — h264_nvenc when the probe succeeds, else x264

Only the codec/quality args live here; surrounding args (-movflags, -pix_fmt,
audio codecs, filters) stay at each call site.
"""
import os
import shutil
import subprocess
import threading
import tempfile

# Quality tiers pinning the historical libx264 settings.
QUALITY = "quality"            # was: -preset medium -crf 18
QUALITY_FAST = "quality_fast"  # was: -preset fast -crf 18
DELIVERY = "delivery"          # was: -preset fast -crf 22

_X264_ARGS = {
    QUALITY: ["-c:v", "libx264", "-preset", "medium", "-crf", "18"],
    QUALITY_FAST: ["-c:v", "libx264", "-preset", "fast", "-crf", "18"],
    DELIVERY: ["-c:v", "libx264", "-preset", "fast", "-crf", "22"],
}

# NVENC -cq is not 1:1 with x264 CRF: benchmarked on the prod GPU (RTX 4000
# Ada), cq ≈ crf + 7 lands in the same file-size ballpark, with vbr + AQ for
# quality. Presets p1-p7: p5 ≈ "medium", p4 ≈ "fast".
# -pix_fmt yuv420p is REQUIRED: with RGB input (the bgr24 rawvideo pipe from
# OpenCV) nvenc otherwise emits H.264 in gbrp/GBR colorspace, which ffmpeg
# reads fine but web players render as a magenta/green mess.
_NVENC_ARGS = {
    QUALITY: ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
              "-rc", "vbr", "-cq", "25", "-b:v", "0",
              "-spatial-aq", "1", "-temporal-aq", "1", "-pix_fmt", "yuv420p"],
    QUALITY_FAST: ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                   "-rc", "vbr", "-cq", "25", "-b:v", "0", "-spatial-aq", "1",
                   "-pix_fmt", "yuv420p"],
    DELIVERY: ["-c:v", "h264_nvenc", "-preset", "p4",
               "-rc", "vbr", "-cq", "29", "-b:v", "0", "-spatial-aq", "1",
               "-pix_fmt", "yuv420p"],
}

# Output args that drop container/stream metadata carried over from the source
# — most notably YouTube's "produced by Google Inc." stream handler, which
# otherwise survives every re-encode (ffmpeg copies input metadata by default)
# and rides into the published clip. The per-stream specifiers are required:
# global -map_metadata -1 alone leaves the audio handler_name intact on a
# stream copy. Empty audio/video specifiers are harmless when a clip lacks that
# stream (ffmpeg ignores them, verified). Spliced in before the output filename
# at each final-artifact producer; kept out of video_encode_args() so that
# stays purely codec/quality args.
METADATA_SCRUB = ["-map_metadata", "-1", "-map_chapters", "-1",
                  "-map_metadata:s:v", "-1", "-map_metadata:s:a", "-1"]

# Loudness normalisation for the delivered clip.
#
# Without this the clip inherits whatever the source was mastered at, so a
# user's clips land anywhere: measured across real delivered clips on
# 26-jul-2026, from -13.8 LUFS on a loud upload down to -28 LUFS on a quiet
# talk. TikTok, Reels and Shorts all normalise playback to roughly -14 LUFS,
# which means the quiet ones just sound thin next to everything else in the
# feed — the loud ones aren't rewarded, the quiet ones are punished.
#
# I=-14 matches the platforms' target, TP=-1.5 leaves headroom so their own
# re-encode can't clip, LRA=11 is the usual allowance for speech. Applied at
# the clip cut, where the audio is being encoded to AAC anyway, so it costs
# nothing extra. AUDIO_NORMALIZE=0 turns it off.
LOUDNORM_FILTER = "loudnorm=I=-14:TP=-1.5:LRA=11"


def audio_encode_args():
    """AAC encode args for a delivered clip, with loudness normalisation.

    Every delivered clip has been shipping at 96kHz instead of the source's
    44.1kHz (confirmed by reproduction, 31-jul-2026) — the loudnorm filter's
    internal true-peak oversampling has no `-ar` pinning it back down
    afterward, so the AAC encoder just runs with whatever rate the
    filtergraph handed it. 96kHz stereo AAC is not a normal delivery format;
    most decoders are tuned for 44.1/48kHz, and this is a real candidate for
    the "sounds phasey/off" quality complaints on delivered audio. Pinned
    to 48000 (a safe, universally-supported rate) regardless of source rate.
    """
    args = ["-ar", "48000", "-c:a", "aac"]
    if os.environ.get("AUDIO_NORMALIZE", "1").strip() != "0":
        args = ["-af", LOUDNORM_FILTER] + args
    return args

_probe_lock = threading.Lock()
_nvenc_ok = None  # None = not probed yet
_announced = False
_gpu_render_lock = threading.Lock()
_gpu_render_ok = None  # None = not probed yet

# GPU decode/filter support (PART 2, 6-aug-2026). Kaggle's ffmpeg build is
# not guaranteed to ship CUDA filters even though NVENC works, so this is
# probed once per process and every call site falls back to CPU decode when
# the build cannot do it. `GPU_RENDER=0` disables; `GPU_RENDER=1` requires
# (warns and falls back if the build lacks CUDA); `auto` uses the probe.
GPU_RENDER_FILTERS = ("scale_cuda", "scale_npp", "overlay_cuda", "crop_cuda")


def _probe_gpu_render():
    """Can THIS ffmpeg decode and filter on the GPU on THIS host?

    Requires BOTH build capability (CUDA hwaccel + a CUDA filter in
    `ffmpeg -filters`) AND a working device: two tiny real operations are
    run — a CUDA decode of a generated file, and scale_cuda on uploaded
    frames. Build support alone is not enough: a container with CUDA filters
    but no NVIDIA device (e.g. a CPU-only backend) must not enable
    `-hwaccel cuda` and then fail every render. The historical CPU path is
    byte-identical when this returns False.
    """
    try:
        hw = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=30)
        # getattr/None guards: tests stub subprocess.run with doubles that
        # carry only returncode; a missing stdout just means "no hwaccel".
        hw_names = str(getattr(hw, "stdout", "") or "").lower()
    except Exception:
        return False
    if not any(a in hw_names for a in ("cuda", "nvdec", "cuvid")):
        print("⚠️ [GPU_RENDER] ffmpeg has no CUDA hwaccel in -hwaccels; "
              "GPU decode unavailable for this build.")
        return False
    try:
        flt = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=30)
        flt_text = str(getattr(flt, "stdout", "") or "")
    except Exception:
        return False
    names = set()
    for line in flt_text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            names.add(parts[1])
    present = sorted(n for n in GPU_RENDER_FILTERS if n in names)
    if not present:
        print("⚠️ [GPU_RENDER] ffmpeg lacks CUDA filters "
              f"({', '.join(GPU_RENDER_FILTERS)}) — GPU decode unavailable.")
        return False
    probe_dir = tempfile.mkdtemp(prefix="gpu_render_probe_")
    try:
        src = os.path.join(probe_dir, "src.mp4")
        ok = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=black:s=64x64:d=0.2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", src],
            capture_output=True, timeout=30)
        if ok.returncode != 0:
            return False
        dec = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
             "-i", src, "-vf", "hwdownload,format=nv12",
             "-f", "null", "-"],
            capture_output=True, timeout=30)
        if dec.returncode != 0:
            _err = str(getattr(dec, "stderr", "") or "")[-400:]
            print(f"⚠️ [GPU_RENDER] real CUDA decode test failed: {_err}")
            return False
        flt = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=red:s=64x64:d=0.2",
             # hwdownload is REQUIRED after scale_cuda. Without it the frames
             # are still in CUDA memory, `format` cannot pull them back, and
             # ffmpeg injects an auto_scaler that can't accept GPU frames:
             #   "Impossible to convert between the formats supported by the
             #    filter 'Parsed_scale_cuda_1' and the filter 'auto_scaler_0'"
             # That aborted the probe on hosts whose GPU decode works fine,
             # so REQUIRE_GPU_DECODE=1 killed the job (and first burned a
             # pointless ffmpeg re-download via _ensure_gpu_decode_ffmpeg).
             # Newer ffmpeg reports the same fault as "Could not open encoder
             # before EOF" / "Nothing was written into output file" instead.
             "-vf", "hwupload_cuda,scale_cuda=32:32,hwdownload,format=nv12",
             "-f", "null", "-"],
            capture_output=True, timeout=30)
        if flt.returncode != 0:
            _err = str(getattr(flt, "stderr", "") or "")[-400:]
            print(f"⚠️ [GPU_RENDER] real scale_cuda test failed: {_err}")
            return False
    except Exception:
        return False
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
    print(f"🎞️ [GPU_RENDER] CUDA decode + filters verified on this device: "
          f"{present}")
    return True


def gpu_render_available():
    """Probe CUDA decode/filter support once and cache it (thread-safe)."""
    global _gpu_render_ok
    if _gpu_render_ok is None:
        mode = os.environ.get("GPU_RENDER", "auto").strip().lower()
        if mode == "0":
            _gpu_render_ok = False
        else:
            with _gpu_render_lock:
                if _gpu_render_ok is None:
                    _gpu_render_ok = _probe_gpu_render()
                    if not _gpu_render_ok and mode == "1":
                        print("⚠️ [GPU_RENDER] GPU_RENDER=1 but this ffmpeg "
                              "build lacks CUDA hwaccel/filters — falling "
                              "back to CPU decode")
    return _gpu_render_ok


def gpu_decode_args(device=None, output_format=False):
    """Input-side ffmpeg args to decode on the GPU; [] = CPU decode.

    ``output_format=True`` keeps the decoded frames on the GPU
    (``-hwaccel_output_format cuda``) for the CUDA filtergraph
    (unified_filtergraph_gpu); the default keeps frames in system memory so
    the CPU filtergraph works unchanged. ``device`` pins a multi-GPU worker
    to a specific card (see gpu_affinity.py; never CUDA_VISIBLE_DEVICES — it
    is read once at CUDA init).
    """
    if not gpu_render_available():
        return []
    args = ["-hwaccel", "cuda", "-extra_hw_frames", "16"]
    if output_format:
        args += ["-hwaccel_output_format", "cuda"]
    if device is not None:
        # Accept either an int (0/1) or gpu_affinity's "cuda:N" string.
        if isinstance(device, str) and device.lower().startswith("cuda:"):
            args += ["-hwaccel_device", device.split(":", 1)[1]]
        else:
            args += ["-hwaccel_device", str(device)]
    return args


def reset_gpu_render_cache():
    """Test hook: forget the cached GPU-render probe result."""
    global _gpu_render_ok
    with _gpu_render_lock:
        _gpu_render_ok = None


def _probe_nvenc():
    """One tiny lavfi encode to prove h264_nvenc works end-to-end.

    NVENC rejects frames smaller than ~145px, so the probe uses 256x256.
    Any failure (no ffmpeg binary, no GPU, no driver libs) means False.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
        "-c:v", "h264_nvenc", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
        )
        return result.returncode == 0
    except Exception:
        return False


def nvenc_available():
    """Probe h264_nvenc once and cache the verdict (thread-safe)."""
    global _nvenc_ok
    if _nvenc_ok is None:
        with _probe_lock:
            if _nvenc_ok is None:
                _nvenc_ok = _probe_nvenc()
    return _nvenc_ok


def reset_encoder_cache():
    """Test hook: forget the cached probe result."""
    global _nvenc_ok, _announced
    with _probe_lock:
        _nvenc_ok = None
        _announced = False


def video_encode_args(tier=QUALITY, device=None):
    """Return the codec/quality args for one encode, honoring FFMPEG_ENCODER."""
    global _announced
    if tier not in _X264_ARGS:
        raise ValueError(f"Unknown encode tier: {tier!r}")

    mode = os.environ.get("FFMPEG_ENCODER", "auto").strip().lower()
    use_nvenc = False
    if mode in ("nvenc", "auto"):
        use_nvenc = nvenc_available()
        if mode == "nvenc" and not use_nvenc:
            print("⚠️ [Encoder] FFMPEG_ENCODER=nvenc but h264_nvenc is not "
                  "usable here — falling back to libx264")

    if not _announced:
        _announced = True
        print(f"🎞️ [Encoder] video encoder: {'h264_nvenc' if use_nvenc else 'libx264'} "
              f"(FFMPEG_ENCODER={mode})")

    args = list((_NVENC_ARGS if use_nvenc else _X264_ARGS)[tier])
    if use_nvenc and device is not None:
        gpu_idx = str(device).split(":", 1)[1] if (isinstance(device, str) and ":" in str(device)) else str(device)
        args = ["-gpu", gpu_idx] + args
    return args
