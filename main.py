import time
import cv2
import scenedetect
import subprocess
import shutil
import tempfile
import argparse
import re
import sys
import glob
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import os
import yt_dlp
# import whisper (replaced by faster_whisper inside function)
from google import genai
from google.genai import types as genai_types

import context_layer
from download_gate import ClientCannotServeFloor, can_serve_hd_floor
import gemini_worker
import gemini_pool
import picker
import source_store
from clip_selection import snap_clip_to_words
from ffmpeg_utils import (video_encode_args, audio_encode_args, QUALITY,
                          QUALITY_FAST, METADATA_SCRUB, gpu_decode_args,
                          gpu_render_available, nvenc_available,
                          reset_gpu_render_cache, prepend_gpu_env)
from pipeline_progress import write_progress as _write_progress
from pipeline_progress import mark_clip_ready as _mark_clip_ready
from pipeline_progress import record_stage_durations
from hardware_defaults import default_clip_workers
import gpu_affinity
from dotenv import load_dotenv
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Encoding-safe console output: the app worker pipes stdout/stderr, and a
# C/ASCII locale can crash a print that contains an emoji BEFORE the pipeline
# even starts — which looks like a silent exit-1 with no log lines. Force
# errors="replace" so log lines are never lost, and install an excepthook so
# any uncaught exception always prints a traceback instead of exiting bare.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


def _excepthook(exc_type, exc_value, exc_tb):
    try:
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_tb)
    except Exception:
        pass


sys.excepthook = _excepthook

# Load environment variables
load_dotenv()

# Kaggle's apt ffmpeg is a CPU-only build; the bootstrap installs a
# decode-capable one to FFMPEG_DIR but only exports PATH inside its own shell.
# Put the nvenc ffmpeg + the NVIDIA runtime libs on PATH / LD_LIBRARY_PATH for
# THIS process (see ffmpeg_utils.prepend_gpu_env) so every ffmpeg/onnxruntime
# call here finds them without relying on the bootstrap's shell environment.
prepend_gpu_env()

# --- Constants ---
ASPECT_RATIO = 9 / 16





# Some sources burn a channel-handle watermark strip into the raw footage
# itself (e.g. a row of social handles across the top, a text sticker along
# the bottom) — genuinely part of the source pixels, not something the
# reframe/caption/vision stages can tell apart from real content. Cropping
# it off HERE, before scene detection/face detection/rendering
# ever run, means every downstream stage only ever sees clean frames — not
# just the final render. Per-source (different creators brand differently),
# so off by default; set for a specific source via env.
SOURCE_LOGO_CROP_TOP_PX = max(int(os.environ.get("SOURCE_LOGO_CROP_TOP_PX", "0")), 0)
SOURCE_LOGO_CROP_BOTTOM_PX = max(int(os.environ.get("SOURCE_LOGO_CROP_BOTTOM_PX", "0")), 0)


def _link_or_copy(src, dst):
    """Hardlink src -> dst with a cross-device copy fallback — multi-GB
    source files must never be copied per job."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


_FFMPEG_DOWNLOAD_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-linux64-gpl.tar.xz")


def _ensure_gpu_decode_ffmpeg():
    """Install the decode-capable ffmpeg when GPU decode is required but the
    binary is missing (GPU-or-nothing contract). Self-healing so a missed or
    failed bootstrap download can't block jobs forever. Returns True when GPU
    decode is available after the attempt."""
    if gpu_render_available():
        return True
    target_dir = os.environ.get("FFMPEG_DIR") or "/kaggle/working/ffmpeg-nvenc"
    bin_path = os.path.join(target_dir, "ffmpeg")
    if os.path.exists(bin_path):
        return False  # present but not decode-capable — can't fix in-process
    try:
        print(f"⬇️  Installing decode-capable ffmpeg (BtbN master build) to "
              f"{target_dir} — GPU-or-nothing requires it...", flush=True)
        os.makedirs(target_dir, exist_ok=True)
        archive = os.path.join(tempfile.gettempdir(), "ffmpeg-nvenc.tar.xz")
        if os.path.exists(archive):
            os.remove(archive)
        curl = subprocess.run(
            ["curl", "-fSL", "--max-time", "300", _FFMPEG_DOWNLOAD_URL,
             "-o", archive],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if curl.returncode != 0 or not os.path.exists(archive):
            raise RuntimeError(f"curl failed ({curl.returncode})")
        import tarfile
        workdir = tempfile.mkdtemp(prefix="ffmpeg_install_")
        with tarfile.open(archive, "r:xz") as tf:
            tf.extractall(workdir)
        matches = glob.glob(os.path.join(workdir, "ffmpeg-*-linux64-gpl", "bin", "ffmpeg"))
        if not matches:
            raise RuntimeError("build archive did not contain ffmpeg")
        src_bin = matches[0]
        src_probe = os.path.join(os.path.dirname(src_bin), "ffprobe")
        _link_or_copy(src_bin, bin_path)
        if os.path.exists(src_probe):
            _link_or_copy(src_probe, os.path.join(target_dir, "ffprobe"))
        os.environ["PATH"] = target_dir + os.pathsep + os.environ.get("PATH", "")
        reset_gpu_render_cache()
        if gpu_render_available():
            print(f"✅ Decode-capable ffmpeg installed and verified — GPU "
                  f"decode is now active.", flush=True)
            return True
        print("⚠️ Installed build did not pass the GPU-decode probe.", flush=True)
        return False
    except Exception as e:
        print(f"⚠️ Could not install decode-capable ffmpeg "
              f"({type(e).__name__}: {e}).", flush=True)
        return False


def _print_pipeline_diagnostics():
    """One-shot startup diagnostics: which ffmpeg binary is in use, GPU
    decode/encode verdicts, torch CUDA. Every run's log then answers "is the
    GPU actually processing this" instead of silently falling back to CPU.
    The probes cache their verdict for the rest of the process, so calling
    them here costs nothing extra (the first render would pay it anyway)."""
    import shutil
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpus = torch.cuda.device_count()
    except Exception as e:
        cuda, gpus = False, 0
        print(f"   ⚠️ torch unavailable ({type(e).__name__}) — GPU check skipped")
    print("🖥️  Pipeline diagnostics:", flush=True)
    print(f"   ffmpeg: {shutil.which('ffmpeg') or 'ffmpeg (PATH)'}", flush=True)
    try:
        decode_ok = gpu_render_available()
        encode_ok = nvenc_available()
    except Exception as e:
        decode_ok = encode_ok = False
        print(f"   ⚠️ ffmpeg probe failed ({type(e).__name__}) — treated as no GPU", flush=True)
    # GPU-or-nothing: if decode is required but missing, try to self-heal by
    # installing the decode-capable build BEFORE judging the run.
    require = (os.environ.get("REQUIRE_GPU_DECODE") or "1").strip().lower()
    if cuda and not decode_ok and require == "1":
        decode_ok = _ensure_gpu_decode_ffmpeg() or decode_ok
        if decode_ok:
            try:
                encode_ok = nvenc_available()
            except Exception:
                pass
    print(f"   GPU decode (NVDEC + CUDA filters): "
          f"{'YES' if decode_ok else 'NO'}", flush=True)
    print(f"   GPU encode (NVENC h264): "
          f"{'YES' if encode_ok else 'NO'}", flush=True)
    print(f"   torch CUDA: {cuda} ({gpus} device(s))", flush=True)
    try:
        import onnxruntime
        _providers = onnxruntime.get_available_providers()
        print(f"   onnxruntime: {onnxruntime.__version__} "
              f"providers={_providers}", flush=True)
        if cuda and "CUDAExecutionProvider" not in _providers:
            print("⚠️ onnxruntime is CPU-only — insightface (face spine) will "
                  "run on CPU. Install onnxruntime-gpu via kaggle_bootstrap.sh "
                  "so the reframe stage uses the GPU.", flush=True)
    except Exception:
        pass
    if cuda and not decode_ok:
        if require == "1":
            # GPU or nothing (owner's explicit contract): a decode-capable
            # ffmpeg is required, full stop. Say exactly what is missing so
            # the fix is one step, not a hunt.
            _nvenc_bin = os.path.join(
                os.environ.get("FFMPEG_DIR") or "/kaggle/working/ffmpeg-nvenc",
                "ffmpeg")
            _state = ("present" if os.path.exists(_nvenc_bin)
                      else "MISSING — kaggle_bootstrap.sh's download failed")
            print("❌ GPU detected but this ffmpeg cannot decode on the GPU "
                  f"(no NVDEC/CUDA hwaccel) and REQUIRE_GPU_DECODE=1 — CPU "
                  "decode is refused. Decode-capable build at "
                  f"{_nvenc_bin}: {_state}. Fix: run kaggle_bootstrap.sh and "
                  "confirm it prints 'CUDA decode (NVDEC + CUDA filters) also "
                  "present'. (REQUIRE_GPU_DECODE=warn runs on CPU with a loud "
                  "warning; 0 disables the check — both only for emergencies.)",
                  flush=True)
            sys.exit(1)
        if require == "warn":
            print("⚠️ GPU present but GPU decode is unavailable in the ffmpeg on "
                  "PATH — decoding on CPU for this run (REQUIRE_GPU_DECODE=warn). "
                  "Install the nvenc build via kaggle_bootstrap.sh for the "
                  "strict GPU-only mode.", flush=True)


def _report_gpu_utilization(output_dir=None):
    """One nvidia-smi snapshot at job end — every run proves both GPUs were
    used instead of trusting it. Prints a summary and writes
    gpu_utilization.json beside the job."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        rows = []
        for line in (out.stdout or "").strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                rows.append({
                    "index": parts[0], "name": parts[1],
                    "utilization_gpu": parts[2], "memory_used_mb": parts[3],
                    "memory_total_mb": parts[4],
                })
        if rows:
            summary = ", ".join(
                f"GPU {r['index']}: {r['utilization_gpu']}%" for r in rows)
            print(f"🎛️  GPU utilization at job end: {summary}", flush=True)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "gpu_utilization.json"),
                          "w") as f:
                    json.dump(rows, f, indent=2)
    except Exception as e:
        print(f"⚠️ GPU utilization snapshot failed "
              f"({type(e).__name__}: {e})", flush=True)


def source_logo_crop_vf_args():
    """ffmpeg -vf args to strip the configured top/bottom watermark bands,
    or [] when neither is configured (the common case)."""
    if not (SOURCE_LOGO_CROP_TOP_PX or SOURCE_LOGO_CROP_BOTTOM_PX):
        return []
    total = SOURCE_LOGO_CROP_TOP_PX + SOURCE_LOGO_CROP_BOTTOM_PX
    return ["-vf", f"crop=iw:ih-{total}:0:{SOURCE_LOGO_CROP_TOP_PX}"]


def detect_scenes(video_path):
    import scene_detection
    return scene_detection.detect_scenes(video_path)

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def _probe_video_specs(path):
    """True resolution/bitrate of a downloaded file (PART 6 download gate).

    Nothing downstream used to check what yt-dlp actually landed: an HD
    format string can still produce a low-bitrate stream, and the fallback
    ladder silently ships ~360p progressive when HD attempts fail. This is
    the measurement that turns "the quality is low" into "the source is
    XxY at Z Mbps" — the ceiling the reframe inherits.
    """
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    size_mb = round(os.path.getsize(path) / (1024 ** 2), 2)
    duration = (frames / fps) if fps and frames > 0 else 0.0
    bitrate_mbps = round(size_mb * 8 / duration, 2) if duration else 0.0
    codec = ""
    pix_fmt = ""
    try:
        _probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=20)
        if _probe.returncode == 0:
            parts = _probe.stdout.strip().split(",")
            codec = parts[0] if parts else ""
            pix_fmt = parts[1] if len(parts) > 1 else ""
    except Exception:
        pass
    return {
        "width": w, "height": h, "fps": round(fps, 2),
        "duration_s": round(duration, 2), "size_mb": size_mb,
        "bitrate_mbps": bitrate_mbps, "codec": codec, "pix_fmt": pix_fmt,
    }


def _download_quality_floor(specs):
    """Is this source below the HD floor? Returns the reason string or None."""
    min_h = int(os.environ.get("MIN_SOURCE_HEIGHT", "1080"))
    min_b = float(os.environ.get("MIN_SOURCE_BITRATE_Mbps", "1.2"))
    if specs["height"] < min_h:
        return (f"height {specs['height']}p < {min_h}p")
    if specs["bitrate_mbps"] < min_b:
        return (f"bitrate {specs['bitrate_mbps']} Mbps < {min_b} Mbps")
    # GPU-or-nothing: even a 1080p source is rejected if its codec cannot be
    # decoded on the GPU (AV1, or 10-bit VP9 on Turing) — otherwise ffmpeg
    # would silently software-decode it during render.
    codec = str(specs.get("codec") or "").lower()
    pix_fmt = str(specs.get("pix_fmt") or "").lower()
    if codec in ("av1",) or codec.startswith("av01"):
        return ("codec AV1 — Turing's NVDEC has no AV1 decoder "
                "(GPU-or-nothing)")
    if codec.startswith(("vp9", "vp09")) and pix_fmt.startswith(
            ("p010", "p016", "yuv420p10")):
        return ("10-bit VP9 — Turing's NVDEC has no VP9 10-bit decoder "
                "(GPU-or-nothing)")
    return None


def _enforce_hd_gate(specs, require_hd=False):
    """PART 6 (6-aug-2026): refuse to proceed on a sub-HD source.

    Returns the floor reason (or None) and raises when ``require_hd`` is set
    and the source is below the floor — the downloader exists to LAND high
    quality, not to report what it landed. ALLOW_LOW_QUALITY_SOURCE=1 keeps
    an explicit escape hatch for uploaders whose source is genuinely soft.
    """
    reason = _download_quality_floor(specs)
    if reason and require_hd and os.environ.get(
            "ALLOW_LOW_QUALITY_SOURCE", "").strip().lower() not in (
            "1", "true", "yes"):
        raise RuntimeError(
            f"HD download failed: the best available source is {reason}. "
            "The reframe inherits the source ceiling, so clips would be soft. "
            "Set ALLOW_LOW_QUALITY_SOURCE=1 to proceed anyway.")
    return reason


# Byte budget for the sanitized video title used as the stem of every derived
# file. Filesystems cap a name in BYTES (255 on ext4), not characters, and the
# pipeline decorates this stem: "_clip_10.mp4" (12), "subtitled_<ts>_" (21),
# "temp_hook_<hex8>_" (19), "autosubs_<ts>_" + ".ass" (24). Budgeting 120 bytes
# leaves room for all of them stacked and still lands well under the limit.
#
# The old cap was 100 CHARACTERS, which is 300 bytes of Bengali or Arabic — over
# the limit before any decoration. It surfaced as OSError 36 killing the hook
# endpoint in prod on 26-jul-2026.
MAX_TITLE_BYTES = 120


def truncate_bytes(text, max_bytes):
    """Trim ``text`` to a byte budget without splitting a multi-byte character."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", "ignore")


def sanitize_filename(filename):
    """Remove invalid characters from filename and bound it for the filesystem."""
    filename = re.sub(r'[<>:"/\\|?*#]', '', filename)
    filename = filename.replace(' ', '_')
    return truncate_bytes(filename, MAX_TITLE_BYTES)


def download_youtube_video(url, output_dir=".", require_hd=False):
    """
    Downloads a YouTube video using yt-dlp.
    Returns the path to the downloaded video and the video title.
    """
    # SSRF guard: block non-http(s) schemes and private/loopback/metadata hosts
    # before handing the URL to yt-dlp.
    from security_utils import assert_public_url
    assert_public_url(url)

    print(f"🔍 Debug: yt-dlp version: {yt_dlp.version.__version__}")
    print("📥 Downloading video from YouTube...")
    step_start_time = time.time()

    # Derived from this file's own location, not hardcoded to /app: /app is
    # only where docker-compose.yml bind-mounts the repo. On Kaggle the repo
    # lives at /kaggle/working/openshorts (see openshorts_kaggle.ipynb), so a
    # literal '/app/cookies.txt' can never exist there and every Kaggle run
    # silently lost the on-disk jar, falling through to the (often stale)
    # YOUTUBE_COOKIES env var or no cookies at all.
    cookies_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cookies.txt')
    cookies_env = os.environ.get("YOUTUBE_COOKIES")
    # auto_refresh_cookies.sh keeps cookies.txt fresh (a real Netscape
    # jar, rewritten every ~20min from a logged-in Chrome session). The
    # YOUTUBE_COOKIES env value is the LEGACY path and can be stale — a
    # 27-char blob was clobbering the fresh 4.9KB jar on every job, which is
    # exactly the "Sign in to confirm you're not a bot" wall (round-5 live
    # failure, 3-aug-2026). The env is now only a fallback for the first run
    # before the refresh loop has ever written a file: never clobber a newer
    # jar with an older value.
    if os.path.exists(cookies_path) and os.path.getsize(cookies_path) > 0:
        print(f"🍪 Using on-disk cookies jar ({os.path.getsize(cookies_path)} bytes)")
    elif cookies_env:
        print("🍪 No cookies file yet — writing YOUTUBE_COOKIES env value...")
        try:
            with open(cookies_path, 'w') as f:
                f.write(cookies_env)
            if os.path.exists(cookies_path):
                 # Never print file CONTENT here: with a headerless cookies
                 # blob this would leak live YouTube session cookies to logs.
                 print(f"   Debug: Cookies file created. Size: {os.path.getsize(cookies_path)} bytes")
        except Exception as e:
            print(f"⚠️ Failed to write cookies file: {e}")
            cookies_path = None
    else:
        cookies_path = None
        print("⚠️ No cookies file and no YOUTUBE_COOKIES env var — downloads may hit YouTube's bot wall.")
    
    # Optional HTTP proxy. Set PROXY_URL to route downloads through it; unset
    # (self-host) goes direct as before.
    _proxy = os.environ.get("PROXY_URL", "").strip() or None
    if _proxy:
        print("🌐 Using proxy for download.")

    # Two download strategies, tried in order so a break in the HD path degrades
    # gracefully instead of failing the whole job: an HD attempt first, then a
    # conservative fallback (also the only strategy for self-host).
    # PO token provider — YouTube increasingly requires one to trust a client
    # as non-bot, especially from datacenter/cloud IPs. HTTP server mode
    # (BGUTIL_BASE_URL, a persistent sidecar — see docker-compose.yml) is
    # strongly preferred: the provider's own docs say the alternative
    # script-per-request mode (BGUTIL_SCRIPT_PATH) "is NOT recommended" for
    # anything beyond occasional single-shot use. IMPORTANT: this must be
    # merged into EVERY attempt's extractor_args below, not just one — a
    # prior version of this code only wired it into the HD attempt, so every
    # other attempt silently ran with no PO token at all (visible in yt-dlp's
    # debug log as "[pot:...] Script path doesn't exist").
    _bgutil_http = os.environ.get("BGUTIL_BASE_URL", "").strip()
    # Fallback: read from .env (bootstrap writes it but env vars don't cross
    # process boundaries — e.g. Kaggle's bash cell → Python kernel).
    if not _bgutil_http:
        _env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.isfile(_env_file):
            with open(_env_file) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line.startswith("BGUTIL_BASE_URL="):
                        _v = _line.split("=", 1)[1].strip().strip('"').strip("'")
                        if _v:
                            _bgutil_http = _v
                            os.environ["BGUTIL_BASE_URL"] = _bgutil_http
                            break
    # Auto-detect: probe loopback addresses on the default port.
    if not _bgutil_http:
        try:
            import urllib.request as _ur
            for _addr in ("127.0.0.1", "[::1]"):
                try:
                    _req = _ur.Request(f"http://{_addr}:4416/ping", method="GET")
                    with _ur.urlopen(_req, timeout=1) as resp:
                        if resp.status == 200:
                            _bgutil_http = f"http://{_addr}:4416"
                            os.environ["BGUTIL_BASE_URL"] = _bgutil_http
                            break
                except Exception:
                    continue
        except Exception:
            pass
    _bgutil_script = os.environ.get("BGUTIL_SCRIPT_PATH", "").strip()
    if _bgutil_http:
        _pot_args = {'youtubepot-bgutilhttp': {'base_url': [_bgutil_http]}}
    elif _bgutil_script:
        _pot_args = {'youtubepot-bgutilscript': {'script_path': [_bgutil_script]}}
    else:
        _pot_args = {}
    hd_args = dict(_pot_args) if _pot_args else None
    fallback_args = {
        'youtube': {
            'player_client': ['tv_embed', 'android', 'mweb', 'web'],
            'player_skip': ['webpage', 'configs'],
        },
        **_pot_args,
    }
    # Client spoofing, no cookies: impersonate YouTube's iOS app instead of the
    # plain web client. iOS goes first — YouTube's SABR-only streaming rollout
    # has been degrading android's format availability. This alone often
    # dodges the "Sign in to confirm you're not a bot" wall that flags
    # datacenter IPs on the web client, without needing a cookies file at all
    # (and a STALE cookies file can make things worse than none, since an
    # invalid session reads as more suspicious than an anonymous request).
    ios_spoof_args = {
        'youtube': {
            'player_client': ['ios', 'android', 'web'],
        },
        **_pot_args,
    }

    # Cap at 720p ONLY when the bytes actually go through the paid proxy — that
    # cap exists to control bandwidth cost, and the direct attempt has none.
    #
    # This is per-attempt on purpose. Deciding it once from `_proxy` capped the
    # DIRECT attempt too, so with DIRECT_FIRST=1 (which serves most downloads)
    # every YouTube source arrived at 720p and, since the reframe inherits the
    # source height, 80% of delivered clips came out 406x720 (audited 25-jul-2026).
    def _hd_fmt_for(proxy):
        # 1080p is the HARD FLOOR (owner, 10-aug-2026): nothing below 1080 is
        # ever accepted. The chain has NO sub-1080 option, so yt-dlp cannot
        # land 720p/480p/360p on the HD attempt; the HD gate then rejects any
        # lower source that slips through the ladder's fallback attempts.
        # Only GPU-decodable codecs are selectable: H.264 first (up to 1080p,
        # always 8-bit), then 8-bit VP9 (vp09.00.10), then H.265. 10-bit VP9
        # (vp09.00.40) and AV1 are excluded — Turing's NVDEC can't decode
        # either, so they would silently CPU-decode.
        # SOURCE_MAX_HEIGHT=0 restores no-cap quality-first.
        _max_h = (os.environ.get("SOURCE_MAX_HEIGHT") or "1440").strip() or "1440"
        if proxy:
            # Same 1080 floor on the proxy path — the old 720p bandwidth cap
            # is gone; the quality contract outranks bandwidth cost.
            _max_h = "1440"
        if _max_h in ("0", "unlimited", "none"):
            return ('bestvideo[vcodec^=avc1][height>=1080]+bestaudio/'
                    'bestvideo[vcodec^=vp09.00.10][height>=1080]+bestaudio/'
                    'bestvideo[vcodec^=h265][height>=1080]+bestaudio/'
                    'bestvideo[vcodec^=hevc][height>=1080]+bestaudio/'
                    'bestvideo[height>=1080][vcodec!=av01][vcodec!=vp09.00.40]+bestaudio/'
                    'best[vcodec!=av01][ext=mp4]/best[vcodec!=av01]')
        return (f'bestvideo[height>={1080}][height<={_max_h}][vcodec^=avc1]+bestaudio/'
                f'bestvideo[height>={1080}][height<={_max_h}][vcodec^=vp09.00.10]+bestaudio/'
                f'bestvideo[height>={1080}][height<={_max_h}][vcodec^=h265]+bestaudio/'
                f'bestvideo[height>={1080}][height<={_max_h}][vcodec^=hevc]+bestaudio/'
                f'bestvideo[height>={1080}][height<={_max_h}][vcodec!=av01][vcodec!=vp09.00.40]+bestaudio/'
                'bestvideo[height>=1080][vcodec^=avc1]+bestaudio/'
                'bestvideo[height>=1080][vcodec!=av01][vcodec!=vp09.00.40]+bestaudio/'
                'best[vcodec!=av01][ext=mp4]/best[vcodec!=av01]')
    fallback_fmt = 'bestvideo+bestaudio/best'

    def _base_opts(extractor_args, proxy, use_cookies=True):
        return {
            'quiet': False, 'verbose': True, 'no_warnings': False,
            'cookiefile': cookies_path if (use_cookies and cookies_path) else None,
            'proxy': proxy, 'socket_timeout': 30, 'retries': 10, 'fragment_retries': 10,
            'nocheckcertificate': True, 'cachedir': False,
            'extractor_args': extractor_args,
            'http_headers': {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ),
            },
        }

    # Wire bytes actually pulled through the (paid) proxy, summed across
    # fragments/streams. Reported to app.py via the PROXY_BYTES= line below.
    _dl_bytes = {"total": 0}

    def _progress_hook(d):
        if d.get('status') == 'finished':
            _dl_bytes["total"] += int(d.get('total_bytes')
                                      or d.get('total_bytes_estimate')
                                      or d.get('downloaded_bytes') or 0)

    def _attempt(extractor_args, fmt, proxy, use_cookies=True,
                 allow_subfloor=False):
        _dl_bytes["total"] = 0
        with yt_dlp.YoutubeDL(_base_opts(extractor_args, proxy, use_cookies)) as ydl:
            info = ydl.extract_info(url, download=False)
        # Pre-download capability gate: this probe already fetched the
        # client's FULL format list — if that list can't serve the HD floor
        # (e.g. a spoofed client that YouTube only offers ~360p), skip the
        # download instead of pulling a low-res file the HD gate would throw
        # away anyway. allow_subfloor keeps the FINAL ladder strategy exempt
        # so the low-quality restore path (ALLOW_LOW_QUALITY_SOURCE=1) still
        # has a file to work with when nothing HD exists.
        if not allow_subfloor and not can_serve_hd_floor(info):
            fmts = info.get("formats") or []
            best = max((f.get("height") or 0) for f in fmts)
            raise ClientCannotServeFloor(best)
        sanitized = sanitize_filename(info.get('title', 'youtube_video'))
        expected = os.path.join(output_dir, f'{sanitized}.mp4')
        if os.path.exists(expected):
            os.remove(expected)
        dl_opts = {
            **_base_opts(extractor_args, proxy, use_cookies),
            'format': fmt,
            'outtmpl': os.path.join(output_dir, f'{sanitized}.%(ext)s'),
            'merge_output_format': 'mp4', 'overwrites': True,
            'progress_hooks': [_progress_hook],
        }
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([url])
        return sanitized

    # DIRECT_FIRST=1: try the server's own IP before spending proxy bandwidth.
    # Needs cookies + a PO-token provider — without both, YouTube flags the
    # datacenter IP after the first request (verified in prod, 21-jul-2026).
    _direct_first = (os.environ.get("DIRECT_FIRST", "").strip() == "1"
                     and _proxy and hd_args and cookies_path)

    # PLAIN ANONYMOUS FIRST — measured, 5-aug-2026.
    #
    # kaggle_download_probe.py ran every strategy end to end (real downloads,
    # not metadata fetches) on Kaggle: plain anonymous with NO extractor args,
    # no cookies and no proxy was the fastest success, and ios/tv_embed also
    # passed. On a host whose IP is not flagged this is strictly the best path
    # — nothing to expire, no PO-token sidecar, no proxy spend — and it was
    # simply missing from this list.
    #
    # It stays first on flagged hosts too (the studio, most datacenters): there
    # it fails in a couple of seconds on the bot wall and the existing
    # ios-spoof/HD/fallback ladder takes over exactly as before. Cheap to try,
    # and it removes the cookie dependency entirely wherever it works.
    #
    # Deliberately use_cookies=False: sending a STALE jar is worse than sending
    # none, because an invalid session reads as more suspicious than an
    # anonymous request (see the ios_spoof_args note above).
    #
    # PO-token + iOS client FIRST (11-aug-2026): on a YouTube-flagged Kaggle
    # IP the old ordering ran plain anonymous first, which hit the bot wall
    # before ever reaching the PO-token path — "worked effortlessly before,
    # now walls" was the IP flag + bad ordering, not a code regression (the
    # same code downloaded fine minutes apart). The spoofed+token path is the
    # strongest anti-bot config, so it goes first; plain anonymous stays as
    # a later fallback for clean IPs.
    attempts = (
        # HD format on purpose: fallback_fmt is 'best[ext=mp4]/best', which is a
        # PRE-MERGED (progressive) stream, and YouTube caps those at ~360-720p.
        # Real HD needs separate bestvideo+bestaudio merged, which _hd_fmt_for
        # builds. Measured 5-aug-2026: with fallback_fmt this attempt pulled a
        # 10.5-minute source in 28.41 MiB (~360p) and the reframe inherited that
        # resolution, so every delivered clip was low quality.
        [('ios-spoof', ios_spoof_args, _hd_fmt_for(None), None, False)]
        + [('anonymous', {}, _hd_fmt_for(None), None, False)]
        + ([('HD-direct', hd_args, _hd_fmt_for(None), None, True)] if _direct_first else [])
        + ([('HD', hd_args, _hd_fmt_for(_proxy), _proxy, True)] if hd_args else [])
        + [('fallback-hd', fallback_args, _hd_fmt_for(_proxy), _proxy, True)]
    )

    sanitized_title = None
    last_err = None
    used_proxy = False
    best_title = None
    best_specs = None
    best_path = None

    def _quality_key(specs):
        """Rough sharpness ordering for choosing between landed files."""
        return (specs["height"], specs["bitrate_mbps"])

    for idx, (label, ea, fmt, proxy, use_cookies) in enumerate(attempts):
        # PART 6 (6-aug-2026): a low-quality success must not stop the ladder
        # when a better (HD-labeled) strategy is still queued — the goal is
        # that HD LANDS, not that the first success ships. Only attempts that
        # can plausibly produce a better stream count as "remaining HD".
        remaining_hd = any("HD" in a[0] for a in attempts[idx + 1:])
        # A 403 on the media fetch is usually transient: the googlevideo URL is
        # bound to the IP that extracted it, and the residential proxy rotates
        # its exit IP between requests. Retrying re-extracts and usually lands
        # on a consistent IP (3 of 62 downloads hit this on 22-jul-2026).
        for retry in range(2):
            try:
                print(f"📥 Download attempt: {label}" + (f" (retry {retry})" if retry else ""))
                sanitized_title = _attempt(ea, fmt, proxy, use_cookies,
                                           allow_subfloor=(
                                               idx == len(attempts) - 1))
                used_proxy = proxy is not None
                print(f"✅ Download succeeded ({label}).")
                # Verify what actually landed BEFORE deciding to proceed.
                _landed = os.path.join(output_dir, f"{sanitized_title}.mp4")
                _specs = _probe_video_specs(_landed)
                if _specs is not None:
                    print(f"📐 Source specs ({label}): {_specs['width']}x"
                          f"{_specs['height']} @ {_specs['fps']}fps · "
                          f"{_specs['bitrate_mbps']} Mbps · "
                          f"codec {_specs.get('codec') or '?'} "
                          f"({_specs.get('pix_fmt') or '?'}) · "
                          f"{_specs['size_mb']} MiB")
                    _reason = _download_quality_floor(_specs)
                    if _reason:
                        print(f"⚠️ LOW-QUALITY SOURCE ({label}): {_reason}")
                        if remaining_hd:
                            # Park this file aside, keep climbing; restore it
                            # at the end if nothing better lands.
                            try:
                                _best = _landed + ".best"
                                shutil.copy2(_landed, _best)
                                best_title, best_specs = sanitized_title, _specs
                                best_path = _best
                                print("   ↪️ Below the HD floor — continuing "
                                      "the ladder for a higher-quality stream.")
                            except OSError as _e:
                                print(f"   ⚠️ Could not park best download ({_e})")
                            sanitized_title = None  # keep climbing
                            break
                break
            except ClientCannotServeFloor as _skip:
                # The probe decided this client can't serve the floor at
                # all; its format list is stable, so retrying is pointless —
                # move to the next strategy without burning a download.
                last_err = _skip
                print(f"   ↪️ {label} can't serve the HD floor "
                      f"(best {_skip.best_height}p) — skipping to the next "
                      "strategy")
                break
            except Exception as e:
                last_err = e
                print(f"⚠️  Download attempt '{label}' failed: {str(e)[:200]}")
                # 403s and YouTube's bot wall ("Sign in to confirm you're not
                # a bot") are often transient IP-level flags — the earlier
                # ladder pass may have succeeded on the same video minutes
                # ago. Retry them once with a short backoff before moving on.
                _msg = str(e)
                retryable = (any(t in _msg for t in (
                    '403', 'Forbidden', 'Sign in to confirm', 'not a bot',
                    'Sign in', 'confirm you'))
                    or 'bot' in _msg.lower())
                if not retryable or retry == 1:
                    break
                # Bot-wall flags usually need a longer cool-down than a 403.
                time.sleep(25 if ('Sign in' in _msg or 'bot' in _msg.lower())
                           else 3)
        if sanitized_title is not None:
            break

    if sanitized_title is None:
        import sys
        _wall = ('Sign in to confirm' in str(last_err) or 'bot' in str(last_err).lower())
        if _wall:
            error_msg = f"""
❌ ================================================================= ❌
❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED — BOT WALL
❌ ================================================================= ❌
YouTube is asking Kaggle's IP to "sign in to confirm you're not a bot".
This is INTERMITTENT (the same video downloaded fine minutes ago) — simply
re-run the job and it often passes. If it keeps happening, the durable fix
is cookies:
  1. Install the "Get cookies.txt LOCALLY" browser extension (Chrome/Edge/
     Firefox) and export cookies for youtube.com as a Netscape-format file.
  2. Paste the file contents as a Kaggle Secret named YOUTUBE_COOKIES
     (Add-ons → Secrets, tick the checkbox) and re-run Cells 1-3 once.
  3. The pipeline writes it to cookies.txt automatically on the next job.
Manual path: download the video and use the 'Upload Video' tab.
Technical Details: {str(last_err)}
"""
        else:
            error_msg = f"""
❌ ================================================================= ❌
❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED (all strategies)
❌ ================================================================= ❌
REASON: YouTube blocked the request or the download tooling is out of date.
👇 SOLUTION FOR USER: download the video manually and use the 'Upload Video' tab.
Technical Details: {str(last_err)}
"""
        print(error_msg, file=sys.stdout)
        print(error_msg, file=sys.stderr)
        sys.stdout.flush(); sys.stderr.flush()
        time.sleep(0.5)
        raise last_err

    downloaded_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    if not os.path.exists(downloaded_file):
        for f in os.listdir(output_dir):
            if f.startswith(sanitized_title) and f.endswith('.mp4'):
                downloaded_file = os.path.join(output_dir, f)
                break

    # Restore the best-quality file if the ladder's last attempt landed worse
    # than a parked one (e.g. an HD attempt succeeded below-floor, then the
    # final fallback overwrote it with something even worse).
    if best_path and best_specs:
        try:
            _final_specs = _probe_video_specs(downloaded_file)
            if (_final_specs is None
                    or _quality_key(_final_specs) < _quality_key(best_specs)):
                os.replace(best_path, downloaded_file)
                print(f"♻️ Restored the best download ({best_specs['width']}x"
                      f"{best_specs['height']} @ "
                      f"{best_specs['bitrate_mbps']} Mbps) over the ladder's "
                      "final attempt.")
        except OSError as e:
            print(f"⚠️ Could not restore best download ({e})")
    elif best_path:
        try:
            os.remove(best_path)
        except OSError:
            pass

    # PART 6 gate (6-aug-2026): measure the FINAL file and refuse to proceed
    # on a source below the HD floor when the caller requires HD. The
    # downloader exists to LAND high quality — warning alone was not enough —
    # but ALLOW_LOW_QUALITY_SOURCE=1 keeps the escape hatch for uploaders
    # whose source is genuinely soft.
    source_specs = _probe_video_specs(downloaded_file)
    if source_specs:
        print(f"📐 Source specs (final): {source_specs['width']}x{source_specs['height']} "
              f"@ {source_specs['fps']}fps · {source_specs['bitrate_mbps']} Mbps · "
              f"codec {source_specs.get('codec') or '?'} "
              f"({source_specs.get('pix_fmt') or '?'}) "
              f"· {source_specs['size_mb']} MiB")
        reason = _enforce_hd_gate(source_specs, require_hd=require_hd)
        if reason:
            print(f"⚠️ LOW-QUALITY SOURCE: {reason} — clips will look soft. "
                  "This is the uploader's source ceiling (or the ladder fell "
                  "back to a pre-merged progressive stream), not the render "
                  "pipeline.")
        try:
            with open(os.path.join(output_dir, "source_specs.json"), "w") as f:
                json.dump(source_specs, f, indent=2)
        except OSError as e:
            print(f"   ⚠️ Could not write source_specs.json ({e})")
    else:
        print("⚠️ Could not probe the downloaded video specs.")

    if used_proxy and _dl_bytes["total"]:
        # Machine-parseable marker consumed by app.py's log reader for the
        # monthly proxy-bandwidth counter. Not shown to clients (log filter).
        # Only emitted when the winning attempt actually went through the
        # proxy — direct-first successes are free bandwidth.
        print(f"PROXY_BYTES={_dl_bytes['total']}")
    print(f"✅ Video downloaded in {time.time() - step_start_time:.2f}s: {downloaded_file}")
    return downloaded_file, sanitized_title

def finalize_clip_passthrough(input_video, final_output_video):
    """Keep the clip's native framing (for horizontal/16:9 output).

    The input is the freshly encoded cut, so a stream-copy remux is enough to
    add +faststart — re-encoding here would only cost time and quality.
    """
    if os.path.exists(final_output_video):
        os.remove(final_output_video)
    print(f"🎬 Passthrough (native framing): {input_video}")
    cmd = [
        'ffmpeg', '-y', '-i', input_video,
        '-c', 'copy', *METADATA_SCRUB, '-movflags', '+faststart',
        final_output_video,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
    print(f"✅ Clip saved to {final_output_video}")
    return True, []


def prepare_caption_burn(clip_path, transcript, clip_start, clip_end,
                         general_ranges=None):
    """Generate one clip's caption ASS + output naming (PART 6.4).

    Returns ``(ass_path, out_path, ass_filter)`` — everything the render or
    the standalone burn needs — or None when captions are skipped (silent
    video, no words in range, AUTO_CAPTIONS=0, or any failure). Extracted
    from auto_caption_clip so the render pass can fold the burn in (one
    encode from the reframed frames) instead of re-encoding the clean clip.
    """
    if os.environ.get("AUTO_CAPTIONS", "1").strip() == "0":
        return None
    if not transcript or not transcript.get('segments'):
        return None  # silent video: nothing to caption
    try:
        import subtitles as _subs
        style = _subs.AUTO_CAPTION_STYLE
        output_dir = os.path.dirname(clip_path)
        stem = os.path.basename(clip_path)
        generation_id = int(time.time())
        # The output name MUST stay exactly "subtitled_<ts>_<clip filename>":
        # the modal's walk-back and _canonical_clip_file both reconstruct the
        # clean original from it, so trimming the stem here would orphan the
        # pair. Length is bounded upstream instead, by MAX_TITLE_BYTES at
        # download time. A legacy clip whose name predates that budget can still
        # overflow — that raises OSError 36, which the except below turns into
        # "ship the clip uncaptioned" rather than a broken filename.
        # The .ass path is interpolated INTO an ffmpeg filter string
        # (-vf ass='...'), where a literal apostrophe closes the quote and
        # breaks the filter. Titles carry apostrophes constantly in English
        # ("Earth's", "Don't"), so this name must stay free of the clip stem —
        # which is exactly why /api/subtitle has always used a neutral
        # "subs_<i>_<ts>.ass". Deriving it from the stem silently cost captions
        # on every apostrophe title until 29-jul-2026.
        #
        # The OUTPUT name still carries the stem, and must: the modal's
        # walk-back and _canonical_clip_file reconstruct the clean original
        # from it. That one is only ever passed as an argv element, never
        # inside a filter string, so quoting never applies to it.
        # Unique per clip, not just per second: clips render in parallel
        # (CLIP_WORKERS), so a bare timestamp would collide and let one clip
        # burn another's captions.
        ass_path = os.path.join(
            output_dir, f"autosubs_{generation_id}_{uuid.uuid4().hex[:8]}.ass")
        out_path = os.path.join(output_dir, f"subtitled_{generation_id}_{stem}")

        # Stroke width is derived from font size (see subtitles.
        # auto_stroke_width / CAPTION_STROKE_RATIO), not a fixed px value,
        # so it stays proportional if font_size ever changes.
        border_width = _subs.auto_stroke_width(style["font_size"])

        if not _subs.generate_ass(
                transcript, clip_start, clip_end, ass_path,
                max_chars=style["max_chars"], max_duration=style["max_duration"],
                alignment=style["alignment"], fontsize=style["font_size"],
                font_name=style["font_name"], font_color=style["font_color"],
                border_color=style["border_color"], border_width=border_width,
                highlight_color=style["highlight_color"], effect=style["effect"],
                base_opacity=style["base_opacity"], uppercase=style["uppercase"],
                margin_v=style.get("margin_v", _subs.SAFE_MARGIN_V),
                general_ranges=general_ranges,
                speaker_colors=style.get("speaker_colors", False),
                letter_spacing_ratio=_subs.CAPTION_LETTER_SPACING_RATIO):
            print("   ℹ️ No words in range — clip ships without captions.")
            return None

        return ass_path, out_path, _subs.ass_filter_string(ass_path)
    except Exception as e:
        print(f"   ⚠️ Auto-captions failed ({type(e).__name__}: {e}) — "
              f"delivering the clip without them.")
        return None


def auto_caption_clip(clip_path, transcript, clip_start, clip_end,
                      general_ranges=None):
    """Burn the default caption style onto a finished clip.

    Captions are mandatory for short-form to land, but they were opt-in behind a
    modal and only 9% of delivered clips ever got them (prod audit, 25-jul-2026).
    So every clip now ships captioned by default.

    The captioned file is written ALONGSIDE the clip as
    ``subtitled_<ts>_<clip>.mp4`` — the same convention /api/subtitle uses — so
    the untouched original stays on disk and re-styling from the modal replaces
    the captions instead of burning a second layer over them.

    Returns the captioned path, or None when captions were skipped (silent
    video, no words in range, AUTO_CAPTIONS=0, or any failure — a caption
    problem must never cost the user the clip they already paid for).
    """
    prep = prepare_caption_burn(clip_path, transcript, clip_start, clip_end,
                                general_ranges)
    if prep is None:
        return None
    ass_path, out_path, _vf = prep
    try:
        import subtitles as _subs
        style = _subs.AUTO_CAPTION_STYLE
        _subs.burn_subtitles(
            clip_path, ass_path, out_path,
            alignment=style["alignment"], fontsize=style["font_size"],
            font_name=style["font_name"], font_color=style["font_color"],
            border_color=style["border_color"],
            border_width=_subs.auto_stroke_width(style["font_size"]))
        print(f"   💬 Captions burned: {os.path.basename(out_path)}")
        return out_path
    except Exception as e:
        print(f"   ⚠️ Auto-captions failed ({type(e).__name__}: {e}) — "
              f"delivering the clip without them.")
        return None


def render_clip(input_video, final_output_video, output_format="auto",
                transcript=None, clip_start=0.0, clip_end=None,
                focus_directives=None, primary_subject_x=None,
                custom_aspect=None, ass_filter=None, captioned_output=None):
    """Route a cut clip through the right renderer for the chosen output format.
    vertical/auto -> 9:16 reframe, square -> 1:1 reframe, horizontal -> keep.

    transcript/clip_start/clip_end are optional — when the caller has them
    (the source video's full transcript plus this clip's absolute time
    range), AssemblyAI-diarized speaker-turn timing informs the speaker
    fusion the reframe engine consumes (see speaker_fusion.py). Without
    them, reframing relies on LR-ASD speaker evidence alone.

    focus_directives (see analyze_scene_context): clip-relative shot direction
    from the scene-context layer — who the camera should be on and why. When
    absent, framing falls back to pure heuristic tracking."""
    if output_format == "horizontal":
        return finalize_clip_passthrough(input_video, final_output_video)
    if output_format == "custom" and custom_aspect:
        aspect = custom_aspect
    else:
        aspect = 1.0 if output_format == "square" else ASPECT_RATIO
    return process_video_to_vertical(input_video, final_output_video, aspect_ratio=aspect,
                                     transcript=transcript, clip_start=clip_start, clip_end=clip_end,
                                     focus_directives=focus_directives,
                                     primary_subject_x=primary_subject_x,
                                     ass_filter=ass_filter,
                                     captioned_output=captioned_output)


# Watermark geometry, as fractions of the clip width/height.
#
# Vertical placement is the whole point: the top and bottom strips of a 9:16
# clip are either black bars or blurred filler, so a mark up
# there is cropped away without touching a single pixel of real footage. At 40%
# of the height it sits inside the content band — a 16:9 source letterboxed
# into 9:16 spans roughly 34%-66% — so removing the mark means cutting into the
# picture. Left-aligned, like OpusClip's.
WATERMARK_WIDTH_RATIO = 0.30
WATERMARK_MARGIN_RATIO = 0.05
WATERMARK_Y_RATIO = 0.40
WATERMARK_OPACITY = 0.85


def apply_watermark(video_path):
    """Burn the OpenShorts watermark into a finished clip (free plan).

    One re-encode pass on the final file so every output format gets the
    mark, and later subtitle/hook re-encodes keep it — they re-encode the
    already-marked pixels.
    """
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "assets", "watermark.png")
    if not os.path.exists(logo_path):
        print(f"   ⚠️ Watermark asset missing ({logo_path}); clip kept unmarked.")
        return False

    # Scale the lockup from the clip's real width: overlay can't read the other
    # input's size, and computing it here avoids the deprecated scale2ref.
    try:
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_path],
            stderr=subprocess.STDOUT, timeout=60,
        ).decode().strip().split("x")
        vw, vh = int(probe[0]), int(probe[1])
    except Exception as e:
        print(f"   ⚠️ Could not probe clip for watermark ({e}); clip kept unmarked.")
        return False

    wm_w = max(80, int(vw * WATERMARK_WIDTH_RATIO))
    x = int(vw * WATERMARK_MARGIN_RATIO)
    y = int(vh * WATERMARK_Y_RATIO)
    filt = (
        f"[1:v]scale={wm_w}:-1,format=rgba,"
        f"colorchannelmixer=aa={WATERMARK_OPACITY}[wm];"
        f"[0:v][wm]overlay=x={x}:y={y}"
    )
    tmp_path = video_path + ".wm.mp4"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", logo_path,
           "-filter_complex", filt,
           *video_encode_args(QUALITY), "-c:a", "copy", *METADATA_SCRUB,
           "-movflags", "+faststart", tmp_path]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            timeout=1800)
    if result.returncode == 0 and os.path.exists(tmp_path):
        os.replace(tmp_path, video_path)
        return True
    err = (result.stderr or b"").decode(errors="ignore")[-300:]
    print(f"   ⚠️ Watermark pass failed (clip kept unmarked): {err}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return False


def process_video_to_vertical(input_video, final_output_video, aspect_ratio=ASPECT_RATIO,
                              transcript=None, clip_start=0.0, clip_end=None,
                              focus_directives=None, primary_subject_x=None,
                              ass_filter=None, captioned_output=None):
    """
    Core logic to reframe a horizontal video to a target aspect ratio.

    v3 is the only reframe engine: it plans the whole shot list up front
    (face_spine SCRFD/ArcFace tracks + LR-ASD speaker evidence + UNISAL
    saliency), renders one static crop per shot, and fails loud on any
    composition violation — there is no v1/v2 fallback engine anymore.
    aspect_ratio: width/height of the output (9/16 vertical, 1.0 square).
    transcript/clip_start/clip_end/focus_directives: optional, see
    render_clip's docstring.
    """
    import reframe_v3
    t0 = time.time()
    result = reframe_v3.render(input_video, final_output_video, aspect_ratio,
                               transcript=transcript, clip_start=clip_start, clip_end=clip_end,
                               focus_directives=focus_directives,
                               primary_subject_x=primary_subject_x,
                               ass_filter=ass_filter,
                               captioned_output=captioned_output)
    print(f"   \u23f1\ufe0f Reframe v3 total: {time.time() - t0:.1f}s")
    return result


def _generate_source_artifacts(source_video, output_dir, frame_count=12):
    """Extract stills + a low-bitrate preview proxy for the Source section.

    URL jobs delete the full-quality original after processing to save disk —
    but the dashboard's Source panel needs something to show. Generate the
    lightweight artifacts (JPEG stills + a ~480p faststart proxy) BEFORE the
    original is removed; the /api/source endpoints then serve these without
    needing the source. Fail-open: any failure only loses the Source panel,
    never the job.
    """
    try:
        frames_dir = os.path.join(output_dir, "source_frames")
        os.makedirs(frames_dir, exist_ok=True)
        if len(glob.glob(os.path.join(frames_dir, "source_*.jpg"))) < frame_count:
            duration = 0.0
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", source_video],
                    capture_output=True, text=True, timeout=30)
                duration = float(probe.stdout.strip() or 0)
            except Exception:
                duration = 0.0
            if duration > 0:
                # Fast extraction: one keyframe-seek per still (-ss BEFORE -i
                # seeks without decoding the whole source — the fps-filter
                # approach decoded the full 46-min video and timed out).
                for idx in range(frame_count):
                    t = duration * (idx + 0.5) / frame_count
                    subprocess.run(
                        ["ffmpeg", "-y", "-loglevel", "error",
                         "-ss", f"{t:.2f}", "-i", source_video,
                         "-frames:v", "1",
                         "-vf", "scale='min(480,iw)':-2",
                         os.path.join(frames_dir, f"source_{idx + 1:02d}.jpg")],
                        check=True, timeout=60)
        preview = os.path.join(output_dir, "source_preview.mp4")
        if not os.path.exists(preview):
            tmp = os.path.join(output_dir, ".source_preview.tmp.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", source_video,
                 "-t", "90", "-vf", "scale='min(720,iw)':-2",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "30",
                 "-an", "-movflags", "+faststart", tmp],
                check=True, timeout=300)
            os.replace(tmp, preview)
        print("🖼️  Source artifacts ready (stills + preview proxy)")
    except Exception as e:
        print(f"⚠️  Source artifacts failed ({type(e).__name__}: {e}) — "
              f"Source panel will be unavailable for this job")


def transcribe_video(video_path):
    print("🎙️  Transcribing video...")
    from transcribe_backends import transcribe_media

    transcript = transcribe_media(video_path)

    print(f"   Detected language '{transcript['language']}', "
          f"{len(transcript['segments'])} segments")
    for segment in transcript['segments']:
        # Print progress to keep user informed (and prevent timeouts feeling)
        print(f"   [{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

    return transcript

def _upload_and_generate(client, model_name, video_path, prompt, schema):
    """Shared upload -> poll-until-ACTIVE -> generate -> cleanup. Same
    mechanism get_visual_clips() uses on a full source video, reused here on
    a short rough-cut candidate clip instead."""
    file_upload = None
    try:
        file_upload = client.files.upload(file=video_path)
        deadline = time.time() + 180
        while True:
            info = client.files.get(name=file_upload.name)
            state = str(getattr(getattr(info, "state", info), "name", "")).upper()
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Gemini could not process the rough-cut clip.")
            if time.time() > deadline:
                raise TimeoutError("Gemini rough-cut processing timed out.")
            time.sleep(2)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema,
            safety_settings=gemini_worker.RELAXED_SAFETY_SETTINGS)
        response = gemini_pool.generate_with_fallback(
            client, model_name, [file_upload, prompt], config=config,
            max_attempts=1, log=lambda msg: print(msg))
        gemini_worker.raise_if_blocked(response)
        parsed_obj = getattr(response, "parsed", None)
        if parsed_obj is not None:
            return parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
        return gemini_worker._parse_json_response_text(gemini_worker._get_response_text(response))
    finally:
        if file_upload is not None:
            try:
                client.files.delete(name=file_upload.name)
            except Exception:
                pass


def _clip_relative_transcript_excerpt(transcript_result, clip_start, clip_end):
    """Transcript lines re-based to clip-relative seconds, so the scene-context
    director's timestamps come back on the same clock the renderer uses."""
    parts = []
    for seg in (transcript_result or {}).get("segments", []):
        seg_start, seg_end = seg.get("start", 0), seg.get("end", 0)
        if seg_end <= clip_start or seg_start >= clip_end:
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        rel = max(0.0, seg_start - clip_start)
        speaker = seg.get("speaker")
        tag = f"[{rel:.1f}s{'' if not speaker else ' ' + str(speaker)}] "
        parts.append(tag + text)
    return "\n".join(parts)


def analyze_scene_context(pool, model_name, clip_path, clip_duration,
                          transcript_result=None, clip_start=0.0, clip_end=None):
    """Third verification layer: after a clip has been selected and its
    boundaries confirmed, have Gemini watch the FINAL streamlined cut and
    direct the camera — who should be on screen when, and why.

    This is the only stage that understands meaning rather than pixels. The
    reframe engine can detect faces and motion but has no way to know that a
    guy grunting and clowning is the REASON everyone is laughing, so it framed
    the people laughing instead of the person causing it (ground-truthed
    31-jul-2026). See gemini_worker.VISION_SCENE_CONTEXT_PROMPT_TEMPLATE for
    the universal rule this encodes.

    Returns a dict {"directives": [...], "primary_subject": str,
    "primary_subject_x": float|None} — with empty directives when unavailable
    (the renderer treats an empty list as "fall back to pure heuristic
    tracking", so this stays a quality layer, never a hard dependency).
    """
    if not pool or not clip_duration:
        return {"directives": [], "primary_subject": "", "primary_subject_x": None}
    key = pool.acquire()
    if not key:
        return []
    try:
        client = gemini_worker.make_client(key)
        excerpt = _clip_relative_transcript_excerpt(
            transcript_result, clip_start,
            clip_end if clip_end is not None else clip_start + clip_duration)
        # ~1 directive per 2s of footage, matching the 1.5-3s shot cadence
        # measured in RESEARCH_viral_clip_patterns.md.
        target_count = max(3, int(round(clip_duration / 2.0)))
        prompt = gemini_worker.VISION_SCENE_CONTEXT_PROMPT_TEMPLATE.format(
            transcript_excerpt=excerpt or "(no transcript available)",
            clip_duration=clip_duration,
            target_count=target_count)
        parsed = _upload_and_generate(
            client, model_name, clip_path, prompt,
            gemini_worker.SceneContextResponse)
    except gemini_worker.GeminiBlockedError:
        return {"directives": [], "primary_subject": "", "primary_subject_x": None}
    except Exception as e:
        if not gemini_pool.is_transient_error(e):  # see confirm_clip_with_vision's twin
            pool.mark_bad(key)
        print(f"⚠️ Scene-context direction failed: {e}")
        return {"directives": [], "primary_subject": "", "primary_subject_x": None}

    directives = (parsed or {}).get("directives") or []
    cleaned = []
    for d in directives:
        try:
            s = max(0.0, float(d.get("start", 0)))
            e = min(float(clip_duration), float(d.get("end", 0)))
            x = float(d.get("x_position", 0.5))
        except (TypeError, ValueError):
            continue
        if e - s < 0.2:
            continue
        cleaned.append({
            "start": s,
            "end": e,
            "subject": str(d.get("subject", ""))[:120],
            # Clamped, not dropped: an out-of-range position is still a usable
            # "far left"/"far right" signal.
            "x_position": min(max(x, 0.0), 1.0),
            "reason": str(d.get("reason", "speaking")),
            "intensity": min(max(float(d.get("intensity", 0.5) or 0.5), 0.0), 1.0),
        })
    cleaned.sort(key=lambda d: d["start"])
    try:
        prim_x = float((parsed or {}).get("primary_subject_x"))
        prim_x = min(max(prim_x, 0.0), 1.0)
    except (TypeError, ValueError):
        prim_x = None
    if cleaned:
        causes = sum(1 for d in cleaned if d["reason"] == "causing_reaction")
        refs = sum(1 for d in cleaned if d["reason"] == "referenced")
        print(f"   🎬 Scene direction: {len(cleaned)} shot(s) "
              f"({causes} cause-of-reaction, {refs} referenced) — "
              f"{(parsed or {}).get('summary', '')[:80]}"
              + (f" | key subject x={prim_x:.2f}" if prim_x is not None else ""))
    return {
        "directives": cleaned,
        "primary_subject": str((parsed or {}).get("primary_subject", ""))[:120],
        "primary_subject_x": prim_x,
    }


def _snap_keep_spans_to_words(keep_spans, words, clip_start, clip_end, min_span_duration=1.0):
    """Snap each picker-proposed keep_span's boundaries onto real word
    edges — same reasoning as clip_selection.snap_clip_to_words (LLMs are
    bad at millisecond arithmetic, word timestamps are ground truth) —
    then clamp to [clip_start, clip_end], drop spans that collapse below
    min_span_duration once snapped, and merge spans left touching/
    overlapping after snapping. Returns a list of [start, end] pairs,
    sorted, non-overlapping. Falls back to unsnapped (but still clamped)
    boundaries when no nearby words exist.
    """
    if not keep_spans:
        return []

    relevant_words = [w for w in words if w.get('e', 0) > clip_start and w.get('s', 0) < clip_end]

    def _nearest_start(t):
        return min(relevant_words, key=lambda w: abs(w['s'] - t))['s'] if relevant_words else t

    def _nearest_end(t):
        return min(relevant_words, key=lambda w: abs(w['e'] - t))['e'] if relevant_words else t

    snapped = []
    for span in keep_spans:
        s = max(clip_start, min(clip_end, float(span.get('start', clip_start))))
        e = max(clip_start, min(clip_end, float(span.get('end', clip_end))))
        if e <= s:
            continue
        s = max(clip_start, _nearest_start(s))
        e = min(clip_end, _nearest_end(e))
        if e - s >= min_span_duration:
            snapped.append([s, e])

    if not snapped:
        return []

    snapped.sort(key=lambda sp: sp[0])
    merged = [snapped[0]]
    for s, e in snapped[1:]:
        if s <= merged[-1][1] + 0.05:  # touching/overlapping once snapped
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _remap_transcript_onto_jump_cut(transcript_result, keep_spans):
    """Build a transcript whose segment/word timestamps are shifted onto
    the JUMP-CUT video's own (shorter) timeline instead of the source
    video's — required for the speaker-binding stages and captions to stay
    in sync once dead-air/cuttable stretches have been physically removed
    from the footage, not just skipped in metadata. Content outside every
    keep_span doesn't exist in the output video, so it's dropped, not just
    hidden.
    """
    offsets = []
    cursor = 0.0
    for s, e in keep_spans:
        offsets.append(cursor)
        cursor += (e - s)

    def _shift(t, s, offset):
        return t - s + offset

    remapped_segments = []
    for seg in transcript_result.get('segments', []):
        seg_start, seg_end = seg.get('start', 0), seg.get('end', 0)
        for (s, e), offset in zip(keep_spans, offsets):
            if seg_end <= s or seg_start >= e:
                continue
            new_words = []
            for w in seg.get('words', []):
                w_start = w.get('start', w.get('s', 0))
                w_end = w.get('end', w.get('e', 0))
                if w_end <= s or w_start >= e:
                    continue
                new_word = dict(w)
                ws = _shift(max(w_start, s), s, offset)
                we = _shift(min(w_end, e), s, offset)
                for key in ('start', 's'):
                    if key in new_word:
                        new_word[key] = ws
                for key in ('end', 'e'):
                    if key in new_word:
                        new_word[key] = we
                new_words.append(new_word)
            new_seg = dict(seg)
            new_seg['start'] = _shift(max(seg_start, s), s, offset)
            new_seg['end'] = _shift(min(seg_end, e), s, offset)
            new_seg['words'] = new_words
            remapped_segments.append(new_seg)

    remapped_transcript = dict(transcript_result)
    remapped_transcript['segments'] = remapped_segments
    return remapped_transcript, cursor


def _build_jump_cut_source(source_video_path, keep_spans, workdir):
    """Extract and concatenate only the keep_spans from the source video —
    the actual jump cut, removing dead air/cuttable stretches physically
    from the footage rather than just picking [start, end] boundaries.
    Each span is cut with the same encode params so the concat demuxer can
    stream-copy them together with no quality loss on the join. Returns
    the combined video's path.
    """
    segment_paths = []
    for idx, (s, e) in enumerate(keep_spans):
        seg_path = os.path.join(workdir, f"keep_{idx:03d}.mp4")
        cmd = ['ffmpeg', '-y',
               *gpu_decode_args(device=gpu_affinity.current_device()),
               '-ss', f'{s:.3f}', '-to', f'{e:.3f}', '-i', source_video_path,
               *source_logo_crop_vf_args(),
               *video_encode_args(QUALITY_FAST, device=gpu_affinity.current_device()), *audio_encode_args(), seg_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=300)
        except subprocess.CalledProcessError as exc:
            _err = (exc.stderr or b"").decode("utf-8", "replace").strip()
            raise RuntimeError(
                f"jump-cut segment extract failed [{s:.1f}s-{e:.1f}s]"
                + (f": {_err[-400:]}" if _err else "")) from exc
        segment_paths.append(seg_path)

    list_path = os.path.join(workdir, 'jump_cut_concat.txt')
    with open(list_path, 'w') as f:
        for p in segment_paths:
            f.write(f"file '{p}'\n")

    combined_path = os.path.join(workdir, 'jump_cut_combined.mp4')
    subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path,
                     '-c', 'copy', combined_path], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
    return combined_path


def _apply_term_corrections(transcript_result, corrections):
    """Fix ASR mishearings of proper nouns/brand terms (picker-flagged,
    e.g. "Clod" -> "Claude") in place, across both segment text and
    word-level tokens — the latter is what captions actually render, so a
    text-only fix would leave the burned-in captions wrong.

    Whole-word, case-insensitive matching so a substring like "cloud" isn't
    clobbered while fixing "clod". Word-level replacement preserves each
    original word's leading-space/punctuation shell (main.py's word dicts
    carry those inside the 'word' string) by only swapping the inner
    alphabetic run.
    """
    for corr in corrections:
        wrong, correct = corr.get("wrong", "").strip(), corr.get("correct", "").strip()
        if not wrong or not correct:
            continue
        pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
        for segment in transcript_result.get('segments', []):
            if segment.get('text'):
                segment['text'] = pattern.sub(correct, segment['text'])
            for word in segment.get('words', []):
                if word.get('word'):
                    word['word'] = pattern.sub(correct, word['word'])


_QUESTION_START_RE = re.compile(
    r'^(who|what|when|where|why|how|do|does|did|are|is|can|could|'
    r'would|will|have|has|tell|walk|so)\b', re.IGNORECASE)
_LEADING_CONJUNCTION_RE = re.compile(
    r'^(and|but|so|because|which|or|yet|then)\b', re.IGNORECASE)
_DEPENDENT_OPENING_RE = re.compile(
    r'^(he|she|it|they|we|you|this|that|i|i\'m|i\'d|i\'ll|it\'s|that\'s|'
    r'daily|weekly|monthly|yearly|once|twice|thrice|three|two|one|four|five|'
    r'six|seven|eight|nine|ten|yes|no|yeah|yep|nope|nah|probably|maybe|'
    r'about|like|minimum|maximum|every|each|always|usually|sometimes)\b',
    re.IGNORECASE)


def _is_question_like(segment_text):
    """Round-5 spec 1.3: a segment is question-like when it ends in a literal
    '?' or starts with an interrogative — catches unpunctuated spoken
    questions ("Tell me how often") that the old endswith('?') check missed."""
    text = str(segment_text or '').strip()
    if not text:
        return False
    return text.endswith('?') or bool(_QUESTION_START_RE.match(text))


def _is_dependent_opening(opening_text):
    """True when a clip opening is a grammatical continuation that NEEDS the
    preceding question to make sense: leading conjunction, bare pronoun, or a
    dangling bare value (number / frequency / rating / yes-no). Self-contained
    claims ("If you don't understand…") must NOT match — that's the
    over-correction regression guard."""
    text = str(opening_text or '').strip()
    if not text:
        return False
    return bool(_LEADING_CONJUNCTION_RE.match(text)
                or _DEPENDENT_OPENING_RE.match(text))


def _sentence_start_at_or_before(transcript_result, t, max_backtrack=6.0):
    """Start of the sentence containing time ``t``, from word punctuation.

    Falls back to ``t`` when no sentence boundary is close enough (unpunctuated
    transcript, or the sentence runs longer than ``max_backtrack``), so this can
    only ever move a boundary EARLIER, never later, and never strands a caller.
    """
    words = []
    for seg in transcript_result.get('segments', []):
        for w in seg.get('words', []) or []:
            words.append({'w': w.get('word', ''), 's': w.get('start', 0),
                          'e': w.get('end', 0)})
    if not words:
        return t
    try:
        from clip_selection import sentence_boundaries
        starts, _ends = sentence_boundaries(words)
    except Exception:
        return t
    earlier = [x for x in starts if x <= t + 0.05 and t - x <= max_backtrack]
    return max(earlier) if earlier else t


def _extend_start_for_preceding_question(candidate, transcript_result,
                                         max_gap=6.0, max_segments=3,
                                         max_prepend=8.0):
    """Deterministic backstop for reply-only / context-dependent openings
    ("Deal breakers? I don't have any", "three times what?").

    Round-5 spec 1.3 — the old version only checked the immediately preceding
    segment for a literal '?'. Real misses: the question split across segments,
    unpunctuated spoken questions, and dangling-answer openings ("Daily, I'd
    say minimum 3 times a day") with no visible question. This version:

    - only rewinds when the OPENING is a dependent shape (leading conjunction,
      bare pronoun, or dangling bare value) — self-contained claims are never
      touched;
    - searches backward up to ``max_segments`` segments for the NEAREST
      question-like segment (ends in '?' or starts with an interrogative);
    - hard-caps the amount of setup prepended (``max_prepend``).

    Records ``candidate['_context_start']`` = the question's start so the word/
    sentence snapper can lock the boundary (clip_selection.snap_clip_to_words'
    ``context_start``), instead of re-truncating it (spec 1.2).
    """
    segments = sorted(
        (s for s in transcript_result.get('segments', []) if str(s.get('text', '')).strip()),
        key=lambda s: s.get('start', 0))
    if not segments:
        return
    start = float(candidate['start'])

    # Opening text: the segment actually being spoken at the candidate start
    # (its end is after the start). The question that ENDS just before the
    # start must not be mistaken for the opening.
    opening = None
    for seg in segments:
        if seg.get('end', 0) > start:
            opening = str(seg.get('text', '')).strip()
            break
    if not _is_dependent_opening(opening):
        return

    # Walk backward from the segment before the start, up to max_segments, and
    # rewind to the NEAREST question-like segment that is close enough.
    before_start = [s for s in segments if s.get('end', 0) <= start]
    for seg in reversed(before_start[-max_segments:]):
        gap = start - seg.get('end', 0)
        if gap < 0 or gap > max_gap:
            continue
        if _is_question_like(seg.get('text', '')):
            question_start = max(0.0, seg.get('start', start))
            # Rewind to the SENTENCE the question belongs to, not the segment.
            # Segments are utterance chunks and routinely split a sentence in
            # half, so "the segment ending in '?'" is often only the question's
            # TAIL. Ground truth from the shipped clip: the question is one
            # sentence at 833.72 ("When you say I can please you more than the
            # weekend, just so he knows, how often do you like it weekly?") but
            # it is split across segments, so rewinding to the segment start
            # (838.14) opened the clip on "like it weekly?" — which tells a
            # cold viewer nothing about what is weekly. The whole point of the
            # rewind is a question the viewer can actually understand.
            question_start = _sentence_start_at_or_before(
                transcript_result, question_start)
            if start - question_start > max_prepend:
                return  # too much setup to drag in
            candidate['start'] = question_start
            candidate['_context_start'] = question_start
            return


def _build_word_list(transcript_result):
    """Ground-truth word list used for snapping cut points. Built AFTER term
    corrections so captions/snapping see the fixed spelling, not the ASR's
    raw mishearing."""
    words = []
    for segment in transcript_result['segments']:
        for word in segment.get('words', []):
            words.append({'w': word['word'], 's': word['start'], 'e': word['end']})
    return words


_scene_boundary_cache = {}


def scene_boundaries_for(video_path):
    """Cached scene boundaries in seconds for one source video.

    Shared by clip selection (PART 3.2, 6-aug-2026) and the renderer so a
    scene clamp never re-detects. Returns [(start_s, end_s), ...] or [] when
    detection fails — the clamp fails open (a boundary problem must never
    kill a job).
    """
    if not video_path:
        return []
    if os.environ.get("SCENE_DETECTION", "1").strip().lower() in (
            "0", "false", "no", "off"):
        print("ℹ️  Scene detection disabled (SCENE_DETECTION=0) — the end-clamp "
              "polish is skipped, clips still render.")
        return []
    try:
        key = (video_path, os.path.getmtime(video_path),
               os.path.getsize(video_path))
    except OSError:
        return []
    if key not in _scene_boundary_cache:
        try:
            scenes, fps = detect_scenes(video_path)
            fps = float(fps) or 30.0
            _scene_boundary_cache[key] = [
                (float(s.get_frames()) / fps, float(e.get_frames()) / fps)
                for s, e in scenes]
        except Exception as e:
            print(f"⚠️ Scene-boundary clamp unavailable ({e})")
            _scene_boundary_cache[key] = []
    return _scene_boundary_cache[key]


def _clamp_candidate_end_to_scene(candidate, scene_bounds):
    """Pull a clip's END back to the last scene boundary at/before it.

    Vision rescue and sentence snapping can extend a clip into the NEXT
    scene, where a different person is already on screen — the "clip has
    ended but you still see another person" complaint (6-aug-2026). Only
    pulls IN, never extends. If no boundary fits inside the clip (or scene
    data is missing), leaves the boundary untouched rather than truncating
    a sentence.
    """
    if not scene_bounds:
        return candidate
    start, end = candidate["start"], candidate["end"]
    is_long = candidate.get("clip_type") == "long_context"
    min_duration = 45.0 if is_long else 15.0
    fits = [b[1] for b in scene_bounds
            if start + min_duration <= b[1] <= end + 0.05]
    if fits:
        candidate["end"] = max(fits)
    return candidate


def _snap_candidates(shorts, words, video_duration):
    """Snap every clip's start/end onto real word boundaries. Long-context
    segments get a higher floor + ceiling (a full arc needs room) and shorts
    get a 120s ceiling (15-90 is the preferred band, 120 is the hard cap)."""
    for s in shorts:
        is_long = s.get("clip_type") == "long_context"
        ns, ne = snap_clip_to_words(
            s.get("start", 0), s.get("end", 0), words, video_duration,
            min_duration=45.0 if is_long else 15.0,
            max_duration=240.0 if is_long else 120.0,
            context_start=s.get("_context_start"))
        s["start"], s["end"] = ns, ne
        s.pop("_context_start", None)
    return shorts


def _dedup_overlapping_clips(shorts):
    """Drop lower-scoring clips when two picks overlap in time.

    The picker dedups at pick time, and the shared tail then MOVES boundaries
    — question extension pulls starts earlier, word/sentence snapping
    re-lands them — so distinct picks can collide into overlapping
    spans after selection. Every short in the list is rendered, so without
    this the same moment ships multiple times and spends GPU + vision/context
    calls on near-identical clips (the "it keeps picking the same thing"
    complaint). Runs on the FINAL boundaries, after every mutation, so it
    catches what the picker's pick-time dedup cannot.

    Keeps the higher-scoring clip (predicted_score, the key both engines
    write). Ties keep the earlier pick.
    """
    if len(shorts) < 2:
        return shorts
    ranked = sorted(enumerate(shorts), key=lambda t: (t[1].get("start", 0), t[0]))
    kept = []
    for idx, clip in ranked:
        prev = kept[-1][1] if kept else None
        if prev is not None and clip["start"] < prev["end"]:
            cur_score = (clip.get("predicted_score")
                         or clip.get("score") or 0)
            prev_score = (prev.get("predicted_score")
                          or prev.get("score") or 0)
            if cur_score > prev_score:
                print(f"   ⚠️ overlapping picks "
                      f"[{prev['start']:.1f}-{prev['end']:.1f}] / "
                      f"[{clip['start']:.1f}-{clip['end']:.1f}] — keeping the "
                      f"higher-scoring one ({cur_score} > {prev_score})")
                kept[-1] = (idx, clip)
            else:
                print(f"   ⚠️ overlapping picks "
                      f"[{prev['start']:.1f}-{prev['end']:.1f}] / "
                      f"[{clip['start']:.1f}-{clip['end']:.1f}] — keeping "
                      f"[{prev['start']:.1f}-{prev['end']:.1f}] "
                      f"({prev_score} >= {cur_score})")
            continue
        kept.append((idx, clip))
    return [c for _, c in kept]


def get_viral_clips(transcript_result, video_duration, source_video_path=None,
                    clip_count=None, long_context_count=0, style_variant="balanced",
                    output_dir=None, context_blob=None):
    """Stage 3 — viral moment selection via the unified picker.

    The picker (picker.py) IS the planner: one Gemini call reads the whole
    transcript plus the pre-loaded context blob (context_layer.py) and
    returns the EXACT requested number of distinct moments — no separate
    engines, no vision confirmation, no slicing. The shared tail below
    (ASR term corrections, question backstop, sentence-anchored snapping,
    scene clamp, overlap dedup) is unchanged; the count the picker promised
    is surfaced on the result so a physical shortfall is never silent.
    """
    # Optional named-identity enrichment: renames anonymous diarized speaker
    # labels where confident. Fails open — see face_id.py.
    if source_video_path and os.environ.get("FACE_ID_DB"):
        try:
            import face_id
            transcript_result, _ = face_id.enrich_if_configured(
                transcript_result, source_video_path)
        except Exception as e:
            print(f"⚠️ Face ID enrichment skipped ({type(e).__name__}: {e})")

    if clip_count is None:
        raise RuntimeError(
            "No clip count provided — auto mode was removed. The picker "
            "fulfills an explicit count at all costs; pass --clip-count.")

    if output_dir:
        _write_progress(output_dir, "analyze", note="picking clips with Gemini",
                        step="picking the viral moments (Gemini)", step_pct=40)
    picker_result = picker.select_viral_clips(
        transcript_result, video_duration, clip_count=clip_count,
        long_context_count=long_context_count, style_variant=style_variant,
        context_blob=context_blob)
    if not picker_result or not picker_result.get("shorts"):
        raise RuntimeError(
            "Clip detection failed — Gemini did not return usable clips for this video.")

    # Repair ASR mishearings FIRST, then build the word list, so captions and
    # cut snapping both see the corrected spelling (same order as before).
    if picker_result.get("term_corrections"):
        _apply_term_corrections(transcript_result, picker_result["term_corrections"])

    words = _build_word_list(transcript_result)
    shorts = picker_result["shorts"]
    for s in shorts:
        _extend_start_for_preceding_question(s, transcript_result)
    _snap_candidates(shorts, words, video_duration)
    if output_dir:
        _write_progress(output_dir, "analyze",
                        note="detecting scene boundaries",
                        step="detecting scene boundaries (only the picked spans)",
                        step_pct=85)
    _scene_bounds = []
    _scene_disabled = os.environ.get("SCENE_DETECTION", "1").strip().lower() in (
        "0", "false", "no", "off")
    if _scene_disabled:
        print("ℹ️  Scene detection disabled (SCENE_DETECTION=0) — the end-clamp "
              "polish is skipped, clips still render.")
    elif source_video_path and shorts:
        # Targeted scan: only the seconds around each picked clip's span —
        # ~10x cheaper than scanning the whole source for the same clamp.
        try:
            import scene_detection as _sd
            _scene_bounds, _ = _sd.detect_scenes_in_ranges(
                source_video_path,
                [(float(s["start"]), float(s["end"])) for s in shorts])
        except Exception as e:
            print(f"⚠️ Scene-boundary clamp unavailable "
                  f"({type(e).__name__}: {str(e)[:150]})")
    for s in shorts:
        _clamp_candidate_end_to_scene(s, _scene_bounds)
    shorts = _dedup_overlapping_clips(shorts)

    result = {
        "shorts": shorts,
        "rejected": [],
        "clip_count_requested": picker_result.get("requested", len(shorts)),
        "clip_count_delivered": len(shorts),
        "clip_count_shortfall": max(0, picker_result.get("requested", len(shorts)) - len(shorts)),
    }
    if picker_result.get("cost_analysis"):
        result["cost_analysis"] = picker_result["cost_analysis"]
    if context_blob and context_blob.get("cost_analysis"):
        _fold_cost(result, context_blob["cost_analysis"])
    if result["clip_count_shortfall"]:
        print(f"⚠️ DELIVERED {len(shorts)}/{result['clip_count_requested']} clip(s) — "
              f"short by {result['clip_count_shortfall']}. Written to metadata, never silent.")
    return result


def _fold_cost(result, extra_cost):
    """Merge the context layer's cost into the picker's cost summary."""
    existing = result.get("cost_analysis")
    if not existing:
        result["cost_analysis"] = extra_cost
        return
    for key in ("input_tokens", "output_tokens", "total_cost"):
        if key in existing and key in extra_cost:
            existing[key] = existing.get(key, 0) + extra_cost.get(key, 0)


def get_visual_clips(video_path, video_duration, language="en"):
    """Clip a SILENT video by vision: Gemini watches the footage and picks the
    most engaging visual moments (no transcript). Returns the same
    {"shorts", "cost_analysis"} shape as get_viral_clips, or None."""
    print("🎥  Silent video — analyzing with Gemini vision (no transcript)...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found.")
        return None
    client = gemini_worker.make_client(api_key)
    model_name = os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'
    print(f"🎥  Model: {model_name} | uploading {os.path.basename(video_path)}…")

    file_upload = None
    try:
        file_upload = client.files.upload(file=video_path)
        deadline = time.time() + 180
        while True:
            info = client.files.get(name=file_upload.name)
            state = str(getattr(getattr(info, "state", info), "name", "")).upper()
            if state == "ACTIVE":
                break
            if state == "FAILED":
                print("❌ Gemini could not process the video.")
                return None
            if time.time() > deadline:
                print("❌ Gemini video processing timed out.")
                return None
            time.sleep(2)

        prompt = gemini_worker.VISUAL_PROMPT_TEMPLATE.format(
            video_duration=video_duration, language=language)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_worker.VisualResponse,
            safety_settings=gemini_worker.RELAXED_SAFETY_SETTINGS,
        )
        response = gemini_pool.generate_with_fallback(
            client, model_name, [file_upload, prompt], config=config,
            max_attempts=1, log=lambda msg: print(msg))
        gemini_worker.raise_if_blocked(response)
        parsed = json.loads(response.text)
        shorts = parsed.get("shorts") or []
        # Clamp to the real duration; drop anything degenerate.
        clean = []
        for s in shorts:
            s["start"] = max(0.0, float(s.get("start", 0)))
            s["end"] = min(float(video_duration), float(s.get("end", 0)))
            if s["end"] - s["start"] >= 1.0:
                clean.append(s)
        if not clean:
            print("⚠️ Vision pass returned no usable clips.")
            return None

        cost = gemini_worker._calculate_cost_analysis(response, model_name)
        if cost:
            print(f"💰 Vision cost ({model_name}): ${cost.get('total_cost', 0):.6f}")
        result = {"shorts": clean}
        if cost:
            result["cost_analysis"] = cost
        return result
    except gemini_worker.GeminiBlockedError as e:
        print(f"🚫 {e}")
        raise
    except Exception as e:
        print(f"❌ Gemini vision error: {e}")
        return None
    finally:
        if file_upload is not None:
            try:
                client.files.delete(name=file_upload.name)
            except Exception:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoCrop-Vertical with Viral Clip Detection.")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help="Path to the input video file.")
    input_group.add_argument('-u', '--url', type=str, help="YouTube URL to download and process.")
    
    parser.add_argument('-o', '--output', type=str, help="Output directory or file (if processing whole video).")
    parser.add_argument('--keep-original', action='store_true', help="Keep the downloaded YouTube video.")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip AI analysis and convert the whole video.")
    parser.add_argument('--clip-count', type=int, default=None,
                        help="Hard target clip count (1-40): narrative is built to "
                             "that number rather than the duration-based auto floor.")
    parser.add_argument('--long-context-clips', type=int, default=0,
                        help="Request up to N full-arc 1-3 min long-context clips "
                             "alongside the tight shorts (0 = off).")
    parser.add_argument('--format', type=str, default="auto", choices=["auto", "vertical", "horizontal", "square", "custom"],
                        help="Output aspect: vertical/auto (9:16), horizontal (keep 16:9), square (1:1).")
    parser.add_argument('--custom-width', type=int, default=None,
                        help="Custom output width (with --format custom); pairs with --custom-height.")
    parser.add_argument('--custom-height', type=int, default=None,
                        help="Custom output height (with --format custom); pairs with --custom-width.")
    parser.add_argument('--style-variant', type=str, default="balanced",
                        choices=["balanced", "high_energy", "story_driven"],
                        help="Narrative-prompt style variant (AI Preferences, round 3 item 6).")
    parser.add_argument('--remove-background-audio', type=str, default="",
                        choices=["", "auto", "isolate", "denoise"],
                        help="Strip background audio from delivered clips. "
                             "'isolate' = Demucs voice separation (removes music); "
                             "'denoise' = fast FFmpeg speech chain (removes hiss/room "
                             "tone only); 'auto' = Demucs when installed, else denoise.")

    args = parser.parse_args()
    remove_background_audio = (args.remove_background_audio or "").strip() or None
    output_format = args.format

    # Show what the GPU is doing for THIS run, before any work starts.
    _print_pipeline_diagnostics()

    # Custom aspect ratio drives the reframe crop shape end-to-end; the
    # pipeline still scales to the delivery floor for quality, same as 9:16.
    output_aspect = None
    if output_format == "custom":
        if not (args.custom_width and args.custom_height):
            print("❌ --format custom requires --custom-width and --custom-height")
            exit(1)
        output_aspect = args.custom_width / args.custom_height

    script_start_time = time.time()
    # Per-stage wall-clock tracking for the ETA estimate (plan round 2,
    # item 3): durations are flushed to stage_durations.json at job end.
    _stage_t0 = time.time()
    _stage_durations = {}

    def _ensure_dir(path: str) -> str:
        """Create directory if missing and return the same path."""
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    # The clip count is ALWAYS explicit — auto mode was removed. Fail fast,
    # before any download or Gemini call: the picker fulfills the requested
    # count at all costs, so it needs a count to fulfill.
    if not args.skip_analysis and args.clip_count is None:
        parser.error(
            "--clip-count is required for clip analysis — auto mode was "
            "removed; the picker fulfills an explicit count at all costs.")
    
    # 1. Get Input Video
    # The pre-download context layer starts the moment the URL is known: the
    # link goes to Gemini (own key) IN PARALLEL with the download below, so
    # by the time the transcript exists the picker already has the brain.
    context_thread = None
    _context_started_at = None
    cached_source = None
    if args.url:
        # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
        # For whole-video runs (--skip-analysis), --output can be a file path.
        if args.output and not args.skip_analysis:
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default "."
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or "."
            else:
                output_dir = "."

        # Per-source persistent reuse (source_store.py): a URL we have seen
        # before skips the download entirely — the source video is hardlinked
        # from the cache, and the transcript + context blob ride along.
        cached_source = source_store.lookup(args.url)
        if cached_source:
            print(f"♻️  Source cached for this URL — reusing the downloaded "
                  f"video (no re-download).")
            input_video = os.path.join(output_dir, cached_source["filename"])
            _link_or_copy(cached_source["video_path"], input_video)
            video_title = cached_source["title"]
            if not video_title:
                video_title = os.path.splitext(cached_source["filename"])[0]

        # Start the pre-download context layer whenever we do not already
        # have a cached blob: the link goes to Gemini (own key) IN PARALLEL
        # with the download below. A cached re-run whose first attempt never
        # landed (504 etc.) must retry here too — otherwise the context stays
        # transcript-only forever for that source.
        if (not args.skip_analysis
                and (cached_source is None
                     or not cached_source.get("context_blob"))):
            context_thread = context_layer.analyze_url_async(
                args.url, os.path.join(output_dir, context_layer.CONTEXT_BLOB_FILENAME))
            _context_started_at = time.time()

        if cached_source is None:
            input_video, video_title = download_youtube_video(
                args.url, output_dir, require_hd=True)
            # Cache the source NOW (hardlink — instant even for multi-GB
            # files) so a later failure in the job never forces a re-download.
            source_store.save_source(args.url, input_video, video_title)
            _write_progress(output_dir, "download", note="source downloaded")
            _stage_durations["download"] = time.time() - _stage_t0
            _stage_t0 = time.time()
    else:
        input_video = args.input
        video_title = os.path.splitext(os.path.basename(input_video))[0]
        
        if args.output and not args.skip_analysis:
            # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default to input dir.
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or os.path.dirname(input_video)
            else:
                output_dir = os.path.dirname(input_video)

    if not os.path.exists(input_video):
        print(f"❌ Input file not found: {input_video}")
        exit(1)

    # 2. Decision: Analyze clips or process whole?
    if args.skip_analysis:
        print("⏩ Skipping analysis, processing entire video...")
        output_file = args.output if args.output else os.path.join(output_dir, f"{video_title}_vertical.mp4")
        render_clip(input_video, output_file, output_format,
                    custom_aspect=output_aspect)
    else:
        # Get duration (needed by both the transcript and the vision path).
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        # 3. Transcribe — unless the video has no audio, in which case fall back
        # to Gemini vision (picks clips from the imagery instead of the speech).
        from transcribe_backends import NoAudioError
        transcript = None
        if cached_source and cached_source.get("transcript"):
            transcript = cached_source["transcript"]
            print(f"♻️  Transcript loaded from cache ({len(transcript.get('segments') or [])} segments) — "
                  f"no transcription API call.")
        else:
            try:
                _write_progress(output_dir, "transcribe", note="transcribing audio",
                                duration_seconds=duration,
                                step="transcribing audio (AssemblyAI)")
                transcript = transcribe_video(input_video)
                if args.url:
                    source_store.save_transcript(args.url, transcript)
            except NoAudioError as e:
                print(f"🔇 {e} — switching to visual analysis.")
        _stage_durations["transcribe"] = time.time() - _stage_t0
        _stage_t0 = time.time()

        # Collect the parallel context layer (5s cap — it must never block
        # the pipeline; the picker is fully functional transcript-only).
        context_blob = None
        if cached_source and cached_source.get("context_blob"):
            context_blob = cached_source["context_blob"]
            print("♻️  Context blob loaded from cache.")
        elif context_thread is not None:
            # The thread had the whole download+transcribe runway on a fresh
            # run; a cached re-run (cached transcript, no download) starts it
            # ~now, so wait out the remainder of a fair budget so the blob
            # can actually land and get cached. join() returns the instant
            # the thread finishes, so a stuck 504 never blocks the job long.
            _ctx_wait = (context_layer.wait_budget(
                time.time() - _context_started_at,
                target=float(os.environ.get("CONTEXT_WAIT_TARGET", "35")))
                if _context_started_at is not None else 5.0)
            context_thread.join(timeout=_ctx_wait)
            context_blob = context_layer.load_context(
                os.path.join(output_dir, context_layer.CONTEXT_BLOB_FILENAME))
            if context_blob and args.url:
                source_store.save_context(args.url, context_blob)
            if not context_blob:
                print(f"ℹ️  Context blob not ready in time "
                      f"(waited {_ctx_wait:.0f}s) — the picker runs "
                      "transcript-only; the count is still fulfilled.")

        # 4. Gemini Analysis (transcript-driven, or vision for silent videos)
        _write_progress(output_dir, "analyze", note="finding the viral moments",
                        step="asking Gemini for the viral moments", step_pct=20)
        if transcript is not None:
            clips_data = get_viral_clips(transcript, duration,
                                         source_video_path=input_video,
                                         clip_count=args.clip_count,
                                         long_context_count=args.long_context_clips,
                                         style_variant=args.style_variant,
                                         output_dir=output_dir,
                                         context_blob=context_blob)
        else:
            clips_data = get_visual_clips(input_video, duration)
        _stage_durations["analyze"] = time.time() - _stage_t0
        _stage_t0 = time.time()

        if not clips_data or 'shorts' not in clips_data:
            # Deliberately fail instead of reframing the whole video: that path
            # wrote no metadata.json, so app.py marked the job failed anyway
            # (app.py:1087) after burning GPU on a render nobody could see.
            raise RuntimeError(
                "Clip detection failed — Gemini did not return usable clips for this video.")
        else:
            print(f"🔥 Found {len(clips_data['shorts'])} clips!")
            print(f"🎯 Next: sentence-snap boundaries → scene boundaries → "
                  f"per-clip render on the GPU (clip 1/{len(clips_data['shorts'])}) "
                  f"→ finalize.")
            _write_progress(output_dir, "render", 0, len(clips_data['shorts']),
                            note="preparing clips",
                            step="preparing clips — per-clip render starts next",
                            step_pct=0)

            # Save metadata. Silent videos have no transcript → no subtitles,
            # which is correct (there's no speech to caption).
            clips_data['transcript'] = transcript or {"language": "none", "segments": []}
            # Source identity for same-source job reuse (round-5 feature: "if
            # the clip already exists, just map it") — /api/process checks this
            # before starting a fresh run of the same URL.
            clips_data['source_url'] = args.url or ""
            clips_data['source_file'] = os.path.basename(input_video)
            # PART 6: surface the verified source specs (written by the
            # download gate) in the job metadata so the dashboard can show
            # "Source: 1080p · 4.3 Mbps" next to the clips instead of leaving
            # quality unexplained.
            _specs_path = os.path.join(output_dir, "source_specs.json")
            if os.path.exists(_specs_path):
                try:
                    with open(_specs_path) as _sf:
                        clips_data['source_specs'] = json.load(_sf)
                except (OSError, ValueError):
                    pass
            metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
            # Round-5 spec 4.1: write atomically (tmp + os.replace) so a crash
            # mid-write never leaves a truncated metadata file that every
            # consumer treats as authoritative. .ready markers gate the poll
            # loop; only ORPHAN markers (whose clip file no longer exists) are
            # purged — a marker whose file is present is a genuinely completed
            # clip and lets this run RESUME it instead of re-rendering
            # (_process_one_clip's skip-existing path).
            for marker in glob.glob(os.path.join(output_dir, "*.ready")):
                clip_file = marker[:-len(".ready")]
                if not os.path.exists(clip_file):
                    try:
                        os.remove(marker)
                    except OSError:
                        pass
            _meta_tmp = metadata_file + ".tmp"
            with open(_meta_tmp, 'w') as f:
                json.dump(clips_data, f, indent=2)
            os.replace(_meta_tmp, metadata_file)
            print(f"   Saved metadata to {metadata_file}")
            _write_progress(output_dir, "render", 0, len(clips_data['shorts']),
                            note="rendering clips",
                            step=f"rendering clip 0/{len(clips_data['shorts'])}",
                            step_pct=0)

            # Ground-truth word list for keep_span snapping — same source
            # get_viral_clips() uses for boundary snapping, rebuilt here
            # since it's cheap pure-Python and not passed out of that call.
            _all_words = []
            if transcript:
                for segment in transcript.get('segments', []):
                    for word in segment.get('words', []):
                        _all_words.append({'w': word.get('word', ''),
                                           's': word.get('start', 0), 'e': word.get('end', 0)})

            # Gemini key pool for the scene-context direction layer, built
            # once and shared across clip workers (it's internally safe to
            # acquire from several threads — that's what it exists for).
            # Named distinctly because `pool` below is the ThreadPoolExecutor.
            vision_pool = gemini_pool.pool_from_env()
            vision_model = os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'

            # 5. Process clips in parallel: each worker cuts + renders one
            # clip. Renders are mostly ffmpeg subprocesses (parallelize well);
            # per-clip face-spine detection runs inside the worker, so
            # nothing is shared across threads.
            def _process_one_clip(i, clip):
                # Bind this worker thread to one GPU. On the measured 2×T4
                # Kaggle run every concurrent clip ran on GPU 0 while GPU 1 sat
                # idle; see gpu_affinity for why this is a thread-local torch
                # device rather than CUDA_VISIBLE_DEVICES. No-op on CPU and
                # single-GPU hosts.
                gpu_affinity.assign_worker(i)
                _clip_gpu = gpu_affinity.current_device()
                start = clip['start']
                end = clip['end']
                print(f"\n🎬 Processing Clip {i+1}: {start}s - {end}s"
                      + (f" (gpu {_clip_gpu})" if _clip_gpu else ""))
                print(f"   Title: {clip.get('video_title_for_youtube_short', 'No Title')}")

                clip_filename = f"{video_title}_clip_{i+1}.mp4"
                clip_temp_path = os.path.join(output_dir, f"temp_{clip_filename}")
                clip_final_path = os.path.join(output_dir, clip_filename)
                jump_cut_workdir = None

                # Resume: if this clip's final file AND its .ready marker both
                # already exist (a previous run completed this exact clip), map
                # to it instead of re-rendering — the poll loop will surface it
                # immediately. Saves GPU time on re-runs of the same source.
                if (os.path.exists(clip_final_path)
                        and os.path.getsize(clip_final_path) > 0
                        and os.path.exists(
                            os.path.join(output_dir, f"{clip_filename}.ready"))):
                    print(f"   ♻️ Clip {i+1} already exists — mapping to "
                          f"{clip_filename} (no re-render)")
                    return True

                # keep_spans marks the sub-ranges the picker judged essential —
                # everything else in [start, end] is dead air/filler to jump-
                # cut out, not just boundary trim (see RESEARCH_viral_clip_
                # patterns.md §4 — 6/6 real published shorts studied do this,
                # the single most consistent editing pattern found).
                keep_spans = _snap_keep_spans_to_words(
                    clip.get('keep_spans') or [], _all_words, start, end)
                total_kept = sum(e - s for s, e in keep_spans)
                # If the picker returned nothing usable, or the kept spans
                # already cover ~all of [start, end], there's nothing to cut
                # — skip the extra re-encode/concat pass entirely.
                do_jump_cut = bool(keep_spans) and total_kept < (end - start) * 0.97

                try:
                    if do_jump_cut:
                        jump_cut_workdir = tempfile.mkdtemp(prefix=f"jumpcut_{i}_")
                        removed = (end - start) - total_kept
                        print(f"   ✂️  Jump-cutting {len(keep_spans)} span(s), "
                              f"removing {removed:.1f}s of dead air/filler")
                        combined_path = _build_jump_cut_source(input_video, keep_spans, jump_cut_workdir)
                        clip_transcript, new_duration = _remap_transcript_onto_jump_cut(
                            transcript or {"segments": []}, keep_spans)
                        # os.replace requires same-filesystem; the jump-cut
                        # workdir (mkdtemp -> /tmp) and output_dir (a bind
                        # mount) are different devices in this container,
                        # so a plain rename fails with EXDEV (confirmed in
                        # prod, 31-jul-2026). shutil.move falls back to
                        # copy+delete across devices.
                        shutil.move(combined_path, clip_temp_path)
                        render_clip_start, render_clip_end = 0.0, new_duration
                    else:
                        # ffmpeg cut — re-encoding for precision on strict seconds
                        cut_command = [
                            'ffmpeg', '-y',
                            *gpu_decode_args(device=gpu_affinity.current_device()),
                            '-ss', str(start),
                            '-to', str(end),
                            '-i', input_video,
                            *source_logo_crop_vf_args(),
                            *video_encode_args(QUALITY_FAST, device=gpu_affinity.current_device()),
                            *audio_encode_args(),
                            clip_temp_path
                        ]
                        _cut = subprocess.run(cut_command, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.PIPE)
                        if _cut.returncode != 0:
                            _err = (_cut.stderr or b"").decode("utf-8", "replace").strip()
                            raise RuntimeError(
                                f"ffmpeg cut failed for clip {i + 1}"
                                + (f": {_err[-400:]}" if _err else ""))
                        clip_transcript = transcript
                        render_clip_start, render_clip_end = start, end

                    # Third verification layer, on the STREAMLINED cut (post
                    # jump-cut) so its timestamps land on the same clock the
                    # renderer uses: Gemini watches the finished clip and
                    # directs the camera — who to be on, and why. See
                    # analyze_scene_context. SCENE_DIRECTION=0 skips it —
                    # faster/cheaper, framing falls back to heuristics.
                    # The local director (speaker fusion + beat planner)
                    # replaced the per-clip Gemini camera-director call —
                    # SCENE_DIRECTION is opt-in now.
                    if os.environ.get("SCENE_DIRECTION", "0").strip().lower() in (
                            "0", "false", "no", "off"):
                        scene_ctx = {"directives": [], "primary_subject": "",
                                     "primary_subject_x": None}
                    else:
                        scene_ctx = analyze_scene_context(
                            vision_pool, vision_model, clip_temp_path,
                            render_clip_end - render_clip_start,
                            transcript_result=clip_transcript,
                            clip_start=render_clip_start, clip_end=render_clip_end)
                    # analyze_scene_context returns [] on failure (fails open)
                    # and a dict on success — guard both so a transient
                    # scene-context 503 can never crash the clip (Clip 4 died
                    # on "list indices must be integers" in the 03:30 run).
                    focus_directives = (scene_ctx.get("directives") or []
                                        if isinstance(scene_ctx, dict) else [])
                    primary_subject_x = (scene_ctx.get("primary_subject_x")
                                         if isinstance(scene_ctx, dict) else None)

                    # PART 6.4 (6-aug-2026): when nothing needs a post-render
                    # pass (no watermark, no audio cleanup), fold the caption
                    # burn INTO the render — one encode from the reframed
                    # frames instead of re-encoding the clean clip. The v3
                    # render fills the frame (no letterbox), so captions use
                    # the full-frame band (general_ranges=[]). Any prep
                    # failure, watermark or audio-cleanup path falls back to
                    # the existing post-render burn.
                    _cap_prep = None
                    if (os.environ.get("WATERMARK") != "1"
                            and not remove_background_audio):
                        try:
                            _cap_prep = prepare_caption_burn(
                                clip_final_path, clip_transcript,
                                render_clip_start, render_clip_end,
                                general_ranges=[])
                        except Exception as e:
                            print(f"   ⚠️ Caption prep failed ({e}) — "
                                  "falling back to the post-render burn")
                            _cap_prep = None

                    success, general_ranges = render_clip(
                        clip_temp_path, clip_final_path, output_format,
                        transcript=clip_transcript, clip_start=render_clip_start,
                        clip_end=render_clip_end,
                        focus_directives=focus_directives,
                        primary_subject_x=primary_subject_x,
                        custom_aspect=output_aspect,
                        ass_filter=_cap_prep[2] if _cap_prep else None,
                        captioned_output=_cap_prep[1] if _cap_prep else None)
                    if success and os.environ.get("WATERMARK") == "1":
                        apply_watermark(clip_final_path)
                    if success:
                        # Background-audio removal FIRST, then captions.
                        #
                        # This order is load-bearing and was wrong until
                        # 6-aug-2026: auto_caption_clip does not modify the clip
                        # in place, it writes a NEW subtitled_<ts>_<name>.mp4
                        # with `-c:a copy`, and app.py serves that derived file
                        # (_canonical_clip_file picks the newest one). Cleaning
                        # the audio afterwards therefore cleaned a file nobody
                        # watches, while the delivered clip kept its original
                        # music — reported from a real job ("I put remove
                        # background audio and I still hear sound in three
                        # clips"), and the log confirmed it: captions burned at
                        # 02:53:46, audio cleaned at 02:53:55, on the wrong file.
                        #
                        # Cleaning first means the caption pass copies the
                        # ALREADY-cleaned audio through. Fails OPEN: a
                        # separation error ships the original audio, never a
                        # silent or missing track.
                        if remove_background_audio:
                            try:
                                import audio_cleanup
                                cleaned = clip_final_path + ".voice.mp4"
                                audio_cleanup.clean_audio(
                                    clip_final_path, cleaned,
                                    mode=remove_background_audio)
                                shutil.move(cleaned, clip_final_path)
                                print(f"   🔇 Background audio removed ({remove_background_audio}).")
                            except Exception as e:
                                print(f"   ⚠️ Background-audio removal failed ({e}); "
                                      "keeping the original audio.")
                                try:
                                    os.remove(clip_final_path + ".voice.mp4")
                                except OSError:
                                    pass
                        # Captions: the single-pass render wrote the
                        # subtitled file alongside the clean one; otherwise
                        # (watermark or audio-cleanup path) burn after, as
                        # before. The canonical file that
                        # app.py serves is the newest subtitled_* file either
                        # way, and the clean original stays for re-styling.
                        _captioned = (os.path.exists(_cap_prep[1])
                                      and os.path.getsize(_cap_prep[1]) > 0
                                      if _cap_prep else False)
                        if _captioned:
                            print(f"   💬 Captions burned (single-pass): "
                                  f"{os.path.basename(_cap_prep[1])}")
                        else:
                            auto_caption_clip(
                                clip_final_path, clip_transcript,
                                render_clip_start, render_clip_end,
                                general_ranges=general_ranges)
                        # Only now — captions burned (or deliberately skipped) and
                        # the file fully written — is the clip safe to surface.
                        # app.py's poll loop gates on this marker.
                        _mark_clip_ready(output_dir, clip_filename)
                        print(f"   ✅ Clip {i+1} ready: {clip_final_path}")
                    return success
                finally:
                    if os.path.exists(clip_temp_path):
                        os.remove(clip_temp_path)
                    if jump_cut_workdir:
                        shutil.rmtree(jump_cut_workdir, ignore_errors=True)

            # 5 on GPU hosts (T4 verification showed idle encode capacity);
            # fewer on CPU hosts, where parallel ffmpeg/whisper threads
            # oversubscribe a small box (see hardware_defaults). An explicit
            # CLIP_WORKERS env var always wins.
            clip_workers = max(
                int(os.environ.get("CLIP_WORKERS") or default_clip_workers()), 1)
            _gpu_plan = gpu_affinity.describe()
            if _gpu_plan:
                print(_gpu_plan)
            shorts = clips_data['shorts']
            _progress_lock = threading.Lock()
            _rendered_count = [0]
            with ThreadPoolExecutor(max_workers=min(clip_workers, len(shorts))) as pool:
                futures = {pool.submit(_process_one_clip, i, clip): i
                           for i, clip in enumerate(shorts)}
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        ok = future.result()
                    except Exception as e:
                        ok = False
                        print(f"   ❌ Clip {i+1} failed: {type(e).__name__}: {e}")
                    if ok:
                        with _progress_lock:
                            _rendered_count[0] += 1
                            _write_progress(output_dir, "render",
                                            _rendered_count[0], len(shorts),
                                            step=f"rendering clip "
                                                 f"{_rendered_count[0]}/{len(shorts)}",
                                            step_pct=100 * _rendered_count[0] // len(shorts))
            _write_progress(output_dir, "finalize", len(shorts), len(shorts),
                            note="job complete",
                            step="finalizing — stitching the clips together",
                            step_pct=100)
            _stage_durations["render"] = time.time() - _stage_t0
            _stage_t0 = time.time()
            _stage_durations["finalize"] = time.time() - _stage_t0
            _report_gpu_utilization(output_dir)

    # Clean up original if requested
    if args.url and not args.keep_original and os.path.exists(input_video):
        # Source panel artifacts (stills + preview proxy) must survive the
        # cleanup — generate them while the original is still on disk.
        _generate_source_artifacts(input_video, output_dir)
        os.remove(input_video)
        print(f"🗑️  Cleaned up downloaded video.")

    # Feed the rolling per-stage averages used by the dashboard's ETA.
    if _stage_durations:
        record_stage_durations(output_dir, _stage_durations)

    total_time = time.time() - script_start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
