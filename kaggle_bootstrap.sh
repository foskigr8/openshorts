#!/usr/bin/env bash
# OpenShorts on Kaggle (or any Docker-less GPU host).
#
# Kaggle has no Docker, so the compose stack cannot be used. This brings up the
# same application natively: one Python process serves BOTH the API and the
# built dashboard (see the single-origin block at the end of app.py), and one
# cloudflared tunnel exposes it.
#
#   bash kaggle_bootstrap.sh            # install, build, serve, tunnel
#   SKIP_INSTALL=1 bash kaggle_bootstrap.sh   # re-run without reinstalling
#
# Expects to be run from the repo root.
set -euo pipefail

PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-/tmp/openshorts-logs}"
mkdir -p "$LOG_DIR"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# --- 1. GPU sanity ---------------------------------------------------------
say "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    echo "    no nvidia-smi — running CPU-only (much slower; LR-ASD ~2x realtime)"
    GPU_COUNT=0
fi

# 6-aug-2026 (PART 2): on a GPU host, prefer CUDA decode offload and CUDA
# Whisper. Both probe and fall back cleanly inside the app when the ffmpeg
# build lacks CUDA, so exporting them unconditionally here is safe on
# CPU-only boxes too.
if [ "${GPU_COUNT:-0}" -gt 0 ]; then
    export GPU_RENDER="${GPU_RENDER:-1}"
    echo "    GPU_RENDER=1 (probe + fallback inside app)"
fi

# 6-aug-2026: use BOTH T4s. CLIP_GPUS was opt-in because sharding broke
# TransNetV2 scene detection (worker thread device vs model device); the
# scene-detection guard in scene_detection.py now pins inference to the
# model's device, so spreading is safe on multi-GPU hosts.
if [ "${GPU_COUNT:-0}" -gt 1 ]; then
    export CLIP_GPUS="${CLIP_GPUS:-0,1}"
    echo "    CLIP_GPUS=0,1 (both GPUs: LR-ASD + ffmpeg per-worker)"
fi

# Option B downloads (owner choice, 9-aug-2026): 1440p cap + VP9/H.264
# preference — both codecs GPU-decode on the T4, AV1 does not. main.py's
# default is already 1440, so this only surfaces the decision in the log.
echo "    SOURCE_MAX_HEIGHT=1440 (Option B: VP9/H.264 <=1440p, GPU-decodable on T4)"

# --- 1b. nvenc-capable ffmpeg -----------------------------------------------
# Kaggle's preinstalled ffmpeg (apt) is a CPU-only build with no --enable-nvenc.
# FFMPEG_ENCODER/GPU_RENDER above tell the app to use nvenc, but ffmpeg_utils.py
# probes the actual binary at runtime and silently falls back to libx264 when
# nvenc isn't there — so every render was CPU-encoded despite the GPU config
# (confirmed: nvidia-smi showed zero encode activity during a job). nvenc only
# needs the driver's libnvidia-encode.so (already present, it's part of the
# driver, not CUDA toolkit), so a static build with nvenc compiled in is a drop-
# in swap — no system packages, no rebuild. Entirely optional: any failure here
# leaves ffmpeg exactly as it was (CPU encode), same as before this block.
if [ "${GPU_COUNT:-0}" -gt 0 ] && [ "${SKIP_FFMPEG_NVENC:-0}" != "1" ]; then
    say "nvenc-capable ffmpeg"
    # Skip ONLY when the ffmpeg on PATH has BOTH GPU encode AND CUDA decode +
    # filters — encode alone (h264_nvenc) is not enough: the pipeline is
    # GPU-or-nothing, so a decode-capable build must be installed even when
    # some other ffmpeg with just NVENC is on PATH.
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc \
        && ffmpeg -hide_banner -hwaccels 2>/dev/null | grep -qiE 'cuda|nvdec|cuvid' \
        && ffmpeg -hide_banner -filters 2>/dev/null | grep -qE 'scale_cuda|scale_npp'; then
        echo "    ffmpeg already has GPU encode + CUDA decode/filters — nothing to do"
        export REQUIRE_GPU_DECODE="1"
        echo "    REQUIRE_GPU_DECODE=1 — strict GPU-or-nothing mode confirmed"
    else
        echo "    CUDA decode/filters missing on the PATH ffmpeg — installing the"
        echo "    full decode-capable build (BtbN master, verified: cuda + scale_cuda + nvenc)"
        FFMPEG_DIR="${FFMPEG_DIR:-/kaggle/working/ffmpeg-nvenc}"
        mkdir -p "$FFMPEG_DIR"
        # Two asset variants (gpl, then gpl-shared) with a retry each — the
        # pipeline is GPU-or-nothing, so a flaky download must never silently
        # leave the host CPU-decode-only.
        FFMPEG_DL_OK=0
        for VARIANT in "ffmpeg-master-latest-linux64-gpl.tar.xz" "ffmpeg-master-latest-linux64-gpl-shared.tar.xz"; do
            for TRY in 1 2; do
                FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/$VARIANT"
                if curl -fsSL --max-time 180 "$FFMPEG_URL" -o /tmp/ffmpeg-nvenc.tar.xz 2>"$LOG_DIR/ffmpeg_nvenc.log" \
                    && tar -xJf /tmp/ffmpeg-nvenc.tar.xz -C /tmp 2>>"$LOG_DIR/ffmpeg_nvenc.log" \
                    && cp /tmp/ffmpeg-*-linux64-gpl*/bin/ffmpeg /tmp/ffmpeg-*-linux64-gpl*/bin/ffprobe "$FFMPEG_DIR/" 2>>"$LOG_DIR/ffmpeg_nvenc.log"; then
                    FFMPEG_DL_OK=1
                    break 2
                fi
                echo "    download attempt $TRY ($VARIANT) failed — retrying"
            done
        done
        rm -rf /tmp/ffmpeg-nvenc.tar.xz /tmp/ffmpeg-*-linux64-gpl* 2>/dev/null
        if [ "$FFMPEG_DL_OK" = "1" ]; then
            export PATH="$FFMPEG_DIR:$PATH"
            if "$FFMPEG_DIR/ffmpeg" -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc; then
                echo "    installed to $FFMPEG_DIR, h264_nvenc confirmed present"
                if "$FFMPEG_DIR/ffmpeg" -hide_banner -hwaccels 2>/dev/null | grep -qiE 'cuda|nvdec|cuvid' \
                    && "$FFMPEG_DIR/ffmpeg" -hide_banner -filters 2>/dev/null | grep -qE 'scale_cuda|scale_npp'; then
                    echo "    CUDA decode (NVDEC + CUDA filters) also present — GPU decode + encode both active"
                    export REQUIRE_GPU_DECODE="1"
                    echo "    REQUIRE_GPU_DECODE=1 — strict GPU-or-nothing mode confirmed"
                else
                    echo "    NOTE: this build has NVENC (GPU encode) but no CUDA decode/filters."
                    echo "          JOBS WILL FAIL at startup (REQUIRE_GPU_DECODE=1, GPU-or-nothing)"
                    echo "          until a decode-capable build is installed."
                fi
                echo "    main.py prepends $FFMPEG_DIR to PATH itself at job start,"
                echo "    so uvicorn does NOT need this shell's PATH export."
            else
                echo "    downloaded build lacks h264_nvenc — JOBS WILL FAIL (GPU-or-nothing)"
            fi
        else
            echo "    download/extract failed after all attempts — JOBS WILL FAIL (GPU-or-nothing)."
            echo "    See $LOG_DIR/ffmpeg_nvenc.log. GitHub must be reachable from Kaggle"
            echo "    (the notebook's own git pull uses it)."
        fi
    fi
fi

# --- 2. Python deps --------------------------------------------------------
# Kaggle's base image already carries torch/opencv/numpy built against its own
# CUDA. Reinstalling those from requirements.txt is slow and can break CUDA, so
# they are held back and only the packages Kaggle lacks are installed.
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    say "Python dependencies (2-5 min)"
    grep -vE '^(torch|torchvision|torchaudio|opencv|numpy|scipy)([=<>~!]|$)' \
        requirements.txt > /tmp/req-kaggle.txt || cp requirements.txt /tmp/req-kaggle.txt
    # Those lines are stripped because Kaggle's image already has them built
    # against Kaggle's own CUDA — which also drops requirements.txt's opencv
    # caps. Put the caps back: insightface pulls albumentations -> albucore,
    # which wants the newest opencv, and opencv 5.x is what drags a numpy
    # major-version jump behind it.
    #
    # CAPS, not pinned versions. An exact pin has to match a release that
    # actually exists on PyPI, and the obvious way to build one is wrong:
    # cv2.__version__ reports "4.13.0" while the distribution is "4.13.0.x", so
    # `opencv-python==4.13.0` is unsatisfiable — and an unsatisfiable pin is
    # ResolutionImpossible, meaning NOTHING installs. Measured on the host
    # 6-aug-2026: no yt-dlp, no ultralytics. A cap is satisfied by whatever
    # is already installed, so pip simply leaves it alone.
    printf 'opencv-python<5\nopencv-contrib-python<5\nopencv-python-headless<5\n' \
        >> /tmp/req-kaggle.txt
    if pip install -q -r /tmp/req-kaggle.txt 2>/tmp/pip-err.log; then
        echo "    dependencies installed"
    else
        echo "    pip install reported errors:"
        grep -iE "conflict is caused by|depends on|The user requested|ResolutionImpossible|ERROR" \
            /tmp/pip-err.log | head -8 | sed 's/^/      /' || true
        echo "    continuing — the import checks below say what actually survived"
    fi

    # Report the versions that actually matter, without asserting a major
    # version: this host ships numpy 2.0.2. The real signal is whether the
    # imports below succeed, not a number.
    python3 - <<'PYVER' || true
for mod in ("numpy", "cv2"):
    try:
        print(f"    {mod} {__import__(mod).__version__}")
    except Exception as e:
        print(f"    {mod} MISSING ({type(e).__name__})")
PYVER

    # Face ID needs an ONNX runtime to execute; requirements.txt deliberately
    # ships only insightface (onnxruntime-gpu is ~2GB and lives in the
    # Dockerfile's GPU block so the CPU image stays slim). Kaggle IS a GPU host,
    # so install it here. Kaggle images ship the CPU onnxruntime, which makes
    # insightface (face spine — the first heavy substage of reframe) run every
    # frame on the CPU while the GPUs idle ("stuck in Reframe engine v3 with
    # GPU at 0"). On a GPU host, force the CUDA build and uninstall the CPU
    # wheel first (both ship the same module).
    say "Face ID runtime (onnxruntime-gpu from requirements-gpu.txt)"
    if [ "${GPU_COUNT:-0}" -gt 0 ]; then
        # The pip nvidia wheels (CUDA-12 libs for onnxruntime) install into
        # site-packages/nvidia/*/lib — put them on the loader path for this
        # shell (main.py re-adds them at job start as a belt-and-suspenders).
        export LD_LIBRARY_PATH="$(python3 -c "
import os, site, glob
dirs = []
for base in site.getsitepackages():
    for p in glob.glob(os.path.join(base, 'nvidia', '*', 'lib')):
        if p not in dirs:
            dirs.append(p)
print(':'.join(dirs))
" 2>/dev/null):$LD_LIBRARY_PATH"
        if python3 -c "import onnxruntime; print('CUDAExecutionProvider' in onnxruntime.get_available_providers())" 2>/dev/null | grep -q True; then
            echo "    onnxruntime-gpu active: CUDAExecutionProvider available"
        else
            echo "    onnxruntime is CPU-only — uninstalling it and installing the"
            echo "    pinned onnxruntime-gpu (requirements-gpu.txt) for GPU face spine"
            pip uninstall -y onnxruntime 2>&1 | tail -1
            if pip install -q -r requirements-gpu.txt 2>&1 | tail -3; then
                if python3 -c "import onnxruntime; print('CUDAExecutionProvider' in onnxruntime.get_available_providers())" 2>/dev/null | grep -q True; then
                    echo "    onnxruntime-gpu installed: CUDAExecutionProvider available — face spine on GPU"
                else
                    echo "    CUDAExecutionProvider listed but the provider's libs must"
                    echo "    actually load — running the real load test:"
                    if python3 - <<'PY'
import ctypes
missing = []
for lib in ("libcublas.so.12", "libcublasLt.so.12", "libcudart.so.12",
            "libcurand.so.10", "libcudnn.so.9"):
    try:
        ctypes.CDLL(lib)
    except OSError:
        missing.append(lib)
print("    missing:", missing or "none — CUDA 12 libs load")
raise SystemExit(1 if missing else 0)
PY
                    then
                        echo "    CUDA 12 libs load — face spine on GPU"
                    else
                        echo "    missing CUDA 12 libs — installing the nvidia cu12 wheels"
                        pip install -q "nvidia-cublas-cu12<13" "nvidia-cudnn-cu12>=9,<10" \
                            "nvidia-cuda-runtime-cu12<13" "nvidia-curand-cu12<13" 2>&1 | tail -3
                        export LD_LIBRARY_PATH="$(python3 -c "
import os, site, glob
dirs = []
for base in site.getsitepackages():
    for p in glob.glob(os.path.join(base, 'nvidia', '*', 'lib')):
        if p not in dirs:
            dirs.append(p)
print(':'.join(dirs))
" 2>/dev/null):$LD_LIBRARY_PATH"
                        if python3 -c "
import ctypes
for lib in ('libcublas.so.12','libcublasLt.so.12','libcudart.so.12','libcudnn.so.9'):
    ctypes.CDLL(lib)
" 2>/dev/null; then
                            echo "    nvidia cu12 wheels installed — CUDA 12 libs load — face spine on GPU"
                        else
                            echo "    still missing CUDA libs — face spine stays CPU (check CUDA version)"
                        fi
                    fi
                fi
            else
                echo "    onnxruntime-gpu install FAILED — reinstalling CPU onnxruntime so face spine still works"
                pip install -q "onnxruntime==1.28.0" 2>&1 | tail -3 || \
                    echo "    CPU onnxruntime reinstall also failed — face ID will stay off"
            fi
        fi
    else
        # CPU-only host: plain onnxruntime (not the 2GB GPU wheel).
        pip install -q "onnxruntime==1.28.0" 2>&1 | tail -3 || \
            echo "    onnxruntime install reported errors — face ID will stay off"
    fi
    if python3 -c "import insightface" >/dev/null 2>&1; then
        echo "    insightface imports"
    else
        echo "    insightface NOT importable — named-speaker enrichment stays off"
        echo "    (the pipeline still runs; speakers stay anonymous)"
    fi

    python3 - <<'PY'
import importlib
# cv2 is in this list because insightface pulls opencv-python-headless,
# which overwrites the cv2 package Kaggle ships. Our usage (VideoCapture,
# cvtColor, resize) is headless-safe and no GUI call exists in this repo,
# but a broken cv2 would otherwise surface much later as a failed render.
for m in ("fastapi", "uvicorn", "yt_dlp", "ultralytics", "torch", "cv2"):
    try:
        importlib.import_module(m)
        print(f"    ok   {m}")
    except Exception as e:
        print(f"    MISS {m}: {type(e).__name__}: {e}")
PY
fi

# --- 3. Dashboard ----------------------------------------------------------
# Built once into dashboard/dist and then served by app.py. No VITE_API_URL is
# set on purpose: config.js falls back to a relative base, so the SPA talks to
# whatever origin serves it — which is what makes one tunnel enough.
# Rebuild when the SOURCE is newer than the build, not just when dist is
# missing. dashboard/dist is gitignored, so a fresh clone always builds — but
# the notebook now pulls into an existing checkout (cell 2), and a UI change
# arriving that way left a stale dist in place reporting "already built". The
# dashboard is compiled, so an un-rebuilt dist serves the OLD interface with
# no sign anything is wrong. REBUILD_DASHBOARD=1 forces it.
dashboard_needs_build() {
    [ ! -d dashboard/dist ] && return 0
    [ "${REBUILD_DASHBOARD:-0}" = "1" ] && return 0
    # Any source file newer than the built index.html.
    newer=$(find dashboard/src dashboard/index.html dashboard/package.json \
                 dashboard/vite.config.js dashboard/seo 2>/dev/null \
                 -newer dashboard/dist/index.html -print -quit)
    [ -n "$newer" ]
}

if dashboard_needs_build; then
    if command -v npm >/dev/null 2>&1; then
        if [ -d dashboard/dist ]; then
            say "Rebuilding dashboard — sources changed since the last build (2-4 min)"
        else
            say "Building dashboard (2-4 min)"
        fi
        if (cd dashboard && npm ci --silent 2>/dev/null || npm install --silent) \
                && (cd dashboard && npm run build --silent); then
            echo "    dashboard build ok"
        else
            # Loud on purpose: app.py serves dist directly, so a failed build
            # means the UI is stale or absent while everything else looks fine.
            echo "    !! DASHBOARD BUILD FAILED — the served UI is stale or missing."
            echo "       Run it by hand for the real error:"
            echo "       cd dashboard && npm run build"
        fi
    else
        echo "    no npm — dashboard cannot be built here."
        echo "    Commit dashboard/dist, or attach it as a Kaggle Dataset."
    fi
else
    say "Dashboard already built (dashboard/dist is newer than the sources)"
fi

# --- 3b. PO token provider (YouTube anti-bot) -------------------------------
# "Sign in to confirm you're not a bot" on EVERY strategy, from a clean Kaggle
# IP with no cookies (measured 6-aug-2026). YouTube increasingly requires a PO
# token to trust a client from a datacenter/cloud IP, and cookies are a poor
# answer here: they expire in ~3h and a STALE jar reads as more suspicious than
# an anonymous request.
#
# The Docker image already solves this (see the Dockerfile's bgutil block) —
# Kaggle simply never had it, so main.py's _pot_args was empty and every
# attempt went in bare. This is the same provider, in the HTTP server mode its
# own docs recommend over script-per-request.
#
# Entirely optional: every failure below leaves the pipeline exactly as it was,
# downloading without a token.
POT_DIR="${POT_DIR:-/kaggle/working/bgutil-provider}"
POT_PORT="${POT_PORT:-4416}"
if [ "${SKIP_POT:-0}" = "1" ]; then
    say "PO token provider (skipped: SKIP_POT=1)"
elif ! command -v npm >/dev/null 2>&1; then
    say "PO token provider"
    echo "    no npm — cannot build the provider; YouTube may hit the bot wall"
else
    say "YouTube PO token provider"
    # System deps for the 'canvas' npm package (native addon).
    # Without these, canvas fails to compile and the server crashes on startup.
    if command -v apt-get >/dev/null 2>&1; then
        apt-get install -y -qq libcairo2-dev libjpeg-dev libpango1.0-dev \
            libgif-dev build-essential g++ pkg-config > "$LOG_DIR/bgutil_deps.log" 2>&1 \
            && echo "    system deps for canvas: installed" \
            || echo "    system deps for canvas: install failed (continuing)"
    fi
    # yt-dlp nightly + the plugin that talks to the provider. The plugin is what
    # makes yt-dlp aware of the token at all.
    pip install -q --upgrade --pre "yt-dlp[default]" bgutil-ytdlp-pot-provider 2>&1 | tail -2 || \
        echo "    plugin install reported errors — continuing"
    if [ ! -f "$POT_DIR/server/build/main.js" ]; then
        echo "    building the provider (~1-2 min, first run only)"
        rm -rf "$POT_DIR"
        if git clone --depth 1 -q https://github.com/Brainicism/bgutil-ytdlp-pot-provider "$POT_DIR" \
            && (cd "$POT_DIR/server" && npm install --include=dev --ignore-scripts=false --no-audit --no-fund > "$LOG_DIR/bgutil_install.log" 2>&1) \
            && (cd "$POT_DIR/server" && npm rebuild >> "$LOG_DIR/bgutil_install.log" 2>&1 || true) \
            && (cd "$POT_DIR/server" && (./node_modules/.bin/tsc || npx --yes tsc) > "$LOG_DIR/bgutil_tsc.log" 2>&1); then
            echo "    provider built"
        else
            echo "    provider build FAILED — check $LOG_DIR/bgutil_install.log and $LOG_DIR/bgutil_tsc.log"
        fi
    else
        echo "    provider already built"
    fi

    # Pre-flight: verify the canvas native addon actually loads.
    # If it doesn't, the server will crash on startup.
    if [ -f "$POT_DIR/server/build/main.js" ]; then
        if ! (cd "$POT_DIR/server" && node -e "require('canvas')" 2>/dev/null); then
            echo "    canvas addon broken — rebuilding..."
            (cd "$POT_DIR/server" && npm rebuild canvas >> "$LOG_DIR/bgutil_install.log" 2>&1)
            if (cd "$POT_DIR/server" && node -e "require('canvas')" 2>/dev/null); then
                echo "    canvas fixed after rebuild"
            else
                echo "    canvas still broken — server will likely crash"
                echo "    see $LOG_DIR/bgutil_install.log for details"
            fi
        else
            echo "    canvas addon: ok"
        fi
    fi

    # Helper: check if the server is responding on any loopback address.
    # The server binds to [::] (IPv6 all-interfaces) which also accepts IPv4
    # on dual-stack systems, but some Kaggle instances might only have one.
    _pot_ping() {
        curl -s --max-time 3 "http://127.0.0.1:$POT_PORT/ping" >/dev/null 2>&1 && return 0
        curl -s --max-time 3 "http://[::1]:$POT_PORT/ping" >/dev/null 2>&1 && return 0
        return 1
    }

    if [ -f "$POT_DIR/server/build/main.js" ]; then
        if _pot_ping; then
            echo "    provider already running on :$POT_PORT"
        else
            setsid nohup node "$POT_DIR/server/build/main.js" --port "$POT_PORT" \
                < /dev/null > "$LOG_DIR/bgutil.log" 2>&1 &
            BGUTIL_PID=$!
            echo "    started provider (pid=$BGUTIL_PID), waiting for it..."
            for _ in $(seq 1 20); do
                _pot_ping && break
                # Check if process died
                if ! kill -0 "$BGUTIL_PID" 2>/dev/null; then
                    echo "    provider process died — check $LOG_DIR/bgutil.log:"
                    tail -5 "$LOG_DIR/bgutil.log" 2>/dev/null | sed 's/^/    /'
                    break
                fi
                sleep 1
            done
        fi
        if _pot_ping; then
            # Figure out which address actually responded for the URL
            if curl -s --max-time 2 "http://127.0.0.1:$POT_PORT/ping" >/dev/null 2>&1; then
                export BGUTIL_BASE_URL="http://127.0.0.1:$POT_PORT"
            else
                export BGUTIL_BASE_URL="http://[::1]:$POT_PORT"
            fi
            grep -q "BGUTIL_BASE_URL=" .env 2>/dev/null && sed -i "s|BGUTIL_BASE_URL=.*|BGUTIL_BASE_URL=\"$BGUTIL_BASE_URL\"|" .env || echo "BGUTIL_BASE_URL=\"$BGUTIL_BASE_URL\"" >> .env
            echo "    provider responding — BGUTIL_BASE_URL=$BGUTIL_BASE_URL"
        else
            echo "    provider did not come up (see $LOG_DIR/bgutil.log)"
            echo "    downloads will run without a PO token"
            [ -f "$LOG_DIR/bgutil.log" ] && echo "    last 5 lines of bgutil.log:" && tail -5 "$LOG_DIR/bgutil.log" | sed 's/^/      /'
        fi
    fi
fi

# --- 4. Secrets ------------------------------------------------------------
# On Kaggle these come from UserSecretsClient (see KAGGLE.md), exported into the
# environment before this script runs. Nothing is written to disk except the
# cookie jar, which yt-dlp needs as a file.
say "Configuration"
[ -n "${GEMINI_API_KEY:-}" ] && echo "    GEMINI_API_KEY: set" || echo "    GEMINI_API_KEY: MISSING (clip selection will fail)"
if [ -n "${GEMINI_API_KEYS:-}" ]; then
    echo "    GEMINI_API_KEYS: $(echo "$GEMINI_API_KEYS" | tr ',' '\n' | grep -c .) extra key(s) — pool rotates on 429/quota"
else
    echo "    GEMINI_API_KEYS: unset (single key; a quota hit stops the job)"
fi
# Stage 3 picker — the unified planner (picker.py). The pre-download context
# layer (context_layer.py) uses its OWN key so it never shares a rate budget
# with the picker; unset, it reuses GEMINI_API_KEY.
if [ -n "${CONTEXT_GEMINI_API_KEY:-}" ]; then
    echo "    CONTEXT_GEMINI_API_KEY: set (dedicated context-layer key)"
else
    echo "    CONTEXT_GEMINI_API_KEY: unset — the context layer will reuse GEMINI_API_KEY"
fi
echo "    Stage 3: picker.py (one pass, whole transcript, explicit clip count only)"
# Transcription backend. Setting ASSEMBLYAI_API_KEY alone does NOTHING —
# transcribe_backends.py:591 picks the backend from TRANSCRIBE_BACKEND, which
# defaults to "whisper". Measured on Kaggle 5-aug-2026: the local whisper path
# took 254s of a ~420s job AND produced no diarization, which costs
# the speaker-binding pipeline its diarized-tier evidence entirely (that run's
# framing line read "lip-sync 97%, directed 3%" with diarized at 0%). On
# multi-speaker footage that is the signal that stops the camera sitting on
# the wrong person.
if [ -n "${ASSEMBLYAI_API_KEY:-}" ]; then
    export TRANSCRIBE_BACKEND="${TRANSCRIBE_BACKEND:-assemblyai}"
    echo "    ASSEMBLYAI_API_KEY: set — TRANSCRIBE_BACKEND=$TRANSCRIBE_BACKEND (API + diarization)"
else
    echo "    ASSEMBLYAI_API_KEY: unset — local whisper (~254s measured, and NO diarization)"
fi
# Authenticates model downloads from the HuggingFace Hub. Only matters on the
# whisper fallback path, which fetches its model from HF; unauthenticated
# requests are rate-limited and slower.
[ -n "${HF_TOKEN:-}" ] && echo "    HF_TOKEN: set" || echo "    HF_TOKEN: unset (HF downloads rate-limited)"
# Persistent storage. /kaggle/working is wiped when the session ends (12h cap,
# and it dies sooner often enough), so without this every clip is lost.
if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_STORAGE_REPO:-}" ]; then
    echo "    storage: huggingface -> $HF_STORAGE_REPO (clips upload as they finish)"
else
    echo "    storage: NOT configured — clips are wiped when this session ends."
    echo "      Set HF_TOKEN (a *write* token) + HF_STORAGE_REPO=<user>/openshorts-clips."
fi
# Named-speaker enrichment. Dormant unless FACE_ID_DB points at a folder of
# {name}.jpg headshots. On a multi-GPU host it defaults to the LAST GPU so it
# does not compete with the render pipeline on GPU 0 (the integration guide's
# recommendation); FACE_ID_CTX overrides.
if [ -n "${FACE_ID_DB:-}" ]; then
    if [ -d "${FACE_ID_DB}" ]; then
        _faces=$(find "${FACE_ID_DB}" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l)
        echo "    face ID: ${FACE_ID_DB} (${_faces} headshot(s)), ctx=${FACE_ID_CTX:-auto}"
        [ "$_faces" -eq 0 ] && echo "      -> no images found; speakers will stay anonymous"
    else
        echo "    face ID: FACE_ID_DB=${FACE_ID_DB} is not a directory — enrichment stays off"
    fi
else
    echo "    face ID: off (set FACE_ID_DB to a folder of {name}.jpg headshots to name speakers)"
fi
# The single most common job-killer on a cloud IP, so it goes in the summary.
if [ -n "${BGUTIL_BASE_URL:-}" ]; then
    echo "    PO token: $BGUTIL_BASE_URL (YouTube anti-bot)"
else
    echo "    PO token: NOT available — 'Sign in to confirm you are not a bot' is likely."
    echo "      Either the provider failed to build above, or SKIP_POT=1."
fi
# Where burned-in captions sit: bottom (default), middle or top.
echo "    CAPTION_POSITION: ${CAPTION_POSITION:-bottom}"
if [ -n "${YOUTUBE_COOKIES:-}" ] && [ ! -s cookies.txt ]; then
    printf '%s' "$YOUTUBE_COOKIES" > cookies.txt
    echo "    cookies.txt: written from YOUTUBE_COOKIES ($(wc -c < cookies.txt) bytes)"
elif [ -s cookies.txt ]; then
    echo "    cookies.txt: present ($(wc -c < cookies.txt) bytes)"
else
    echo "    cookies.txt: MISSING — YouTube downloads will hit the bot wall"
fi

# GPU-appropriate defaults, mirroring docker-compose.gpu.yml.
if [ "$GPU_COUNT" -gt 0 ]; then
    export FFMPEG_ENCODER="${FFMPEG_ENCODER:-nvenc}"
    export WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
    export USE_ASD="${USE_ASD:-1}"
    echo "    encoder=nvenc  whisper_device=cuda  gpus=$GPU_COUNT"
    if python3 -c "import onnxruntime; print('CUDAExecutionProvider' in onnxruntime.get_available_providers())" 2>/dev/null | grep -q True; then
        echo "    onnxruntime: CUDAExecutionProvider ACTIVE (face spine on GPU)"
    else
        echo "    onnxruntime: CUDAExecutionProvider MISSING (face spine would run on CPU!)"
    fi
fi
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/output}"
mkdir -p "$OUTPUT_DIR" uploads

# --- 5. API + dashboard ----------------------------------------------------
say "Starting OpenShorts on :$PORT"
pkill -f "uvicorn app:app" 2>/dev/null || true
nohup python3 -m uvicorn app:app --host 0.0.0.0 --port "$PORT" \
    > "$LOG_DIR/backend.log" 2>&1 &
for _ in $(seq 1 60); do
    curl -sf "http://localhost:$PORT/api/system" >/dev/null 2>&1 && break
    sleep 2
done
if curl -sf "http://localhost:$PORT/api/system" >/dev/null 2>&1; then
    echo "    backend: up"
else
    echo "    backend: FAILED — last log lines:"; tail -20 "$LOG_DIR/backend.log"; exit 1
fi

# --- 6. Tunnel -------------------------------------------------------------
# Kaggle allows no inbound connections, so the UI is reached through an
# outbound tunnel. The quick tunnel needs no account; its hostname changes
# every session, which is the main ergonomic cost of this setup.
say "Public URL"
if ! command -v cloudflared >/dev/null 2>&1; then
    curl -sL -o /tmp/cloudflared \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x /tmp/cloudflared
    CF=/tmp/cloudflared
else
    CF=cloudflared
fi
pkill -f "cloudflared tunnel" 2>/dev/null || true
nohup "$CF" tunnel --url "http://localhost:$PORT" --no-autoupdate \
    > "$LOG_DIR/tunnel.log" 2>&1 &
URL=""
for _ in $(seq 1 45); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/tunnel.log" 2>/dev/null | head -1) || true
    [ -n "$URL" ] && break
    sleep 2
done
if [ -n "$URL" ]; then
    printf '\n    \033[1;32m%s\033[0m\n\n' "$URL"
    echo "    Open that in a browser. Logs: $LOG_DIR/{backend,tunnel}.log"
else
    echo "    tunnel did not report a URL — check $LOG_DIR/tunnel.log"
    tail -20 "$LOG_DIR/tunnel.log"
fi
