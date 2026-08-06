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

# --- 2. Python deps --------------------------------------------------------
# Kaggle's base image already carries torch/opencv/numpy built against its own
# CUDA. Reinstalling those from requirements.txt is slow and can break CUDA, so
# they are held back and only the packages Kaggle lacks are installed.
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    say "Python dependencies (2-5 min)"
    grep -vE '^(torch|torchvision|torchaudio|opencv|numpy|scipy)([=<>~!]|$)' \
        requirements.txt > /tmp/req-kaggle.txt || cp requirements.txt /tmp/req-kaggle.txt
    # Those lines are stripped because Kaggle's image already has them built
    # against its own CUDA — but stripping them also removes requirements.txt's
    # `numpy<2` GUARD, and a TRANSITIVE dependency can still drag numpy 2.x in.
    # Measured 6-aug-2026: insightface unconstrained resolves to numpy 2.5.1 +
    # opencv-python-headless 5.x, which breaks mediapipe, ultralytics and the
    # rest of Kaggle's preinstalled stack. So pin the held-back packages to the
    # versions ALREADY on this host and make pip resolve around them.
    python3 - <<'PYPIN' > /tmp/constraints-kaggle.txt || true
for name, mod in (("numpy", "numpy"), ("scipy", "scipy"),
                  ("opencv-python", "cv2"), ("pandas", "pandas")):
    try:
        m = __import__(mod)
        print(f"{name}=={m.__version__}")
    except Exception:
        pass
PYPIN
    echo "    holding these at the versions Kaggle already ships:"
    sed 's/^/      /' /tmp/constraints-kaggle.txt || true
    pip install -q -c /tmp/constraints-kaggle.txt -r /tmp/req-kaggle.txt 2>&1 | tail -5 || {
        echo "    pip install reported errors — continuing, but expect import failures"; }
    # protobuf conflict repair.
    #
    # Kaggle's image ships TensorFlow whose generated *_pb2.py files require
    # protobuf >= 5.27 (that is when `runtime_version` appeared). mediapipe
    # pins protobuf < 5, so installing it downgrades protobuf and BREAKS
    # TensorFlow. mediapipe then imports tasks.python -> tensorflow ->
    # ImportError: cannot import name 'runtime_version' from 'google.protobuf'
    # — which is what killed the first Kaggle job (5-aug-2026).
    #
    # The two packages cannot both be satisfied, so one has to go. This
    # pipeline does not use TensorFlow at all (torch does the ML work), and
    # mediapipe only touches it for doc annotations, so removing TF is the
    # cheap resolution. Remedies are applied in order and IMPORT IS RETESTED
    # after each, because the right fix depends on the image version and
    # guessing a pin from outside Kaggle is how this broke in the first place.
    # NOTE: every command in this block is guarded with `|| true`.
    # The script runs under `set -euo pipefail`, and the whole point here is to
    # run commands that FAIL — probing a broken import, uninstalling a package
    # that may not be present. Without the guards, `set -e` kills the script on
    # the first probe and the repair silently never runs, which is exactly what
    # happened on the first attempt (5-aug-2026): output stopped dead after
    # "attempting repair" with no error shown.
    # The guard is only worth having if it is checked. A numpy 2.x bump breaks
    # mediapipe/ultralytics in ways that surface much later as a failed render.
    _np=$(python3 -c "import numpy;print(numpy.__version__)" 2>/dev/null || echo "missing")
    case "$_np" in
        1.*) echo "    numpy $_np — guard held" ;;
        *)   echo "    !! numpy is $_np — something upgraded it past the <2 guard."
             echo "       mediapipe/ultralytics will misbehave. Fix before rendering:"
             echo "       pip install 'numpy<2'" ;;
    esac

    say "Checking mediapipe imports"
    _mp_ok() { python3 -c "import mediapipe" >/dev/null 2>&1; }
    if ! _mp_ok; then
        echo "    mediapipe import failed — attempting repair"
        _err=$(python3 -c "import mediapipe" 2>&1 | tail -1) || true
        echo "    $_err"
        if echo "$_err" | grep -q "runtime_version\|protobuf" 2>/dev/null; then
            echo "    remedy 1/2: removing tensorflow (unused by this pipeline)"
            pip uninstall -y -q tensorflow tensorflow-cpu tensorflow-gpu 2>/dev/null || true
        fi
        if ! _mp_ok; then
            echo "    remedy 2/2: reinstalling mediapipe against the current protobuf"
            pip install -q --force-reinstall --no-deps mediapipe 2>&1 | tail -2 || true
            pip install -q "protobuf<5" 2>&1 | tail -2 || true
        fi
        if _mp_ok; then
            echo "    repaired: mediapipe imports"
        else
            echo "    STILL BROKEN — face detection will not work. Last error:"
            python3 -c "import mediapipe" 2>&1 | tail -3 || true
        fi
    else
        echo "    mediapipe imports"
    fi

    # Face ID needs an ONNX runtime to execute; requirements.txt deliberately
    # ships only insightface (onnxruntime-gpu is ~2GB and lives in the
    # Dockerfile's GPU block so the CPU image stays slim). Kaggle IS a GPU host,
    # so install it here — and only if it is not already importable, because
    # Kaggle images sometimes carry onnxruntime already and installing both the
    # CPU and GPU wheels puts two copies of the same module on the path.
    say "Face ID runtime (onnxruntime)"
    if python3 -c "import onnxruntime" >/dev/null 2>&1; then
        echo "    onnxruntime already present ($(python3 -c 'import onnxruntime;print(onnxruntime.__version__)' 2>/dev/null))"
    else
        pip install -q -c /tmp/constraints-kaggle.txt onnxruntime-gpu 2>&1 | tail -3 || \
            echo "    onnxruntime-gpu install reported errors — face ID will stay off"
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
for m in ("fastapi", "uvicorn", "yt_dlp", "mediapipe", "ultralytics", "torch", "cv2"):
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
# Stage 3 engine. The viral-clip-finder engine ships its methodology as
# markdown files next to the code; if the clone is missing them (a partial
# checkout, or a .gitignore that swallowed the folder) the engine reports
# itself unavailable and the job quietly drops back to the older narrative
# path — a silent quality regression that looks like a working run. Say so
# here instead, while someone is still reading the output.
VIRAL_ENGINE="${VIRAL_ENGINE:-auto}"
export VIRAL_ENGINE
echo "    VIRAL_ENGINE: $VIRAL_ENGINE"
if [ -f viral_clip_finder_skill/SKILL.md ]; then
    echo "    viral-clip-finder skill: present ($(ls viral_clip_finder_skill/references/*.md 2>/dev/null | wc -l) reference docs)"
else
    echo "    viral-clip-finder skill: MISSING viral_clip_finder_skill/SKILL.md"
    if [ "$VIRAL_ENGINE" = "skill" ]; then
        echo "    -> VIRAL_ENGINE=skill cannot run without it. Jobs will fail."
    else
        echo "    -> Stage 3 will fall back to the narrative engine (older, weaker selection)."
    fi
fi
# The narrative engine is Gemini-only via its OWN key and has no fallback to
# GEMINI_API_KEY (deepseek_worker._narrative_provider_candidates). Unset, it
# contributes nothing — worth knowing, because "auto" then has no safety net
# under the skill engine.
if [ -n "${NARRATIVE_GEMINI_API_KEY:-}" ]; then
    echo "    NARRATIVE_GEMINI_API_KEY: set (narrative fallback available)"
else
    echo "    NARRATIVE_GEMINI_API_KEY: unset — the narrative fallback is inert;"
    echo "      the skill engine runs on GEMINI_API_KEY and is the only Stage 3 path"
fi
# Transcription backend. Setting ASSEMBLYAI_API_KEY alone does NOTHING —
# transcribe_backends.py:591 picks the backend from TRANSCRIBE_BACKEND, which
# defaults to "whisper". Measured on Kaggle 5-aug-2026: the local whisper path
# took 254s of a ~420s job AND produced no diarization, which costs
# subject_policy its TIER_DIARIZED evidence entirely (that run's framing line
# read "lip-sync 97%, directed 3%" with diarized at 0%). On multi-speaker
# footage that is the signal that stops the camera sitting on the wrong person.
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
    export YOLO_DEVICE="${YOLO_DEVICE:-0}"
    export USE_ASD="${USE_ASD:-1}"
    echo "    encoder=nvenc  yolo_device=0  gpus=$GPU_COUNT"
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
