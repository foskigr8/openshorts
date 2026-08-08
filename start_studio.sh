#!/bin/bash
# One-shot startup for the OpenShorts studio after a stop/restart — folds in
# every manual step from RUNBOOK.md sections 1-4 and 6.3 so this can be run
# directly in a terminal without asking an AI session to walk through it
# each time. Safe to re-run any time (idempotent): fixing already-correct
# permissions, re-registering already-registered ports, and starting an
# already-running cookie refresher are all no-ops here, not errors.
set -uo pipefail
cd /teamspace/studios/this_studio/openshorts

echo "==> Fixing uploads/output permissions (survives a VM restart resetting them)"
mkdir -p uploads
sudo setfacl -R -m u:999:rwx uploads output . 2>&1 | grep -v "^$" || true
sudo setfacl -R -d -m u:999:rwx uploads output . 2>&1 | grep -v "^$" || true

echo "==> Starting containers (docker compose up -d)"
if docker info 2>/dev/null | grep -q 'nvidia'; then
  echo "    NVIDIA runtime detected -> applying docker-compose.gpu.yml"
  COMPOSE_CMD=(docker compose -f docker-compose.yml -f docker-compose.gpu.yml)
else
  echo "    No NVIDIA runtime (CPU host) -> base compose only"
  COMPOSE_CMD=(docker compose)
fi
if docker image inspect openshorts-backend >/dev/null 2>&1; then
  # Warm host: images exist — plain up, no rebuild (fast restarts).
  "${COMPOSE_CMD[@]}" up -d
else
  # Fresh VM after a studio restart: images were wiped — build them.
  echo "    backend image missing -> building (first boot can take a while)"
  "${COMPOSE_CMD[@]}" up -d --build
fi

# First boot after a fresh studio start can take a few seconds for the
# backend to come up; a permission-reset crash-loop looks the same from
# `docker ps` (container just restarts immediately), so give it a moment
# and then restart once if it's not settled — the permission fix above
# already applied, so a restart is what actually clears the crash-loop.
echo "==> Waiting for backend to settle..."
sleep 5
if ! docker ps --format '{{.Names}} {{.Status}}' | grep -q '^openshorts-backend.*Up'; then
  echo "    backend not up yet, restarting once after the permission fix"
  docker restart openshorts-backend
  sleep 5
fi

echo "==> Container status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo "==> Registering Lightning ports (5175 dashboard, 8000 API, 3100 Remotion)"
python3 - <<'EOF'
from lightning_sdk import Studio
try:
    s = Studio()
    eps = s.add_ports([5175, 8000, 3100])
    for e in eps:
        print("   ", e.name, e.ports, e.urls)
except Exception as e:
    # "already exists" (409) just means it's already registered -- fine.
    msg = str(e)
    if "already exists" in msg or "409" in msg:
        print("    ports already registered, skipping")
    else:
        print("    port registration failed:", e)
EOF

echo "==> Persistent Chrome for cookie refresh (Xvfb + remote-debugging-port=9223)"
if pgrep -f "remote-debugging-port=9223" > /dev/null; then
  echo "    already running"
else
  # Needed by refresh_youtube_cookies.py over CDP. Two real failure modes
  # hit on a fresh VM, both handled here:
  # 1. libatk-bridge2.0-0 missing -> chrome fails to launch at all.
  # 2. .cookie_profile/SingletonLock left over from a PREVIOUS studio VM
  #    (different hostname/pid) -> chrome refuses to start, thinking
  #    another instance owns the profile. Safe to clear: that VM is gone.
  if ! pgrep -f "Xvfb :99" > /dev/null; then
    Xvfb :99 -screen 0 1280x800x24 > /tmp/xvfb.log 2>&1 &
    disown
    sleep 2
  fi
  if ! ldconfig -p | grep -q libatk-bridge-2.0.so.0; then
    sudo apt-get install -y libatk-bridge2.0-0 > /tmp/apt_atk.log 2>&1
  fi
  rm -f /teamspace/studios/this_studio/.cookie_profile/SingletonLock \
        /teamspace/studios/this_studio/.cookie_profile/SingletonSocket \
        /teamspace/studios/this_studio/.cookie_profile/SingletonCookie
  DISPLAY=:99 /home/zeus/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome \
    --user-data-dir=/teamspace/studios/this_studio/.cookie_profile \
    --remote-debugging-port=9223 --remote-debugging-address=0.0.0.0 \
    --no-sandbox --disable-dev-shm-usage --no-first-run \
    --no-default-browser-check --disable-fre \
    --window-size=1280,800 --window-position=0,0 --start-maximized \
    "https://accounts.google.com/ServiceLogin?service=youtube" \
    > /tmp/chrome_cookie.log 2>&1 &
  disown
  sleep 3
  if pgrep -f "remote-debugging-port=9223" > /dev/null; then
    echo "    started"
  else
    echo "    FAILED -- check /tmp/chrome_cookie.log"
  fi
fi

echo "==> Cookie auto-refresh loop (YouTube cookies rotate roughly hourly)"
if pgrep -f auto_refresh_cookies.sh > /dev/null; then
  echo "    already running"
else
  nohup /teamspace/studios/this_studio/openshorts/auto_refresh_cookies.sh > /dev/null 2>&1 &
  disown
  echo "    started"
fi

# ---------------------------------------------------------------------------
# Cookie preflight. This is the step that makes "Sign in to confirm you're not
# a bot" a solved problem instead of a recurring one. Starting the refresh loop
# is NOT the same as having working cookies: if the persistent Chrome profile
# has been logged out by Google, every refresh fails and .env silently keeps a
# stale/half jar. So: refresh once synchronously, then VERIFY, and if it's not
# usable, bring up the noVNC bridge and say exactly what to do.
# ---------------------------------------------------------------------------
echo "==> Verifying YouTube cookies"
python3 refresh_youtube_cookies.py 2>&1 | sed 's/^/    /' || true

if python3 cookie_health.py 2>&1 | sed 's/^/    /'; then
  echo "    cookies are complete — YouTube downloads should work."
else
  echo ""
  echo "    ================================================================"
  echo "    ACTION NEEDED: the persistent Chrome profile is not logged in to"
  echo "    Google, so YouTube downloads WILL fail with the bot check."
  echo "    ================================================================"
  # Bring up the VNC bridge automatically so logging in is a click, not a
  # scavenger hunt through RUNBOOK.md.
  if ! pgrep -f "x11vnc -display :99" > /dev/null; then
    x11vnc -display :99 -forever -shared -nopw -rfbport 5900 > /tmp/x11vnc.log 2>&1 &
    disown
    sleep 1
  fi
  if ! pgrep -f "websockify --web=/usr/share/novnc" > /dev/null; then
    python3 /usr/bin/websockify --web=/usr/share/novnc 6080 localhost:5900 > /tmp/novnc.log 2>&1 &
    disown
    sleep 1
  fi
  python3 - <<'EOF' || true
from lightning_sdk import Studio
try:
    s = Studio()
    for e in s.add_ports([6080]):
        for u in (e.urls or []):
            print(f"    -> open {u}/vnc.html?autoconnect=true&resize=scale")
except Exception as exc:
    msg = str(exc)
    if "already exists" in msg or "409" in msg:
        print("    -> noVNC port 6080 already registered; open its URL "
              "with /vnc.html?autoconnect=true&resize=scale")
    else:
        print("    -> could not register noVNC port:", exc)
EOF
  echo "    Log into the Google account in that browser window, then re-run"
  echo "    this script. Nothing else is needed — the loop takes over after."
fi

# ---------------------------------------------------------------------------
# Pre-warm the voice-separation model. The weights (~80MB) are fetched from
# Hugging Face on first use and cached under ./.cache/huggingface, which lives
# on the bind mount and therefore survives container recreation. Without this,
# the very first job that enables "remove background audio" pays the download
# mid-render — and fails that clip outright if HF is rate-limiting. Warming it
# here means the feature is ready, not dormant.
# ---------------------------------------------------------------------------
echo "==> Pre-warming the background-audio (voice separation) model"
docker exec openshorts-backend python3 -c "
import sys
try:
    import audio_cleanup
    if not audio_cleanup.demucs_available():
        print('    demucs not installed -> the FFmpeg denoise fallback will be used')
        sys.exit(0)
    from demucs.pretrained import get_model
    get_model('htdemucs')
    print('    voice-separation model ready (music removal available)')
except Exception as e:
    print('    pre-warm failed (%s); first use will fetch it on demand' % e)
" 2>&1 | grep -v "^Warning: You are sending unauthenticated" || \
  echo "    backend not reachable — skipped"

echo ""
echo "==> Health summary"
docker ps --format '{{.Names}}' | grep -q openshorts-backend \
  && echo "    backend      : up" || echo "    backend      : DOWN"
docker ps --format '{{.Names}}' | grep -q openshorts-bgutil-pot \
  && echo "    po-token     : up" || echo "    po-token     : DOWN (bot checks more likely)"
pgrep -f "remote-debugging-port=9223" > /dev/null \
  && echo "    cookie chrome: up" || echo "    cookie chrome: DOWN"
pgrep -f auto_refresh_cookies.sh > /dev/null \
  && echo "    refresh loop : up" || echo "    refresh loop : DOWN"
python3 cookie_health.py > /dev/null 2>&1 \
  && echo "    cookies      : complete" || echo "    cookies      : INCOMPLETE (see above)"
docker exec openshorts-backend python3 -c "import audio_cleanup,sys; sys.exit(0 if audio_cleanup.demucs_available() else 1)" 2>/dev/null \
  && echo "    music removal: ready (demucs)" || echo "    music removal: fallback only (denoise, no music removal)"

echo ""
echo "==> Dashboard: https://5175-01kyqmvb3hgqwxbq17mxew92x4.cloudspaces.litng.ai"
echo "==> If that 404s, the studio's public URL changed -- re-run this script,"
echo "    or check the printed port URLs above for the new one."
