#!/usr/bin/env bash
# Render the dashboard headlessly and save screenshots.
#
# Why this exists: every layout regression in this app so far — the mangled
# clip strip, the 800px-tall preview, the workspace that did not fit a 1080p
# laptop — passed `npm run build` and `npm run lint` without complaint. Neither
# of those tools can see. This one can.
#
#   ./scripts/shoot.sh                  # home, finished job, running job
#   ./scripts/shoot.sh 1440 900         # at a different viewport
#
# Output: dashboard/.shots/*.png
#
# The states are seeded through localStorage (the same key ProjectContext
# persists to), so no backend is needed — videos will show as unavailable,
# which is fine: this is about layout, not playback.
set -euo pipefail
cd "$(dirname "$0")/.."

W="${1:-1920}"
H="${2:-900}"          # 900 ≈ a 1080p laptop once browser chrome is gone
PORT=5199
OUT=".shots"
CHROME="$(command -v google-chrome || command -v chromium || command -v chromium-browser)"
[ -n "$CHROME" ] || { echo "no chrome/chromium on PATH"; exit 1; }

npm run build >/dev/null
mkdir -p "$OUT"

seed() {   # $1 = filename, $2 = project JSON
  cat > "dist/$1" <<EOF
<!doctype html><meta charset="utf-8"><body><script>
localStorage.setItem('os_projects_v2', JSON.stringify({demo: $2}));
localStorage.setItem('os_projects_v2:active', 'demo');
location.replace('/');
</script></body>
EOF
}

CLIPS='[{"start":30,"end":60,"video_url":"/videos/demo/c1.mp4","video_title_for_youtube_short":"Worst Dating Show Excuse Ever"},
        {"start":90,"end":120,"video_url":"/videos/demo/c2.mp4","video_title_for_youtube_short":"The Gaslighting Masterclass Nobody Asked For"},
        {"start":150,"end":180,"video_url":"/videos/demo/c3.mp4","video_title_for_youtube_short":"The Virgin Contestant Bizarre Lies"},
        {"start":210,"end":240,"video_url":"/videos/demo/c4.mp4","video_title_for_youtube_short":"He Said WHAT About Her Shoes"}]'

seed seed-done.html "{\"id\":\"demo\",\"title\":\"https://youtu.be/demo\",\"status\":\"complete\",
  \"source\":{\"type\":\"url\",\"payload\":\"https://youtu.be/demo\"},
  \"progress\":{\"stage\":\"finalize\",\"overall_pct\":100,\"clips_done\":4,\"clips_total\":4},
  \"results\":{\"clips\":$CLIPS},\"stageDurations\":{\"download\":42,\"transcribe\":88,\"analyze\":31,\"render\":210},
  \"requestedClipCount\":4,\"createdAt\":0}"

seed seed-run.html "{\"id\":\"demo\",\"title\":\"https://youtu.be/demo\",\"status\":\"processing\",
  \"source\":{\"type\":\"url\",\"payload\":\"https://youtu.be/demo\"},
  \"progress\":{\"stage\":\"render\",\"overall_pct\":62,\"clips_done\":2,\"clips_total\":6,
               \"step\":\"rendering clip 3/6\",\"step_pct\":44,\"eta_seconds\":412},
  \"results\":{\"clips\":$CLIPS},\"stageDurations\":{\"download\":42,\"transcribe\":88,\"analyze\":31},
  \"requestedClipCount\":6,\"createdAt\":0}"

npx vite preview --port "$PORT" --strictPort >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 20); do
  curl -sf -o /dev/null "http://localhost:$PORT/" && break || sleep 0.5
done

shot() {
  "$CHROME" --headless=new --disable-gpu --no-sandbox --hide-scrollbars \
    --window-size="$W,$H" --virtual-time-budget=9000 \
    --screenshot="$OUT/$1.png" "http://localhost:$PORT/$2" 2>/dev/null || true
  echo "  $OUT/$1.png"
}

echo "shooting at ${W}x${H}:"
shot home ""
shot done seed-done.html
shot running seed-run.html
