#!/bin/bash
# Keeps .env's YOUTUBE_COOKIES fresh automatically so a session doesn't need to
# manually re-run the refresh before every job. Purely a script — no LLM.
#
# Self-healing: the refresher talks to a persistent Chrome over CDP, and if that
# Chrome dies (VM hiccup, OOM, stray pkill) every subsequent refresh fails
# forever and cookies silently rot until a job hits the bot wall. So each cycle
# checks Chrome is alive and relaunches it if not. A failed refresh still leaves
# .env untouched — the existing cookies stay as the natural fallback.
cd /teamspace/studios/this_studio/openshorts

CHROME_BIN=/home/zeus/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome
PROFILE=/teamspace/studios/this_studio/.cookie_profile

ensure_chrome() {
  pgrep -f "remote-debugging-port=9223" > /dev/null && return 0
  echo "[$(date -Is)] chrome not running -> relaunching"
  pgrep -f "Xvfb :99" > /dev/null || { Xvfb :99 -screen 0 1280x800x24 > /tmp/xvfb.log 2>&1 & sleep 2; }
  # Stale single-instance locks from a previous VM stop Chrome from starting.
  rm -f "$PROFILE/SingletonLock" "$PROFILE/SingletonSocket" "$PROFILE/SingletonCookie"
  DISPLAY=:99 "$CHROME_BIN" \
    --user-data-dir="$PROFILE" \
    --remote-debugging-port=9223 --remote-debugging-address=0.0.0.0 \
    --no-sandbox --disable-dev-shm-usage --no-first-run \
    --no-default-browser-check --disable-fre \
    --window-size=1280,800 --window-position=0,0 --start-maximized \
    "https://www.youtube.com" > /tmp/chrome_cookie.log 2>&1 &
  disown
  sleep 5
}

while true; do
  ensure_chrome
  python3 refresh_youtube_cookies.py
  # Surface the resulting jar's health in the same log, so diagnosing "why did
  # downloads start failing" is one `tail` rather than an investigation.
  python3 cookie_health.py
  sleep 1200  # 20 minutes — cookies observed to survive roughly an hour
done
