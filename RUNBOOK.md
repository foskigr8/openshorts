# Running OpenShorts in this studio (self-serve runbook)

> **OBSOLETE in part (8-aug-2026):** the caption/reframe notes referencing
> `reframe_v2` predate the v3-only cleanup — `reframe_v3` is the only
> reframe engine. The run/restart steps are still current.

Quick reference for starting/restarting the app without help. Everything here
runs from a terminal in this workspace, inside `/teamspace/studios/this_studio/openshorts`.

## 1. Start the app

```bash
cd /teamspace/studios/this_studio/openshorts
docker compose up -d
```

`-d` runs it in the background so the terminal stays free. First run after a
fresh studio boot will rebuild images (slow, several minutes); after that it's
fast (seconds).

Check it's actually up:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

You want to see `openshorts-backend`, `openshorts-frontend`, and
`openshorts-renderer` all `Up`.

## 2. If the backend crash-loops with a `PermissionError: 'uploads'` (or similar)

This happens whenever the studio's underlying VM restarts — it resets file
ownership on the `uploads/`, `output/`, `cookies.txt` files, which breaks the
container's non-root user's ability to write there. Fix:

```bash
cd /teamspace/studios/this_studio/openshorts
mkdir -p uploads
sudo setfacl -R -m u:999:rwx uploads output .
sudo setfacl -R -d -m u:999:rwx uploads output .
docker restart openshorts-backend
```

(`setfacl` needs the `acl` package — if the command isn't found, run
`sudo apt-get install -y acl` first.)

## 3. Open the dashboard in your browser

This studio's containers aren't reachable at plain `localhost` from your
browser — they need to be registered as exposed ports with Lightning, and
then opened through Lightning's proxy URL.

**If the ports are already registered** (they should stay registered across
container restarts, since registration is by port number): just go to

```
https://5175-01kyqmvb3hgqwxbq17mxew92x4.cloudspaces.litng.ai
```

**If that 404s / doesn't load** (e.g. a brand new studio, or ports somehow
got cleared), re-register them from a Python shell in this workspace:

```bash
python3 -c "
from lightning_sdk import Studio
s = Studio()
eps = s.add_ports([5175, 8000, 3100])
for e in eps:
    print(e.name, e.ports, e.urls)
"
```

That prints the URLs to use — port 5175 is the dashboard, 8000 is the raw
API, 3100 is the Remotion render service (rarely needed directly).

One gotcha we hit once: Vite's dev server rejects unrecognized hostnames by
default. If you ever see "Blocked request... not allowed" in the browser
instead of the app, it means `dashboard/vite.config.js`'s `allowedHosts`
list needs the Lightning domain in it — it's already there
(`.cloudspaces.litng.ai`), so this shouldn't recur unless that file gets
reverted.

## 4. Stop everything

```bash
docker compose down
```

## 5. Rebuild after code changes

If you (or I) edit backend Python files, no rebuild is needed — the backend
container bind-mounts the repo live, just restart it:

```bash
docker restart openshorts-backend
```

Frontend (`dashboard/`) also bind-mounts and hot-reloads via Vite — usually
no restart needed at all, just refresh the browser tab.

If `requirements.txt`, `package.json`, or the `Dockerfile`s themselves
change, you do need a real rebuild:

```bash
docker compose up -d --build
```

## 6. Resuming after a full studio stop/restart (checklist)

Everything in `/teamspace/studios/this_studio` (this repo, `.env`, the
`.cookie_profile/` browser profile) is on persistent disk and survives a
studio stop. Docker containers, and any plain background process (Xvfb,
Chrome, x11vnc, websockify), do NOT — they need to be started again. In
order:

1. Fix permissions (see §2), then `docker compose up -d`.
2. Re-register Lightning ports if the dashboard 404s (see §3) — also add
   `6080` to the port list if you need the browser/noVNC bridge below.
3. **YouTube cookies are the flakiest part.** They rotate/expire within
   roughly an hour of being written, sometimes sooner, regardless of source
   — this is a Google security behavior tied to network/IP mismatch, not a
   bug in this repo. **Auto-refresh now runs on its own** (31-jul-2026,
   pure shell script, zero LLM/token cost per refresh — see
   `auto_refresh_cookies.sh`, launched with `nohup ... & disown` so it
   survives the invoking shell exiting):
   ```bash
   nohup /teamspace/studios/this_studio/openshorts/auto_refresh_cookies.sh > /dev/null 2>&1 &
   disown
   ```
   Re-run this after every studio restart (it does NOT survive a stop —
   check first with `ps aux | grep auto_refresh_cookies`). It loops
   `refresh_youtube_cookies.py` every 20 minutes, logging to
   `/tmp/auto_refresh_cookies.log`; a failed refresh (browser not logged
   in) just leaves `.env`'s existing cookies in place — no special handling
   needed, that's the natural fallback. `main.py` runs as a fresh
   subprocess per job and calls `load_dotenv()` at import, so it always
   picks up whatever's currently on disk — no cache-staleness concern.
   This talks to the persistent Chrome instance below over CDP — it must be
   running first. If the log shows "No logged-in session detected," the
   browser itself needs restarting (and possibly a re-login) per step 4.
4. **Persistent same-IP browser for cookie automation** (only needed if step
   3's Chrome isn't already running — check with
   `ps aux | grep remote-debugging-port=9223`):
   ```bash
   Xvfb :99 -screen 0 1280x800x24 &
   DISPLAY=:99 /home/zeus/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome \
     --user-data-dir=/teamspace/studios/this_studio/.cookie_profile \
     --remote-debugging-port=9223 --remote-debugging-address=0.0.0.0 \
     --no-sandbox --disable-dev-shm-usage --no-first-run \
     --no-default-browser-check --disable-fre \
     --window-size=1280,800 --window-position=0,0 --start-maximized \
     https://accounts.google.com/ServiceLogin?service=youtube &
   x11vnc -display :99 -forever -shared -nopw -rfbport 5900 &
   python3 /usr/bin/websockify --web=/usr/share/novnc 6080 localhost:5900 &
   ```
   The `.cookie_profile` directory persists on disk, so the Google login
   should already be there — no re-login needed unless Google force-logged
   it out. If a login IS needed, open the noVNC URL (port 6080, registered
   per step 2) at `/vnc.html?autoconnect=true&resize=scale` and log in once
   through the actual browser GUI.
5. **`video-analyzer` Gemini-vision skill** — installed at
   `.claude/skills/video-analyzer/` (repo-adjacent, persists). Needs
   `google-genai` importable by the host's `python3` (the conda `cloudspace`
   env) and `GEMINI_API_KEY` exported in the shell that invokes it — if a
   fresh studio boot doesn't have the package, `pip install google-genai
   --break-system-packages`.
6. **Server-side fallback API keys** for self-testing without the browser
   (Gemini/DeepSeek/AssemblyAI) already live in `.env` — BYOK headers from
   the dashboard still take priority, this is only a fallback so pipeline
   runs can be triggered directly (`docker exec ... python3 main.py -u
   <url> ...`) without a browser session.

## 7. Session status (31-jul-2026) — what's been fixed vs. still open

Built across this session (spans two studio-stop pauses — see git-free
history in this file since there's no git repo here): AssemblyAI
transcription + DeepSeek narrative-arc clip selection (now `deepseek-v4-pro`,
was `deepseek-v4-flash`) + Gemini Vision confirmation pipeline, audio-informed
speaker-camera tracking, automated YouTube cookie refresh (§6.3-4), and the
`video-analyzer` skill for self-verifying rendered output by actually
watching it — the single most load-bearing workflow change this session.
**Whenever the user says "analyze the video," use that skill with a pointed
critique prompt (hook/framing/captions/content), not its generic summary
prompt** — the generic prompt has repeatedly rated broken output as fine.

**Fixed and verified this session** (each confirmed via a real pipeline run
+ independent video-analyzer critique, not just unit tests):
- AssemblyAI single-utterance segmentation collapse.
- Vision-rejection fallback that used to ship the rejected clip anyway —
  now fails closed.
- DeepSeek only ever proposing one candidate — prompt now asks for 2-5.
- A prompt-template bug (unescaped `{...}` in prose) that would have
  crashed every DeepSeek call — caught immediately by the test suite.
- **Ad/sponsor-segment exclusion**: DeepSeek picked a sponsor-read/survey-
  scam segment as its best candidate because ad copy scores well on pure
  hook/payoff structure. Added an AD/SPONSOR SEGMENT EXCLUSION rule (judges
  by subject matter — is this about transacting with a third party — not
  tone) that runs before any other selection rule. Confirmed working on a
  second real video with an actual embedded "make $750 fast" segment.
- **Hook rule, twice-revised**: v1 required a hook to always be self-
  contained ("Get 'em, boy!" fails). v2 added a deterministic code backstop
  (`main._extend_start_for_preceding_question`) for the specific "reply to
  an unheard question" pattern, since the LLM didn't reliably self-check
  even with the rule in the prompt. **v3 (current, most important
  revision)**: the real test isn't "does it need the line before it," it's
  "is the opening line a SELF-CONTAINED CLAIM (fine to open cold) vs. a bare
  REPLY FRAGMENT (never fine alone)." A user-supplied reference clip showed
  professionally-edited clips in this exact genre routinely open cold on a
  claim with zero setup — the old rule was overcorrecting. Both
  `deepseek_worker.py`'s REAL HOOK RULE and `gemini_worker.py`'s vision
  hook-check were rewritten to this framework and made consistent with each
  other (they'd drifted — the Gemini side still cited an example the
  DeepSeek side now treats as a GOOD hook).
- **Bounded retry on vision rejection**: the context/hook check's own
  contract has it return `approved: false` alongside a fix-suggesting delta
  (not only when unfixable) — but the code applied the delta and dropped
  the clip anyway without ever checking whether the fix worked. Real
  candidates that WERE successfully rescued were still logged "still
  rejected." Added one bounded retry (`main._context_check_once` +
  retry logic in `confirm_clip_with_vision`) — re-checks only after
  boundaries actually moved, capped at one retry, so cost stays controlled.
- Wrong reframe mode for small reaction-cam face bubbles (1 face, tiny
  relative to frame) — now correctly routes to GENERAL instead of TRACK.
  (A SEPARATE, still-open framing issue remains for genuine group scenes —
  see below.)
- Vision rejection reasons logged instead of discarded, plus a
  `🔧 Vision adjusted [...] -> [...]` / `🔁 Retry context check...` log
  trail so debugging doesn't require re-deriving from raw transcript grep.
- **Gemini PROHIBITED_CONTENT false-positives**: this pipeline's actual
  content (mainstream reality/dating-show footage) tripped Gemini's DEFAULT
  safety thresholds outright, silently dropping that candidate's context
  check. Added `gemini_worker.RELAXED_SAFETY_SETTINGS` (BLOCK_ONLY_HIGH on
  sexual/harassment/hate/dangerous categories — Google's absolute policy
  floor for actually-prohibited content, e.g. CSAM, can't be configured
  off and still applies) to every Gemini call in the pipeline.
- **Caption "stacking" bug** (user-reported, confirmed via screenshot):
  captions used a fixed bottom-of-frame margin regardless of reframe mode.
  During a GENERAL-layout scene (content shrunk to ~42% height, vertically
  centered, blurred fill above/below — see `reframe_v2.general_filtergraph`)
  the caption floated in the blur band well below the actual content,
  reading as a second stacked panel — especially jarring right after a
  TRACK scene (full-height, caption looks normal) in the same clip.
  `reframe_v2.render()` now returns clip-relative GENERAL time-ranges
  alongside its success bool (**breaking return-shape change**: `render()`,
  `render_clip()`, and `process_video_to_vertical()` all now return
  `(success, general_ranges)` tuples, not a bare bool — check any new
  caller). `subtitles.generate_ass()` takes an optional `general_ranges`
  param and gives lines inside those ranges a per-event MarginV override
  landing them in the bottom slice of the actual (smaller, centered)
  content box instead of the empty space below it.

**Verified end-to-end on real content**:
- DeepSeek term-correction pass — fires reliably on real audio (e.g.
  "Brent Faiyaz" mis-heard as "Brent Faiyad", "Pledge of Allegiance" as
  "presidigence"), applied to both segment text and burned captions.
- Ad-exclusion, hook rule v2, and reframe small-face fix all confirmed on
  a second real video (`test_fixtures/pop_balloon_v2.mp4`, kept on disk —
  reuse via `-i` instead of re-downloading to dodge cookie rotation).

**Not yet re-verified against real content** (implemented + unit-tested,
correct by inspection, blocked on a studio pause mid-run):
- Hook rule v3 (self-contained-claim framework) and the caption GENERAL-
  margin fix were both built in the same push that was running when this
  pause happened — a pipeline run was in progress (3 candidates approved,
  clip 1 fully rendered+captioned, clips 2-3 still reframing) but never
  independently critiqued via video-analyzer. **Next step on resume**: let
  that run finish (or re-run — source is `test_fixtures/pop_balloon_v2.mp4`,
  no re-download needed) and check the delivered clips for (a) whether the
  hook now reads as strong per the v3 framework, (b) whether any GENERAL-
  mode caption in the output actually sits inside the content box now.

**Scoped but not started** — the "editing style" gap, from a user-supplied
reference clip analyzed via video-analyzer (31-jul-2026): the reference is a
heavily hand-composited edit (external reaction-meme inserts, B-roll
cutaways, mid-sentence jump cuts removing internal dead air — not just
boundary trimming, punch-in zoom on dramatic lines, sound effects). User's
explicit scope call: **build mid-sentence jump-cutting (remove internal
dead air within a clip, not just its start/end) and punch-zoom next; reaction-
meme/B-roll compositing and sound effects are explicitly deferred** ("we are
not doing reaction meme b-rolls just yet... sound effect much later, we need
to get the clip right first"). Neither jump-cutting nor punch-zoom has been
started.

**Still fully unbuilt**: the dynamic split-screen/solo-shot camera-switching
feature (3-rule spec, discussed but deferred pending the above) — my
recommendation when raised: build v1 on AssemblyAI speaker-overlap detection
(cheap, already-available signal) rather than true visual reaction detection
(needs face-mesh landmarks we don't have) as a first pass.
