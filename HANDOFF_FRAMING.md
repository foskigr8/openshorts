# Task: finish the OpenShorts framing rebuild and validate it on Pop The Balloon

You are continuing work in `/teamspace/studios/this_studio/openshorts`.
Branch `session/framing-work`, HEAD `0b55031`. `main` is clean at `be4dd06`.
**Do not roll back to main.** The owner's instruction was: no going back, make it better.

Read this whole document before running anything. It contains measurements that
took hours to obtain — do not re-derive them, and do not contradict them without
new evidence.

---

## 0. How to run anything (get this right first)

`/app` inside the container is **NOT a bind mount**. Editing a file on the host
does nothing until you copy it in:

```bash
docker cp reframe_v2.py openshorts-backend:/app/
docker cp subject_policy.py openshorts-backend:/app/
docker cp tests/test_subject_policy.py openshorts-backend:/app/tests/
docker exec openshorts-backend sh -c 'cd /app && python3 -m pytest tests/ -q'
```

Baseline: **650 tests pass.** Any drop is a regression you caused.

Containers: `openshorts-backend`, `openshorts-frontend` (port 5175),
`openshorts-bgutil-pot`. UI history API: `curl -s http://localhost:8000/api/history`.

---

## 1. What was rebuilt, and the evidence behind it

The camera framed whoever had the biggest face — on a hosted show, the person
holding the microphone. Every previous attempt tuned boost constants and
measured no change, because the mechanism, not the tuning, was wrong.

Old design: candidate score = face AREA, multiplied by each signal (lip-sync,
diarization, mouth motion, directive), summed per identity with 0.85 decay,
plus a 1.6-2.2x sticky bonus for the incumbent.

Measured on a 20s / 9-shot / 121-detection-frame segment:

| experiment | speaker framed |
|---|---|
| baseline | 61% |
| lip-sync boost 4x -> **1000x** (250x!) | 70% |
| boost 4x, all hysteresis removed | **97%** |

Conclusions, all measured:
- **The boost knobs were never the lever.** A decaying sum reaches ~6.7x its
  input; a merely 2x-bigger incumbent is unreachable by any single-frame boost.
- **Hysteresis damped signal and noise equally.** It cost 27 points of accuracy.
- **Fighting a bad signal caused the jitter it was meant to prevent.** Simulated
  over the same frames, size-scoring produced 15 camera jumps at 44.9% accuracy;
  following lip-sync produced **9** — fewer — at 100%. The source has 9 shots.

### The replacement: `subject_policy.py` (new, pure, fully unit-tested)

Evidence tiers, strongest first, with hysteresis proportional to weakness:

```
tier 1 TIER_ASD        lip-sync (LR-ASD)     switch immediately
tier 2 TIER_DIARIZED   diarized speaker id   switch immediately
tier 3 TIER_DIRECTIVE  Gemini shot direction respects the shot-hold floor
tier 4 TIER_MOUTH      mouth motion          needs MOUTH_CONFIRM_SAMPLES agreeing
tier 5 TIER_HOLD       hold whoever is framed
tier 6 TIER_SIZE       biggest/most central  last resort only
```

`SubjectPolicy.decide(candidates, Evidence(...), frame_number, frame_width)`
returns `(box, id, tier)`. It returns the **caller's own box object** — downstream
bookkeeping matches on box identity, so never copy it.

Result on the real render path: **61% -> 96%**, and the tier mix became
`lip-sync 83%, diarized 7%, held 9%, size 1%` (size decided 61% of frames before).

### Three defects that had to be fixed for the decision to reach the screen

1. **`USE_ASD` shipped defaulting to `0`.** LR-ASD locates the speaker in 19 of
   20 seconds and was switched off in production. Now defaults to `1`
   (`reframe_v2.py`).
2. **Ids cannot define "same subject."** `identity_tracker` minted a fresh
   negative provisional id every frame, and detection supplies both a ~135px
   MediaPipe face box and a ~624x384 YOLO body box for one person (IoU 0.08).
   One motionless person produced ids `0,-1,0,-3,0,-4` across seven detections.
   Fixed by `subject_policy.same_subject()` (geometric) and provisional-id
   inheritance by overlap (`identity_tracker.PROVISIONAL_MATCH_IOU`).
3. **An ambiguous LR-ASD position match is not evidence.** In a crowded shot its
   box sits equidistant from several faces and the match hopped across five
   people at 0.17s intervals. `ASD_MATCH_MARGIN = 1.6` now rejects it.

Also: the renderer used to force a hard re-snap on `target_id != last_target_id`,
re-injecting flicker after the policy had correctly decided. The cut signal is
now `subject_changed`, derived from `policy.shot_started`.

---

## 2. THE MAIN TASK — validate on Pop The Balloon

The owner's judgement, which is correct: the clip tested so far
(`Blind_Dating_Girls_By_Celebrity_Lookalikes`) is an unfair test, because
contestants hold **printed celebrity photos in front of their own faces**.
Pop The Balloon is the real target content.

### 2a. Re-download the source

- URL: `https://youtu.be/ua9Z0Lq3QVA`
- Job dir `output/adcd8a49-28c5-4c82-b806-2b08c7e63bbb/` still holds the
  **full diarized transcript** (1877 segments, speakers A-H) in
  `Ep_115_Pop_The_Balloon_Or_Find_Love__With_Arlette_Amuli_metadata.json`,
  plus two selected spans: **481.28-608.26** and **2222.807-2250.88**.
  The source `.mp4` itself was deleted. Only the video needs re-downloading.

**Two real yt-dlp findings from this session — apply both:**

1. The bgutil PO-token plugin **does not read the `BGUTIL_BASE_URL` env var**
   that `docker-compose.yml` sets. It silently defaults to `127.0.0.1:4416` and
   fails. It must be passed as an extractor arg, and the old arg name is now
   deprecated:

   ```
   --extractor-args "youtubepot-bgutilhttp:base_url=http://bgutil-pot:4416"
   ```
   (The deprecated name `youtube:getpot_bgutil_baseurl` is *rejected*, not just
   warned about.) Verify with `-v`: the `pot:bgutil:http` warnings must vanish.
   **This has never worked in this deployment** — it is very likely the root of
   the recurring yt-dlp failures. Consider wiring the correct arg into
   `main.py`/wherever yt-dlp is invoked, and into the compose file docs.

2. `/app/cookies.txt` is **expired** — YouTube returns
   `Sign in to confirm you're not a bot` even with a fresh-looking file.
   `refresh_youtube_cookies.py` writes a file but cannot revive a dead Google
   session. The owner says the cookie problem was solved before; re-run that
   flow and confirm the jar actually authenticates before blaming anything else.

Working command shape once cookies are valid:

```bash
docker exec openshorts-backend sh -c 'cd /tmp && yt-dlp \
  --cookies /app/cookies.txt \
  --extractor-args "youtube:player_client=web_safari,tv;fetch_pot=always" \
  --extractor-args "youtubepot-bgutilhttp:base_url=http://bgutil-pot:4416" \
  --download-sections "*481-513" --force-keyframes-at-cuts \
  -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]" \
  --merge-output-format mp4 -o "/tmp/ptb_raw.%(ext)s" \
  "https://youtu.be/ua9Z0Lq3QVA"'
```

### 2b. Render old vs new on the same span

The old engine is recoverable from git — use it, do not guess at it:

```bash
mkdir -p /tmp/oldengine
for f in reframe_v2.py main.py identity_tracker.py; do
  git show 2530352:$f > /tmp/oldengine/$f
  docker cp /tmp/oldengine/$f openshorts-backend:/tmp/oldengine/$f
done
# run old with:  PYTHONPATH=/tmp/oldengine:/app
```

Render **both** over the same span, passing the real transcript (see 2c), then
put all three (side-by-side, new, old) into the UI history so the owner can watch
them. A working recipe for that is in section 4.

### 2c. Call `render()` the way `main.py` calls it — this is non-negotiable

An entire earlier round of measurements was invalidated because `render()` was
called without `transcript=`, silently disabling diarization, binding and
directives. **If a measurement does not call the code the way `main.py:1603`
calls it, it is not a measurement.**

```python
reframe_v2.render(input_video, out_path, aspect_ratio,
                  transcript=transcript, clip_start=..., clip_end=...,
                  focus_directives=..., primary_subject_x=...)
```

Ready-made harnesses already exist inside the container:
`/tmp/policycheck.py` (framing accuracy), `/tmp/jitcmp.py` (cut/jitter profile),
`/tmp/cutsource.py` (per-cut trace with tiers), `/tmp/scenecheck.py` (per-shot
candidate counts + ASD coverage). Point them at the new clip.

---

## 3. Known-open problems (in priority order)

### 3.1 Printed/photo faces are framed as if they were people — HIGHEST VALUE

Independently confirmed by video review of **both** engines on the Blind Dating
clip: the camera frames Ariana Grande / Angelina Jolie / Scarlett Johansson
posters while the real contestants are cropped off below the frame. This exactly
matches the owner's complaint: *"showing a face in the first one, then underneath
it's not showing anything at all in frame."*

This is a **detection-layer** problem, upstream of the policy. Every face
candidate in that shot is fake, so better evidence routing cannot help.

Proposed fix — **measure before building**:
- A printed face never moves its mouth across its entire lifetime and never
  registers on lip-sync. Confirm that first (instrument `mouth_activity` and ASD
  per track id over a full clip) rather than assuming it.
- Machinery already exists and is only wired to split cells:
  `id_seen_counts` + `SPLIT_MIN_TRACK_DETECTIONS` in `reframe_v2.py`.
- Second half: when every candidate is rejected as a photo, the correct shot is
  the **wide lineup**, not a tight crop on cardboard. There is currently no
  "no valid subject -> widen" path; it falls through to biggest-box.

Note this may be far less severe on Pop The Balloon (no props). Measure on the
real content before investing in it.

### 3.2 Heads clipped at the top edge

Reported on both engines. Unresolved whether `stabilize_box` (blending a face box
with a much larger body box, which drags the aim centre downward) contributes.
Test by rendering with `POLICY_BOX_BLEND=1.0` (reproduces old snap behaviour) and
comparing head-room.

### 3.3 Do NOT tune against the pixel-delta jitter metric

`/tmp/jitcmp.py` reports source 8 cuts / 0 rapid pairs; old engine 14 / 3; new
15 / 8. **That metric failed to register two changes known to alter the output**,
and a shot-length sweep moved rapid pairs 6/6/10/6 while costing 7 points of
accuracy — i.e. cut frequency is not the jitter source. Treat it as a smoke
alarm, not a target. Ground truth is watching the video.

---

## 4. How to verify (do all three, in this order)

1. **Unit tests**: 650 must pass.
2. **Real-path measurement**: `/tmp/policycheck.py`. Report the tier mix line
   (`Framing evidence: ...`) — if `size` climbs above a few percent, something
   upstream broke.
3. **Watch it.** Metrics were wrong twice this session; the video was right both
   times.

```bash
export GEMINI_API_KEY="$(grep -m1 '^GEMINI_API_KEY=' .env | cut -d= -f2- | tr -d '"')"
python3 ~/.claude/skills/video-analyzer/scripts/analyze_video.py <clip.mp4> --fps 6 \
  --prompt "Harsh editor review of an auto-reframed vertical clip. With timestamps:
  (1) is the visible person the one SPEAKING, or is the camera on the host/mic-holder?
  (2) jitter: framing twitching/resizing on the same person, or cuts <0.7s apart?
  (3) heads cropped off, framing on empty background, or centred on the GAP between people?
  (4) total cut count; which felt jarring? End with a one-line verdict: watchable, yes or no."
```

**Put the results in the UI so the owner can actually watch them.** Working recipe:
create `output/<job-id>/` containing `<base>_clip_N.mp4` (+ empty `.mp4.ready`
files), a `<base>_metadata.json` with a `shorts` array (one entry per clip, each
with `title`/`start`/`end`), and `progress.json` = `{"stage": "finalize"}`.
Create it **inside the container** (`docker exec`) — the host cannot write to
`output/`. Then confirm via `curl -s http://localhost:8000/api/history`.

---

## 5. Rules

- **No shortcuts.** Do not stub, do not weaken a test to make it green.
- Every fix needs a test that fails before and passes after. Name which test
  proves which fix.
- Verify by running, not by reading. "This should work" is not acceptable.
- **Fail open**: any new signal must degrade to current behaviour on error.
- Report measured numbers for every claim. If something did not work, say so —
  two fixes this session produced no measurable change and saying so plainly was
  more useful than the fixes.
- One render per output path. Two concurrent writers produced a truncated file
  and a round of garbage numbers this session.
- Beware simulations: an offline simulator keyed on **position** while production
  keyed on **id** predicted 9 switches where production had 25. A simulation that
  abstracts away the thing production actually uses will confirm whatever you
  hoped.

## 6. Definition of done

- 650+ tests green.
- Pop The Balloon rendered by both engines over the same span, both in the UI.
- Tier mix reported; `size` stays ~1%.
- Video-analyzer verdict for both, with timestamps.
- A clear statement of whether the new engine is better on the **real** content,
  backed by the video review — not only by the accuracy number.
