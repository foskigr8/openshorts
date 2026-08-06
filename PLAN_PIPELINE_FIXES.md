# Plan: fix the OpenShorts pipeline — the complete problem register

**Audience:** the implementing engineer (Gemini).
**Repo:** `github.com/foskigr8/openshorts`, branch `session/framing-work`.
**Baseline:** 881 tests passing (`python3 -m pytest tests/ -q`; several test
files need OpenCV and will not collect without it). A drop is a regression you
caused.

This is the **owner's problem register** — every problem reported across the
project, in one place, with the owner's own words, the evidence, what is known
about the cause, where the code lives, and how to prove a fix.

## Nothing in this document is fixed

Read that again, because it governs how the whole document is written.

Code has been changed against most of these problems. Tests were written and
they pass. **None of it has been confirmed in a delivered clip.** The owner
watched the output after several of these changes landed and the verdict was
still *"horrible trash."*

So every problem below is **OPEN**. Where code has already been changed, that is
recorded as *"code changed, unconfirmed"* — which tells you **where to look
first**, not that the problem is solved. Treat a prior change as a hypothesis
somebody wrote down, not as a result. Several of these problems have already
been "fixed" more than once.

**A problem stops being open when the owner sees it working in a rendered clip
or a real run — not when a test passes.**

---

## 0. How to work on this codebase

These are not style preferences. Every one was paid for.

1. **Verify by running, not by reading.** Every "measured" claim here came from
   a real run with a date. Match that standard and attach your own dates.

2. **Watch the video before claiming anything about framing.** Metrics have been
   wrong twice on this project while the video was right both times. Use the
   `video-analyzer` skill with a **pointed critique prompt** — the default
   summary prompt made a visibly broken clip look fine.
   (`HANDOFF_FRAMING.md` §2e, §4.)

3. **A passing test is not a fixed problem.** This codebase has 881 tests and
   the delivered clips are still unusable. Tests stop regressions; they do not
   demonstrate quality. Every item below has a verification step involving real
   output — do that one.

4. **Do NOT tune against the pixel-delta jitter metric.** It failed to register
   two changes known to alter the output. Smoke alarm, not a target.
   (`HANDOFF_FRAMING.md` §3.3.)

5. **A simulation that abstracts away what production uses will confirm whatever
   you hoped.** An offline simulator keyed on position predicted 9 camera
   switches where production gave 25, because production keyed on track id.

6. **Fail loudly. If you must fail open, say so in the log the owner reads.**
   The most expensive pattern in this project is a stage that degrades quietly
   and ships a plausible-looking bad clip — see §6.

7. **One render per output path.** Two concurrent writers produced a truncated
   file and a round of garbage numbers.

8. **Never commit secrets.** `cookies.txt` and `.env` are gitignored. Never
   print a clone URL containing a token.

### The frame contract

The working frame is **3:4 content inside a 9:16 canvas**. Every framing
judgement is against that box — not the 9:16 canvas, not the source. "Out of
frame" means out of the 3:4 content box.

---

## PRIORITY 0 — GPU rendering

> *"When rendering was going on, only the CPU was spiking. What's the meaning of
> that? We have dual GPU and you're telling me rendering takes CPU. CPU is
> spiking to 378%, and GPU is just there doing nothing. We have a full stack we
> are not even utilizing."*

**OPEN.** A GPU path exists in the code and **has never been observed working on
the real host.** The owner's measurement — CPU at 378%, GPUs idle — is the only
measurement anyone has.

### Where the code is

| Piece | Location |
|---|---|
| Capability probe (real CUDA decode + `scale_cuda` test, not a version string) | `ffmpeg_utils.py:103-191` |
| `-hwaccel cuda`, `-hwaccel_output_format cuda`, `-extra_hw_frames`, `-hwaccel_device` | `ffmpeg_utils.py:200-216` |
| CUDA filtergraph (`scale_cuda` + `overlay_cuda`) | `reframe_v2.unified_filtergraph_gpu:398` |
| CPU filtergraph (fallback) | `reframe_v2.unified_filtergraph:347` |
| NVENC encode | `ffmpeg_utils.video_encode_args:265` |
| Per-worker device → ffmpeg | `gpu_affinity.current_device()` → `-hwaccel_device` |

The probe deliberately tests an actual decode and an actual `scale_cuda`, because
a build can advertise `-hwaccel cuda` and then fail every render. Do not replace
it with a version check.

### Known to remain on CPU

- **MediaPipe face detection is CPU-only** — no GPU wheel exists. A render stays
  partly CPU-bound. If it dominates once the GPU path is confirmed, the options
  are a larger `DETECT_STRIDE` (`main.py:1077`) or replacing MediaPipe. Separate
  decision; do not start there.
- **The video is decoded twice** — once for detection, once for the render.
  Detection downscales to 640px (`main.DETECT_MAX_WIDTH:1071`) and samples every
  4th frame. One decode feeding both is the real win.
- **TransNetV2 scene detection is not device-aware** — see 0.4.

### The work

**0.1 — Establish what the GPU path actually does on Kaggle.** Nothing else here
matters until this is answered. Run a job and report:
- the probe's own log line — did it enable CUDA, or fall back, and why;
- `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1` during
  the render;
- reframe wall-clock against the 6-aug-2026 baseline: **75.4s for two clips**
  (25s and 38s of output);
- CPU% against the 378% baseline.

**If the probe falls back to CPU, that fallback reason is the entire task.**

**0.2 — Prove GPU output is visually identical to CPU output** on the same
input. `scale_cuda`/`overlay_cuda` are not pixel-equivalent to `scale`/`overlay`.
Compare frames. A faster render that looks worse is a regression.

**0.3 — Remove the double decode**, once 0.1 and 0.2 are green.

**0.4 — Make TransNetV2 device-aware, then re-enable sharding.** Sharding is
opt-in (`CLIP_GPUS`) because it broke scene detection on the 6-aug-2026 run:

```
TransNetV2 scene detection failed (RuntimeError: Expected all tensors to be on
the same device, but got weight is on cuda:0, different from other tensors on
cuda:1) — falling back to PySceneDetect
```

The model loads once on the default device; a worker pinned to `cuda:1` feeds it
tensors from the wrong one. It **failed open** — worse scene boundaries, no error
(§6). Fix the device handling, prove it, then flip the default back on in
`gpu_affinity.available_devices()`.

---

## PRIORITY 1 — Framing and camera

> *"It's horrible trash. The tracking, which is the camera, why is it not
> stabilized? It was just one subject, the subject was moving around. I know it
> can be tracked and stabilized. But instead it's shaking, it's jittering."*

> *"The crop, aside the jittering, it just completely cropped the face. It's
> showing the body as the subject is talking."*

> *"I did not have this problem when I was using the regular OpenShorts before I
> modified it. So are we really regressing or improving?"*

**OPEN — this is why the clips are unusable.**

Read `HANDOFF_FRAMING.md` before starting: §1 is why the size-accumulator design
was replaced, §2h is the owner's own spec, §3 lists known-open defects. Do not
re-derive any of it.

### 1.1 Jitter inside a held shot

A held shot must be **pixel-locked**.

*Code changed, unconfirmed:* `CROP_SUPERSAMPLE` (`main.py:230`) defaults to 2, so
an eased follow steps at 0.5px instead of a whole pixel. **The owner watched
clips after this change and still reported jitter** — so either it does not help,
or it is not the dominant source.

Two sources remain untouched (`HANDOFF_FRAMING.md` §2h(2)):

- **Long-shot follow.** `main.SmoothedCameraman.get_crop_box`, driven by
  `LONG_SHOT_FOLLOW_SECONDS` (`main.py:137`, 3.0) and `LONG_FOLLOW_RATE`
  (`main.py:162`, 0.04). After 3s a held shot eases toward the subject
  continuously; a vision review called it *"drifting… feels robotic."*
  **Continuous correction is inherently visible.** Prefer correcting only when
  the subject approaches the crop edge; otherwise stay locked.
- **`stabilize_box`** (`subject_policy.py:190`, `POLICY_BOX_BLEND=0.35`) blends
  the MediaPipe face box toward the YOLO body box for the same person, which lags
  the aim and wanders. Consider aiming at the **face** box and using the blend
  for size only.

The owner has pre-empted the usual excuse: *"If the excuse is that it's because
of the cuts, like it cut the dead air — that is no excuse, because you can cut
dead air, join it together, and then still track it nicely."* Correct. The camera
solve runs on the concatenated timeline (`main._build_jump_cut_source:2914`); a
seam must not produce a slide.

Measure with rendered rects — consecutive-frame crop-x deltas inside one shot
should be zero or monotonic and tiny; any oscillation is the bug. Then watch it
anyway.

### 1.2 The crop loses the face and frames the body

Not covered by the existing handoff. Subject is talking; frame shows torso.

Leads, in order of suspicion:
- `stabilize_box` blending toward the YOLO **body** box drags the aim centre down
  — the same mechanism suspected behind "heads clipped at the top edge"
  (`HANDOFF_FRAMING.md` §3.2). Test with `POLICY_BOX_BLEND=1.0`.
- The clip reporting this had `key subject x=0.22` — subject far off-centre.
- Its framing evidence: `lip-sync 31%, diarized 50%, held 16%, size 1%`. Low
  lip-sync means the strongest signal was mostly unavailable.

**Run the old-vs-new comparison first** (`HANDOFF_FRAMING.md` §2b). The owner says
stock OpenShorts did not do this. If it is a regression, the diff is the answer
and that is far cheaper than a redesign.

### 1.3 The clip opens on the wrong person

> *"The clip started with a new person, but this person did not start the frame
> because the old person was covering the frame."*

Traced decision-by-decision in `HANDOFF_FRAMING.md` §2h(1): the opening shot
commits on frame 0's **instantaneous** evidence, which can be unanimous and
wrong, and `ABSOLUTE_MIN_SHOT_SECONDS` (`subject_policy.py:94`, 1.5s) then
protects the mistake for the whole opening beat.

Resolve the opening shot from the first turn's evidence **in aggregate** (majority
over the first ~1s) before committing frame 0. **Do not weaken
`ABSOLUTE_MIN_SHOT_SECONDS`** — the floor is right, the initial choice is wrong.

### 1.4 The clip ends showing the next scene's person

> *"The clip has ended, but then you will still see another person that is
> supposed to start in the next scene. That's terrible clipping."*

The end boundary lands after a scene cut. Ends are chosen by word-boundary
snapping (`clip_selection.snap_clip_to_words` via `main._snap_candidates`) with no
awareness of scene boundaries.

When a scene cut falls within a small window before the chosen end, pull the end
back to just before that cut. Scene boundaries are already computed and reach the
render — the *selection* stage just does not use them. Consider the same for the
incoming boundary (§1.3).

### 1.5 Face ID should drive framing, with duration and ratios

> *"I thought we had face ID to detect who and who is talking. Don't the face ID
> carry duration? If the face ID don't carry duration, then it's useless. Because
> face ID has to show duration, ratios too."*

Stated precisely:
- `face_id.build_face_trajectory` **does** return duration — per person
  `{"name", "confidence", "on_screen": [[start, end], …]}` — and every sample
  carries a `bbox`, so position is there too.
- `speaking` is deliberately **empty**: LR-ASD is clip-scoped, so face_id has no
  speaking ranges.
- **Face ID does not drive the camera at all.** Its only consumer is the
  transcript speaker rename feeding Stage 3
  (`viral_clip_finder.format_transcript_for_skill`). Framing comes from
  `subject_policy.py` via LR-ASD + diarization.
- It has been **off in every run so far** (`FACE_ID_DB` unset).

The owner's expectation is a **feature that does not exist**. Building it means
routing the face trajectory into `subject_policy` as a named-identity evidence
tier, so the camera can hold a named person across a shot. `insightface` is
installed and was verified working standalone (11 identifications at 0.993
confidence on a real video), running on the last GPU so it does not contend with
the render.

**Decide explicitly whether to build this.** §1.1–1.4 may fix the visible symptoms
without it.

### 1.6 Standing framing rules — do not violate

- **Hard cuts, not pans.** Four reference shorts analysed 31-jul-2026, re-verified
  1-aug-2026: **100% hard cuts, zero pans.** Every historical stutter/jitter
  complaint traced to sliding a crop window over static tripod footage — no
  parallax, so it reads as the picture sliding. `CAMERA_STYLE` (`main.py:215`)
  defaults to `"cut"`. Keep it.
- Shots hold 1–4s. A reaction gets an instant cut, holds 1–2s, cuts back.
- Focal-length changes only at a cut, never as a continuous move.
- **Reaction cutaways are a semantic trigger, not a motion trigger** (§2h(3)).
  Currently OFF (`REACTION_CUTS=0`) because mouth-motion spikes stole the frame
  from the speaker. The owner wants them, but only when dialogue deliberately
  draws attention to someone — a speaker says another person did something,
  comments on their behaviour, points at or names them. Use the transcript and
  the existing `referenced` / `causing_reaction` directives from `gemini_worker`.
- **Printed/photo faces get framed as people** (§3.1) — confirmed on the Blind
  Dating clip: the camera framed posters while real contestants were cropped off.
  Detection-layer problem. Measure on the target content before investing.
- **Heads clipped at the top edge** (§3.2).

---

## PRIORITY 2 — Clip selection and boundaries

> *"I don't think this pipeline we did is still starting videos with
> mid-sentences, no context to it. What are we doing?"*

> *"Now that we have the skill, I don't think it should be giving us issues
> because I know the skill is good. So I think the workflow is not implementing
> things properly."*

**OPEN. The owner's diagnosis was correct.** On the 6-aug-2026 run the skill
engine ran, produced clips, and its entire response was discarded:

```
02:51:23  ❌ Viral Clip Finder error: ValidationError ... clips.0.reaction_cam
02:51:23  🐋 Gemini (narrative):  ← silently fell back to the OLD selector
02:52:05  🔥 Found 2 clips!
```

Two clips from a 10-minute source, chosen by the pre-upgrade engine.

*Code changed, unconfirmed:* the response schema now tolerates nulls. **No one
has yet seen the skill engine deliver clips end-to-end.**

### 2.1 Confirm which engine is running — before anything else in this section

Look for `🔥 Viral Clip Finder (gemini): N clip(s), M rejected candidate(s)`. If
you see `🐋 Gemini (narrative)`, the skill engine failed and you are debugging the
wrong component. Consider making `VIRAL_ENGINE=skill` the default so a failure is
loud rather than a silent downgrade (§6).

**Every complaint in 2.2 and 2.3 must be re-judged on a run where the skill engine
actually lands.** They may be artefacts of the fallback.

### 2.2 Mid-sentence starts with no context

Machinery exists — verify it fires rather than rebuilding:
- `main._extend_start_for_preceding_question` rewinds to a preceding question when
  the opening line is dependent, including to the **sentence** start.
- `clip_selection.snap_clip_to_words` snaps to real word edges.
- The Gemini Vision confirmation pass reviews the opening.

If the skill returns a mid-sentence start and nothing rewinds it, that is the gap.

### 2.3 Too few clips

Two clips from a 10-minute source. The count directive scales with duration
(`viral_clip_finder._candidate_count_directive`:
`max(3, min(15, duration//90 + 2))` → ~8 for 10 minutes). The fallback produced 2.
Re-measure with the skill engine running.

---

## PRIORITY 3 — Output quality

> *"The quality has not improved, it's still low quality."*

**OPEN, and under-specified — get the owner to point at a specific artefact before
optimising blind.**

Facts:
- The source downloaded at 1080p (`f137` + `f140`), so the input is not the
  ceiling.
- Encode is `h264_nvenc -preset p5 -tune hq -rc vbr -cq 25`. `cq 25` is moderate;
  19–21 costs size and gains fidelity.
- Every derivation re-encodes: reframe → captions → audio clean. Audio cleaning
  copies the video stream; the caption burn re-encodes.
- Check whether the 3:4 content box is being **upscaled** from a crop smaller than
  the output resolution. If so, no encoder setting fixes it.

Ask which it is: soft/blurry (upscale), blocky (bitrate), or washed out
(colour/range). Three different bugs.

---

## PRIORITY 4 — Projects, history, logs, captions, storage

**All OPEN. Code has been changed against every item here and none of it has been
confirmed by the owner in practice.** Where a change exists it is noted so you
know where to look — and so that if the symptom persists you investigate that
change rather than rebuilding from scratch.

### 4.1 Every run must be its own project

> *"When I finish a job and I want to redo another job, I copy the link, and then
> I go back to the same job that I finished. Each project should be treated
> individually. Only when I specify to reopen project should it be shown."*

Caused by same-source reuse (`app.py:1892`, `_find_completed_job_for_source` at
`app.py:474`), which matched on `source_url` and returned the finished job.

*Code changed, unconfirmed:* the frontend now sends `force_new=1` on every URL
submission; reuse remains for callers that omit it. **Verify by submitting the
same link twice and confirming two separate projects.**

### 4.2 Date without time

> *"The date should be accurate, the date and time… even if it's the same link, if
> the project is run at the same time, you should show the same time."*

*Code changed, unconfirmed:* `HistoryTab.jsx:77` formats with hour and minute.
**Verify two runs of one link are distinguishable in the UI.**

### 4.3 Logs must survive job completion

> *"When a project finishes, the logs should not vanish… so I can easily copy it."*

Jobs live in an in-memory dict, so logs died with the process and with cleanup.

*Code changed, unconfirmed:* `GET /api/jobs/{job_id}/logs` serves the persisted
`logs.jsonl`, History has a "logs" button, and the HF backup uploads
`jobs/<job_id>/logs.jsonl` so a wiped session cannot take the diagnosis with it.
**Verify: kill the Kaggle session, start a fresh one, download the previous run's
log.**

### 4.4 Caption position does not work

> *"I put the caption in bottom, upper, whatever — only center. It doesn't work."*

Two causes were found: the per-job margin was overwritten during
blurred-background scenes, and the content-box floor used a raised inset that
dominated the requested value, landing "bottom" ~40% up the frame.

*Code changed, unconfirmed:* the floor now only clears the blur band. **Verify by
rendering the same clip at bottom / raised / middle and watching all three.**

### 4.5 Background audio removal does not work

> *"I put remove background audio and I think it did not work — I was hearing
> sound in three audios I played."*

Confirmed from the log: captions burned at 02:53:46, audio cleaned at 02:53:55.
Captions write a **new** `subtitled_*.mp4` with `-c:a copy`, and that derived file
is what gets served — so the cleaned audio went to a file nobody watches.

*Code changed, unconfirmed:* cleaning now runs before captions. **Verify by
listening to a delivered clip.**

### 4.6 Clips and History on a fresh Kaggle session

`/kaggle/working` is wiped at session end.

*Code changed, unconfirmed:* clips upload to HF as each finishes, and History falls
back to listing the HF repo with playback routed through the restore endpoint.
**Verify: finish a job, kill the session, start a new one, open History, play a
clip.**

### 4.7 Thumbnails disappear mid-sentence — needs a repro

Ambiguous between thumbnails vanishing from the History/results grid and a
generated thumbnail image being cut off. Get a screenshot and the exact screen
before touching `thumbnail.py`.

### 4.8 A visible logo in the output — needs a repro

Recorded as outstanding in commit `e5171fb`. Get a clip and a timestamp.

---

## PRIORITY 5 — Model, cost, Stage 3 reliability

> *"Is it that we're hitting quota limit? Is the model not strong enough? We can
> add more free models — Grok, others — or I can pay for a cheap one."*

**OPEN. It is not quota** — there is not a single 429 in the logs. What the
6-aug-2026 run shows:
- a **190-second read timeout** on a **189 KB prompt** to `gemini-3.1-flash-lite`
  — a small fast model given a large reasoning task under a strict JSON contract;
- two schema rejections, each burning a full retry at that size.

Cheap things first:
1. `VCF_REFERENCES=lean` — prompt ~189 KB → ~85 KB, keeping the scoring rubric and
   anti-patterns. One env var.
2. A **stronger model for Stage 3 only** (`NARRATIVE_GEMINI_MODEL` /
   `GEMINI_MODEL`). Stage 3 runs once per job; vision confirmation runs per clip.
   Pay for quality on the once-per-job call.
3. Raise the 180s timeout (`deepseek_worker.py:676`) if large prompts persist.

**Adding more free providers is the wrong move now** — it spreads the same
weakness across more endpoints. The transport is already provider-agnostic (any
OpenAI-compatible `base_url`), so adding one later is cheap if it is ever
genuinely needed.

---

## 6. The silent-degradation problem

The most expensive pattern here is not a crash. It is a stage that fails, recovers
quietly, and ships a plausible-looking bad clip. Four confirmed:

1. **Stage 3 fell back to the old engine** on a schema error. The owner watched
   clips from the wrong selector and concluded the skill was bad.
2. **Scene detection fell back to PySceneDetect** on a CUDA device mismatch —
   worse boundaries, no error.
3. **Background-audio removal cleaned a file nobody watches** — the log said it
   succeeded; the delivered clip still had music.
4. **Face ID silently stayed off** — no `FACE_ID_DB`, no message.

**When you add a fallback, print a line saying the output is degraded and why.**
Keep the fallback where it protects a job from a transient failure — but make it
visible in the log the owner reads.

---

## 7. Where code has already been changed (unconfirmed)

These commits exist on `session/framing-work`. **This is a map of where to look,
not a list of solved problems.** If a symptom persists, start by checking whether
the change actually does what its commit message claims — on real output, not in a
test.

| Commit | What it attempted |
|---|---|
| `98df856` | Stage 3 skill engine could never take effect — `keep_spans` emitted as `[start, end]` pairs where consumers read `{"start", "end"}` dicts |
| `381f544` | Kaggle findings, per-job caption placement, HF persistent storage |
| `6c41585` | Dashboard served stale UI after a `git pull` |
| `e400598` | `insightface` installable; face ID on the last GPU |
| `5704c71` | Kaggle dependency install failed outright — caps, not pins |
| `a0ddd8e` | YouTube bot wall — PO token provider |
| `6e2ed31` | Stage 3 response discarded over `"reaction_cam": null` |
| `cea64ca` | GPU sharding broke scene detection; audio cleaned the wrong file; caption placement overridden |
| `ee1cf95` | Whole-pixel crop stepping — `CROP_SUPERSAMPLE=2` |
| `e5171fb` | Caption "bottom" landing ~40% up; History blank on a fresh session; no logs download |
| `99e5d51` | Job logs lived only on the Kaggle disk |

Verified in isolation only — standalone, never in a delivered clip: face ID
identification, HF storage configuration, both T4s visible, NVENC present,
AssemblyAI with diarization.

---

## 8. Suggested order

1. **§0.1–0.2** — find out what the GPU path does on the real host. Owner's stated
   priority, completely unproven, and it shortens every later test cycle.
2. **§1.1 jitter** and **§1.2 face crop** — why clips are unusable. Run the
   old-vs-new engine comparison FIRST (`HANDOFF_FRAMING.md` §2b): the owner says
   stock OpenShorts did not do this, and a regression diff is far cheaper than a
   redesign.
3. **§2.1** — confirm the skill engine lands, then re-judge §2.2 and §2.3.
4. **§1.3 opening shot**, **§1.4 end bleed** — boundary logic, causes known.
5. **§4.1–4.6** — confirm the recent changes in real output. Cheap to check, and
   several are things the owner has already reported more than once.
6. **§0.3**, **§0.4**, **§3**, **§5**.
7. **§4.7**, **§4.8** — need a repro from the owner.
8. **§1.5 face-ID-driven framing** — a feature, not a bug fix. Only if the
   symptoms survive §1.1–1.4.

## 9. Definition of done

Not "tests pass". For each problem:

- A **rendered clip** demonstrates the fix, watched with the video-analyzer skill
  and a pointed prompt — camera locked within every held shot, the speaking
  subject's face inside the 3:4 box, the clip opening on the speaker, no
  next-scene person at the end.
- A **real Kaggle job** produces clips from the **skill** engine.
- `nvidia-smi` shows real GPU utilisation during a render, with before/after
  wall-clock reported.
- Caption placement, background-audio removal, per-run projects, log persistence
  and History-after-restart confirmed **in the delivered artefact**.
- The full test suite still passes, with new tests for changed behaviour.
