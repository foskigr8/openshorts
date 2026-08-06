# Plan: fix the OpenShorts pipeline — the complete complaint register

**Audience:** the implementing engineer (Gemini).
**Repo:** `github.com/foskigr8/openshorts`, branch `session/framing-work`.
**Baseline:** 881 tests passing
(`python3 -m pytest tests/ -q`, minus the two cv2-dependent files if OpenCV is
absent locally). A drop is a regression you caused.

This document is the **owner's complaint register** — every problem reported
across the whole project, in one place, each with the evidence behind it, what
is already known about the cause, where the code lives, and how to prove the
fix. It supersedes nothing: `HANDOFF_FRAMING.md`, `KAGGLE_OPTIMIZATION_PLAN.md`
and `PLAN_CAPTIONS_AND_STORAGE.md` remain correct for their own areas and are
cross-referenced throughout.

Read §0 before touching anything. The traps listed there have each cost this
project multiple days already.

---

## 0. How to work on this codebase

These are not style preferences. Every one of them is a lesson paid for.

1. **Verify by running, not by reading.** Every claim in this document that
   says "measured" came from a real run with a date attached. Match that
   standard, and attach your own dates.

2. **Watch the video before declaring anything about framing fixed.** Metrics
   have been wrong twice on this project while the video was right both times.
   Use the `video-analyzer` skill with a **pointed critique prompt** — not the
   default summary prompt, which made a visibly broken clip look fine.
   (`HANDOFF_FRAMING.md` §2e, §4.)

3. **Do NOT tune against the pixel-delta jitter metric.** It failed to register
   two changes known to alter the output. Smoke alarm, not a target.
   (`HANDOFF_FRAMING.md` §3.3.)

4. **A simulation that abstracts away what production actually uses will
   confirm whatever you hoped.** An offline simulator keyed on position
   predicted 9 camera switches where production gave 25, because production
   keyed on track id.

5. **Fail loudly, not open — or if you fail open, say so at the top of the
   log.** The single most expensive pattern in this project is a stage that
   degrades silently and produces a plausible-looking bad clip. Three separate
   instances are catalogued in §6.

6. **One render per output path.** Two concurrent writers produced a truncated
   file and a round of garbage numbers.

7. **Never commit secrets.** `cookies.txt` and `.env` are gitignored. Never
   print a clone URL containing a token.

### The frame contract

The working frame is **3:4 content inside a 9:16 canvas**. That is the frame
every framing decision is judged against — not the 9:16 canvas, and not the
source. When this document says "the subject is out of frame", it means out of
the 3:4 content box.

---

## PRIORITY 0 — GPU rendering (explicit owner request)

> *"When rendering was going on, only the CPU was spiking. What's the meaning of
> that? We have dual GPU and you're telling me rendering takes CPU. CPU is
> spiking to 378%, and GPU is just there doing nothing. We have a full stack we
> are not even utilizing."*

**Status: a GPU path has been BUILT since that complaint and is NOT yet
verified on the real host. Your first job is to prove it, not to rebuild it.**

### What exists now

| Piece | Where |
|---|---|
| Build/device capability probe (real CUDA decode + `scale_cuda` test, not a version string) | `ffmpeg_utils.py:103-191` |
| `-hwaccel cuda`, `-hwaccel_output_format cuda`, `-extra_hw_frames`, `-hwaccel_device` | `ffmpeg_utils.py:200-216` |
| CUDA filtergraph (`scale_cuda` + `overlay_cuda`) | `reframe_v2.unified_filtergraph_gpu:398` |
| CPU filtergraph (unchanged fallback) | `reframe_v2.unified_filtergraph:347` |
| NVENC encode | `ffmpeg_utils.video_encode_args:265` |
| Per-worker device passed to ffmpeg | `gpu_affinity.current_device()` → `-hwaccel_device` |

The probe is deliberately strict — a build can advertise `-hwaccel cuda` and
then fail every render — so it tests an actual decode and an actual
`scale_cuda` before enabling the GPU path. Trust it; do not replace it with a
version check.

### What is still CPU, by design or by omission

- **MediaPipe face detection is CPU-only.** Its Python wheel has no GPU build.
  This will keep a render partly CPU-bound. Do not burn time trying to move it;
  if it dominates after the GPU path is confirmed, the options are a larger
  `DETECT_STRIDE` (`main.py:1077`) or replacing MediaPipe — a separate decision.
- **The video is decoded twice** — once for detection, once for the render.
  Detection already downscales to 640px (`main.DETECT_MAX_WIDTH:1071`) and
  samples every 4th frame. A single decode feeding both is the next real win.
- **TransNetV2 scene detection is not device-aware**, which is why GPU sharding
  is currently opt-in (see below).

### The work, in order

**0.1 — VERIFY THE GPU PATH ON KAGGLE. Nothing else in this section matters
until this is done.** Run a job and report:
- the probe's own log line (did it enable the GPU path, or fall back, and why),
- `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1` sampled
  during the render,
- wall-clock for the reframe stage vs the 6-aug-2026 baseline of **75.4s for two
  clips** (25s and 38s of output),
- CPU% during the render vs the 378% baseline.

If the probe falls back to CPU, the fallback reason IS the bug — fix that first.

**0.2 — Prove the GPU output is visually identical to the CPU output** on the
same input. Hardware scalers differ subtly from swscale; `scale_cuda` and
`overlay_cuda` are not pixel-equivalent to `scale`/`overlay`. Compare frames.
A faster render that looks worse is a regression, and the owner will notice.

**0.3 — Kill the double decode** (see above), once 0.1 and 0.2 are green.

**0.4 — Fix TransNetV2 device handling, then re-enable sharding.** Sharding is
opt-in (`CLIP_GPUS`) because it broke scene detection:

```
TransNetV2 scene detection failed (RuntimeError: Expected all tensors to be on
the same device, but got weight is on cuda:0, different from other tensors on
cuda:1) — falling back to PySceneDetect
```

The model loads once on the default device; a worker pinned to `cuda:1` then
feeds it tensors from the wrong one. Make the load device-aware (or pin scene
detection to the default device explicitly), prove it, then flip the default
back on in `gpu_affinity.available_devices()`. Note it **failed open** — worse
scene boundaries, no error (see §6).

## PRIORITY 1 — Framing and camera (the reason clips are unusable)

> *"It's horrible trash. The tracking, which is the camera, why is it not
> stabilized? It was just one subject, the subject was moving around. I know it
> can be tracked and stabilized. But instead it's shaking, it's jittering."*

> *"The crop, aside the jittering, it just completely cropped the face. It's
> showing the body as the subject is talking."*

> *"I did not have this problem when I was using the regular OpenShorts before I
> modified it. So are we really regressing or improving?"*

**Status: open, and this is what makes clips unusable.** One contributing cause
(whole-pixel crop stepping) was fixed in `ee1cf95`; everything else below is
untouched. The owner watched clips AFTER that fix and still called the tracking
"horrible trash", so treat the jitter as unresolved until a video says otherwise.

Read `HANDOFF_FRAMING.md` in full before starting — §1 explains why the old
size-accumulator design was replaced, §2h is the owner's own spec for the next
three fixes, and §3 lists known-open defects. Do not re-derive any of it.

### 1.1 Jitter inside a held shot — HIGHEST PRIORITY

**Already fixed (do not redo):** whole-pixel crop stepping. `CROP_SUPERSAMPLE`
(`main.py:230`) now defaults to 2 on every host, so an eased follow steps at
0.5px instead of 1px. That addressed *stepping granularity* only — if jitter
survives, it is one of the two sources below, which are untouched.

A held shot must be **pixel-locked**. Two known sources, both already isolated
(`HANDOFF_FRAMING.md` §2h(2)):

- **Long-shot follow.** `main.SmoothedCameraman.get_crop_box`, driven by
  `LONG_SHOT_FOLLOW_SECONDS` (`main.py:137`, default 3.0) and `LONG_FOLLOW_RATE`
  (`main.py:162`, default 0.04). After 3s a held shot begins easing toward the
  subject continuously. A vision review called it *"drifting… feels robotic."*
  **Continuous correction is inherently visible.** Preferred fix: only correct
  when the subject approaches the crop edge; otherwise stay locked.
- **`stabilize_box`** (`subject_policy.py:190`, `POLICY_BOX_BLEND=0.35`) blends
  a MediaPipe face box toward a YOLO body box for the same person. This lags
  the aim and wanders. Consider aiming at the **face** box and using the blend
  for size only.

**The owner has pre-empted the usual excuse:** *"If the excuse is that it's
because of the cuts, like it cut the dead air — that is no excuse, because you
can cut dead air, join it together, and then still track it nicely so
everything will be smooth."* This is correct. Jump-cutting concatenates
segments (`main._build_jump_cut_source:2654`) and the camera solve runs on the
concatenated timeline; a seam must not produce a slide.

**Measure with rendered rects, not by eye:** consecutive-frame crop-x deltas
inside a single shot should be zero, or monotonic and tiny. Any oscillation is
the bug. Then watch the video anyway.

### 1.2 The crop loses the face and frames the body

New in the 6-aug-2026 run and **not** covered by the existing handoff. The
subject is talking and the frame shows their torso.

Leads, in order of suspicion:

- `stabilize_box` blending toward the YOLO **body** box drags the aim centre
  downward — the same mechanism suspected in §3.2 of the handoff ("heads
  clipped at the top edge"). Test with `POLICY_BOX_BLEND=1.0`.
- The clip in question reported `key subject x=0.22` — subject far off-centre.
- Framing evidence for that clip: `lip-sync 31%, diarized 50%, held 16%,
  size 1%`. Low lip-sync means the strongest signal was mostly unavailable.

Regression check the owner explicitly raised: **compare against stock
OpenShorts on the same source.** `HANDOFF_FRAMING.md` §2b documents how to run
the old engine side by side. If the old engine keeps the face and the new one
does not, that is a regression and the diff between them is the answer.

### 1.3 The clip opens on the wrong person

> *"The clip started with a new person, but this person did not start the frame
> because the old person was covering the frame."*

Already traced decision-by-decision in `HANDOFF_FRAMING.md` §2h(1): the opening
shot commits on frame 0's **instantaneous** evidence, which can be unanimous
and wrong, and `ABSOLUTE_MIN_SHOT_SECONDS` (`subject_policy.py:94`, 1.5s) then
protects the mistake for the whole opening beat.

**Fix:** resolve the opening shot from the first turn's evidence **in
aggregate** (majority over the first ~1s of samples) before committing frame 0.
**Do not weaken `ABSOLUTE_MIN_SHOT_SECONDS`** — the floor is correct; the
initial choice is wrong.

### 1.4 The clip ends showing the next scene's person

> *"The clip has ended, but then you will still see another person that is
> supposed to start in the next scene. That's terrible clipping."*

New, not previously recorded. The clip's end boundary is landing after a scene
cut. The end is currently chosen by word-boundary snapping
(`clip_selection.snap_clip_to_words`, applied in `main._snap_candidates`) with
no awareness of scene boundaries.

**Fix direction:** when a scene cut falls within a small window before the
chosen end, pull the end back to just before that cut. Scene boundaries are
already computed (TransNetV2 / PySceneDetect) — the render receives them, but
the *selection* stage does not use them for the outgoing boundary. Same
treatment is worth considering for the incoming boundary (§1.3).

### 1.5 Face ID must drive framing, with duration and ratios

> *"I thought we had face ID to detect who and who is talking. Don't the face ID
> carry duration? If the face ID don't carry duration, then it's useless.
> Because face ID has to show duration, ratios too. Duration — where each person
> started talking from. Ratios — calculating where the person's face is."*

**Current reality, stated precisely:**

- `face_id.py` **does** produce duration: `build_face_trajectory` returns per
  person `{"name", "confidence", "on_screen": [[start, end], …], "speaking": []}`.
- It **does** produce position: every sample carries a `bbox`.
- But `speaking` is deliberately left **empty** — LR-ASD is clip-scoped by
  design, so face_id has no speaking ranges.
- And critically: **face ID does not drive the camera at all.** Its only
  consumer is the transcript speaker rename that feeds Stage 3
  (`viral_clip_finder.format_transcript_for_skill`). Framing is decided by
  `subject_policy.py` from LR-ASD + diarization.
- It was **off** in every run so far (`FACE_ID_DB` unset).

**The owner's expectation is a feature that does not exist yet.** Building it
means routing the face trajectory into `subject_policy` as an additional
evidence tier — named identity, with on-screen ranges and bbox ratios — so the
camera can hold a *named* person across a shot. `insightface` is now installed
and verified working end-to-end (see §7), and it runs on the last GPU by
default so it does not contend with the render.

**Decide explicitly whether to build this**, and say so — it is a real feature,
not a bug fix, and §1.1–1.4 may fix the visible symptoms without it.

### 1.6 Standing framing rules (do not violate)

- **Hard cuts, not pans.** Analysis of four reference shorts (31-jul-2026,
  re-verified 1-aug-2026) found **100% hard cuts, zero pans**. Every historical
  "stutter/jitter/freeze" complaint traced back to sliding a crop window over
  static tripod footage: no parallax, so it reads as the picture sliding.
  `CAMERA_STYLE` (`main.py:215`) defaults to `"cut"`. Keep it that way.
- Shots hold 1–4s; a reaction gets an instant cut, holds 1–2s, cuts back.
- Focal-length changes only at a cut, never as a continuous move.
- **Reaction cutaways are a semantic trigger, not a motion trigger**
  (`HANDOFF_FRAMING.md` §2h(3)). They are currently OFF (`REACTION_CUTS=0`)
  because mouth-motion spikes stole the frame from the speaker. The owner wants
  them, but only when dialogue deliberately draws attention to someone: a
  speaker says another person did something, comments on their behaviour,
  points at or names them. Use the transcript and the existing
  `referenced` / `causing_reaction` directives from `gemini_worker`.
- **Printed/photo faces are framed as if they were people**
  (`HANDOFF_FRAMING.md` §3.1) — confirmed on the Blind Dating clip, where the
  camera framed posters while real contestants were cropped off. Detection-layer
  problem. May be far less severe on the target content; measure before
  investing.
- **Heads clipped at the top edge** (`HANDOFF_FRAMING.md` §3.2).

---

## PRIORITY 2 — Clip selection and boundaries

> *"I don't think this pipeline we did is still starting videos with
> mid-sentences, no context to it. It's just like, what are we doing?"*

> *"Now that we have the skill, I don't think it should be giving us issues
> because I know the skill is good. So I think the workflow is not implementing
> things properly."*

**The owner's diagnosis was correct.** In the 6-aug-2026 run the skill engine
ran, produced clips, and its entire response was thrown away:

```
02:51:23  ❌ Viral Clip Finder error: ValidationError ... clips.0.reaction_cam
02:51:23  🐋 Gemini (narrative):  ← silently fell back to the OLD selector
02:52:05  🔥 Found 2 clips!
```

Two clips from a 10-minute source, chosen by the pre-upgrade engine. **Fixed in
`6e2ed31`** (null-tolerant schema) — but the run that produced the owner's
complaints predates the fix, so **every clip-quality complaint in this section
must be re-evaluated on a run where the skill engine actually lands.**

### 2.1 Confirm the skill engine is the one running

Before investigating any selection complaint, check the log for
`🔥 Viral Clip Finder (gemini): N clip(s), M rejected candidate(s)`. If you see
`🐋 Gemini (narrative)` instead, the skill engine failed and you are debugging
the wrong component. Consider making `VIRAL_ENGINE=skill` the default so a
failure is loud instead of a silent downgrade (see §6).

### 2.2 Mid-sentence starts with no context

Machinery already exists and should be verified rather than rebuilt:

- `main._extend_start_for_preceding_question` rewinds the start to a preceding
  question when the opening line is dependent, including rewinding to the
  **sentence** start rather than the segment start.
- `clip_selection.snap_clip_to_words` snaps boundaries to real word edges.
- The Gemini Vision confirmation pass reviews the opening.

Check on a skill-engine run whether these are firing. If the skill returns a
start mid-sentence and nothing rewinds it, that is the gap.

### 2.3 Too few clips

Two clips from a 10-minute source. The count directive scales with duration
(`viral_clip_finder._candidate_count_directive`: `max(3, min(15, duration//90 + 2))`
→ ~8 for a 10-minute video). The narrative fallback produced 2. Re-measure with
the skill engine actually running.

---

## PRIORITY 3 — Output quality

> *"The quality has not improved, it's still low quality."*

**Status: open, and under-specified — get the owner to point at a specific
artefact before optimising blind.**

Facts to work from:
- The source downloaded at 1080p (`f137` video + `f140` audio) in the measured
  run, so the input is not the ceiling.
- Encode is `h264_nvenc -preset p5 -tune hq -rc vbr -cq 25`. `cq 25` is
  moderate; lowering it (e.g. 19–21) costs file size and gains fidelity.
- Every derivation is a re-encode: reframe → captions → audio clean. Each
  generation loses quality. Audio cleaning now copies the video stream
  (`-c:v copy`), but the caption burn re-encodes.
- Consider whether the 3:4 content box is being upscaled from a crop smaller
  than the output resolution — that would be the real cause and no encoder
  setting fixes it.

**Ask which it is:** soft/blurry (upscale), blocky (bitrate), or washed out
(colour/range). Those are three different bugs.

---

## PRIORITY 4 — Projects, history and logs

**Most of this section has been fixed since the complaint. Verify rather than
rebuild; only §4.4 is still open.**

### 4.1 Every run must be its own project — FIXED, verify

> *"When I finish a job and I want to redo another job, I copy the link, and
> then I go back to the same job that I finished. Each project should be treated
> individually. Only when I specify to reopen project should it be shown."*

The frontend now sends `force_new=1` on every URL submission; reuse happens only
via History's explicit "reopen project" (`restoreProject`). The reuse path
survives for callers that omit the flag (`app.py:1892`). **Verify by submitting
the same link twice and confirming two separate projects.**

### 4.2 Date without time — FIXED, verify

`HistoryTab.jsx:77` now formats with `hour`/`minute`. Two runs of the same link
should be distinguishable at a glance.

### 4.3 Logs must survive job completion — FIXED, verify

> *"When a project finishes, the logs should not vanish… so I can easily copy
> it."*

`GET /api/jobs/{job_id}/logs` serves the persisted `logs.jsonl` as a download,
History has a "logs" button, and `_hf_backup_ready_clips` uploads
`jobs/<job_id>/logs.jsonl` to HF so a wiped Kaggle session can no longer take
the diagnosis with it — the logs endpoint restores from HF when the local copy
is gone. **Verify on a fresh session: kill the Kaggle session, start a new one,
and download the previous run's log.**

### 4.4 Thumbnails disappear mid-sentence — OPEN, needs a repro

> *"I noticed the thumbnails disappear mid sentence."*

**Under-specified. Do not guess.** Ambiguous between clip thumbnails vanishing
from the History/results grid and a generated thumbnail image being cut off.
Get a screenshot and the exact screen before touching `thumbnail.py`.

### 4.5 Visible logo — OPEN, needs a repro

Recorded as outstanding in commit `e5171fb`. A logo is appearing in output that
should not be there. Get a clip and a timestamp.

## PRIORITY 5 — Model, cost and reliability of Stage 3

> *"Is it that we're hitting quota limit? What's happening? Is it that the model
> is not strong enough? Because if the model is the problem then we can add more
> free models — Grok, others as good as Gemini — or I can pay for a cheap one."*

**It is not quota.** There is not a single 429 in the logs. What the 6-aug-2026
run shows is:

- A **190-second read timeout** on a **189 KB prompt** to
  `gemini-3.1-flash-lite` — a small, fast model given a very large reasoning
  task with a strict JSON contract.
- Two schema rejections (now fixed) that each burned a full retry at that size.

**Do the cheap things first, in this order:**

1. `VCF_REFERENCES=lean` — drops the prompt from ~189 KB to ~85 KB by keeping
   the scoring rubric and anti-patterns and dropping the pattern/niche
   catalogues. One env var.
2. Use a **stronger model for Stage 3 only** (`NARRATIVE_GEMINI_MODEL` /
   `GEMINI_MODEL`). Stage 3 runs once per job; vision confirmation runs per
   clip. Paying for quality on the once-per-job call is the efficient trade.
3. Raise the 180s HTTP timeout in `deepseek_worker._run_deepseek_stage:676` if
   large prompts remain.

**Adding more free providers is the wrong move right now** — it spreads the
same weakness across more endpoints. The transport is already provider-agnostic
(any OpenAI-compatible `base_url`), so adding one later is easy if it is ever
actually needed.

---

## 6. The silent-degradation problem (read this even if you skip everything else)

The most expensive pattern in this project is not a crash. It is a stage that
fails, recovers quietly, and ships a plausible-looking bad clip. Four confirmed
instances:

1. **Stage 3 fell back to the old engine** on a schema error. The owner watched
   clips from the wrong selector and concluded the skill was bad.
2. **Scene detection fell back to PySceneDetect** on a CUDA device mismatch.
   Worse cut boundaries, no error.
3. **Background-audio removal cleaned a file nobody watches** — captions write a
   *new* `subtitled_*.mp4` with `-c:a copy`, and that derived file is what gets
   served. The owner heard music in three clips that the log said were cleaned.
4. **Face ID silently stayed off** — no `FACE_ID_DB`, no message, anonymous
   speakers.

**When you add a fallback, print a line that says the output is degraded and
why.** Where a fallback exists to protect a job from a transient failure, keep
it — but make it visible in the job log the owner reads, not just in a comment.

---

## 7. Already fixed — do not re-fix, do verify

All on `session/framing-work`. If a symptom below reappears, it is a
**regression**, not an unfixed bug.

| Commit | Fix |
|---|---|
| `98df856` | Stage 3 skill engine could never take effect — `keep_spans` emitted as `[start, end]` pairs where every consumer reads `{"start", "end"}` dicts |
| `381f544` | Kaggle findings + per-job caption placement + HF persistent storage |
| `6c41585` | Dashboard served stale UI after a `git pull` (rebuilt only when `dist` was missing) |
| `e400598` | `insightface` installable; face ID defaults to the last GPU |
| `5704c71` | Kaggle dependency install failed outright (`ResolutionImpossible`) — caps, not pins |
| `a0ddd8e` | YouTube bot wall — PO token provider on Kaggle |
| `6e2ed31` | Stage 3 response discarded over `"reaction_cam": null` |
| `cea64ca` | GPU sharding broke scene detection (now opt-in); audio cleaned the wrong file; caption placement overridden in GENERAL scenes |
| `ee1cf95` | Whole-pixel crop stepping — `CROP_SUPERSAMPLE=2` everywhere, so follow shots step 0.5px |
| `e5171fb` | Caption "bottom" landed ~40% up the frame; History blank on a fresh Kaggle session; no logs-download endpoint |
| `99e5d51` | Job logs lived only on the Kaggle disk — now backed up to HF and restorable |

Verified working end-to-end on real input: face ID (11 identifications at 0.993
confidence on a real video, speaker renamed, named speaker reaching the Stage 3
prompt), HF storage configuration, both T4s visible, NVENC, AssemblyAI with
diarization.

**Not yet verified on a real run:** clips surviving a Kaggle session restart via
HF storage, and per-job caption placement in a rendered video.

---

## 8. Suggested order of work

1. **§0.1 + §0.2 — verify the GPU path already built, on the real host.** It is
   the owner's stated priority, it is unproven, and confirming it shortens every
   test cycle that follows. If the probe is falling back to CPU, that is the
   whole task.
2. **§1.1 jitter** and **§1.2 face crop.** These are why clips are unusable.
   Run the old-vs-new engine comparison (`HANDOFF_FRAMING.md` §2b) FIRST — the
   owner says stock OpenShorts did not do this, and if it is a regression the
   diff is the answer, which is far cheaper than a redesign.
3. **§2.1** — confirm the skill engine actually lands, then re-judge §2.2 and
   §2.3 on that run. Several selection complaints may already be gone, because
   the run that produced them used the fallback engine.
4. **§1.3 opening shot** and **§1.4 end bleed** — boundary logic, causes known.
5. **Verify the recent fixes on a real run**: §4.1 (same link → new project),
   §4.2 (date + time), §4.3 (logs survive a session restart), captions at
   bottom, and clips restoring from HF into History.
6. **§0.3 double decode**, **§0.4 re-enable sharding**, **§3 quality**,
   **§5 model prompt/size** — after the above.
7. **§4.4 thumbnails** and **§4.5 logo** — need a repro from the owner first.
8. **§1.5 face-ID-driven framing** — a real feature, not a bug fix. Only after
   §1.1–1.4, and only if the symptoms survive them.

## 9. Definition of done

- The full test suite passes, with new tests for every behaviour changed.
- A real end-to-end job on Kaggle produces clips from the **skill** engine.
- A rendered clip has been **watched** (video-analyzer, pointed prompt) and the
  camera is locked within every held shot, the speaking subject's face is in the
  3:4 content box, the clip opens on the speaker, and no next-scene person
  appears at the end.
- `nvidia-smi` shows real GPU utilisation during a render, with before/after
  wall-clock numbers reported.
- Background-audio removal, caption placement, and per-run projects verified in
  the delivered artefact — not in the log.
