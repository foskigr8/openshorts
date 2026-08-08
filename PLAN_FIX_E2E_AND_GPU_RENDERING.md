# Plan: fix the first real end-to-end run — stabilization, GPU rendering, boundaries, dashboard

> **OBSOLETE in part (8-aug-2026):** this plan targeted the v2 engine
> (`reframe_v2`, MediaPipe/YOLO) and its supersample path. Both were removed —
> `reframe_v3` is the only reframe engine. `REFRAME_DUMP_PATH` still exists
> (now produced by `reframe_v3.render`); the rest is historical.

> **STATUS (6-aug-2026):** written at the point the previous Claude session hit
> its limit (the request was *"write a plan for deepseek containg what to do to
> fix it and yes i want gpu rendering to save time"*). This replaces the
> hand-wavy wish-list from the voice note with a code-grounded plan: every item
> below names the file/line where the change lands and the measurement that
> proves it.
>
> **Implementation status (later same day):**
> - **DONE, tested (860 passing):** PART 1 (supersampling grid + target EMA +
>   follow-step floor + REFRAME_DUMP_PATH + head containment), PART 3
>   (scene-boundary clamp + vision different-speaker-open/end-bleed rules),
>   PART 4 (merge_face_id_with_tracker + named-speaker binding fallback),
>   PART 5 (per-run projects via `force_new`, date+time, persisted per-job
>   logs, cached poster endpoint), PART 6 (download gate: ffprobe specs +
>   LOW-QUALITY SOURCE warning + metadata; fallback relabeled) and PART 2's
>   safe layer (GPU_RENDER probe, `-hwaccel cuda` decode offload on all four
>   ffmpeg call sites, Demucs `--device cuda`).
> - **Implemented but must be validated on the 2×T4 box (the probe's real
>   device test cannot pass on a CPU-only host):** PART 2.3's CUDA filtergraph
>   (scale_cuda/overlay_cuda with hwdownload around the CPU-only crop/blur/
>   grade) — gated on a REAL device probe and retried with the CPU graph on
>   any failure, so a GPU edge case degrades instead of breaking a job. The
>   probe itself was hardened: build support alone no longer enables
>   `-hwaccel cuda` (the studio container has CUDA filters but no device and
>   correctly stays on CPU). PART 6.4 (single-encode caption burn) remains
>   open — the 3–4 re-encode generations are secondary to the download gate.
> - **PART 6 hardened per owner direction ("make sure high quality lands"):**
>   a below-floor success no longer ends the ladder — it parks the file aside
>   and keeps trying HD strategies, restoring the best at the end; and the
>   pipeline now FAILS the job when the best available source is still below
>   the floor (`ALLOW_LOW_QUALITY_SOURCE=1` is the escape hatch) instead of
>   shipping soft clips with a warning. The HD format string also accepts
>   vp9/av01 streams when avc1 HD is unavailable.
> - **PART 6.4 now implemented (one-pass captions):** the reframe render
>   encodes the clean clip AND the captioned clip from the same reframed
>   frames (single ffmpeg invocation, ass filter on the second output), so
>   the captioned file loses one generation instead of two and the separate
>   caption-burn process is gone. The old post-render burn remains the
>   automatic fallback whenever a watermark or audio cleanup runs, or when
>   the v1 engine takes over — file conventions (clean original +
>   subtitled_<ts>_<clip> canonical) are unchanged.
> - Kaggle notebook env cell: set `GPU_RENDER=1`, `CROP_SUPERSAMPLE=2` on the
>   T4 host and confirm the probe line appears in the job log.
>
> **Do not re-fix what is already fixed.** The three defects from the first
> real end-to-end run landed in `cea64ca` (6-aug-2026) and the skill-engine
> null fix in `6e2ed31`:
> - **Scene detection** — GPU sharding broke TransNetV2 (tensor on the wrong
>   device); sharding is now opt-in via `CLIP_GPUS`.
> - **Background-audio removal** — ran *after* caption burn, cleaning a file
>   nobody watches; it now runs first in the clip worker
>   (`main.py:3464`).
> - **Caption position** — the content-box margin overrode the per-job choice;
>   it is now a floor (`subtitles.py`, `general_ranges` handling).
> - **`reaction_cam: null`** — a null in an unused field threw away the whole
>   skill response; the fix in `6e2ed31` means the viral-clip-finder skill
>   actually runs now. The complaint clips were selected by the **old
>   fallback engine** because the skill never ran.

**Audience:** implementing engineer (DeepSeek or otherwise).
**Repo:** `github.com/foskigr8/openshorts`, branch `session/framing-work`.
If you lack repo access, ask the owner to add you — every change is in this
repo. Read `KAGGLE.md`, `KAGGLE_OPTIMIZATION_PLAN.md` and
`HANDOFF_FRAMING.md` first; §2e of the last one lists measurement traps that
have already cost this project real time (Gemini spatial claims must be
cross-checked against pixels, and re-encoding hides generational loss).

Baseline: **654 passed, 21 skipped** on the host (`python3 -m pytest tests/ -q`
in `openshorts/`, 6-aug-2026). A drop is a regression you caused.

Canonical reproduction for every item: the same source used in the complaint
run — YouTube `wuZ9vuq2oWE`, `output/c745ab5f-2360-4a14-a8fd-3c4bee57af6b/`.
Every fix that changes rendered output must be demonstrated on that video, not
on a synthetic clip.

---

## What the plan fixes, mapped to the complaint

| complaint (voice note, 6-aug-2026) | root cause | fix |
|---|---|---|
| camera jitters/shakes on one moving subject; "why is it not stabilized" | eased follow pans step at whole-pixel crop granularity; detector noise enters the target path | PART 1 — smooth the target trajectory + sub-pixel crop stepping |
| CPU pegged at 378%, both GPUs idle during render | no hwaccel decode; crop/scale/overlay/gblur/ass all run on CPU; only NVENC encode is GPU | PART 2 — GPU decode + GPU filters + GPU Demucs |
| clips start mid-sentence, no context | the run fell back to the old engine (skill never ran — now fixed); sentence snapping exists but needs a scene-boundary clamp | PART 3 — verify with the fixed engine, then clamp/vision rules |
| clip ends showing the next person's scene | vision rescue extends `end` past the last scene cut; next speaker already on screen | PART 3 — scene-boundary clamp + end-of-clip vision rejection |
| face ID "doesn't carry duration"; who's talking isn't driving the camera | face ID only renames transcript speakers; framing is LR-ASD/MediaPipe only | PART 4 — bind face-ID tracks into the framing policy |
| project handling: same link reopens the finished job; date only, no time; logs vanish; thumbnails disappear | per-run projects not modeled; logs live in memory; thumbnails tied to transient files | PART 5 — dashboard/project/log/thumbnail fixes |
| quality "still low" | the **source download** is the ceiling: nothing verifies what resolution/bitrate yt-dlp actually got, and the fallback ladder silently accepts ~360p progressive streams | PART 6 — verify/enforce HD at the download gate (re-encode passes are secondary) |

---

# PART 1 — Kill the camera jitter (stabilization)

## Context

The camera path is already eased, but only in two narrow places:

- `CAMERA_STYLE="cut"` (default, `main.py:217`) — subject switches are hard
  cuts; between cuts the shot is *locked*. This is deliberate (measured
  31-jul-2026 against four real shorts) and is **not** the jitter source.
- Long-shot follow (`main.py:137-174`, `LONG_FOLLOW_*`) pans an
  acceleration-limited exponential chase (`SmoothedCameraman._eased_step`,
  `main.py:354`) so a *single moving subject* is followed.

The jitter on follow shots has three mechanical causes, in order of
importance:

1. **Whole-pixel crop stepping.** ffmpeg's `crop` filter only takes integer
   coordinates, and the eased path's own floor is 1 px/frame
   (`EASE_MIN_STEP`, `main.py:232`). On a ~660px-wide crop scaled down to a
   ~400px output, 1 source px is a visible discrete jump, and sub-pixel
   crawl is rendered as freeze-then-jump (documented at `main.py:228-238`).
2. **Detector noise enters the chase.** Target updates come from MediaPipe
   detections every `DETECT_STRIDE` frames; box-center jitter between
   detections feeds the chase directly. Jump-confirmation filters *teleports*
   (`main.py:345-371`) but not ordinary ±2–5px box wobble, which the eased
   chase dutifully follows.
3. **Whole-pixel quantization + non-2× scaling** make the visible step
   size non-uniform (a 1px source step maps to 0.5–0.7 output px depending on
   position parity), which reads as uneven judder rather than smooth motion.

## What to do

### 1.1 Instrument the path first (do not tune blind)

Add `REFRAME_DUMP_PATH=<dir>` to `reframe_v2.render` (`reframe_v2.py:2806`):
write `rects` (the per-frame eased rects) and the raw target centers as
`np.save` files. Render the complaint clip's clip 2 (`588.5s–626.2s` of
`wuZ9vuq2oWE`) and diff target vs. emitted crop: the diff shows how much of
the jitter is detector noise (cause 2) vs. quantization (causes 1/3).

### 1.2 Smooth the *target*, not the emitted path

The chase already smooths the emitted path; feeding it a noisy target makes
it chase noise. Apply a one-sided smoothing filter to `target_center_x/y`
inside `SmoothedCameraman.update_target` (`main.py:480-586`) before the
chase: a 5-tap centered Savitzky-Golay on the last detections, or an
exponential moving average with `alpha ≈ LONG_FOLLOW_RATE`. Keep the
existing jump-confirm gate; this only smooths ordinary box wobble.

### 1.3 Sub-pixel stepping via 2× supersampling

The honest fix for integer-crop judder: render the crop at 2× source
resolution, then downscale to the delivery size — so a 1px eased step
becomes 0.5 source px and parity-dependent steps disappear.

- In `unified_filtergraph` / `unified_split_filtergraph`
  (`reframe_v2.py:335`, `:383`): insert `scale=iw*2:ih*2:flags=bicubic` after
  the source input, multiply the sendcmd crop coordinates by 2
  (`dedupe_sendcmd_lines`, `reframe_v2.py:84`), and let the existing final
  `scale={fg_w}:{fg_h}` produce the delivery size. GPU makes this nearly
  free (see PART 2: `scale_cuda` instead of CPU `scale`).
- This subsumes the `EASE_MIN_STEP=1.0` floor: with 2× supersampling the
  floor should drop to 0.5 px (still no sub-pixel stall, but half the step).

### 1.4 Acceptance for this part

- Pixel-level check, not eyeballing: dump the crop path with 1.1 and assert
  the per-frame delta of the emitted x is ≤ 2 source px with no
  freeze-then-jump pattern on the complaint clip's follow segments.
- Re-render clip 2 and confirm no frame shows the face clipped (see the
  "face cropped / showing the body" complaint — the crop must center on the
  face centroid with safe margins, mirroring the split-cell final-crop logic
  at `reframe_v2.py:238`).

### 1.5 Keep the face fully in frame (the "it cropped the face, it's showing
the body" complaint)

Owner report (6-aug-2026): *"the crop, aside the jittering, it just completely
cropped the face. It's showing the body as the subject is talking."* Two
distinct defects, both worth their own fix:

- **The crop y is anchored to the subject box, not the face.** When evidence
  is weak (lip-sync 31% on clip 2) the tracker can hold a stale/lower target
  and the crop centers on the torso. The head anchor constant exists
  (`CAMERA_HEAD_ANCHOR`, `main.py:183`, with the top-third grid target
  `CAMERA_HEAD_Y` at `main.py:190`); enforce it on every framed shot — while
  a subject is identified and speaking, re-center the crop y on the head
  anchor every detection, and never let the emitted rect drift below it.
- **Zoomed-in crops can exceed the headroom.** When `current_zoom` tightens
  the crop, the y-center math (`SmoothedCameraman`, `main.py:475-520`) must
  guarantee the face box stays inside the crop with ≥ the safe margin the
  split cells already enforce (`reframe_v2.py:238`). Add a pixel check to the
  acceptance: for every frame of the complaint clip, the face box fully
  inside the emitted crop.

Acceptance: a frame-by-frame overlay (face box + crop rect burned into a
debug render) on `wuZ9vuq2oWE` clip 2 shows the face inside the crop 100% of
frames.

---

# PART 2 — GPU rendering (owner request: "yes i want gpu rendering to save time")

## Context

Current state, verified by grep on 6-aug-2026:

- **No hwaccel anywhere.** No `hwaccel`/`cuvid`/`nvdec`/`scale_cuda`/
  `scale_npp` in `reframe_v2.py`, `ffmpeg_utils.py`, `main.py` or
  `asd_worker.py`. Decode, crop, scale, split, overlay, `gblur sigma=24`,
  `eq`, `unsharp` and the `ass` caption burn all run on CPU.
- **NVENC is already live** (`ffmpeg_utils.py:133` `video_encode_args`,
  `_NVENC_ARGS` `:50`; `_probe_nvenc` `:107`). The Kaggle log shows
  `video encoder: h264_nvenc` — do not replace it with x264.
- LR-ASD + YOLO use the GPU (torch); MediaPipe is CPU-only by design.
- Demucs (background-audio removal) runs on CPU in `audio_cleanup.py:63-106`.
- So "CPU 378%, GPUs idle" is architecture, not a bug — and the fix is the
  plan below, starting with a probe because Kaggle's ffmpeg build is not
  guaranteed to ship CUDA filters.

## What to do

### 2.1 Probe first — one helper, one env var

Add to `ffmpeg_utils.py`:

- `gpu_render_available()` — a single cached probe (mirror
  `_probe_nvenc`): run `ffmpeg -hwaccels` for `cuda`/`nvdec`, then check
  `ffmpeg -filters` for the filters this build actually has:
  `scale_cuda`, `scale_npp`, `overlay_cuda` (if present), `crop_cuda` (if
  present — do not assume). Report exactly what exists.
- `GPU_RENDER` env var, `auto|0|1`, `auto` = probe result. The CPU
  filtergraph stays the default path until the GPU path passes the same
  tests.
- If Kaggle's ffmpeg lacks CUDA decode/filters, add an optional bootstrap
  step in `kaggle_bootstrap.sh` to install a static ffmpeg build with CUDA
  (BtbN nvidia build or conda-forge `ffmpeg` with the nvidia channel); keep
  NVENC-only mode as the guaranteed fallback.

### 2.2 GPU decode everywhere clips are processed

Add `-hwaccel cuda -hwaccel_output_format cuda -extra_hw_frames 16` (or
`-hwaccel nvdec` where the build prefers it) to every clip-processing
ffmpeg invocation:

- initial clip cut — `main.py:3419` (`cut_command`)
- jump-cut spans — `main.py:2667` (`_build_jump_cut_source`)
- reframe render — `reframe_v2.py:3023`
- caption burn — `subtitles.py:861` (`burn_subtitles`)

On multi-GPU hosts the worker threads must pin the device: use
`-hwaccel_device {n}` with the thread-local GPU index from `gpu_affinity.py`
(do **not** use `CUDA_VISIBLE_DEVICES` per worker — see
`KAGGLE_OPTIMIZATION_PLAN.md` Finding 4; CUDA reads it once at init).

### 2.3 GPU filtergraph for the reframe render

In `reframe_v2.render` (`reframe_v2.py:3023`), when `GPU_RENDER` is on:

- Keep frames on the GPU through the hot path: `scale_cuda`/`scale_npp` for
  the background blow-up and the foreground scale, `overlay_cuda` for the
  two overlay steps if the build has it.
- `crop` (sendcmd-driven) and `ass` have no reliable CUDA variants: keep
  them CPU, but minimize the CPU surface:
  - `hwdownload` → `crop` → `hwupload_cuda` only around the crop when
    `crop_cuda` is absent (crop is cheap; decode/scale/overlay/blur are
    not).
  - `ass` burn stays CPU (libass has no CUDA) — but with PART 6 it runs
    once inside the same graph, then `hwupload_cuda` before the NVENC
    encode.
- Replace the `gblur sigma=24` on full-size background frames
  (`reframe_v2.py:367`) with blur-on-small: scale the background to ~1/8,
  `gblur`, scale back up. The background is deliberately out of focus; this
  removes the single most expensive CPU filter even in CPU mode.

### 2.4 Demucs on GPU

In `audio_cleanup.py:63-106`, pass the CUDA device to Demucs when available
(`DEMUCS_DEVICE=cuda` with CPU fallback — mirror the torch device handling
in `asd_worker.py:138`). This moves "remove background audio" off the CPU.

### 2.5 Verification

- `/usr/bin/time -v` a fixed 60s clip render before/after on the 2×T4 host;
  record wall time, max CPU%, and GPU utilization (`nvidia-smi dmon`).
  Target: render wall time cut by ≥40% on GPU hosts, CPU% no longer pegged.
- Same test suite green (654 tests); a `GPU_RENDER=1` CI-style smoke render
  on the host, comparing frame hashes vs. `GPU_RENDER=0` (allow tiny NVENC
  nondeterminism — compare a downscaled frame hash, not exact bytes).
- Update the Kaggle notebook's env cell to set `GPU_RENDER=1` and the
  bootstrap to export it, then one real Kaggle run of `wuZ9vuq2oWE`.

---

# PART 3 — Clip boundaries: mid-sentence starts, no context, next-scene bleed

## Context

The infrastructure for good boundaries already exists and is used:

- `clip_selection.sentence_boundaries` (`clip_selection.py:124`) extracts
  real sentence edges from punctuated word tokens.
- `snap_clip_to_words` (`clip_selection.py:188`) snaps starts to sentence
  starts and ends to sentence ends, with a `context_start` back-off for
  Q&A.
- `_snap_candidates` (`main.py:2896`) applies it to every engine's output,
  and Gemini vision confirmation (`confirm_clip_with_vision`,
  `main.py:2390`) rescues boundaries, including pulling the start earlier
  for missing context (`main.py:2476`).

The complaint run never exercised any of this — the skill response was
thrown away and the old engine picked the clips. **Step 3.1 is a
re-run, not a code change.** After that, two real holes remain:

1. The end rescue can extend a clip *past* the last scene cut into the next
   person's shot (the "clip has ended but you still see another person"
   complaint). Nothing clamps `end` to a scene boundary.
2. A clip can open with a different person's face filling the frame for a
   beat before the actual speaker appears (the "old person was covering the
   frame" complaint). Vision rescue only checks *context*, not
   "who is physically on screen at frame 0".

## What to do

### 3.1 Re-run with the fixed engine before changing code

Reproduce `wuZ9vuq2oWE` end-to-end on the current branch (skill engine now
active). Check the log for `Viral Clip Finder` (not the `Gemini (narrative)`
fallback) and confirm whether boundary complaints persist. Every item below
presumes they do.

### 3.2 Scene-boundary clamp on clip ends

- Add a cached `scene_boundaries(source)` helper (the work is already done
  per render inside `reframe_v2.render` via `main.detect_scenes`; hoist it
  so selection and render share one result).
- After `_snap_candidates` (`main.py:2896-2907`), clamp each candidate's
  `end` to the last scene boundary ≤ `end` (never extend past a scene cut).
  If that clamp drops the clip below `min_duration`, keep the snapped end
  but let vision re-review (see 3.3) — never silently ship a truncated
  sentence.

### 3.3 Vision confirmation rules for frame-0 and frame-last

Extend `confirm_clip_with_vision` (`main.py:2390`) with two rejection
reasons, mirroring the existing retry pattern (`main.py:2498-2510`):

- **"different-speaker open"** — a face other than the selected speaker
  fills the frame at the clip's start; rescue by pulling `start` later (the
  existing rescue only ever pulls *earlier*).
- **"next-scene bleed"** — an identified face other than the selected
  speaker is on screen in the final ~0.5s; rescue by pulling `end` back to
  the scene boundary, never forward.

### 3.4 Face-ID on-screen ranges as a boundary constraint

When `FACE_ID_DB` is set, `on_screen` ranges (already produced by
`face_id.py` and documented in `viral_clip_finder_skill/references/
integration-openshorts.md:425-461`) constrain snapping: never extend a clip
end past the selected speaker's `on_screen` end, and prefer starts where the
speaker is already on screen. This is the same data PART 4 uses to drive the
camera — one source, two consumers.

### 3.5 "Is the model strong enough?" — make the engine visible, don't chase
models

Owner report (6-aug-2026): *"is it that the model is not strong enough? ... we
can add more free models from Grok, or other generals as good as Gemini, or
pay for a cheap model."* The plan's diagnosis (already landed): the clip
selection skill **never ran** in the complaint job — the `reaction_cam: null`
validation error threw away the response and the pipeline fell back to the
pre-upgrade narrative engine (`main.py:2935-2945`). That is a wiring bug, not
a model-capability ceiling, and the fix (`6e2ed31`) is in the branch. To make
this verifiable instead of debatable:

- Log, per job, which engine selected each clip (`Viral Clip Finder (skill)`
  vs `Gemini (narrative)` fallback) and which model/role ran selection and
  vision confirmation. Today the log shows the provider line but not a
  per-clip engine attribution in the metadata.
- Keep the existing provider ladder (Gemini → DeepSeek with transient
  retries) and expose per-role `MODEL` env overrides (selection/vision) so a
  different model — Grok, a paid tier, whatever — can be swapped in without
  code changes. No new provider integration is planned until the skill-run
  fix is verified on a real run (PART 3.1).

The quota/depth question is separate: if a provider's output is truncated or
validation errors recur after the fix, that is a retry/robustness bug in the
worker (`gemini_worker.py`), not a reason to change models.

---

# PART 4 — Face ID drives the camera (feature, not bug)

## Context

Today `FACE_ID_DB` only **renames transcript speakers**:
`face_id.enrich_if_configured` (`main.py:2923-2942`) feeds names into the
skill prompt. It never touches framing. Framing is LR-ASD lip-sync + MediaPipe
boxes + `SpeakerTracker` (`reframe_v2.py:2880-2900`); when lip-sync evidence
is weak (the complaint clip shows `lip-sync 31%, diarized 50%`, key subject
`x=0.22`) the camera hunts.

The skill's integration doc already sketches the missing piece —
`merge_face_id_with_tracker` (`integration-openshorts.md:425-461`): bind
insightface track ids to identity-tracker track ids, producing
`{track_id -> {name, confidence, on_screen}}`.

## What to do

1. Implement `merge_face_id_with_tracker` in `face_id.py` (per the reference
   doc), keyed off the same `identity_tracker` used by framing
   (`reframe_v2.py:2910-2920`).
2. Add a **known-speaker tier** to the subject policy: a candidate whose
   identity matches the currently-speaking diarized/face-ID speaker and is
   within their `on_screen` range outranks anonymous candidates, without
   bypassing the switch cooldown (the same "unlock, never force" pattern
   used for ASD turns at `reframe_v2.py:2886-2895`).
3. Keep it opt-in and fail-open: everything above activates only when
   `FACE_ID_DB` is set; without it, behavior is byte-identical to today.

## Acceptance

On the complaint video with `FACE_ID_DB` set, the clip for the speaker at
`588.5s–626.2s` must hold that speaker (verified by pixel check of the
face-track) for ≥90% of frames, and `on_screen` durations must appear in the
job metadata.

---

# PART 5 — Dashboard: per-run projects, date+time, persisted logs, thumbnails

## Context (all four are real, reported 6-aug-2026)

- **Same link reopens the finished job.** Submitting a URL after a completed
  job lands on the old project (`App.jsx:363-416` `restoreProject` /
  `openProject`). The user wants a **new** project per run; only an explicit
  "reopen" should restore.
- **Date only, no time.** `created_at` is displayed date-truncated; the user
  explicitly wants date *and* time, and two runs of the same link must be
  distinguishable.
- **Logs vanish.** Logs live in the in-memory job record (`app.py:1390`
  `_log_entry` appends to the job dict). After restart/cleanup they're gone —
  the UI even says "No logs retained for this run" (`App.jsx:416`).
- **Thumbnails disappear.** Thumbnails are generated from transient clip
  files; when a clip is re-captioned/re-rendered or the job dir is cleaned
  (`cleanup_jobs`, `app.py:946`), the thumbnail 404s mid-session.

## What to do

### 5.1 Projects are runs; reopening is explicit

- Model each submission as a new project row (job id = run id). In
  `App.jsx`, replace the auto-restore path (`App.jsx:379-397`) with:
  submitted URL + "New project" starts a fresh run; History rows get a
  separate "Reopen" action that calls the existing `restoreProject`
  (`App.jsx:365`).
- Format `created_at` with date+time everywhere in `HistoryTab.jsx` and the
  project rail; key the display by run timestamp so same-link runs are
  visually distinct.

### 5.2 Persist logs per job

- Append every `_log_entry` (`app.py:1390`) to
  `output/<job_id>/logs.jsonl` (same timestamped entries, append-only,
  flushed per write). On completion, write a final snapshot
  `output/<job_id>/logs.json` for the archive.
- Serve persisted logs for archived/restored projects from that file
  (the restore path at `App.jsx:409` already reads `data.logs`; the API just
  needs to source them from disk instead of the in-memory dict). Recovered
  jobs (`_recover_jobs_from_disk`, `app.py:679`) should also replay the log
  file rather than the single "recovered" line.

### 5.3 Thumbnails survive and regenerate

- Generate thumbnails from the **canonical final clip** after
  `_mark_clip_ready` (not from `clip_temp_path`, which is deleted), named
  `clip_{i}_thumb.jpg` in the job dir (not a timestamped transient).
- Backfill: when `/thumbnails/<job>/<name>` misses, regenerate from the
  canonical clip on demand (the source still exists; one `ffmpeg -ss`
  still-frame extract, pattern already in `main.py:2075`).

---

# PART 6 — Source quality: the download is the ceiling

## Context

Owner report (6-aug-2026): *"the quality problem is that it didn't download in
HD to begin with — the file size was very low."* The evidence in the complaint
run's own log agrees, with an important nuance:

```
02:47:06  [download] 0.0% of 103.71MiB ... 100% of 103.71MiB in 00:00:12
02:47:07  [download] 0.0% of 9.67MiB ...    100% of 9.67MiB
02:47:08  Deleting original file .../People_will_look_up_to_you.f137.mp4
02:47:08  Deleting original file .../People_will_look_up_to_you.f140-17.m4a
02:47:08  ✅ Download succeeded (anonymous).
```

The anonymous **HD attempt** (`main.py:1525`, `_hd_fmt_for` → bestvideo
`[vcodec^=avc1][height<=1080]` + bestaudio) is what ran — `f137` is YouTube's
1080p H.264 stream. So the downloader *requested* HD and got it. The problem
is the other half of the sentence: **~103.71 MiB for a 10.5-minute source ≈
1.4 Mbps, which is a low-bitrate 1080p stream** (typical 1080p is 2.5–8 Mbps).
Either the uploader's source is genuinely low-bitrate (very common for
slideshow/motivational content), or a better stream exists and the selector
picked a weak one. Nothing in the pipeline checks, because:

1. **No post-download verification.** After `✅ Download succeeded`, the job
   never ffprobes the file. Resolution, bitrate and codec are never logged,
   so a 360p fallback or a soft 1080p stream ships silently as "HD".
2. **The fallback ladder still ends in ~360p.** `fallback_fmt =
   'best[ext=mp4]/best'` (`main.py:1492`) is a pre-merged progressive stream,
   which YouTube caps at ~360–720p. If the anonymous HD attempt fails on a
   flagged host, the ladder quietly lands there and the job proceeds on
   low-res source with only a "Download attempt 'fallback'" line in the log.
3. **The reframe inherits the source ceiling.** The crop is derived from
   source height (`reframe_v2.py` `SmoothedCameraman`); a 720p or 360p
   source can never deliver a sharp 9:16 clip, no matter how good the
   renderer is. GPU rendering and re-encode hygiene cannot fix this — only
   the download gate can.

Secondary (still real, now much smaller): every clip is lossy-encoded 3–4
times — initial cut (`main.py:3419`) → jump-cut spans (`main.py:2667`) →
reframe render (`reframe_v2.py:3023`) → caption burn (`subtitles.py:861`).
That adds generational softness on top of a weak source, but it is **not**
the primary quality problem the owner is reporting.

## What to do

### 6.1 Measure the real format table first

From the **Kaggle host** (unflagged IP — this studio host is bot-walled,
verified 6-aug-2026), run the exact selector the pipeline uses:

```
yt-dlp -F --extractor-args "youtube:player_client=tv_embed,android,mweb,web" \
  "https://www.youtube.com/watch?v=wuZ9vuq2oWE"
```

Record, per format id: resolution, fps, `~#` bitrate, size estimate, codec.
This tells us whether the complaint video's `f137` really is the best
available stream or whether a higher-bitrate option exists (e.g. vp9/av01,
1080p60, or a higher-bitrate avc1 id) that the `[vcodec^=avc1]` filter
unnecessarily excludes.

### 6.2 Verify every download after it finishes

- After `✅ Download succeeded` (`main.py:1540`), ffprobe the merged file and
  log `resolution`, `fps`, `bitrate`, `codec`, `size` on their own line (the
  same style as the existing `📥`/`✅` markers, so it shows up in the job
  logs the user reads).
- Add `MIN_SOURCE_HEIGHT` (default 720) and `MIN_SOURCE_BITRATE_Mbps`
  (default ~1.5) env gates. Below the floor: log a loud `⚠️ LOW-QUALITY
  SOURCE: <WxH> @ <X> Mbps — clips will look soft` warning AND try the next
  ladder attempt that can produce a better stream (see 6.3). If no better
  stream exists (the uploader's ceiling), surface the warning in the
  dashboard logs so the user knows the clip quality is source-limited, not
  pipeline-limited.
- Store the verified resolution/bitrate in the job metadata (`metadata.json`
  fields) so the dashboard can show "Source: 1080p · 1.4 Mbps" next to the
  clips.

### 6.3 Never silently ship the 360p fallback

- Re-order the ladder so an HD attempt **with cookies + PO token**
  (`main.py:1526-1534`, `hd_args`/`_direct_first`) runs before the
  progressive-stream fallback, not just before it on proxy hosts.
- Label the fallback attempt `fallback (LOW-RES PROGRESSIVE)` in the log and
  append a job-log warning: *"delivering from a pre-merged stream (~360-720p
  ceiling); HD was unavailable on every attempt."*
- Consider relaxing `[vcodec^=avc1]` when the measurement in 6.1 shows a
  meaningfully better non-avc1 stream (vp9/av01): the merge already handles
  it, and `-movflags +faststart` output is agnostic to input codec.

### 6.4 One lossy encode per clip (the secondary quality item)

Once the source is actually HD, cut the 3–4 encode generations down so the
source sharpness survives to delivery:

- Burn captions inside the reframe render: `render()` already receives
  `transcript`, `clip_start`, `clip_end`; extend it with an optional
  `ass_path` and append `ass=...` to the filter_complex (`reframe_v2.py:3023`),
  eliminating the standalone caption-burn re-encode (`subtitles.py:861`).
  Generate the ASS from the **remapped** transcript after a jump-cut
  (`_remap_transcript_onto_jump_cut`, `main.py:3413`) so caption timing
  matches the render clock.
- Keep the initial cut and jump-cut spans (`main.py:2667`, `main.py:3419`)
  as the only intermediates, at crf 14–16 (or NVENC `-cq 22`); the reframe
  pass becomes the single lossy delivery encode.

## Acceptance

- A download of `wuZ9vuq2oWE` logs its true resolution + bitrate on the
  Kaggle host, and the plan records what the best available stream is.
- A synthetic test (mocked yt-dlp info dict) proves: a 360p/720p/low-bitrate
  result triggers the loud warning; a low-res fallback is never silent; and
  the verified specs land in job metadata.
- Re-render the complaint video; the delivered clips' sharpness is capped by
  the source, and the dashboard shows the source specs so the owner can tell
  the difference.

---

# PART 7 — Acceptance checklist (run in order)

1. `python3 -m pytest tests/ -q` → 654 passed, 21 skipped (no drops).
2. Host GPU probe: `ffmpeg -hwaccels` + filter list recorded in the commit
   message; `GPU_RENDER` behavior on a host *without* CUDA filters falls back
   to the current CPU graph cleanly.
3. Render the complaint video `wuZ9vuq2oWE` end-to-end with the fixed skill
   engine. Log must show `Viral Clip Finder` succeeded (no
   `Gemini (narrative)` fallback, no `reaction_cam` validation error).
4. The download log line shows the verified source specs (resolution,
   bitrate) and the source passes the HD floor — or a loud `LOW-QUALITY
   SOURCE` warning appears with the measured numbers, per PART 6.
5. Watch the delivered clips: no mid-sentence starts, no next-scene bleed at
   clip end, no face clipping, no 1px judder on the follow segments, captions
   at the position chosen in the UI.
6. Background-audio removal verified by waveform (not by ear): silence below
   60 Hz and no music bed in the delivered file.
7. A second run of the same URL creates a **new** project row with date+time;
   the finished job's logs survive a server restart and are visible in
   History; thumbnails survive re-captioning.
8. GPU hosts: render wall time and CPU% measured before/after (PART 2.5
   numbers reported in the PR/commit).

---

## Explicit non-goals (do not do)

- Do not re-enable per-clip GPU sharding of scene detection (broke TransNetV2;
  keep `CLIP_GPUS` opt-in).
- Do not move the background-audio cleanup back after caption burn.
- Do not replace NVENC with x264, or `h264_nvenc` with a raw-frame pipe, to
  "fix quality" — see PART 6 for the real fix.
- Do not add camera *pans* between subjects: `CAMERA_STYLE="cut"` is the
  measured right default; the stabilization work is for follow shots only.
- Do not change the skill prompt or disable the skill engine because a run
  looks bad — the fix for the bad run was the null fix, already landed.
- Do not change the 3:4 working frame: the owner's spec is "the working frame
  is 3:4, in a 9:16 canvas" — `UNIFIED_CROP_RATIO = 3/4`
  (`reframe_v2.py:324`) is the fixed crop shape; nothing in this plan alters
  it.
