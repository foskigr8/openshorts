# Master plan: OpenShorts framing rebuild — full build history + everything left to do

**This is the single entry point.** It supersedes `HANDOFF_PHASE5.md` (kept
in the repo, now just a pointer here) and is written to be executed by
anyone/anything with zero prior context — no chat history, no memory of this
session. Read it top to bottom before running anything.

Repo: `foskigr8/openshorts`. Branch: `claude/gemini-vision-clip-picking-bikvuy`.
`CLAUDE.md` at repo root is the permanent project reference and stays
current — this document is the one-time build log + remaining work for one
specific effort (the framing/reframing rebuild), written in full because the
owner needs to hand this off to another engine (DeepSeek) and cannot spend
more tokens re-explaining it.

> **STATUS (8-aug-2026): the rebuild is shipped end-to-end. `reframe_v3` is
> now the ONLY reframe engine.** `reframe_v2.py`, `subject_policy.py`,
> `tools_centrecheck.py`, `SmoothedCameraman`, `SpeakerTracker`,
> `DETECT_LOCK`, the MediaPipe/YOLO detectors, and the `REFRAME_ENGINE`
> switch were all deleted — there is no engine selection and no v3→v2
> fallback. The Kaggle notebook is workflow-only (secrets → clone →
> bootstrap → smoke test). Everything in this doc that refers to v1/v2 as
> "current" describes the historical state before this cleanup; read it as
> the build log it is, not as instructions to resurrect old engines.

---

## 0. What this whole effort is, in one paragraph

The camera used to frame whoever had the **biggest face**, not whoever
mattered — wrong person framed in multi-person shots, crops landing *between*
two speakers during cross-talk ("half a person" cut off), visible jitter
because the crop was re-decided **every single frame**, reactively, even when
nothing about who should be on screen had changed. The fix being built here
plans the **whole shot list up front** (a handful of static crops, cut
between them, never panned within one) and aims the camera using **both** who
is talking (active-speaker detection) **and** where a human eye would
actually look (trained saliency), because the old engine's only question —
"who is talking" — is the wrong question during a reaction (someone pops a
balloon, the speaker is not the attraction).

Everything below (§1-§5) is **done, tested, committed, pushed**, and §6
(Phase 5: wiring v3 into an actual render) shipped with the v3-only cleanup
on 8-aug-2026 — `main.py` now calls `reframe_v3.render` directly and the old
engines are gone. §7-§9 cover three problems discovered along the way; each
was addressed during the build and is recorded here as history.

---

## 1. Full build history — every phase, in order

Read the files themselves; each has a "WHY THIS EXISTS" docstring at the top
with more depth than this summary. This section exists so nobody has to
reconstruct *why* a design choice was made from the diff alone.

### Phase 0 — `eval/` (measurement harness)
Framing-quality measurement tooling. The standing rule that came out of this
phase and applies to everything after it: **every framing fix ships with a
metric that moved, or it is not a fix.** Also in this phase: nvenc (GPU video
encoding) was missing on the Kaggle host — renders were silently falling back
to CPU encoding. Fixed in `kaggle_bootstrap.sh`.

### Phase 1 — `face_spine.py` (343 lines)
ONE shared face-track spine, replacing two previously-independent trackers
(LR-ASD had its own ByteTrack instance, the renderer had a second unrelated
one — matching between them happened by pixel position every frame, with a
decisiveness gate that silently discarded evidence when two candidates looked
similar). Pipeline: SCRFD detection -> ByteTrack (per-scene) -> ArcFace
re-identification (merges the same person's tracks back together across a
scene cut). Output:
```python
{track_id: {"frames": [t, ...], "boxes": [(x, y, w, h), ...],   # PIXELS
            "landmarks": [...], "embeddings": [...], "det_scores": [...]}}
```
**Boxes are pixel coordinates**, straight from InsightFace's bbox — this
convention holds through every module downstream; nothing normalizes them
except at the one boundary described in Phase 4b below. Deliberately does NOT
include YOLO/body boxes (a body box's centre sits far from a face box's
centre for the same person; mixing them was the root of a real instability —
see the module's own docstring for detail).

### Phase 2 — ASD (active speaker detection) bake-off
`eval/asd_bakeoff.py`, model-agnostic scoring harness comparing LR-ASD vs
LoCoNet vs a UniTalk model. **Decision: LR-ASD.** Vendored into
`vendor/lrasd/` (MIT). Recorded so it is not silently re-litigated.

### Phase 3 — `speaker_fusion.py` (233 lines)
Binds LR-ASD's "who is talking" output to a `face_spine` track ID **once per
clip**, not by fragile pixel-position matching every frame. Output:
`per_second_active_track: List[Optional[int]]`.
Known, documented limitation: this is 1-second granularity today (LR-ASD's
own scores are 25fps and could be used at finer resolution later — flagged,
not fixed, in this phase; `shot_planner.py` was deliberately built to accept
arbitrary-resolution `(timestamp, value)` samples so this can be upgraded
later without changing the planner).

### Phase 4 — `shot_planner.py` (528 lines)
Plans the WHOLE shot list up front: a small number of
`Shot(start, end, shot_type, track_ids, crop_rect)` entries, each with ONE
static crop for its whole duration — nothing left to re-aim within a shot, so
jitter isn't reduced, it's **structurally impossible**. Key pieces:
- `plan_shots_from_samples` / `plan_shots` — noisy per-second samples -> shot list
- `crop_rect_for_track(spine_tracks, track_id, start, end)` — the **median**
  box of a track within a span (robust to one noisy/occluded frame; a
  min/max union would not be). Returns a *subject* box, not yet a *camera*
  crop (Phase 4b does that conversion).
- `insert_reaction_shots` / `is_directive_corroborated` — lets a Gemini
  "something happened here" directive insert a brief reaction cutaway, but
  ONLY when corroborated by real motion evidence. This exists because of a
  **documented real failure**: Gemini once hallucinated a `causing_reaction`
  beat with nobody actually reacting at that timestamp. The corroboration
  gate is deliberately loose (`MIN_CORROBORATION_DETECTIONS = 1`) because a
  false rejection (blocking a genuine but subtle reaction) is worse than a
  false acceptance here — a rejected directive just falls back to the base
  shot, never a broken render.
- `apply_two_shot` — widens a shot to a two-person composition given an
  addressee signal (who a speaker is talking to). That signal itself
  (head pose / 6DRepNet) is **not built** — this function takes it as a plain
  argument so wiring it in later needs no redesign here. Do not build the
  addressee detector as part of Phase 5 unless explicitly asked; it is a
  separate, later piece of work.

### Phase 4b — composition (this session's main work)

**Step 0 — `vendor/pyautoflip/`.** The owner pointed at the `pyautoflip`
library (PyPI, MIT, github.com/AhmedHisham1/pyautoflip, "inspired by"
Google's AutoFlip) and asked why hand-write cropping when a library exists.
**The full source was read before deciding** (not just the README) — see
§2 for the complete reasoning. Two pieces were vendored verbatim (kept
byte-identical to upstream so future releases stay diffable):
- `saliency_detector.py` + `unisal.onnx` (+`.data`, ~13MB) — UNISAL
  ("UNIfied SALiency"), a model trained on human eye-tracking data, ~17ms/frame
  on CPU via ONNX Runtime. Answers "where would a human look" — the signal
  that covers reactions active-speaker detection cannot see.
- `split_screen.py` — `find_split_faces` (pure geometry: do two faces need a
  split?) and `render_split_screen_from_centers` (renders the 2-panel layout).

Everything else in that library — its cropper, its camera-motion/smoothing
handler, its padding function — was read and **rejected**, each for a
specific found defect (full detail, keep it there, don't duplicate: see
`vendor/pyautoflip/README.md`). Short version: its "hold still" mode averages
key-frame positions (on two speakers that's the gap *between* them — the
exact bug this whole rebuild exists to fix), its crop window has zero
vertical composition ability, and its padding stretches the image rather than
letterboxing it.

Also decisive: pyautoflip cannot be a pip dependency — `numpy>=1.24`
(unbounded, resolves to numpy 2.x) and `torch>=2.11` reproduce the exact
dependency-chain breakage `CLAUDE.md` already documents for `insightface`
(see the `numpy<2` guard in `requirements.txt`). The two vendored files add
**zero** new dependencies (they import only `cv2`/`numpy`/`onnxruntime`, all
already in the stack).

**Steps 1-4 — `reframe_v3.py` (435 lines).** Turns a shot's subject box into
an actual camera crop, and decides single/two-shot/split layout. Pure numpy
(no GPU needed for the geometry; only `split_centers` touches vendored code
that needs opencv).
- `build_attention_map(saliency, faces)` — fuses UNISAL saliency with
  role-weighted face boxes (`WeightedFace(box, role)`: `ROLE_SPEAKER` /
  `ROLE_REACTOR` / `ROLE_BYSTANDER`). Speaker/reactor get boosted; bystanders
  are suppressed **multiplicatively** (`x0.25`) — deliberately different from
  pyautoflip's `get_composite_mask`, which only ever *raises* values
  (`np.maximum`), so a bystander on a bright background keeps full saliency
  and still drags the crop toward them. That IS the biggest/brightest-face
  bug. Scaling down is what actually removes them from contention.
- `crop_rect_containing(subject, frame_w, frame_h, aspect, head_y=0.36)` —
  the crop geometry. Places the subject's head at 36% down the frame
  (matches `main.CAMERA_HEAD_Y`, already tuned on this footage) whenever the
  crop is tighter than full height. **Containment is the input constraint,
  not a property checked after the fact** — the subject-plus-margins box
  drives the geometry, so "half a person" becomes a failing assertion, not a
  tuning problem.
- `decide_layout(subjects, speaker_shares, frame_w, frame_h)` — chooses
  `LAYOUT_SINGLE` / `LAYOUT_TWO_SHOT` / `LAYOUT_SPLIT`. Deliberately
  **editorial, not just geometric**: two far-apart faces only trigger a split
  if the speaker-share evidence says both people are genuinely live in the
  exchange (`MIN_SECOND_SPEAKER_SHARE = 0.25`) — otherwise (one person
  holding the floor) it stays a single, because giving half the frame to a
  silent listener wastes the format.

**Step 6 — validation.** `validate_composition(shots, frame_w, frame_h)`
raises `CompositionError` (lists every violation found, not just the first)
rather than silently repairing anything — containment failures, aspect
mismatches, crops escaping the frame, shots too short. Consistent with the
"no silent fallback" rule: every framing bug this rebuild fixes shipped
silently before (a crop that cut a person in half still produced a playable
file), so a violation here should stop the job loudly.

**41 tests, `tests/test_reframe_v3.py`, all passing.** Plus 14 in
`tests/test_vendor_pyautoflip.py` pinning the vendored pieces' contract (so a
future pyautoflip version bump that changes behaviour fails a test instead of
silently changing how clips look). **160 tests green** across the whole
rebuild suite (`test_eval_metrics.py`, `test_face_spine.py`,
`test_asd_bakeoff.py`, `test_speaker_fusion.py`, `test_shot_planner.py`,
`test_vendor_pyautoflip.py`, `test_reframe_v3.py`).

---

## 2. Full reasoning on the pyautoflip decision (for anyone who questions it later)

This gets its own section because it was the single most scrutinized decision
in the session and the owner explicitly wanted the reasoning kept, not just
the conclusion.

The claim "AutoFlip / pyautoflip has years of training behind its cropper" is
**false as stated**. Google's own AutoFlip documentation describes its
cropping as a hand-written heuristic: per-object-type weights and a
`motion_stabilization_threshold_percent` deciding stable-vs-track. What was
actually trained-for-years in that lineage is the **detectors** (face/object
finders) — and this rebuild already replaced those, in Phase 1, with newer
models (SCRFD + ArcFace) than what AutoFlip used. MediaPipe also
**discontinued AutoFlip support on 1 March 2023** — it's a legacy solution.

`pyautoflip` (the PyPI package actually reviewed, since the original AutoFlip
C++ repo is not directly usable here) is a ~1-year-old reimplementation
"inspired by" AutoFlip, not Google's own battle-tested code. Reading its
actual source (not just its README, which oversold what it does) found:
1. Its `STATIONARY` camera mode literally computes
   `avg_x = sum(key_xs) / len(key_xs)` — the average of where it looked
   across a scene. Two speakers on opposite sides of frame -> the average
   crop sits on the empty space between them. This is not a hypothetical; the
   whole rebuild started because production clips actually did this.
2. `compute_crop_window` always returns `y=0, height=full_frame` — it can
   only pan left/right, never adjust vertical head placement.
3. `apply_padding_to_crop` doesn't letterbox; it stretches the crop to fill
   the output then darkens bars over the *distorted* result — faces get
   squashed exactly in the two-person wide-crop case.

So the honest framing is: **there was no proven cropper to take.** There was
a proven *saliency model* (UNISAL, genuinely trained on human eye-tracking
data) and a genuinely useful, detector-agnostic split-screen geometry
function. Both were taken. The cropping geometry that was hand-written
(`reframe_v3.crop_rect_containing`) is ~60 lines of arithmetic (place a box
containing the subject at a given aspect ratio and head height) — there is no
version of that which benefits from being "trained," and no library
encodes it better than tuning it on this specific footage, which is what the
existing `main.py` constants (`CAMERA_HEAD_ANCHOR`, `CAMERA_HEAD_Y`,
`COMPOSE_SIDE_BAND`) already represent.

**This is falsifiable and cheap to check**, if anyone wants to re-litigate
it: `pip install pyautoflip` (NOT into this repo's environment — it will
break numpy/torch pins, use a throwaway venv or the Kaggle scratch space) and
run its CLI against the exact same clip `reframe_v3` produces, side by side.
If pyautoflip's end-to-end output looks better, this decision was wrong and
should be revisited. Nobody has done this comparison yet — it is cheap and
would be worth doing once Phase 5 produces a v3 render to compare against.

---

## 3. Architecture diagram (the whole chain)

```
face_spine.py        (Phase 1)   WHO: tracked face boxes, whole clip, pixel coords
        |
speaker_fusion.py     (Phase 3)   WHO IS TALKING: one speaker<->track binding/clip
        |
shot_planner.py        (Phase 4)   WHO HOLDS THE FRAME, HOW LONG: static shot list
        |
reframe_v3.py            (Phase 4b) WHAT THE CROP LOOKS LIKE: saliency + geometry + layout
        |
   >>>>>>>>>>>>>  NOTHING CALLS THIS YET  <<<<<<<<<<<<<     <- Phase 5 starts here (§6)
        |
   render_clip() in main.py -> ships a video
```

---

## 4. Three real bugs found while trying to validate this on Kaggle

These are separate from the framing rebuild itself but block validating it,
so they need fixing first or alongside Phase 5. All three were found by
reading the actual code, not guessed.

### 4a. `main.py:1526` — cookies path hardcoded to the Docker container path — **FIXED**

Was:
```python
cookies_path = '/app/cookies.txt'
```
`/app` is where `docker-compose.yml` bind-mounts the repo **in the Docker
deployment only**. On Kaggle, the notebook clones to
`/kaggle/working/openshorts` (see `openshorts_kaggle.ipynb` cell 2,
`DEST = "/kaggle/working/openshorts"`). `/app/cookies.txt` could never exist
on Kaggle — every Kaggle run was silently falling through to the
`YOUTUBE_COOKIES` env var (legacy path, often stale) or to no cookies at all.

**Fixed** (this commit) to derive the path from the repo's own location:
```python
cookies_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cookies.txt')
```
Resolves to `/app/cookies.txt` in Docker (unchanged there) and to
`/kaggle/working/openshorts/cookies.txt` on Kaggle.

**Found but deliberately NOT touched:** `auto_refresh_cookies.sh` (the script
meant to keep a fresh jar on disk) hardcodes a *third*, different path —
`cd /teamspace/studios/this_studio/openshorts` — which is a Lightning
Studios path, not Docker or Kaggle. It's unclear whether this script is even
still in active use or which environment it targets today. **Do not
guess-fix this** without confirming with the owner which environments
actually run it; it may be dead, or it may need its own Kaggle-aware path
resolution similar to the fix above. Flagged, not resolved.

### 4b. The download failure itself — how to actually diagnose it, not guess

`kaggle_smoke_test.py` already has real, working diagnostic logic for
exactly this (`_youtube` and `_po_token` checks, `CHECKS` list). **Before
changing anything, run:**
```bash
python3 kaggle_smoke_test.py
```
on the Kaggle host and read the **"PO token provider"** and **"YouTube
download"** rows specifically. Per the smoke test's own accumulated
knowledge (comments dated 5-8 aug 2026, from real incidents):
- If **PO token provider** fails: this is "the usual cause of a failed
  YouTube download from a cloud IP" (the test's own words) — YouTube's bot
  wall returns "Sign in to confirm you're not a bot" without one. Fix:
  re-run `kaggle_bootstrap.sh`, read its "YouTube PO token provider" section,
  check `/tmp/openshorts-logs/bgutil.log` for why the provider process died.
- If PO token is fine but **YouTube download** still fails: read the
  `detail` the check prints (last 200 chars of yt-dlp's stderr) — do not
  guess, the actual error string tells you which of several failure modes
  it is (bot wall, geo-block, private/deleted video, rate limit, cookie jar
  expired — different fixes for each).
- If a `cookies.txt` jar exists but is old: it has a **~3 hour lifetime**
  per the smoke test's own comment; re-paste a fresh `YOUTUBE_COOKIES` value
  if the PO token path isn't sufficient on its own.

**Do not treat "download failed" as one bug** — fix §4a first (it's a
guaranteed, permanent gap on Kaggle regardless of everything else), then run
the smoke test and follow whichever specific branch it reports.

### 4c. The notebook doesn't exercise the real pipeline or this session's work

**Status (8-aug-2026): owner decision — the notebook is workflow-only.**
The render/comparison test cells were removed from `openshorts_kaggle.ipynb`
entirely: the notebook now runs secrets → clone → bootstrap → smoke test and
prints the public studio URL, and rendering is tested through the real app
workflow, not notebook scaffolding. The old test cell called
`reframe_v2.render(..., 0.75)` directly — 3:4, not the shipped 9:16 format,
and with no transcript — so removing it also removed a wrong-format test path.
The v2-vs-v3 comparison described below is still the right acceptance method
(§9 item 5), just done via the app rather than a notebook cell.

`openshorts_kaggle.ipynb` cell 8 (the "end-to-end render test") does this:
```python
SRC = "/kaggle/working/test.mp4"          # manually uploaded, not downloaded
import reframe_v2 as r                     # hardcoded to the OLD engine
r.render(SRC, "/kaggle/working/test_vertical.mp4", 0.75)
```
Two problems: (1) it requires a manual local upload, so it never exercises
`main.py`'s actual YouTube download path — meaning **the download failure
the owner hit could not have been meaningfully diagnosed from this cell even
if it had "worked."** The cookie/PO-token logic lives entirely in
`main.py`'s download function, which this cell never touches. (2) it is
hardcoded to `reframe_v2`, so it cannot validate anything built in Phase 4b —
there is currently no cell in the notebook that can render with
`REFRAME_ENGINE=v3` at all, because that engine does not exist yet (Phase 5).

**This is not "the notebook is broken," it's "the notebook was never updated
for this session's work."** Two notebook changes are needed, and both belong
alongside Phase 5, not before it (no point building a validation cell for an
engine that doesn't exist yet):
1. Add a real end-to-end cell that calls `main.py`'s actual `get_transcript`
   -> `download_video` (or whatever the real entry point is called — check
   `main.py` for the actual function name at the top of the pipeline) path
   with a real YouTube URL, so the cookie/PO-token machinery is genuinely
   exercised and any download failure shows up with full context (not just
   "it failed").
2. Once Phase 5 exists, add a cell that renders the same clip twice — once
   with `REFRAME_ENGINE=v2` (env var unset/default) and once with
   `REFRAME_ENGINE=v3` — and downloads/displays both `.mp4` outputs so they
   can be watched side by side. This is the actual acceptance test for
   everything in this document; no automated metric substitutes for it.

---

## 5. Testing (what already exists, run this first before writing new code)

```bash
# Rebuild-phase suite — pure Python, no GPU, no heavy deps:
python3 -m pytest tests/test_eval_metrics.py tests/test_face_spine.py \
  tests/test_asd_bakeoff.py tests/test_speaker_fusion.py \
  tests/test_shot_planner.py tests/test_vendor_pyautoflip.py \
  tests/test_reframe_v3.py -q
# Expect: 160 passed (at handoff time — if this regresses, something broke)
```

**Sandbox note:** a bare dev container may be missing `opencv-python`,
`onnxruntime`, and the app's full deps (PIL, pydantic, httpx, sqlalchemy,
scenedetect, python-dotenv). The two vendor/reframe_v3 test files skip
gracefully via `pytest.importorskip` when `cv2`/`onnxruntime` are absent —
expected, not a failure. Full `tests/` collection in a bare container will
show many unrelated `ModuleNotFoundError`s for app-level deps — pre-existing,
unrelated to this work. Real validation needs the actual
`pip install -r requirements.txt` environment (Docker or Kaggle).

**None of the 160 tests render an actual video.** They are unit tests for
geometry and logic. The real acceptance test, once Phase 5 exists, is
watching a real render — see §4c point 2 and §6's own note on this.

---

## 6. Phase 5 — the one thing left: wire it into the renderer

**Status (8-aug-2026): implemented.** `reframe_v3.render` exists and the
`REFRAME_ENGINE=v3` branch is wired into `main.py` (uncommitted as of this
note; `git status` shows `main.py`, `reframe_v3.py`,
`openshorts_kaggle.ipynb`, `tests/test_reframe_v3.py` modified). What §6
describes below is the design the implementation follows; the remaining
validation is §9 item 5 — watching a real render on Kaggle. Two fixes beyond
the original §6 spec shipped with it: WIDE shots (planner-produced, no
subjects) render as a full-height centre hold instead of failing, and the
per-shot attention map now actually nudges the crop within containment slack
(previously it was computed and discarded — the reaction case had no effect).

The renderer still runs the old per-frame engine (`reframe_v2`) by default;
`REFRAME_ENGINE=v3` opts into the new one. The rest of this section is the
design the implementation follows.

### 6.1 The hook point already exists

`main.py`, around line 2112, inside `reframe_video_with_scene_detection` (or
whatever it's named at the call site — check the function containing this
block):
```python
if os.environ.get("REFRAME_ENGINE", "v2").strip().lower() != "v1":
    try:
        import reframe_v2
        result = reframe_v2.render(input_video, final_output_video, aspect_ratio,
                                   transcript=transcript, clip_start=clip_start, clip_end=clip_end,
                                   focus_directives=focus_directives,
                                   primary_subject_x=primary_subject_x,
                                   ass_filter=ass_filter,
                                   captioned_output=captioned_output)
        return result
    except Exception as e:
        print(f"Reframe v2 failed ... falling back to v1 frame loop")
    # falls through to the old v1 per-frame loop below
```
`REFRAME_ENGINE` is already an env-var switch (currently only v1 vs v2).
Phase 5 adds a third branch checked **before** the v2 branch:
```python
if os.environ.get("REFRAME_ENGINE", "v2").strip().lower() == "v3":
    import reframe_v3
    result = reframe_v3.render(input_video, final_output_video, aspect_ratio,
                               transcript=transcript, clip_start=clip_start, clip_end=clip_end,
                               focus_directives=focus_directives,
                               primary_subject_x=primary_subject_x,
                               ass_filter=ass_filter,
                               captioned_output=captioned_output)
    return result
# ... existing v2/v1 branches unchanged below ...
```
Same call signature as `reframe_v2.render` so `render_clip` (the actual
caller, further up the stack) needs **no change** beyond this new branch.
**Do not wrap the v3 call in a try/except that silently falls back to v2**
the way v2 falls back to v1 — see §6.3 point 5, this is a deliberate,
different choice for v3 and needs the owner's confirmation if changed.

### 6.2 What `reframe_v3.render(...)` (does not exist yet — write it) has to do

1. Run `face_spine.extract_raw_tracks` + `merge_tracks_by_identity` on the
   clip — or reuse tracks if `render_clip`'s caller already computed them
   upstream (check for this before detecting twice; it's wasteful and this
   pipeline already has a documented issue with redundant GPU work).
2. Run `speaker_fusion` to get `per_second_active_track`.
3. Call `shot_planner.plan_shots(...)` to get the `Shot` list.
4. Per shot: sample a few frames, run the vendored `SaliencyDetector`
   (`vendor.pyautoflip.saliency_detector.SaliencyDetector`), build a
   `WeightedFace` list (`ROLE_SPEAKER` from the shot's `track_ids`,
   `ROLE_REACTOR` for any corroborated reaction directive from
   `shot_planner.insert_reaction_shots`, `ROLE_BYSTANDER` for everyone else),
   call `reframe_v3.build_attention_map` -> `attention_center` ->
   `crop_rect_containing`. Call `decide_layout` for any shot with 2+ subjects.
5. Call `reframe_v3.validate_composition` on the full plan. **Let it raise.**
   Per §6.3 point 5, do not catch this and fall back — confirm with the owner
   before changing this if it seems wrong once real footage is tried.
6. Render: hold each shot's `crop_rect` constant, apply via ffmpeg. **Reuse
   `reframe_v2`'s ffmpeg filter construction** rather than reimplementing a
   frame-loop cropper from scratch — read `reframe_v2.render` (starts at
   `reframe_v2.py:2998`) for its existing crop/pad filtergraph approach
   before writing anything new here.
7. For `LAYOUT_SPLIT` shots specifically: use `reframe_v3.split_centers` +
   the vendored `render_split_screen_from_centers`, which currently expects a
   numpy BGR frame array, not an ffmpeg filter expression. Decide: render
   split shots via a Python frame loop (simplest, matches what the vendored
   function expects directly — **recommended starting point**, get something
   watchable first) or reimplement the split as an ffmpeg filtergraph
   (faster, more consistent with the rest of the pipeline, more work — do
   this later if the frame-loop path is too slow in practice, not before).

### 6.3 Open decisions / known gaps — do not silently resolve these, flag them

1. **Mid-shot subject movement.** `crop_rect_for_track` uses one median box
   per shot. If a subject moves far enough within a shot, containment can
   genuinely fail. `validate_composition` is *designed* to catch this and
   raise — that's correct behavior, not a bug to suppress. The real fix
   (split the shot, or allow one slow linear pan) is **not implemented**.
   Phase 5 does not need to solve this, but must not swallow the resulting
   `CompositionError` — let it surface so it's visible how often this
   actually happens on real footage before deciding whether it needs solving.
2. **Captions and per-shot aspect ratio.** `subtitles.py` derives
   `UNIFIED_CONTENT_HEIGHT_RATIO` from the crop ratio. If shots within one
   clip end up with genuinely different aspect ratios (e.g. a tight single vs
   a wide two-shot), captions would move mid-clip unless every shot is
   letterboxed into ONE content band per whole clip (sized to the widest
   aspect the clip actually uses). Do not silently change subtitle
   positioning to work around this — flag it if it comes up.
3. **Audio reaction detection (PANNs/YAMNet)** was planned as a further
   signal (WHEN a reaction happened, from audio, independent of UNISAL's
   WHERE) but was never started — it needs torch plus a ~300MB checkpoint not
   available in the environment this was built in, so it was deliberately
   left unwritten rather than shipped unverified. **Not required for Phase
   5** — UNISAL already covers the visual half of "something is happening
   here, not on the speaker."
4. **Redundant detection.** Check whether `render_clip`'s caller already has
   face-tracking data from an earlier pipeline stage (e.g. Stage 3's vision
   confirmation pass) before running `face_spine` again inside
   `reframe_v3.render` — this pipeline has had real problems with duplicate
   GPU work before (see `gpu_affinity.py`'s existence).
5. **Fallback behavior on a v3 failure**, stated once, deliberately, so it
   isn't re-decided ad hoc: v2 silently falls back to v1 on ANY exception,
   which is appropriate there (v1 is strictly worse but always works, and a
   render failing outright is worse than a mediocre one shipping). v3 is
   different — its whole reason for existing is fixing bugs that used to
   ship silently. Catching `CompositionError` (or anything else) and quietly
   falling back to v2 would resurrect exactly the failure mode this rebuild
   was built to kill. Default recommendation: **do not add a silent v3->v2
   fallback.** If Phase 5 implementation reveals a case where this default is
   wrong (e.g. some fraction of real clips genuinely can't be planned by v3
   and blocking the whole job is unacceptable), that is a real product
   decision — surface it explicitly rather than deciding it in a try/except.

---

## 7. Explicitly OUT of scope — do not touch these while doing the above

- **Stage 3 (clip selection — `viral_clip_finder.py`, `get_viral_clips` in
  `main.py`, docstring literally says "Stage 3 — viral moment selection").**
  This decides **which moments** become clips. Built before this rebuild
  started (commit `98df856`); none of Phases 0-4b have touched it, they only
  change **how** an already-chosen clip is cropped. The owner has explicitly
  parked further Stage 3 work for a separate session — do not conflate the
  two, and do not "improve" Stage 3 as a side effect of Phase 5 work.
  (Also worth knowing: `CLAUDE.md`'s top-of-file pipeline list numbers stages
  1-11 differently from how "Stage 3" is used everywhere else in the code —
  the code/skill/bootstrap usage of "Stage 3" means the AI-analysis/clip-
  selection step, which is item #4 in that top list. This is a real,
  acknowledged inconsistency in the docs, flagged for a future fix, not
  something to silently "correct" by renaming things.)
- **Phase 6 (GPU utilization, both T4s) and Phase 7 (style profile from
  reference shorts)** — separate, later, unrelated phases.
- **Building the addressee/head-pose detector** for `apply_two_shot` — not
  part of Phase 5; that function already accepts the signal as a plain
  argument for when it exists.

---

## 8. Quick orientation commands

```bash
git log --oneline -15                       # this rebuild's full commit history
cat vendor/pyautoflip/README.md             # what was/wasn't taken from pyautoflip, and why
grep -n "REFRAME_ENGINE" main.py            # the exact hook point for the v3 branch
sed -n '2095,2130p' main.py                 # the v2/v1 branch to mirror for v3
sed -n '2998,3060p' reframe_v2.py           # reframe_v2.render's signature/ffmpeg approach to reuse
grep -n "cookies_path" main.py              # the /app hardcode described in §4a
python3 kaggle_smoke_test.py                # run this FIRST on the Kaggle host, before changing anything
python3 -m pytest tests/test_reframe_v3.py tests/test_shot_planner.py \
  tests/test_vendor_pyautoflip.py -q        # confirm the existing 160-test baseline still holds
```

---

## 9. Suggested order of operations for whoever picks this up

1. ~~Fix §4a (the `/app` cookies hardcode)~~ — **done**, see §4a.
2. Run `kaggle_smoke_test.py` on the actual Kaggle host, follow §4b's
   diagnostic branches for whatever it reports now that §4a is fixed.
3. Build Phase 5 per §6 — write `reframe_v3.render(...)`, wire the
   `REFRAME_ENGINE=v3` branch into `main.py`.
4. Update the notebook per §4c: a real download-path e2e cell, then a v2-vs-v3
   side-by-side render cell once v3 exists.
5. Run it on a real clip — ideally a Pop The Balloon span where a reaction,
   not the active speaker, is the moment, since that's the specific case this
   whole rebuild targets — and **watch it**. No automated metric in this repo
   substitutes for that; `eval/` can supplement it with a number, not replace
   the watching.
