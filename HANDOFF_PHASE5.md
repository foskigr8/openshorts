# Task: wire the v3 framing engine into the renderer (Phase 5), then validate on real footage

You are continuing work on OpenShorts, an AI vertical-video clipper (long
YouTube video -> 9:16 shorts). This document is self-contained — read it
fully before running anything, you do not need prior conversation context.

Repo: `foskigr8/openshorts`. Branch: `claude/gemini-vision-clip-picking-bikvuy`,
HEAD `d7edd04` at handoff time. Base it off `main` if that branch is gone; the
work described here lives in the commits listed in §2, all still on the
branch's history — do not re-derive them, read the code.

**Read `CLAUDE.md` first** (repo root) — it is the standing project reference
and stays current; this document is a one-time handoff for finishing one
specific piece of it (the "Framing rebuild (v3)" section there is the
permanent summary of what's described in full below).

---

## 0. The one-sentence status

**A tested composition library exists (`reframe_v3.py` + `shot_planner.py`)
but nothing calls it yet.** The renderer still runs the old per-frame engine.
Phase 5 is wiring them together so a real clip comes out the other end.

---

## 1. Why this rebuild exists (the bug, in plain terms)

The camera used to frame whoever had the **biggest face**, not whoever
mattered. On a multi-person show this means: the wrong person gets framed when
two people are similar distance from camera, the crop lands *between* two
speakers during a cross-talk moment ("half a person" cut off), and there is
visible jitter because the old engine re-decides the crop **every frame**,
reactively — so even when nothing about who should be on screen changed, the
crop is always slightly re-aiming.

The owner's own framing of the fix: *"if you cut something, remove that end,
and the person is in a stationary place, there is no need for you to move the
person."* I.e.: decide the whole shot list up front, hold each crop
perfectly still for its duration, cut (don't pan) when the subject changes.

A second, related problem: the old engine only asks "who is talking"
(active-speaker detection). That is the wrong question during a **reaction**
— someone pops a balloon, someone winces, and the speaker is not the
attraction. This rebuild adds a real saliency signal (UNISAL, trained on human
eye-tracking) so the camera can also aim at "where would a human look," not
only "who is talking."

---

## 2. What already exists — the full chain, file by file

Read these in this order; each one's own docstring explains its reasoning in
more depth than this summary does.

```
face_spine.py       (Phase 1)  -- WHO: tracked face boxes across the clip
        |
speaker_fusion.py    (Phase 3)  -- WHO IS TALKING: one speaker<->track binding
        |
shot_planner.py      (Phase 4)  -- WHO HOLDS THE FRAME, HOW LONG: shot list
        |
reframe_v3.py         (Phase 4b) -- WHAT THE CROP LOOKS LIKE: geometry + saliency
        |
   >>> NOTHING CALLS THIS YET <<<  <-- Phase 5 starts here
```

### `face_spine.py` (343 lines)
SCRFD face detection + ByteTrack (short-term, per-scene) + ArcFace re-identification
(merges tracks across a cut back into the same person). Output:
```python
{track_id: {"frames": [t, ...], "boxes": [(x, y, w, h), ...],  # PIXELS, not normalized
            "landmarks": [...], "embeddings": [...], "det_scores": [...]}}
```
Two-pass: `extract_raw_tracks` (pass 1, per-scene tracks) then
`merge_tracks_by_identity` (pass 2, re-id merge). **Boxes are pixel
coordinates in the source frame**, e.g. from InsightFace `bbox` directly —
this convention is used everywhere downstream, do not normalize it.

### `speaker_fusion.py` (233 lines)
Binds LR-ASD's active-speaker output to a `face_spine` track ID, once per
clip rather than by pixel-position matching every frame (the old, fragile
approach). Produces `per_second_active_track: List[Optional[int]]`.

### `shot_planner.py` (528 lines)
The shot-list planner. Core idea: turn noisy per-second speaker samples into
a small number of `Shot(start, end, shot_type, track_ids, crop_rect)` entries,
each with ONE static crop for its whole duration. Key functions:
- `plan_shots_from_samples` / `plan_shots` — samples -> shot list
- `crop_rect_for_track(spine_tracks, track_id, start, end)` — the **median**
  box of that track within `[start, end)` (robust to one noisy/occluded
  frame; union would not be). This is a *subject* box, not yet a *camera*
  crop — see reframe_v3 below for the distinction.
- `insert_reaction_shots` / `is_directive_corroborated` — lets a Gemini
  "something happened here" directive insert a brief reaction cutaway, but
  ONLY if corroborated by actual motion evidence (a real, documented failure:
  Gemini once hallucinated a reaction with nobody actually reacting — see
  `is_directive_corroborated`'s docstring for the incident).
- `apply_two_shot` — widens a single shot to a two-person composition when an
  addressee signal says who a speaker is talking to (that signal itself does
  not exist yet — head pose / 6DRepNet, not built. `apply_two_shot` takes it
  as a plain argument so wiring it in later needs no redesign here).

### `reframe_v3.py` (435 lines) — **the newest piece, Phase 4b**
Turns a shot's *subject* box into an actual *camera crop*, and decides layout
(single / two-shot / split-screen). This is a pure-numpy library — no GPU, no
opencv needed for most of it (only `split_centers` touches the vendored
split-screen code, which needs opencv).

Three things it does that nothing before it did:

1. **`build_attention_map(saliency, faces)`** — fuses a UNISAL saliency map
   (see `vendor/pyautoflip/`, below) with role-weighted face boxes
   (`WeightedFace(box, role)`, roles are `ROLE_SPEAKER` / `ROLE_REACTOR` /
   `ROLE_BYSTANDER`). Speaker and reactor roles get boosted; bystanders are
   **suppressed multiplicatively** (`x0.25`, `BYSTANDER_SUPPRESSION`) — this
   is a deliberate departure from the pyautoflip library this was inspired by
   (see §3): its `get_composite_mask` only ever *raises* values
   (`np.maximum`), so a bystander standing on a bright background keeps full
   saliency and still drags the crop toward them — that IS the
   biggest/brightest-face-wins bug this rebuild exists to fix. Scaling down
   is what actually removes a bystander from contention.

2. **`crop_rect_containing(subject, frame_w, frame_h, aspect, head_y=0.36)`**
   — the actual crop geometry. Places the subject with their head at 36% down
   the frame (`DEFAULT_HEAD_Y`, matches `main.CAMERA_HEAD_Y`, already tuned on
   this footage) whenever the crop is tighter than full frame height.
   Guarantees **containment** — the subject-plus-margins box is the *input*
   constraint to the geometry, not a property checked afterward. `contains()`
   is the containment predicate, used both here and by validation.

3. **`decide_layout(subjects, speaker_shares, frame_w, frame_h)`** — chooses
   `LAYOUT_SINGLE` / `LAYOUT_TWO_SHOT` / `LAYOUT_SPLIT`. This is
   deliberately **editorial, not just geometric**: two far-apart faces trigger
   a split *only* if the speaker-share evidence says both people are
   genuinely live in the exchange (`MIN_SECOND_SPEAKER_SHARE = 0.25`) —
   otherwise (one person holding the floor) it stays a single, because
   dedicating half the frame to a silent listener wastes the format.
   `split_centers()` then delegates the actual panel geometry to the vendored
   `find_split_faces` / `render_split_screen_from_centers`.

4. **`validate_composition(shots, frame_w, frame_h)`** raises
   `CompositionError` (does NOT auto-repair) listing every violation found —
   containment failures, aspect mismatches, crops escaping the frame, shots
   too short. This mirrors the project's "no silent fallback" rule: the
   framing bugs this rebuild fixes all shipped silently before (a crop that
   cut a person in half still produced a playable file), so a violation here
   should stop the job loudly, not degrade quietly.

**41 tests, all passing, in `tests/test_reframe_v3.py`.** They are pure numpy
and run without a GPU — good regression coverage for any change here.

---

## 3. `vendor/pyautoflip/` — what was taken from an existing library, and what wasn't

The owner pointed at `pyautoflip` (PyPI, MIT, github.com/AhmedHisham1/pyautoflip,
"inspired by" Google's AutoFlip) and asked why we'd hand-write cropping when a
library exists. **The source was read in full before deciding**, and the
answer is: two specific pieces were worth taking, the actual cropper was not.

Taken (vendored verbatim into `vendor/pyautoflip/`, byte-identical to
upstream so it stays diffable against future releases — see
`vendor/pyautoflip/README.md` for full provenance):
- **`saliency_detector.py` + `unisal.onnx` (+`.data`, ~13MB weights)** — the
  UNISAL saliency model, ~17ms/frame on CPU via ONNX Runtime. Imports only
  `cv2`/`numpy`/`onnxruntime`, no pyautoflip-internal coupling, so it vendors
  cleanly.
- **`split_screen.py`** — `find_split_faces` (pure geometry: do 2+ faces
  need a split?) and `render_split_screen_from_centers` (renders the 2-panel
  layout). Pure functions of face rectangles, no detector coupling.

Rejected, with the specific bug found in each (full detail in
`vendor/pyautoflip/README.md` — **read it before taking any more of that
library**, so a real defect doesn't get re-imported later):
- their `CameraMotionHandler` STATIONARY mode **averages key-frame
  positions** — on two speakers that lands the crop *between* them, i.e. the
  exact "half a person" bug this whole rebuild started from.
- `compute_crop_window` always returns `y=0, height=full_frame` — no
  vertical composition at all, cannot express head-anchoring.
- `apply_padding_to_crop` does not actually letterbox — it stretches the
  crop to fill the output and darkens bands over the *distorted* result.

**Also decisive:** `pyautoflip` cannot be a pip dependency — it requires
`numpy>=1.24` (unbounded, resolves to numpy 2.x) and `torch>=2.11`, which is
the exact dependency-chain breakage `CLAUDE.md` already documents for
`insightface` (see the `numpy<2` guard line in `requirements.txt`). Vendoring
only the two standalone files adds **zero** new dependencies to the project.

---

## 4. What Phase 5 actually needs to do

**The hook point already exists and is easy to find.** `main.py` around line
2112:
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
`REFRAME_ENGINE` is already an env-var switch (v1 vs v2 today). Phase 5 adds
a third branch: `REFRAME_ENGINE=v3` calls a new `reframe_v3.render(...)`
entry point (does not exist yet — needs to be written) with the same
signature as `reframe_v2.render`, so `render_clip` (the caller) needs no
change beyond the new branch.

**What that new `render(...)` entry point has to do, concretely:**
1. Run `face_spine.extract_raw_tracks` + `merge_tracks_by_identity` on the
   clip (or accept already-computed tracks — check if `render_clip`'s caller
   already has this from an earlier stage, to avoid detecting twice).
2. Run speaker_fusion to get `per_second_active_track`.
3. Call `shot_planner.plan_shots(...)` to get the `Shot` list.
4. For each shot: sample a few frames, run the vendored `SaliencyDetector`,
   build `WeightedFace` list (speaker role from the shot's `track_ids`,
   reactor role from any corroborated reaction directive, bystander for
   everyone else), call `reframe_v3.build_attention_map` +
   `attention_center` + `crop_rect_containing`, and `decide_layout` for
   multi-subject shots.
5. Call `reframe_v3.validate_composition` on the full plan — **let it raise**.
   Per the no-silent-fallback rule, do NOT wrap this in a try/except that
   falls back to v2 the way the v2->v1 fallback above does; a composition
   violation should surface, not hide (confirm this with the owner if
   unsure — it's a real design choice, not an assumption to make silently).
6. Render: for each shot, hold `crop_rect` constant, apply it via ffmpeg
   (reuse `reframe_v2`'s ffmpeg-native crop/pad pipeline where possible
   rather than reimplementing frame-loop cropping — check
   `reframe_v2.render`'s ffmpeg filter construction first).
7. For `LAYOUT_SPLIT` shots: use `reframe_v3.split_centers` + the vendored
   `render_split_screen_from_centers`. Note this one currently needs a
   frame array (numpy, BGR) rather than an ffmpeg filter expression — decide
   whether to render split shots via a Python frame loop (simplest, matches
   what the vendored function expects) or to reimplement the split as an
   ffmpeg filtergraph (faster, more consistent with the rest of the
   pipeline, more work). Recommend starting with the frame-loop path to get
   *something watchable*, optimize later if it's too slow.

**Known open gap, flag but don't silently paper over:** a subject who moves a
lot *within* one shot only has `crop_rect_for_track`'s median box today — if
they walk far enough, `validate_composition`'s containment check SHOULD catch
it (that's exactly what it's for) and raise. When that happens, the real fix
is either splitting the shot into two, or allowing one slow linear pan across
it — neither is implemented. Phase 5 does not need to solve this, but should
not swallow the resulting `CompositionError` either; let it surface so it's
visible how often it actually happens on real footage.

---

## 5. Testing

```bash
# Rebuild-phase suite (pure Python, no GPU, no heavy deps needed):
python3 -m pytest tests/test_eval_metrics.py tests/test_face_spine.py \
  tests/test_asd_bakeoff.py tests/test_speaker_fusion.py \
  tests/test_shot_planner.py tests/test_vendor_pyautoflip.py \
  tests/test_reframe_v3.py -q
# Expect: 160 passed (at handoff time)
```

**Sandbox note:** a bare dev container may be missing `opencv-python`,
`onnxruntime`, and the app's full deps (PIL, pydantic, httpx, sqlalchemy,
scenedetect, python-dotenv). `test_vendor_pyautoflip.py` and parts of
`test_reframe_v3.py` skip gracefully via `pytest.importorskip` if
`cv2`/`onnxruntime` are missing — that's expected and not a failure. Full
`tests/` collection will show many unrelated `ModuleNotFoundError` for
app-level deps in a bare container; that's pre-existing, not something Phase
5 introduced. Real validation needs the actual `pip install -r
requirements.txt` environment or the Docker/Kaggle host.

**The tests above are unit tests for geometry/logic — none of them render an
actual video.** After wiring Phase 5, the real acceptance test is: run a real
clip end to end with `REFRAME_ENGINE=v3` and **watch it** — ideally a Pop The
Balloon span where a reaction (not the active speaker) is the moment, since
that's the specific case this rebuild was built to fix and no automated
metric substitutes for watching it. `eval/` (Phase 0 harness) has the
measurement tooling if a quantifiable before/after is wanted; per the
project's standing rule ("every fix ships with a metric that moved, or it is
not a fix"), use it if practical, but the owner watching the output is the
final gate regardless.

---

## 6. Explicitly OUT of scope for Phase 5 (don't touch)

- **Stage 3 (clip selection — `viral_clip_finder.py`, `get_viral_clips` in
  `main.py`)**. This decides *which moments* become clips. It was built
  before this rebuild started (commit `98df856`) and this entire rebuild
  (Phases 0-4b) has never touched it — it only changes *how* an already-chosen
  clip is cropped. The owner has explicitly parked further Stage 3 work for
  later; do not conflate the two.
- **Captions.** `subtitles.py` derives `UNIFIED_CONTENT_HEIGHT_RATIO` from the
  crop ratio; if Phase 5 introduces per-shot aspect ratios that vary within
  one clip (e.g. wide vs tight crops), captions would move mid-clip unless
  handled as ONE content band per whole clip (the widest aspect the clip
  uses), with every shot letterboxed into that same band. Flag this rather
  than silently changing subtitle positioning.
- **Audio reaction detection (PANNs/YAMNet)** — was planned as a later step
  (WHEN a reaction happened, from audio, independent of UNISAL's WHERE) but
  not started: needs torch + a ~300MB checkpoint not available in the
  handoff sandbox, so it was deliberately left unwritten rather than shipped
  unverified. Not blocking Phase 5.
- **Phase 6 (GPU utilization, both T4s) and Phase 7 (style profile from
  reference shorts)** — separate, later phases, unrelated to this handoff.

---

## 7. Quick orientation commands

```bash
git log --oneline -12                    # this rebuild's commit history
cat vendor/pyautoflip/README.md          # what was/wasn't taken from pyautoflip, and why
grep -n "REFRAME_ENGINE" main.py         # the exact hook point for the v3 branch
sed -n '2095,2130p' main.py              # the v2/v1 branch to mirror for v3
sed -n '2998,3060p' reframe_v2.py        # reframe_v2.render's signature/ffmpeg approach to reuse
```
