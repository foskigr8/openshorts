# Next phases: GPU parity, audio reactions, style tuning

**Self-contained — read this without needing prior chat history.** For the
full build history of how the framing rebuild got here (Phases 0-4b, Phase 5),
read `MASTER_PLAN_FRAMING_REBUILD.md` first — it's now a historical build log
(v3 is the only reframe engine; v1/v2/`subject_policy`/MediaPipe were deleted
entirely). This document covers what's left, and it's real work found by
reading the current code, not a recycled TODO list.

Repo: `foskigr8/openshorts`. Branch: `claude/gemini-vision-clip-picking-bikvuy`.
`CLAUDE.md` at repo root is the permanent project reference.

---

## 0. The actual highest-priority item is not code

Before touching anything below: **has a real clip been rendered on Kaggle
and watched?** Nothing in this document matters if `reframe_v3` hasn't been
exercised on real footage yet. If that hasn't happened, do it first — ideally
a Pop The Balloon span where a reaction, not the active speaker, is the
moment, since that's the specific case the whole rebuild targets. No
automated test in this repo substitutes for watching it. If it HAS happened
and something looked wrong, fix that before starting new phases — a
composition bug in already-shipped code outranks a new feature.

The three sections below are ordered by priority assuming the watch-test is
clean or already handled separately.

---

## 1. Phase 6 — GPU parity for the v3 path (do this first of the three)

### Why this is real, not speculative — and a warning about trusting docstrings

`gpu_affinity.py`'s own docstring claims per-worker GPU assignment is fully
wired through "the reframe, clip-cut and caption-burn call sites." **Do not
trust that claim just because it's written down.** It describes the old
`reframe_v2` path, and per the owner's explicit correction, v2 never actually
achieved multi-GPU utilization in practice — the docstring describes intent
or an unverified belief, not a measured result. Treat every claim in this
section, including ones from this document, as needing its own
`nvidia-smi` proof, not as settled fact because code exists that looks like
it should work.

Reading the current `reframe_v3.py` shows two concrete gaps versus what that
docstring describes:

**1a. `asd_worker.score_clip` is called without a device.**
```python
# reframe_v3.py, inside render():
asd_boxes = asd_worker.score_clip(
    input_video, face_spine.detect_faces_per_frame).get("per_second_box") or []
```
`score_clip`'s signature (`asd_worker.py:237`) accepts `device=None` and
`ASDScorer.__init__` (line 131-139) defaults an unset device to
`"cuda" if torch.cuda.is_available() else "cpu"` — i.e. **always the default
CUDA device (GPU 0)**, regardless of which worker thread is running. On a
2×T4 host with concurrent clip workers, every clip's LR-ASD scoring pass
piles onto GPU 0 while GPU 1 sits idle for this stage — precisely the
imbalance `gpu_affinity.py` was built to fix, just not fixed here.

**Fix:** pass the current worker's device through:
```python
import gpu_affinity
...
asd_boxes = asd_worker.score_clip(
    input_video, face_spine.detect_faces_per_frame,
    device=gpu_affinity.current_device()).get("per_second_box") or []
```
`face_spine.build_face_spine(input_video)` (called just above it in the same
function) at least *resolves* a device via `face_spine._resolve_ctx_id`
falling back to `gpu_affinity.current_device()` when none is passed
(`face_spine.py:81-98`) — but resolving a ctx_id is not the same as proving
InsightFace's ONNX session actually executes on that GPU. **Verify this one
too, empirically, rather than assuming it already works** because the code
looks right — check GPU utilization during the face-spine stage specifically
(not just the render as a whole) as part of §1's `nvidia-smi` verification
below. If it turns out face_spine ISN'T actually landing on the assigned GPU
either, fix that here rather than treating it as a solved reference case.

**1b. `reframe_v3`'s own ffmpeg subprocess calls never resolve a worker GPU.**
Both render paths build raw `ffmpeg` command lists with no hwaccel args at
all:
```python
# _render_regular (reframe_v3.py:697-713)
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", input_video,
       "-filter_complex", graph, "-map", "[v]", "-map", "0:a?",
       *video_encode_args(QUALITY_FAST), "-c:a", "copy", *METADATA_SCRUB,
       "-movflags", "+faststart", output_video]
```
```python
# _render_with_splits (reframe_v3.py:716-769) — same gap, both the
# VideoCapture/VideoWriter setup and the final audio-mux ffmpeg call
```
Neither reads `gpu_affinity.current_device()`. Every v3 render's ffmpeg
encode stage runs on whatever device ffmpeg defaults to, not the worker
thread's assigned GPU — the same class of bug §1a describes, in a different
call site.

**Do NOT copy `reframe_v2.py`'s approach to this.** An earlier version of
this document pointed at v2's old `worker_gpu = gpu_affinity.current_device()`
wiring as "the working pattern to port." **That was wrong.** The code being
present in v2 is not evidence it worked — nobody ever confirmed with
`nvidia-smi` that GPU 1 actually got used for v2's ffmpeg stage, and per the
owner's explicit correction, it did not: v2 never actually utilized the
second GPU, regardless of what its code or `gpu_affinity.py`'s docstring
claimed. Do not read v2's source for this at all. This rebuild is being done
from the ground up specifically so it isn't quietly inheriting v2's unproven
assumptions — copying its ffmpeg/CUDA wiring here would be exactly that
mistake again, just in v3's code instead of v2's.

**Build this fresh, and prove it, don't assume it:**
1. Establish what ffmpeg actually needs to run its decode/filter/encode
   pipeline on a specific GPU — this means real CUDA hwaccel flags
   (`-hwaccel cuda -hwaccel_device N` on the input side, and/or CUDA-aware
   filters if the filtergraph should also run on-GPU) threaded from
   `gpu_affinity.current_device()` into the `cmd` list in both
   `_render_regular` and `_render_with_splits`. Verify the exact flags
   against current ffmpeg/nvenc documentation or a working local test, not
   against any code that already exists in this repo's history.
2. **Prove it empirically before calling it done.** Run 2+ concurrent clip
   renders on the actual 2×T4 Kaggle host and watch `nvidia-smi` DURING the
   render — both GPUs must show real utilization, not just be theoretically
   addressed by a flag. A flag that's present but silently ignored by ffmpeg
   (wrong syntax, unsupported filter, falls back to CPU) is functionally the
   same failure as not having it at all, and is exactly the kind of gap that
   went unnoticed in v2 for as long as it did.
3. Check `ffmpeg_utils.video_encode_args` for what it already owns re:
   device selection — but verify its actual behavior by reading it and
   testing it, not by assuming it's correct because it exists.

### Verification for Phase 6
No unit test can prove GPU placement — this needs a real 2×T4 host.
```bash
# On the Kaggle host, submit 2+ clips concurrently, then check:
nvidia-smi   # both GPUs should show utilization during the render, not just GPU 0
```
`gpu_affinity.py`'s docstring documents the exact measured baseline (job
`b86b8c5a`, 2×T4, "three clips rendered concurrently and all three ran on GPU
0") — reproduce a similar concurrent-clip test and confirm GPU 1 is no longer
idle. This is a metric-that-moved requirement per the project's standing
rule: report the before/after `nvidia-smi` utilization split, not just "it
compiled."

---

## 2. Phase 4b step 5 — audio reaction detection (PANNs/YAMNet)

### Why this exists
`shot_planner.insert_reaction_shots` currently corroborates a Gemini
"something happened here" directive using only `is_directive_corroborated`
(`shot_planner.py:386`) — face-spine presence + positional motion, a cheap
proxy that was flagged from the start as weak (see that function's own
docstring: "not a real gesture/expression detector"). `reframe_v3`'s UNISAL
saliency covers the **visual** half of "what would a human look at" but
nothing yet covers **audio** — laughter, applause, a gasp, a loud sound — which
is often the actual trigger for a reaction moment and is completely
independent of the video signal (a real second vote, not a duplicate of
saliency).

### What to build
A new `audio_reactions.py` using **PANNs** (Pretrained Audio Neural
Networks, AudioSet-trained, 527 sound event classes including laughter,
applause, screaming, gasp-adjacent classes) or **YAMNet** as a lighter
alternative if PANNs' dependency weight is a problem — evaluate both, this
repo has no PANNs/YAMNet integration to reuse or copy from yet.

Needed:
1. Extract audio from a clip span (ffmpeg, same pattern as `asd_worker.py`'s
   `_extract_wav`).
2. Run the model, get per-second (or finer) event scores for the relevant
   classes.
3. Expose a function like `detect_reaction_events(video_path, start, end) ->
   List[Tuple[float, float, str, float]]` (start, end, event_label,
   confidence) that `shot_planner.py` or `reframe_v3.py` can call.
4. Feed this into `is_directive_corroborated` (or a sibling function) as a
   **third, independent signal alongside presence+motion** — per the
   corroboration gate's own stated philosophy (loose on purpose, false
   rejection is worse than false acceptance), audio evidence should make
   corroboration MORE likely to pass when a directive's timestamp has a real
   audio event, not replace the existing checks.

### Constraints, stated up front so they aren't rediscovered the hard way
- Needs `torch` (already a dependency) plus a PANNs checkpoint
  (~300MB-ish depending on model variant) that is not currently vendored or
  downloaded anywhere in this repo. Verify the checkpoint's actual size and
  license before committing it into `vendor/` (follow the same pattern as
  `vendor/lrasd/` and `vendor/pyautoflip/` — MIT/Apache-compatible only,
  provenance recorded in a README, byte-identical to upstream where feasible).
- This was deliberately not built earlier in the session because the
  environment it was built in couldn't run/verify a torch+checkpoint model.
  **Do not ship this unverified** — the standing rule from the vendored
  UNISAL work applies here too: write a smoke test that loads the model and
  confirms it fires on a synthetic or real clip with a known laugh/applause
  moment before wiring it into the render path.
- Corroboration is currently synchronous, cheap, and adds "no latency and no
  API cost" per its own docstring. A torch model load changes that
  cost/latency profile — measure it, and if it's expensive, consider caching
  the model load process-wide (same pattern as `SaliencyDetector._session` in
  `vendor/pyautoflip/saliency_detector.py`, a shared class-level session) so
  it isn't reloaded per clip.

### Verification
```bash
python3 -m pytest tests/test_audio_reactions.py -q   # write this alongside the module
```
Plus the eval harness: does adding this signal change
`is_directive_corroborated`'s accept/reject rate on the same footage Phase 4
was tuned against, in the direction expected (more real reactions accepted,
hallucinated ones still rejected)? Per the project's standing rule, this
ships with a metric that moved or it isn't a fix.

---

## 3. Phase 7 — style profile from reference shorts

**This is genuinely underspecified — flag that rather than inventing scope.**
Unlike §1 and §2, there is no existing code, constant, or module in this repo
that partially implements this or defines what "style profile" or "reference
shorts" concretely means (no `style_profile.py`, no reference-clip directory,
no related env var). Before writing any code for this phase:

1. Ask the owner what "reference shorts" means concretely — a folder of
   competitor clips to match crop tightness/pacing against? A per-client
   brand style (specific head-anchor values, caption style, hook timing)?
   Something else? The other two phases in this document have enough
   grounding in existing code to start immediately; this one does not.
2. Once scoped, the natural integration points are the same tuned constants
   `reframe_v3.py` already exposes (`DEFAULT_HEAD_Y`, `DEFAULT_SIDE_MARGIN`,
   `DEFAULT_VERT_MARGIN`, `MAX_TWO_SHOT_WIDTH_FRAC`,
   `MIN_SECOND_SPEAKER_SHARE`) — a "style profile" most plausibly becomes a
   named bundle of overrides for these, selected per job, rather than a new
   subsystem. Confirm this assumption with the owner before building it.

Do not start implementation on this phase without that scoping conversation.
Treat §1 and §2 as the actual actionable work in this document.

---

## 4. Explicitly out of scope — do not touch

- **Stage 3 (clip selection)** — `picker.py` + `context_layer.py`,
  orchestrated by `get_viral_clips` in `main.py`. Rebuilt as its own effort
  (one Gemini planning pass over the whole transcript + a pre-download
  context brain; explicit clip count fulfilled at all costs; no vision
  confirmation, no skill/narrative engines). Unrelated to framing; do not
  rewire it while working on §1/§2.
- **Captions** (`subtitles.py`) — no changes needed for §1 or §2 above. If
  §2's audio work somehow touches subtitle timing, stop and flag it; that
  would be a scope violation of what was asked for.
- **Re-litigating the vendored pyautoflip decision** — settled, with reasons,
  in `vendor/pyautoflip/README.md` and `MASTER_PLAN_FRAMING_REBUILD.md` §2.
  Don't reopen it without new evidence (e.g. an actual side-by-side render
  comparison, which nobody has done yet — see that section for how to run
  one cheaply if genuinely curious).
- **Reintroducing a v1/v2 fallback for v3.** `reframe_v3.render`'s docstring
  is explicit: "there is no v3→v2 fallback." v2 doesn't exist anymore to
  fall back to, and that was a deliberate choice, not an oversight — a
  composition bug should stop a job loudly, not silently degrade to
  something that no longer exists in the codebase anyway.

---

## 5. Quick orientation commands

```bash
git log --oneline -20                        # full history through the v2 removal
cat vendor/pyautoflip/README.md              # pyautoflip decision, if §4 comes up
grep -n "def score_clip" -A 15 asd_worker.py # device= param §1a needs to use
grep -n "gpu_affinity" face_spine.py         # ctx_id resolution to verify empirically, §1a
nvidia-smi                                    # the only real source of truth for §1 — not git history
python3 -m pytest tests/test_reframe_v3.py tests/test_shot_planner.py \
  tests/test_face_spine.py -q                # confirm the 175-test baseline still holds
```

**Do not run `git show` against any pre-deletion `reframe_v2.py` commit for
GPU/ffmpeg wiring guidance.** That file's approach to this problem is
unverified and, per the owner, did not actually work — reading it for this
purpose is how the mistake in an earlier version of this document happened.
