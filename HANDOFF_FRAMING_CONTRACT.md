# HANDOFF — framing contract implementation

Plan: `PLAN_FRAMING_CONTRACT.md`. Evidence: `docs/framing_evidence/`.
**Captions are out of scope by owner request — `subtitles.py` and
`_vsplit_caption_ass` are untouched.** Invariant I8 is not implemented.

## Status against the plan's §7 order of work

| # | Step | Status |
|---|---|---|
| 1 | `framing_contract.py` + unit tests | **done** |
| 2 | `eval/framing_audit.py` gate | **done**, calibrated |
| 3 | Wire contract into single/two-shot path | **done** |
| 4 | Panels + `smart_crop` + aspect-preserving `_recontain` | **done** |
| 5 | Split admission (§5.4) | **done — all 8 rows** |
| 6 | Speaker lock (§5.6) | **done** except the `IDENTITY_CONFIRM` flip — see below |
| 7 | Cut grid + captions | **done**; captions out of scope by request |
| 8 | Delete dead knobs / flip the gate | **done** except flipping `FRAMING_STRICT` |

`775 passed` on the runnable suite; the only 4 failures are pre-existing
`google.genai` import errors in `test_identity_confirm.py`, unrelated.

### Two deliberate non-changes

**`IDENTITY_CONFIRM` still defaults to `0`.** Plan §5.6 says make ASR-first the
default. `CLAUDE.md` records that the owner turned it OFF after reviewing
rendered clips ("the per-clip maps can't reliably tell speakers apart in a
crowd"). Flipping a default that was disabled on the strength of watched
footage, without being able to re-render and check, is not a call to make from
a test suite. It is a one-line change in `reframe_v3.render` when you want it —
but the no-guess rule below fixes clip 2's actual symptom regardless of which
identity source wins, so try that first and see whether the flip is still
needed.

**`FRAMING_STRICT` still defaults to `0` (warn).** Headroom and containment
violations are already always fatal. Promoting the source-dependent invariants
to fatal before a clean re-render would fail jobs on footage nobody has looked
at yet. Flip it once step 1 below is green.

**Nothing here has been proven on real footage yet.** Everything is unit-level
plus the gate calibration against the five existing clips.

### §5.4 admission gate, row by row

| Test | Threshold | Where |
|---|---|---|
| speaker switches in the window | ≥ 3 | `MIN_EXCHANGE_SWITCHES` |
| window duration | 2.5–10.0s | `min_span_s` / `max_split_span` defaults |
| complete sentences in the window | ≥ 2 | `_window_is_legible` (fails OPEN with no transcript) |
| both tracks present | ≥ 70% each | `_track_present_in_span` |
| both tracks pass I2/I3/I4 in their panel | hard | `_split_geometry_holds` |
| face-size ratio between panels | ≤ 1.6x | via `check_panels` |
| head boxes inside either panel | ≤ 2 | via `check_panels` |
| one crop holds both at ≥ 0.12 face height | fails | `two_shot_holds_both` |
| total split time in the clip | ≤ 60% | `_warn_on_split_share` — **logs, never downgrades** |

### §7 step 7, item by item

- `min_shot_seconds` 1.2 → 1.8 — done (all five call sites + `reframe_v3`).
- `MAX_SHOT_SECONDS` 8 → 14 — done.
- `snap_shots_to_speech` — done, plus `word_gaps` to derive cut candidates from
  ASR word timings. Wired into `reframe_v3.render` after the beat planner;
  forced boundaries and the clip's own in/out never move.
- `merge_short_runs` forced-boundary absorption — done
  (`MIN_FORCED_SHOT_SECONDS = 0.8`). This is the flash-frame fix.
- Captions (§5.5) — deliberately skipped.

### §7 step 8, item by item

- Deleted: `DEFAULT_HEAD_Y`, `DEFAULT_SIDE_MARGIN`, `DEFAULT_VERT_MARGIN`,
  `VSPLIT_HEADROOM`, `VSPLIT_SIDE_MARGIN`, `VSPLIT_VERT_MARGIN`.
- **Kept on purpose:** `MIN_CROP_FRAC` and `VSPLIT_PANEL_MIN_FRAC`. The plan
  listed these for deletion, but they are still load-bearing — they are the
  blur floor, and the resolution budget in §5.7 depends on them. Deleting them
  would remove the only thing stopping a distant face being upscaled into mush.
- `FRAMING_STRICT` exists but defaults to `0` (warn). Not flipped.

## What was built

| File | What it does |
|---|---|
| `framing_contract.py` | **New.** Owns all crop geometry. `head_box` (I1), `frame_subject`/`frame_single`/`frame_panel`, `look_room_dir`, `check` (I0/I2/I3/I4/I5/I6), `check_panels` (I7), `two_shot_holds_both`, `achievable_face_frac`. |
| `eval/framing_audit.py` + `eval/models/*.onnx` | **New.** The gate. Calibrated so both reference clips PASS and all three flop clips FAIL. |
| `tests/test_framing_contract.py` | **New.** Parametrised sweeps + the legacy-formula regression. |
| `tests/test_framing_audit.py` | **New.** Freezes the calibration baselines. |
| `reframe_v3.py` | Margin constants deleted; `crop_rect_containing` reimplemented over the contract with an explicit `layout=`; `_wide43_rect` anchors on the subject; `_clip_and_filter_box` rejects placards/lens-fillers/specks; `_track_boxes_for_shot` uses a p90 envelope + 70% presence gate; single path applies look-room; `validate_composition` no longer exempts `LAYOUT_SPLIT`. |
| `smart_crop.py` | Tracker rewritten over `frame_panel`; `_recontain` preserves aspect; containment valve measures the HEAD box. |
| `shot_planner.py` | Split admission gate (all 8 rows); `word_gaps` + `snap_shots_to_speech`; `MIN_FORCED_SHOT_SECONDS` flash-frame fix. |
| `speaker_fusion.py` | The no-guess rule (`unmapped_policy`), `speaker_lock_score` logged per clip, re-binding on by default. |

## The gate, calibrated

```
PASS  reference A   eyeline 0.218  face 0.149  empty 10%  split  2%  med.shot 2.90s
PASS  reference B   eyeline 0.220  face 0.163  empty  0%  split  0%  med.shot 3.27s
FAIL  flop clip 1   eyeline 0.562  face 0.219  empty  0%  split 100% seam-through-face 68%
FAIL  flop clip 2   eyeline 0.412  face 0.092  empty 21%  29% of shots under 1s
FAIL  flop clip 4   eyeline 0.385  face 0.100  empty 27%  40% of shots under 1s
```

    python3 eval/framing_audit.py CLIP.mp4 [--strict] [--json out.json]

~35s per clip. `--strict` exits 1 on any failure.

## READ THIS FIRST — the all-wide regression

A run of this code sent **every** binding wide. That was my no-guess rule
working exactly as written and the binding underneath failing: LR-ASD ran and
the spine found 6-11 faces, but no diarized speaker mapped to a face, so every
labelled second became a WIDE. The rule was right about a stray unmapped
second and wrong about a whole clip — an all-wide render is not a clip. Three
changes:

1. **Safety valve.** Past `SPEAKER_UNMAPPED_MAX_WIDE` (default 0.5) of
   labelled seconds going wide, the fusion is broken rather than cautious.
   It falls back to the LR-ASD prediction and says so loudly. The framing will
   be a guess — but a guess beats an unwatchable clip while the real bug
   gets found.
2. **Plurality binding.** A 60% majority is a high bar with 6-11 faces: the
   vote for one speaker scatters across neighbours and nothing clears it. A
   track leading the runner-up by 2x now binds even at 40%. A dead 50/50 tie
   still binds nothing — that is the case the majority bar was written for.
3. **Diagnostics.** Every clip now prints where the signal dies:

   ```
   🔗 binding: 41/58s diarized, 12s matched to a face, 9s could vote,
      0/3 label(s) bound
      · SPEAKER_01: best track 4 at 33% of 6 vote(s) across 4 track(s) — no binding
   ```

   Read it like this: **`diarized` near zero** → diarization is the problem,
   not the fusion. **`matched` far below `diarized`** → the ASD boxes are not
   landing on the spine's faces (a coordinate-space or scaling bug), and no
   threshold change will help. **`could vote` healthy but nothing bound** →
   it really is the agreement bar, and the plurality tier should now catch it.

**Note:** those clips also crashed at `plan_shots` immediately after this log
(the `max_shot_seconds` TypeError, since fixed), so **no render of this state
has ever been seen** — only the planning log. Treat the next run as the first
real look.

## Remaining work, in priority order

1. **Re-render the three flop clips on the GPU host and re-run the audit.**
   Nothing below is worth doing until this says the geometry actually landed.
   Expected: clip 1 seam-through-face 68% → 0 and most of it becoming
   two-shots; clips 2/4 eyeline 0.39-0.41 → ~0.22. If clip 4's split share
   goes UP, that is correct — its far-apart pairs should now split rather
   than shrink into a wide.
2. **Read the two new log lines.** Every clip now prints
   `🎯 speaker lock NN%` (target ≥ 85%) and, when relevant,
   `⚠️ framing-contract violation(s)` and `⚠️ split screen on NN%`. Those three
   tell you what to fix next without watching a frame.
3. **Then flip `FRAMING_STRICT=1`** in CI, and reconsider `IDENTITY_CONFIRM`.

### New env defaults worth knowing

| Var | Was | Now | Why |
|---|---|---|---|
| `MAX_SHOT_SECONDS` | 8 | **14** | the reference holds payoffs for 10.9-17.8s; 8 made that impossible |
| `SPLIT_MIN_SPAN_S` | 2.0 | **2.5** | a split you cannot read is a glitch |
| `SPLIT_MAX_SPAN_S` | 12 | **10** | longer stops being an exchange |
| `SPEAKER_REBIND_SECONDS` | 0 | **4** | one wrong binding used to hold the whole clip |
| `SPEAKER_UNMAPPED_POLICY` | — | **`wide`** | new: the no-guess rule |
| `MAX_TWO_SHOT_WIDTH_FRAC` | 0.72 | **0.82** | the reference holds pairs across 80% of frame width |
| `min_shot_seconds` | 1.2 | **1.8** | reference median shot is 2.9-3.3s |

`MAX_SHOT_SECONDS` and `min_shot_seconds` together will visibly change pacing —
expect fewer, longer shots. That is the intent (§1.2), but it is the change
most likely to look "wrong" at first glance next to the old output.

## Watch out

- `min_crop_frac` (0.45) floors panel crops for faces below ~0.135 of source
  height, so the panel reads smaller than 0.30 there. That is the resolution
  budget working, not a bug — tests must pick face sizes accordingly.
- `check()` exempts the eyeline when a crop is clamped against a frame edge,
  headroom when `crop_y == 0`, and I6 when the crop is clamped against a SIDE.
  All three are the source's framing, not ours.
- Several test fixtures used 50x60px faces in a 1080p frame. That is ~0.055 of
  frame height — a face no crop can frame to the contract. They were updated to
  realistic 150px mid-shots; if a new test "unexpectedly" refuses to split,
  check the face size first.
