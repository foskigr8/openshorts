# PLAN — The Framing Contract

**Status:** proposed, supersedes the framing sections of `MASTER_PLAN_FRAMING_REBUILD.md`,
`HANDOFF_FRAMING.md` and `99_PROBLEMS_A_FIX_AINT_1.md`.
**Written from:** measured geometry of two reference shorts (`video-refrence/`, 695 sampled
frames) vs. the three failing renders (`videosflop/`, 228 sampled frames).
**Goal:** stop tuning. Replace "knobs that mostly work" with a small set of *arithmetic
invariants* that make the reported failures impossible to render, and a gate that fails the
job when they are violated.

---

## 0. Why this document exists

Every previous framing fix has been a new heuristic layered on the last one. The engine now
has ~25 environment knobs governing composition and it still ships clips where a head is
sheared by the split seam. The reason is structural: **the code's only hard guarantee is
`contains(crop, face_box)`** — is the face box inside the crop rectangle. That predicate is
satisfied by a crop with *one pixel* of headroom, by a crop centred on an empty aisle, and by
a crop where the face box is inside but the person's hair is not (the detector box excludes
hair). Every failure below passes `contains()`.

The fix is not another knob. It is to replace containment with a **framing contract**: a
handful of ratios, measured off footage that works, checked as assertions.

---

## 1. What the reference actually does (measured, not eyeballed)

Two reference shorts, 720×1280, sampled at 2 fps, faces detected with YuNet
(`face_detection_yunet_2023mar`, conf 0.7). n = 695 frames with a face.

### 1.1 Subject framing

| Quantity (dominant face) | p5 | p25 | **p50** | p75 | p95 |
|---|---|---|---|---|---|
| face box height ÷ frame height | 0.08 | 0.14 | **0.155** | 0.17 | 0.20 |
| face box width ÷ frame width | 0.11 | 0.16 | **0.19** | 0.22 | 0.24 |
| eyeline y ÷ frame height | 0.17 | 0.20 | **0.217** | 0.24 | 0.29 |
| face top y ÷ frame height | 0.10 | 0.13 | **0.152** | 0.18 | 0.24 |
| face centre x ÷ frame width | 0.26 | 0.44 | **0.54** | 0.60 | 0.77 |

Read that as four rules:

1. **The face is ~15% of frame height.** Not 30% (passport), not 6% (a dot in a group).
   The band is remarkably tight: the middle 50% of all frames sit between 0.14 and 0.17.
2. **The eyeline sits at ~22% of frame height** — higher than the rule-of-thirds line, which
   is the vertical-video convention. IQR 0.20–0.24. This is the single most consistent number
   in the whole reference.
3. **Headroom above the hair is real and never zero.** Face top at 0.152 with a face 0.155
   tall means roughly 15% of the frame is above the eyebrows — hair plus air.
4. **Horizontal centring is loose, deliberately.** Median 0.54, IQR 0.44–0.60. The subject
   drifts off-centre to leave look-room on the side they face. It is *never* the frame edge
   and *never* the empty half.

Dominant-face boxes clipped by a frame edge: **2.9%** and **0.0%**. Effectively never.

### 1.2 Sequence framing (the cut grid)

| | Ref A | Ref B |
|---|---|---|
| duration | 215.4s | 160.9s |
| cuts | 56 | 35 |
| cuts / minute | 15.6 | 13.1 |
| **median shot** | **2.90s** | **3.27s** |
| mean shot | 3.78s | 4.47s |
| shots < 1.0s | 3 (5%) | 4 (11%) |
| shots > 6.0s | 9 (16%) | 10 (28%) |
| longest hold | 17.8s | 17.0s |

The rhythm is **not** uniform fast cutting. It is a 3-second base pulse with deliberate long
holds on the payoff — 8s, 10.9s, 12.3s, 17.0s, 17.8s. Our engine caps every shot at
`MAX_SHOT_SECONDS=8` (`reframe_v3.py:1385`), so it is structurally incapable of the reference's
most-used dramatic device. The sub-1s shots in the reference are whip/flash transitions
between segments, not speaker cuts.

### 1.3 Action framing — the grammar the reference uses instead of split screen

Faces per frame: 1 face in ~50% of frames, **2 faces in ~35%, 3+ in ~13%**.

There is no split screen anywhere in either reference. The conversation reads as a
conversation because of **over-the-shoulder shot–reverse-shot**: when A speaks, the camera
frames A at the numbers in §1.1 *with B's shoulder or back-of-head as foreground mass on one
side of the frame*. The listener is present as a compositional element, not as a second panel.

The other three devices:

- **Establishing group wide** every ~8–10 shots, 2–3s, showing the whole line-up. Used to
  re-anchor geography, then straight back to OTS.
- **Motivated insert** (a fist, a balloon) — rare, always tied to something just said.
- **The mic as the pointer.** In this format the microphone physically indicates the speaker.
  It is the most reliable "who is talking" signal in the frame and we do not use it at all.

**Conclusion for us — with the honest caveat.** We cannot *select* an over-the-shoulder shot.
The reference's OTS coverage is a production fact: multiple cameras, shot that way on the day.
We get one source frame per moment, already picked, and our only lever is where we put a crop
inside it. So the takeaway is **not** "cut to OTS". It is two things we can actually control:

1. **Crop placement, not shot selection.** When the source frame already contains the listener
   next to the speaker — which in this format it usually does — a single crop biased toward the
   speaker *reproduces the OTS composition as a side effect*: speaker at the numbers in §1.1,
   listener's shoulder or back-of-head occupying the near edge. The rule that produces it is
   look-room: offset the speaker away from frame centre in the direction they face, so the crop
   naturally eats the space the other person occupies instead of trimming them out. Today's
   engine centres on the speaker's box (or worse, the union's centre), which either cuts the
   listener off entirely or centres the gap between them.
2. **Split screen where it is the *right* answer, not where it is the easy one.** The split
   exists to solve a specific problem this reference format never had to face: two people who
   matter simultaneously and are **too far apart for one crop**. The alternative in that
   situation is the 4:3 wide — which is exactly how clips 2 and 4 ended up with 0.09-height
   faces, unreadable on a phone. Two panels at `PANEL_FACE_FRAC = 0.30` put each person at
   **0.15 of output height — the reference number** (§1.1). *A correct split is the only layout
   that preserves subject size when the subjects are far apart.* That makes it a quality tool
   and, done right, the most engaging device we have: both reactions land at once, and the cut
   in and out of it is a beat in itself.

   What has to stop is the split being reached for when a **two-shot would have worked** —
   clip 1's participants are sitting side by side in one plane, and 61% of its frames put the
   seam through a face. That is the admission gate's job (§5.4), not a quota. Fix the geometry
   and apply the two-shot test first, and the split rate self-regulates to the moments that
   actually need it.

Where the source is a locked group wide and nobody is framed usably, no crop policy saves it —
see §5.7, that is a clip-selection problem, not a framing one.

---

## 2. What our clips do (same measurement, same detector)

| | Ref (median) | clip 1 | clip 2 | clip 4 |
|---|---|---|---|---|
| eyeline y | **0.217** | 0.571 | 0.414 | 0.395 |
| face h ÷ frame h | **0.155** | 0.215 | 0.092 | 0.094 |
| face centre x IQR | **0.44–0.60** | 0.56–0.67 | 0.31–0.69 | 0.38–0.79 |
| dominant face clipped | **0–3%** | 1.8% | 6.7% | **15.5%** |
| dominant face straddling the seam (y = H/2) | 0 / 4 | **35 / 57** | 4 / 75 | 10 / 58 |
| frames with no face at all | 0.5% | 1.7% | 10.7% | **32.6%** |

Three distinct diseases, all visible in the numbers:

- **clip 1 — the split seam runs through faces.** 61% of sampled frames have the dominant
  face crossing y = H/2. At full resolution: the bottom panel's subject has their forehead cut
  off (`00003`, `00015`, `00053`); the caption sits *on the seam*, i.e. on the top panel
  subject's chin. Faces are also ~40% too large (0.215 vs 0.155) — the "passport photo"
  effect.
- **clip 2 / clip 4 — the subject is too small and too low.** Face height 0.09, eyeline 0.40.
  That is a wide shot with a person in the middle of it, held for seconds.
- **clip 4 — a third of the clip has no face on screen at all.** Sampled frames `00017`–
  `00025` are a teal curtain and a shoulder. `00027`–`00035` are a full-frame close-up of the
  *back* of a head. This is not a framing quality problem, it is a crop pointed at nothing.

---

## 3. Root cause, traced to the line

### 3.1 The head-cut in split panels is arithmetic, not tuning

Notation: `h_f`, `w_f` = detected face box height/width. For SCRFD/YuNet, `w_f ≈ 0.8·h_f`, the
box spans roughly eyebrows→chin, and **the hair sits about `0.35·h_f` above the box top**.

`crop_rect_containing` (`reframe_v3.py:216`) with `DEFAULT_VERT_MARGIN = 0.35`,
`DEFAULT_SIDE_MARGIN = 0.55`, `DEFAULT_HEAD_Y = 0.36` (`reframe_v3.py:181–188`):

```
need_h = h_f · (1 + 2·0.35) = 1.70·h_f
need_w = w_f · (1 + 2·0.55) = 2.10·w_f
crop_h = max(need_h, need_w / aspect)
crop_top = face_centre_y − 0.36 · crop_h
```

For a **9:16 single** (`aspect = 0.5625`): `crop_h = max(1.70·h_f, 2.10·0.8·h_f/0.5625) = 2.99·h_f`.
Crop top is `0.36·2.99 = 1.076·h_f` above the face centre; the hair top is `0.85·h_f` above it.
Headroom above the hair = `0.23·h_f` = **+7.5% of crop height**. Marginal, survives. Note the
face also comes out at `1/2.99 = 0.335` of crop height — 2.2× the reference's 0.155.

For a **vsplit panel** (`aspect = 0.5625 · 2 = 1.125`, `reframe_v3.py:785`) the *width*
constraint stops binding, because a 9:8 panel is nearly twice as wide relative to its height:

```
crop_h = max(1.70·h_f, 2.10·0.8·h_f / 1.125 = 1.49·h_f) = 1.70·h_f
crop_top = face_centre_y − 0.36 · 1.70·h_f = face_centre_y − 0.612·h_f
hair top  = face_centre_y − 0.850·h_f
```

**The hair top is `0.238·h_f` ABOVE the crop top — headroom is `−14.0%` of the panel height.
The head is cut. Always. By construction, for every split panel this engine has ever rendered.**

Two consequences fall out of the same two lines:

- Face height inside the panel = `h_f / 1.70·h_f = 0.588` of panel height = **0.294 of output
  height**, versus the reference's 0.155. Exactly the measured clip-1 median of 0.215–0.279.
- `_enforce_min_crop` (`reframe_v3.py:193`, `MIN_CROP_FRAC = 0.45`) is applied on the single
  path (`reframe_v3.py:842`) but **not** on either vsplit panel path (`:786`, `:822`). So
  panels are permitted to be tighter than any other shot type in the engine.

The render-time tracker repeats the same mistake differently. `smart_crop.PanelTracker._target_crop`
(`smart_crop.py:167`): `y = sy − sh · headroom` with `headroom = 0.18`. The crop top is placed
`0.18·h_f` above the **face box** top — but the hair is `0.35·h_f` above it. Headroom is
negative by `0.17·h_f` regardless of how tall `crop_h` ends up. Meanwhile `crop_h` is floored at
`0.45·frame_h`, so the anchor and the size are computed against different references.

### 3.2 The validator does not check the thing that breaks

`validate_composition` (`reframe_v3.py:455`):

- `LAYOUT_SPLIT` is `continue`d outright (`:485–487`) — **no checks at all**.
- `LAYOUT_VSPLIT` checks only `contains(panel, face_box)` (`:497–502`). A panel whose top edge
  bisects the subject's hair contains the face box and passes.
- Nothing checks face size, eyeline, headroom, whether the crop centre is on anything, or
  whether the crop contains a face *at all*.

That is why every one of these clips rendered without an error.

### 3.3 Gap-centred crops

`crop_rect_containing:254` — `crop_x = subject_cx − crop_w/2` where `subject` is
`union_box(*subjects)` (`:835`). For two people with space between them the union centre **is**
the gap. `_wide43_rect:273` does the same thing for the 4:3 wide. Clip 4's "crop on the empty
space between participants" is this line.

### 3.4 Subjectless and back-of-head crops

- `_compose_shot:742` — when `_track_boxes_for_shot` returns nothing, it emits a 4:3 wide
  **centred on the frame** (`:753`). If the action is off to one side you get a curtain. There
  is no "does this crop contain a face" check.
- `_track_boxes_for_shot:654` unions the subject's boxes over the whole shot at 0.5s steps.
  A subject who turns away mid-shot contributes boxes only while facing camera; the static crop
  then holds on that region for the seconds they are turned around. Clip 4's back-of-head
  close-up. There is no *presence* requirement — a track detected in 2 of 12 sampled instants
  is treated the same as one detected in 12.

### 3.5 Split screen fires far too often

`plan_conversation_beats` (`shot_planner.py:744`) splits whenever `find_exchange_windows`
(`:593`) finds ≥ 2 speaker switches inside a ≥ 2.0s window, both tracks have *any* detection in
the span (`_track_present_in_span:706` — `any()`, one frame is enough), and
`_tracks_close_enough` (`:719`, via `crop_rect_for_track`) says no. That test uses **median boxes**, so two people
sitting side by side in the same plane fail it whenever one of them briefly leans. There is no
test that both people are *comparably framed* (one panel a close-up, the other a wide group is
allowed), no test that the split lasts long enough to be legible, and no cap on how much of a
clip may be split.

### 3.6 The seam is where the captions go

`_vsplit_caption_ass` (`reframe_v3.py:1015`) forces captions **MIDDLE** during a vsplit, and
`VSPLIT_BAND_FRAC = 0` means there is no band — so "middle" is exactly the seam, which is
exactly where the bottom panel's subject's head is (a panel puts its face at ~0.34 of panel
height = ~0.67 of frame; the top panel's chin is just above 0.50). Every caption in clip 1 sits
on a face. The reference puts captions at **0.62–0.72 of frame height** — well below a face at
0.22, and never near a seam.

### 3.7 Cut grid

`min_shot_seconds = 1.2` (`shot_planner.py:274`, `:334`, `:484`, `:555`, `:748`) against a
reference median of 2.9–3.3s; `MAX_SHOT_SECONDS = 8` against reference holds of up to 17.8s.
`merge_short_runs` (`:172`) refuses to reabsorb across a forced boundary (`:198`), so a scene
cut inside a run still emits a sub-second shot — the flash frames.

---

## 4. The contract

Nine invariants. Everything else is implementation. A render that violates any of them is a
bug, not a taste disagreement.

> **I1 — HEAD, NOT FACE.** No composition function may consume a raw detector box. Every crop
> is computed from a **head box** derived once at the boundary:
> `head = (x − 0.30·w, y − 0.35·h, 1.60·w, 1.50·h)`, clipped to frame.
>
> **I2 — HEADROOM.** For SINGLE, TWO_SHOT and every split panel, the distance from the crop
> top to the head-box top is **≥ 6% of the crop height** (target 10%). No exceptions, no
> layouts exempt.
>
> **I3 — EYELINE.** `eyeline_y = face_top + 0.42·h_f` of the *bound speaker* lands at
> `0.22 ± 0.06` of the output frame height for SINGLE/TWO_SHOT, and at `0.34 ± 0.08` of the
> panel height for split panels.
>
> **I4 — SUBJECT SIZE.** The bound speaker's face box occupies `0.155 × [0.65, 1.45]` of the
> output height → **0.10 ≤ face_h/out_h ≤ 0.225**. Outside that band the layout is wrong, not
> the crop.
>
> **I5 — NEVER FRAME NOTHING.** Every rendered second contains at least one head box whose
> height ≥ 0.06 of output height, fully inside the frame. A crop that satisfies nothing else
> must satisfy this.
>
> **I6 — NEVER CENTRE THE GAP.** For SINGLE and split panels, the crop's centre x lies inside
> the subject's head box. For TWO_SHOT and WIDE, at least one head box centre lies within the
> middle 60% of the crop width.
>
> **I7 — SPLIT IS SYMMETRIC OR IT IS NOT A SPLIT.** The two panels' face heights differ by
> ≤ 1.6×, both panels satisfy I2/I3/I4, and neither panel's subject is a crowd
> (≥ 3 head boxes in one panel disqualifies the split).
>
> **I8 — CAPTIONS ARE A SECOND CROP.** The caption bounding box intersects no head box and
> never crosses a panel seam. Default band: `[0.62, 0.78]` of output height; on a vsplit,
> `[0.86, 0.94]` (below the bottom panel's subject).
>
> **I9 — THE CUT FOLLOWS THE SENTENCE.** Every shot boundary that is not a source scene cut
> lands on a word gap ≥ 120ms. Clip in/out land on sentence boundaries. No shot shorter than
> 1.8s except a forced scene cut, which must still be ≥ 0.8s or be absorbed.

---

## 5. Implementation

### 5.1 New module: `framing_contract.py`

One place that owns the geometry. Pure functions, no I/O, unit-testable without a GPU.

```python
# --- the measured constants (§1.1). Changing these changes the look; changing
# --- anything else is a bug fix.
HEAD_FROM_FACE   = (0.30, 0.35, 1.60, 1.50)  # dx·w, dy·h, sx, sy   (I1)
EYELINE_IN_FACE  = 0.42        # eyeline as a fraction down the face box
SINGLE_FACE_FRAC = 0.155       # face height / output height          (I4)
SINGLE_EYE_Y     = 0.22        # eyeline / output height              (I3)
PANEL_FACE_FRAC  = 0.30        # face height / PANEL height           (I4, = 0.15 of output)
PANEL_EYE_Y      = 0.34        # eyeline / panel height               (I3)
HEADROOM_MIN     = 0.06        # crop-top → head-top, / crop height   (I2)
LOOK_ROOM        = 0.12        # max centre-x offset, / crop width (see note below)
FACE_FRAC_BAND   = (0.10, 0.225)
MIN_HEAD_FRAC    = 0.06        # I5 floor

def head_box(face: Box, frame_w: int, frame_h: int) -> Box: ...
def frame_subject(head: Box, face: Box, frame_w, frame_h, aspect,
                  face_frac: float, eye_y: float) -> Rect:
    """THE crop function. Size from face_frac, position from eye_y.
    Replaces crop_rect_containing for every layout."""
def check(crop: Rect, heads: [Box], faces: [Box], layout: str,
          frame_w, frame_h) -> list[str]:
    """Return the list of violated invariants (I1..I7). Empty == valid."""
```

`frame_subject` is four lines and derives everything from the two ratios:

```
crop_h  = clamp(h_f / face_frac, MIN_CROP_FRAC·frame_h, frame_h)
crop_w  = crop_h · aspect                      # clamped to frame, aspect preserved
eye_px  = face_y + EYELINE_IN_FACE · h_f
crop_y  = eye_px − eye_y · crop_h
crop_x  = face_cx − crop_w/2 + look_room_shift # shift toward the side they face
```

Sanity check with the reference numbers — single, `face_frac=0.155`, `eye_y=0.22`:
`crop_h = 6.45·h_f`; crop top is `1.42·h_f` above the eyeline; the hair is `0.77·h_f` above it;
headroom `= 0.65·h_f = 10.1%` of crop height. ✅ I2.
Panel, `face_frac=0.30`, `eye_y=0.34`: `crop_h = 3.33·h_f`, headroom `= 0.363·h_f = 10.9%`. ✅

Both layouts land on the same ~10% headroom from independent ratios — that is the check that
the constants are self-consistent.

`LOOK_ROOM = 0.12` is likewise bounded by I6 rather than fighting it: a single's crop width is
`3.63·h_f`, so the maximum shift is `0.435·h_f`, while the head box half-width is `0.64·h_f`.
The crop centre therefore stays inside the head box at full look-room. The reference's measured
centre-x IQR (0.44–0.60, §1.1) is `±0.08` of frame width — `0.12` gives the engine slightly more
room than the reference uses, which is the right side to err on.

### 5.2 `reframe_v3.py`

| Line(s) | Change |
|---|---|
| `181–188` | Delete `DEFAULT_HEAD_Y`, `DEFAULT_SIDE_MARGIN`, `DEFAULT_VERT_MARGIN`. They are the head-cut. |
| `216 crop_rect_containing` | Keep the name, reimplement as a thin wrapper over `framing_contract.frame_subject`. Every caller gets the fix for free. |
| `263 _wide43_rect` | Centre on the **speaker's** head box, not the union centre; then expand to contain the others. Enforces I6. |
| `627 _clip_and_filter_box` | Add: reject boxes with aspect outside `[0.55, 1.35]` (signs, placards, the "Red Flag" cards) and boxes whose area is > 12% of the frame (the back-of-head close-up). Add a `min_frac` floor so a 20px face never becomes a subject. |
| `654 _track_boxes_for_shot` | Add a **presence gate**: if the track has detections in < 70% of the sampled instants of the shot, it is not eligible to be a tight single — return `None` and let the layout fall back. Use the **p90 box, not the union**, so one bad frame does not inflate the crop. |
| `729 _compose_shot:742` | The no-subject branch must satisfy I5: pick the 4:3 wide that contains the most head boxes; if there are none at all, hold the previous shot's crop rather than emitting a curtain. |
| `776–798`, `802–833` | Both vsplit paths call `frame_subject(..., PANEL_FACE_FRAC, PANEL_EYE_Y)` and then `check(...)`. Any violation → downgrade to TWO_SHOT, then WIDE. Never render an invalid panel. |
| `834–861` | Single path calls `frame_subject(..., SINGLE_FACE_FRAC, SINGLE_EYE_Y)`; `attention_shifted_crop` becomes a bounded **look-room** shift (≤ `LOOK_ROOM`) instead of a free saliency aim. Direction: toward the nearest other head box within 1.5·crop_w, else toward the face's gaze side (SCRFD's 5 landmarks give it for free — eye-midpoint vs. nose x), else no shift. This is the rule that reproduces the reference's OTS composition (§1.3) without needing OTS coverage. |
| `455 validate_composition` | Delete the `LAYOUT_SPLIT` exemption (`485–487`). Run `framing_contract.check` on every layout including panels. Keep raising `CompositionError`. |
| `918 _crop_resize_panel` | Make the letterbox branch (`935`) unreachable: `_recontain` must preserve aspect (§5.3). If it still fires, raise — a black bar between panels is a bug, not a fallback. |
| `1015 _vsplit_caption_ass` | Invert: captions during a vsplit go to the **bottom band** `[0.86, 0.94]`, never MIDDLE. See §5.5. |
| `1385` | `MAX_SHOT_SECONDS` default `8 → 14`, and exempt a shot whose speaker never changes (the reference's 17s holds). |

### 5.3 `smart_crop.py`

- `167` — `y = sy − sh·headroom` becomes `crop_y = eye_px − PANEL_EYE_Y·crop_h`. The anchor and
  the size must be derived from the same ratio or I2 cannot hold.
- `139 _target_crop` — size from `PANEL_FACE_FRAC`, not from margins.
- `59 _recontain` — **must preserve aspect.** Today it grows `w` and `h` independently, which
  drifts the panel aspect past the ±10% tolerance at `reframe_v3.py:935` and produces the black
  bar seen in clip 2. Grow the *smaller* dimension to the aspect-correct size instead.
- `172 step` — add an assertion: the returned crop satisfies I2/I3 for `box`. On violation,
  snap to `frame_subject(box)` rather than gliding. The tracker is allowed to be less smooth;
  it is not allowed to cut a head.
- Add `VSPLIT_TRACK=0` to the eval matrix. If tracked panels cannot hold the contract, static
  panels are the correct default — a static, correct panel beats a gliding, wrong one.

### 5.4 `shot_planner.py` — split-screen admission

The goal here is **not fewer splits — it is only correct ones.** Today the gate is permissive
in the wrong direction: it admits windows the geometry can't serve (crowd panels, mismatched
scales, one-frame "presence") and it fires on side-by-side pairs a two-shot would hold better.
Tighten those and the split lands where it belongs. A window becomes a VSPLIT **only if all of
these hold**:

| Test | Threshold | Rationale |
|---|---|---|
| speaker switches in the window | ≥ 3 | 2 switches is one handover, not a back-and-forth |
| window duration | 2.5–10.0s **and** ≥ 2 complete sentences | a split that flashes past is a glitch; a split you can read is the device. Longer than 10s stops being an exchange. |
| both tracks **present** | ≥ 70% of sampled instants each | `_track_present_in_span:706` currently accepts one frame |
| both tracks pass I2/I3/I4 in their panel | hard | the geometry must work *before* we commit |
| face-size ratio between panels | ≤ 1.6× | I7 — bans "one close-up + one tiny group" |
| head boxes inside either panel | ≤ 2 | I7 — bans a crowd panel |
| one 9:16 or 4:3 crop can hold both **at ≥ 0.12 face height each** | **fails** | this is the far-apart test, and the size floor is the point of it. A two-shot that shrinks both people below 0.12 is not "holding both" — that is clip 4's wide, and the split beats it. |
| total split time in the clip | ≤ 60% (**diagnostic**) | not a taste quota. Above this, the two-shot test is almost certainly failing — clip 1 ran ~90%. Log it loudly; do not silently downgrade a split that passed every test above. |

Everything that fails admission falls through to, in order: **TWO_SHOT → look-room SINGLE →
4:3 WIDE**. The middle option is what produces the reference's over-the-shoulder look when the
source frame allows it (§1.3) — it is a crop placement, not a shot we choose.

Also in this file:

- `84 MAX_TWO_SHOT_WIDTH_FRAC` `0.72 → 0.82`. The reference happily holds two people across
  80% of frame width; raising this converts most current splits into two-shots, which is the
  outcome we want.
- `719 _tracks_close_enough` — use the p90 box over the window, not the median instant.
- `min_shot_seconds` default `1.2 → 1.8` at all five call sites.
- `merge_short_runs:199–201` — a forced-boundary run shorter than 0.8s is still absorbed. Only a
  boundary that is *both* a scene cut and ≥ 0.8s survives. Kills the flash frames.
- New: `snap_shots_to_speech(shots, transcript)` — move every non-scene-cut boundary to the
  nearest word gap ≥ 120ms within ±0.4s. This is I9 at the shot level; the existing
  sentence-anchored snap in `main.py:1521` only covers the clip in/out.

### 5.5 `subtitles.py`

- `333` — keep `CAPTION_POSITION` as the user preference, but clamp the resolved band to
  `[0.62, 0.78]` of output height (the reference band) unless the user explicitly overrides.
- `481` — the "MIDDLE during a split" rule is the caption-on-face bug. Replace with: during a
  vsplit, `\an2` with `MarginV` computed to place the text block in `[0.86, 0.94]`.
- New: pass the composed shot list into caption generation and, per caption line, assert the
  text box intersects no head box (I8). On collision, lift the line by 6% of frame height, up
  to twice; then shorten the line. Log every collision — a collision that cannot be resolved is
  a shot-planning bug worth seeing.
- Cap at **3 words / line, 1 line** during vsplits (the reference uses 1–4 words), so the box
  stays small enough to have somewhere safe to go.

### 5.6 `speaker_fusion.py` — picture follows the diarized speaker

Clip 2's "host audio over a group picture" is the highest-value fix and it is upstream of all
the geometry.

- Make **ASR-first the default**, not `IDENTITY_CONFIRM=0`. The transcript decides *who*; the
  identity map decides *which face*; LR-ASD is the tiebreaker among candidate faces, not the
  primary. This inverts the current default (CLAUDE.md:208).
- **No-guess rule.** If the diarized speaker has no confidently mapped face on screen, the
  layout is WIDE. Never the nearest torso, never the largest face, never the previous binding.
  Add `SPEAKER_UNMAPPED_POLICY=wide|hold` (default `wide`).
- Turn `SPEAKER_REBIND_SECONDS` back on (default `4` / window `8`). One wrong binding for a
  whole clip is worse than one rebind.
- Log a per-clip **speaker-lock score**: fraction of seconds where the rendered subject's track
  equals the diarized speaker's mapped track. This is the number to move; it should be ≥ 0.85
  before any further framing work is worth doing.

### 5.7 The resolution budget — say no instead of faking it

`crop_h = h_f / face_frac` has a floor: `MIN_CROP_FRAC · frame_h`. On a 1080p source with the
0.45 floor, the tightest legal crop is 486px tall, so **a face smaller than ~75px (0.070 of
source height) can never reach the reference's 0.155** without upscaling past the blur floor.

That is not a bug to tune around — it is the source telling you the camera was far away.
Clips 2 and 4 measured 0.092 face height for exactly this reason: the underlying footage is a
group wide, and no crop can invent a medium shot out of it. The correct behaviours, in order:

1. Frame the 4:3 WIDE honestly and let it be a wide (it will read as an establishing shot,
   which the reference also uses).
2. Prefer a different moment: feed `face_h/frame_h` back into the picker as a **framing
   feasibility signal**, so a span where every candidate speaker is < 0.07 of source height
   scores lower. A clip that can't be framed shouldn't be picked.
3. Never upscale past `MIN_CROP_FRAC` to hit the ratio. A blurry correct-looking crop is worse
   than a sharp wide.

Log per clip: `framing_feasible = fraction of seconds where the bound speaker's face ≥ 0.07 of
source height`. Below 0.5, the clip is a wide-shot clip and should be planned as one.

---

## 6. The gate — `eval/framing_audit.py`

A plan that cannot be checked is another round of back-and-forth. Build the checker with the
plan, not after it.

**Input:** a rendered `.mp4` (+ optionally its shot plan JSON).
**Method:** sample at 2 fps, detect faces (YuNet, bundled — no GPU needed), detect cuts
(`ffmpeg select=gt(scene,0.25)`), OCR-free caption box detection via the burned-in ASS
timings when the plan is available.
**Output:** the §2 table plus a pass/fail per invariant.

Ship thresholds (a clip fails the gate if any is breached):

| Check | Fail if |
|---|---|
| I2 headroom | any frame with dominant head box within 4% of crop top |
| I3 eyeline | median eyeline outside `[0.16, 0.30]` |
| I4 face size | median face height outside `[0.10, 0.225]` |
| I5 empty frames | > 3% of frames with no head box ≥ 0.06 |
| I5 seam straddle | > 2% of frames with a face crossing y = H/2 |
| I6 gap centring | > 5% of frames with crop centre in no head box (single/panel) |
| I7 panel asymmetry | any vsplit second with panel face ratio > 1.6 |
| I8 caption collision | any caption line intersecting a head box |
| I9 cut grid | median shot < 2.0s, or > 8% of shots < 1.0s |
| split share (**warn, not fail**) | vsplit > 60% of clip duration — signals the two-shot test is broken, not that the split is |

Run it over the three `videosflop/` clips first — they must all fail, with the failures
matching §2. A gate that passes the known-bad clips is not a gate. Then wire it into
`main.py`'s finalize step behind `FRAMING_AUDIT=1` (warn) → `=2` (fail the job).

---

## 7. Order of work

Do these in order. Each one is independently shippable and each makes the next one measurable.

1. **`framing_contract.py` + unit tests.** Pure geometry, no pipeline. Tests assert I2/I3/I4
   hold for face boxes spanning 0.04–0.35 of frame height, at 9:16 and 9:8 aspects. *(~1 day)*
2. **`eval/framing_audit.py`.** Prove it fails all three flop clips and passes both reference
   clips. This is the regression gate for everything after. *(~1 day)*
3. **Wire the contract into the single/two-shot path** (`reframe_v3.py:834–861`) and delete the
   old margin constants. Re-render clip 2 and clip 4; eyeline should move 0.40 → 0.22 and face
   size 0.09 → 0.15. *(~1 day)*
4. **Panels.** Both vsplit paths + `smart_crop` + the aspect-preserving `_recontain`. Re-render
   clip 1; seam straddle should go 61% → 0. *(~1–2 days)*
5. **Split admission** (`shot_planner.py`, §5.4). Expect clip 1's side-by-side pairs to become
   two-shots *and* clip 4's far-apart wides to become splits — the rate should move in both
   directions. If splits only go down, the far-apart test (`≥ 0.12 face height each`) is
   mis-wired.
   *(~1 day)*
6. **Speaker lock** (`speaker_fusion.py`, §5.6). Measure the lock score before and after.
   *(~2 days)*
7. **Cut grid + captions** (§5.4 tail, §5.5). *(~1 day)*
8. **Flip the gate to fail-the-job** and delete the knobs the contract makes meaningless:
   `VSPLIT_HEADROOM`, `VSPLIT_SIDE_MARGIN`, `VSPLIT_VERT_MARGIN`, `VSPLIT_PANEL_MIN_FRAC`,
   `MIN_CROP_FRAC`, `DEFAULT_HEAD_Y`. Six fewer things to be wrong.

**Do not** spend time on encode quality, GPU utilisation or model upgrades until step 6 is
green. A sharp, fast render of the wrong person is still the wrong person.

---

## 8. Definition of done

Both reference clips pass the gate unchanged (it must not be tuned to our output). All three
`videosflop/` clips, re-rendered from the same sources, pass the gate. Speaker-lock ≥ 0.85.
Median shot length 2.4–3.6s. Every split screen that renders shows **two comparably framed,
uncut heads at ~0.15 of output height each** — the pass condition is the quality of the split,
not its quantity. Zero seam-straddling faces, zero caption/face collisions.

---

### Appendix A — evidence

Contact sheets (one frame per detected shot) in `docs/framing_evidence/`:

| File | What to look at |
|---|---|
| `reference_A_shots.jpg`, `reference_B_shots.jpg` | The target grammar: OTS shot–reverse-shot, listener as foreground mass, eyeline high and constant, captions on the torso. No split screen anywhere. Note this is shot coverage we can't select — we approximate the *composition* via look-room crop placement (§1.3). |
| `flop_clip1_shots.jpg` | Split screen as the default. Panels framed at 2× the reference face size. |
| `flop_clip1_seam_headcut.jpg` | Four frames at full resolution with the seam drawn in red. `00003`, `00015`, `00053`: the lower panel's subject is decapitated by the panel top edge. `00023`: the upper panel centres on the back of a head while the actual speaker sits tiny in the corner. Every caption sits on a face. |
| `flop_clip4_shots.jpg` | `00017`–`00025`: a teal curtain, no person. `00027`–`00035`: full-frame back-of-head. `00047`–`00077`: letterboxed 4:3 wide with 0.09-height faces, held for ~15s. |

### Appendix B — reproducing the measurements

```bash
# cuts
ffmpeg -v error -i CLIP.mp4 -vf "select='gt(scene,0.25)',metadata=print:file=-" \
       -an -f null - 2>&1 | grep pts_time
# frames + faces (YuNet model: opencv_zoo/models/face_detection_yunet)
ffmpeg -v error -i CLIP.mp4 -vf fps=2 -q:v 3 out/%05d.jpg
python3 eval/framing_audit.py out/          # once §6 exists
```

### Appendix C — the numbers to hold in your head

```
face height  = 0.155 of frame height     (band 0.10 – 0.225)
eyeline      = 0.22  of frame height     (band 0.16 – 0.30)
headroom     = 0.10  of crop height      (floor 0.06)
median shot  = 2.9s                      (floor 1.8s, ceiling 14s)
split panel  = 0.30 of PANEL height      (= 0.15 of output — same as a single)
split screen = for people too far apart to share a crop; unlimited if every
               panel holds the numbers above. It is the only layout that
               keeps subject size when subjects are far apart.
```
