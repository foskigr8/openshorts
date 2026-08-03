# Viral dating-panel short patterns — research notes

Built 31-jul-2026 from 7 real videos, independently analyzed via the
`video-analyzer` skill (not guessed, not from one example): 4 real published
"Pop the Balloon or Find Love" shorts, 2 from a different show ("A Perfect
Match" / UpDating), plus a 40s sample of OUR OWN raw source episode to check
whether the same structure exists upstream of any editing. Source URLs are
in the download script this analysis was run from (`research_clips/` dir,
gitignored — not checked in, this file is the durable artifact).

Goal: find what's common across different shows/creators (not just one
reference clip) so DeepSeek/reframe decisions are grounded in a real sample,
generalizable beyond this one show.

## 1. The single most important finding: our source footage is ALREADY multi-cam

Checked directly against our own `test_fixtures/pop_balloon_v2.mp4` (the raw
full-episode source we clip FROM, not a pre-edited short): it contains real
camera CUTS between a fixed wide shot and a medium two-shot, natively, in
the raw footage. This is not something we have to synthesize — TransNetV2
scene detection is very likely already catching these as scene boundaries
(confirmed 8 scenes detected in a 100s segment of clip 1's span).

**This reframes the "framing is broken, doesn't track" finding entirely.**
The published viral shorts are NOT built by smoothly panning a camera
across one continuous wide shot — every single one of the 6 studied shorts
is edited by CUTTING between pre-existing camera angles (wide / two-shot /
tight single / reaction shot), the same way our raw source itself is shot.
The fix is not "add panning to the reframe engine" — it's "stop discarding
information the source already gives us at each cut, and crop each already-
distinct shot correctly for what it actually shows."

Concretely, for `analyze_scenes_strategy` / `general_filtergraph` in
`reframe_v2.py`:
- GENERAL mode's own crop math currently throws away ~25% of width for a
  16:9 source at `GENERAL_CONTENT_HEIGHT_RATIO=0.42` (confirmed by tracing
  the filtergraph math — scaled width exceeds output width, so
  `crop=w=min(iw,{out_w})` actually crops, contrary to the code comment's
  assumption). That's a real, confirmed bug independent of the multi-cam
  finding — see the framing-bug conversation this session for the trace.
- GENERAL mode hardcodes `center_x = orig_w/2` — the crop is centered on
  the geometric middle of the frame, with ZERO awareness of where the
  actual people are. For a real wide shot (host + line of contestants),
  the geometric center is not necessarily where the relevant action is.
  Since real cuts already provide "movement" (a new shot appears, not a
  smooth pan), GENERAL mode doesn't need continuous panning — it needs ONE
  correct static crop decision per scene, based on where faces/subjects
  actually are in THAT shot, not a hardcoded blind center.
- TRACK mode's continuous panning is likely the right behavior only for
  scenes that are themselves a continuously-moving/handheld shot, or a
  single subject moving within a static camera's frame — not for scenes
  that are already a professionally fixed camera angle.

**Net implication**: the reframe fix isn't "build camera movement we don't
have" — it's "correctly crop each shot the source editor already gave us,"
which is a smaller, more tractable fix than inventing tracking from
scratch, and matches what every single reference clip actually does.

## 2. Hook — confirmed, refines the existing DeepSeek REAL HOOK RULE

All 6 clips open COLD with zero setup — no exceptions. But every single
opening line is one of:
- A direct, specific question ("How many kids do y'all got?", "Jessica were
  you interested in Ariel?", "What would you do if salary didn't matter?")
- A high-conflict/high-stakes statement ("You're just a smidgen too light
  skin for me")
- A physical/visual demonstration that speaks for itself (balloon pops,
  entrance)

None of the 6 open on a bare reply fragment ("I'm not." / "A 9? That's
high.") — this independently confirms the REAL HOOK RULE v3 distinction
(self-contained claim vs. bare fragment) already in `deepseek_worker.py` and
`gemini_worker.py`. No further prompt change indicated here — this is
confirmation, not a new finding.

## 3. Structure — a consistent 4-6 beat arc, every time

Hook → Setup/Investigation → Escalation/Turn → (sometimes a
Negotiation/Reveal beat) → Payoff, landing exactly on the emotional peak.
Two recurring named beats worth encoding explicitly:
- **"The Turn"**: a specific pivot moment (a reveal, a reversal, a
  flirtatious escalation) roughly 60-80% through the clip, distinct from
  the final payoff.
- **A "buffer" reaction shot** (0.5-1s, silent) sometimes placed as the very
  last frame specifically so the clip loops back into its own cold-open
  hook without feeling jarring (`ptb_3`, explicit example).

## 4. Pacing — the most universal, most actionable finding

**Every single one of the 6 clips has zero dead air and mid-sentence jump
cuts removing pauses/fillers WITHIN the clip, not just at its start/end
boundaries.** This is not occasional — it is present in 6/6 real examples,
making it the single most consistent, most confirmed pattern in this whole
research pass. This is the deferred "mid-sentence jump-cutting" feature
from earlier in the session — this research is the evidence that it's not
a nice-to-have, it's the #1 differentiator between what we currently ship
(one untouched contiguous span) and every real example studied.

Two specific jump-cut techniques named across multiple clips:
- Cutting to the reactor/listener's face BEFORE the speaker finishes their
  sentence ("ping-pong" editing) — visually covers the audio cut.
- Overlapping crowd/reaction noise under the next line of dialogue instead
  of waiting for it to die down.

## 5. Ending — always cuts at the peak, never trails

100% of the 6 clips end within 0-1s of the actual emotional/narrative peak
(a reveal, a kiss, a comeback, a mutual match). Several explicitly note the
cut happens "before the energy can dip." This matches the existing
NARRATIVE-ARC RULE's "once the payoff has landed, STOP" — confirmed, not
new, but worth having real-example backing for.

## 6. Framing/camera — see §1 for the core finding. Additional details:

- **Reaction shots follow emotional relevance, not just who's talking.**
  `perfectmatch_1` explicitly cuts to Alexa (the current partner) reacting
  WHILE Jessica is speaking about her — the camera shows who the moment is
  ABOUT, not just who's making noise. This is a materially different (and
  harder) signal than "who is the active speaker" — ties directly to the
  split-screen/reaction-cut feature already discussed and deferred this
  session. Worth keeping in mind as the real target behavior when that
  feature gets built, rather than pure speaker-diarization-driven switching.
- **Two-shots stay wide specifically to capture chemistry** (both people in
  frame together) during flirtatious/mutual moments — not every multi-
  person moment should collapse to a single tight crop.
- **Occasional digital "punch-in" zoom** on a serious or intense line for
  emphasis — a genuinely new technique not currently in our reframe engine
  at all (separate from the deferred pan/zoom split-screen feature).

## 7. Confirmed-common but explicitly deferred (do not build yet)

- **External reaction-meme inserts** (stock laughing-man clips etc.) — seen
  in 2/6 clips here (`ptb_4`, plus the earlier single reference clip). Real
  and recurring, not a one-off, but explicitly deferred by user decision
  this session ("we are not doing reaction meme b-rolls just yet").
- **Captions** — consistent style across all 6 (large bold sans-serif,
  1-3 word pop-on, word-perfect sync, color-highlighted key/controversial
  words, center-lower position) — close to what `subtitles.py` already
  does. Deferred per explicit user call this session (`AUTO_CAPTIONS=0`
  while clip selection/framing/audio get fixed first).

## 8. What this means for current work, in priority order

1. **Mid-sentence jump-cutting** (§4) — highest-confidence, most universal
   finding. Not yet built.
2. **GENERAL-mode crop fix** (§1) — two concrete, scoped bugs: the width-
   loss crop math, and the hardcoded blind geometric center instead of a
   real face-aware static crop decision per scene. Not yet built.
3. **Ending precision** (§5) / **hook precision** (§2) — already
   substantially addressed by existing prompt rules; this research
   confirms them rather than surfacing new gaps.
4. Reaction-relevance camera cuts (§6) and punch-in zoom (§6) — real
   patterns, bigger scope, fold into the already-deferred split-screen/
   reaction-camera feature discussion rather than building separately.
