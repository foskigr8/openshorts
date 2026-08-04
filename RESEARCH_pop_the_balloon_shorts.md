# Editing rules from top-performing Pop The Balloon shorts — 4 Aug 2026

Analyzed with the video-analyzer skill (Gemini 3.1-flash-lite, 6 fps) four
viral vertical shorts from the niche, chosen to span the reaction types the
owner cares about:

| short | views | moment type |
|---|---|---|
| `IpMXVhzc8og` | 8.5M | banter / "was she testing him" |
| `UgvA3cIJWSk` | 7.9M | insults + still gets the girl |
| `Rwl2ZOPGajE` | 6.3M | balloon-pull-out gag |
| `7sRh-5EKsos` | 2.9M | sympathy / rejection |

All four converge on the same editing grammar. The reference shorts are the
show's own hand-edited cuts, so this is ground truth for how this genre is
supposed to be cut.

## The reaction grammar (the part our engine is missing)

1. **Cut to the reaction within 0.2–0.5s of a trigger line** — a jab, a
   laugh, a shock, a balloon pop. The reaction is the payoff; the speaker is
   just the setup.
2. **Reaction shots are punctuation, not scenes**: 0.5–1.5s, then back to
   the speaker. Never hold a reaction longer than ~1.5s.
3. **Reaction padding**: every 3–4 speaker cuts, insert a reaction face even
   if the speaker is still talking (J-cut: their audio continues over the
   reaction).
4. **On the punchline, the reactor wins**: cut to the person being insulted
   or the group laughing, not the person delivering the line.
5. **Eyes follow tension**: if the speaker names/mentions another person,
   cut to that person immediately.
6. **Silence is dead air**: a speaker pause >0.5–1s forces a cut — to a
   listener's face, or to the group.
7. **Emotional beats are the exception**: when someone is being vulnerable,
   HOLD them until their emotional unit finishes — no reaction cuts mid-
   sentence. Comedy cuts fast; sympathy holds.
8. **Balloon pops get their own tight shot**, then the reactor's face.

## The framing grammar

9. **Head-and-shoulders is the default crop**; tight face only for emphasis
   (punchlines, name/age answers); full body only for movement/action.
10. **Consistent headroom** (~10% of screen above the head) when switching
    between subjects of different heights.
11. **Monologue stability**: hold a stable head-and-shoulders during
    continuous sentences — no re-centering twitch mid-sentence.
12. **Group shots** to establish context at the start, at tone shifts, and
    as a check-in roughly every 10s.
13. **Host as buffer**: use the host's face for 0.5s between two different
    speakers to reset.
14. **Speaker dominance ~80%**: in 3+ person talk, the active speaker gets
    most of the screen; non-speakers appear only when emotive.

## Cut timing

15. **J-cuts**: new speaker's audio starts ~0.2s before the video cut lands
    on them.
16. **Banter toggle**: overlapping talk cuts every ~0.8s between the two
    speakers to dramatise conflict.
17. **Cut on breath**; avoid cutting mid-sentence.
18. **Energy rule**: a shot over ~2s needs a visual change — different angle,
    slow push-in, or a reaction insert.

## Gap analysis vs the current engine (session/framing-work, 3 Aug)

| rule | engine today | gap |
|---|---|---|
| follow the speaker | ✅ 98%/97% ASD match | — |
| monologue stability | ✅ held 69%/86% tier | — |
| reaction cuts 0.2–0.5s after trigger | ❌ none | **new reaction signal needed** |
| reaction ≤1.5s then back to speaker | ❌ none | **bounded reaction mode needed** |
| punchline → reactor wins | ❌ camera stays on speaker | reaction trigger covers this |
| eyes follow tension (name → cut to them) | ❌ no semantic link | future: transcript directive |
| silence → cut | ❌ holds through pauses | reaction-on-pause trigger |
| emotional beats hold | ⚠️ holds but for no semantic reason | needs transcript/emotion signal |
| balloon pop shot | ❌ no pop detector | future: audio/visual pop event |
| head-and-shoulders default, headroom | ⚠️ 3:4 crop, complaints of tight crops | measure headroom, adjust crop pad |
| group check-in / host buffer | ❌ no group shot in 3:4 pipeline | future: widen path |

The first implementation target (measured below) is the reaction tier: a
non-speaker whose mouth/expression spikes while the speaker is quiet gets a
bounded 0.5–1.5s shot, then the policy returns to the speaker. Everything
else stays untouched until it is separately measured.

## Implemented 4 Aug 2026 (reaction tier + head anchor)

### Reaction tier (subject_policy.py, tier 7)

Trigger, measured on Pop The Balloon span 2 (28s, 211 decision frames):
non-speaker `mouth_activity` runs ~0.07 median but reaction spikes reach
0.18–0.32, while the framed speaker's mouth drops below ~0.08 during the
beat after a line. So:

- `REACTION_MOUTH_ACTIVITY` 0.18 (a face must spike this hard to count),
- `REACTION_SPEAKER_QUIET` 0.08 (the framed person must be quiet),
- `REACTION_MIN_HOLD` 0.45s / `REACTION_MAX_HOLD` 1.3s (reference shorts
  cap reaction shots at ~1.5s and use them as punctuation),
- `REACTION_COOLDOWN` 2.2s (a laughing group cannot chain face to face),
- the lip-sync/diarized speaker is excluded from reactors (a mid-sentence
  speaker is not a reaction — found live: the first version cut to the
  speaker himself during the callout).

Fail-open: candidates without `mouth_activity` produce no reactions, and all
thresholds are env-tunable.

Measured on span 2: tier mix went from `lip-sync 86%, diarized 7%, held 7%,
size 0%` to `lip-sync 65%, diarized 7%, held 10%, reaction 18%, size 0%`.
The ASD-match metric (policycheck) drops 97% → 74% **because reaction shots
deliberately frame non-speakers** — that is the measured cost of the edit.
Reaction cuts land at src 2223.7, 2228.7, 2232.4, 2245.4, 2249.0; frame
cross-checks confirm they show the laughing/shocked blue-dress woman while
the man speaks, at head-and-shoulders with headroom.

### Head anchor (main.py `CAMERA_HEAD_ANCHOR` 0.16)

The unified 3:4 crop centred on the detection box's middle; for a YOLO
head-and-chest box that puts the head at the top edge and cuts it off
(measured: "top of head cropped" at src 2240.0). Anchoring the crop centre
at 0.16 down the box (the split-cell path's ground-truthed fix) restored
>15% headroom on the reaction shots. Residual: tight zooms on face boxes
can still clamp headroom when the subject sits high in the source frame —
next iteration (zoom cap / headroom-preserving rect).

### Tests

6 new subject-policy tests (reaction fires and returns; no reaction while
the speaker mouths; no interruption of fresh shots; bounded + no chaining;
speaker's own face ignored; identified speaker excluded; missing mouth data
fails open) + 1 new camera test (head anchor) + 1 updated zoom-y expectation.
Full suite: **658 passed** (was 650).

## v5 — the owner's cutting-dynamics spec (4 Aug 2026, second pass)

The owner clarified the target as pure cutting dynamics: hard cuts only, show
the speaker, show the person being talked to in thin intervals, never a shot
under ~1.5-2s, ignore short "yeah/right" agreements, and use the 3:4 width
for two-shots instead of ping-ponging. Every rule below is implemented,
env-tunable, and unit-tested (667 tests now pass):

1. **Minimum shot floor**: `POLICY_ABSOLUTE_MIN_SHOT` default **1.5s**,
   regardless of who is speaking. Measured on span 2: every engine shot is
   1.47s+ (44 frames @29.97), **0 rapid pairs <0.7s** (was 9-10). The one
   shorter gap is a source-cut-aligned re-decision (the director already cut).
2. **Short-agreement suppression**: transcript turns under
   `SHORT_UTTERANCE` (1.5s) are never proposed as the next speaker.
3. **Reaction shots** (expression tracking): non-speaker mouth spike >=0.18
   while the framed person is quiet >=0.35s -> bounded 2s cut, then back.
4. **J-cut pre-roll**: `J_CUT_PRE_ROLL` (0.5s) — the next long turn is
   proposed before its audio starts (TIER_JCUT outranks lip-sync in the
   window).
5. **Fatigue cut**: `FATIGUE_CUT_SECONDS` (8s) — a lock-on past 8s forces a
   2s cutaway to another visible person (backstop; did not fire on these
   clips because reactions/banter already broke every long hold).
6. **3:4 framing**: head anchor (CAMERA_HEAD_ANCHOR 0.16 down the box) placed
   `CAMERA_HEAD_Y` (0.36) down the crop — eyes on the top-third line whenever
   the source frame has the room (a subject near the top/bottom edge clamps;
   you cannot invent headroom the source does not contain).

### Measured tradeoff (v5 on the real clips)

| | span 1 | span 2 |
|---|---|---|
| tier mix | lip-sync 33%, diarized 17%, held 15%, reaction 28%, j-cut 5%, size 0% | lip-sync 34%, diarized 7%, held 50%, reaction 7%, j-cut 2%, size 0% |
| ASD-match | 98% -> **55%** | 97% -> **40%** |
| engine shots <1.5s | 0 | 0 (one 1.07s at a source cut) |
| rapid pairs <0.7s | 0 | 0 |

The floor is the cost: on fast banter the camera holds the current shot the
floor's length before switching, so it lags the conversation (the ASD-match
drop is exactly that lag). The owner's own tip 1.3 — during rapid banter,
default to a WIDER 3:4 crop that includes both speakers — is the compensating
move and is the named next step. Laughter/gasp audio override (tip 2.2) and
hand-gesture widening (tip 4.2) remain future work (they need audio-feature
and hand-detection signals the pipeline does not yet compute).

Watch it: UI job `ptb-v5-4aug` (clips 1/3 = v2-vs-v5 side-by-side, 2/4 = v5
full). Harsh-editor review of v5 span 2: no engine-created sub-1.5s shots and
no rapid pairs, but the reviewer still flags the source's own fast cuts and
the lag on quick speaker changes — the two-shot is the missing half.

## v6 — production regressions fixed (4 Aug 2026, owner feedback)

Owner review of the first production 9:16 delivery found real regressions.
Each was reproduced, fixed, and unit-tested (672 tests):

1. **Captions dropped to the bottom** — my render script never threaded
   `general_ranges` into `auto_caption_clip`, so the per-line middle-band
   MarginV (114) was lost and captions fell to the bottom letterbox. Fixed:
   `render_clip`'s `(success, general_ranges)` now flows into captioning
   exactly like `main.py:3145-3158`. Verified: `.ass` Dialogue lines carry
   MarginV=114 again, matching the old production clips.
2. **Reactions never fired during a monologue** — the trigger required the
   speaker to be quiet, so a 20s monologue showed no listener at all. Added
   `REACTION_DURING_SPEECH_AGE` (2s): once a shot has aged past the punchline
   window, a listener's mouth spike earns a cut even while the speaker
   mouths; fresh shots still need the quiet beat. Span-2 reaction share
   doubled 7% -> 14%.
3. **Reactions hit bystanders, not the partner** — `_best_reactor` now
   prefers the last OTHER subject framed (the conversational partner) over
   the loudest random face (owner: "reaction shots must be of Speaker B, not
   random bystanders"). Fatigue cutaways use the same priority.
4. **Fatigue cut never fired** — reactions reset `shot_started`, starving the
   8s clock. The speaker-hold clock (`_speaker_hold_start`) now only starts
   on a new speaker shot, so a long monologue still hits the fatigue cutaway.
5. **Two-shot jitter** — the banter two-shot recomputed + re-snapped the crop
   every detection stride. It is now composed ONCE on entry and LOCKED until
   the two-shot ends (zero movement while both people stay in frame).
6. **"Ghost cut" at 1:21** — verified in the SOURCE: the show itself cuts to
   a B-roll insert of the red-suit man at src 562.6-564.1. The engine
   faithfully followed a source cut; the tight reframe made it look like a
   glitch. Not an engine flash.

Residual: the Gemini scene-directive tier still reads 0% — directives bind by
x-position and are deliberately outranked by lip-sync/diarization in the
policy. The Gemini visual layer is honored as a POST-RENDER narration/confirm
step on the delivered clips (the "what's happening + is the framing on the
right person" review), which is what the owner remembers from the old
workflow. Deliverable: `ptb-prod-4aug` (two 9:16 captioned clips).
