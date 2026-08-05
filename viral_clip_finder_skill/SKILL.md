---
name: viral-clip-finder
description: The mechanism layer for finding viral moments in long-form transcripts. Use this skill ANY time the user has a timecoded transcript (with or without speaker diarization) from a podcast, interview, stream, vlog, keynote, standup set, panel, or any long-form recording, and wants to extract short-form viral clips (15-30s) for TikTok, YouTube Shorts, Instagram Reels, or Twitter/X video. Also trigger when the user mentions "find viral moments", "clip this", "what should I cut", "shorts from this podcast", "find the hooks", "rank these clips", "what's the viral moment here", "extract clips from transcript", or is sitting on a pile of transcripts and needs to decide what to actually publish. This skill encodes the *judgment layer* — named viral frameworks, an 8-axis scoring rubric, anti-pattern detection, and per-niche playbooks — not the technical transcription pipeline. Use it even when the user doesn't explicitly ask for "viral" — if they have a transcript and want clips, this is the skill.
---

# Viral Clip Finder

## What this skill is — and what it isn't

This skill is the **judgment layer** that decides which 15–30 second windows of a long-form transcript have real viral potential. It assumes transcription, diarization, and any audio processing have already happened upstream in your pipeline. You are not transcribing. You are not slicing audio. You are *reading the transcript and deciding what is worth cutting*.

The reason most "find viral clips" tools fail is that they optimize for surface signals — keyword matching, "this speaker got louder", "this is a question", emotional word counts. Those signals are noise. Viral potential comes from **narrative mechanics**: the structural relationship between the opening line, the tension it creates, and the payoff it delivers in the next 15–30 seconds. A loud sentence with no payoff is just noise. A quiet, deliberate sentence that opens a loop the viewer needs closed — that's a clip.

This skill encodes those mechanics as a small set of named patterns, a scoring rubric that weights them, an anti-pattern detector that filters out false positives, and per-niche playbooks so you can tune judgment to the genre of the source material.

## When to use this skill

Use it when the user provides:
- A timecoded transcript (timestamps like `[00:14:32]` or `00:14:32.500` per line or paragraph)
- Speaker labels from diarization (e.g. `Speaker A:`, `Joe:`, `[S1]`)
- Optional audio cues embedded inline (`[LAUGHTER]`, `[PAUSE]`, `[LOUD]`, `[CROSSTALK]`)
- Source context (podcast name, niche, target platform) — helpful but not required

If the user only has raw audio or video with no transcript, do not run this skill yet — point them back to their transcription step. This skill is downstream of transcription.

## Input contract

The skill expects a transcript that resembles this shape:

```
[00:00:00] Host: Welcome back to the show, today we have...
[00:00:14] Guest: Thanks for having me.
[00:00:16] Host: So I want to start with something controversial...
[00:00:22] Guest: Look, the reason most people fail at this isn't laziness. [PAUSE] It's that they're optimizing for the wrong metric entirely.
[00:00:31] Host: What do you mean?
[00:00:33] Guest: I mean everyone tracks hours. Nobody tracks decisions per hour. And decisions are the only thing that actually compounds.
```

Acceptable variants:
- Brackets vs. plain timestamps
- Speaker names vs. `Speaker A/B`
- Per-line vs. per-paragraph timestamps (handle both)
- VTT/SRT-style formatting — strip the formatting, keep timing + text + speaker

If timestamps are missing entirely, the skill can still reason over content (and flag in output that timing is approximate), but it works best with timing.

### Enriched input (optional, unlocks more patterns)

The skill works on plain transcripts. But if your pipeline can supply **named-speaker identity** (via face recognition, voice fingerprinting, or manual labeling), the skill unlocks higher-value capabilities. The enriched input format is:

```
[00:00:00] Joe Rogan: Welcome back to the show...
[00:00:14] Elon Musk: Thanks for having me.
[00:00:16] Joe Rogan: So I want to start with something controversial...
```

Plus an optional **face trajectory** sidecar (JSON or inline), describing when each named person is on-camera and when they are actively speaking:

```json
{
  "identities": [
    {
      "name": "Joe Rogan",
      "on_screen": [["00:00:00", "00:14:32"], ["00:15:01", "00:31:44"]],
      "speaking": [["00:00:00", "00:00:14"], ["00:00:16", "00:00:22"]]
    },
    {
      "name": "Elon Musk",
      "on_screen": [["00:00:08", "00:47:22"]],
      "speaking": [["00:00:14", "00:00:16"], ["00:00:22", "00:00:33"]]
    }
  ]
}
```

The `speaking` ranges are critical — they tell the skill's cut brief which face to recommend the camera frame at any given moment. They come from your lip-sync model (e.g. LR-ASD) cross-referenced with face ID.

#### Why named identity matters (reframed)

The PRIMARY reason to add face ID is **speaker-attribution correctness**, not celebrity identification. Here's the problem it solves:

When your pipeline produces a diarized transcript, the speaker labels (Speaker A, Speaker B) come from voice fingerprinting. When your lip-sync model (LR-ASD) decides who's actively speaking, it's correlating mouth movement to audio. These two signals often DISAGREE — especially on:
- Code-switched speech (speaker switches language mid-sentence)
- Heavy accents (ASR misattributes)
- Overlapping audio (two people talking at once)
- Quick back-and-forth (speaker changes every 1-2 seconds)

When they disagree, the camera gets confused. The diarization says "Speaker A is talking" so the camera frames Face A, but the lip-sync says "Face B is producing audio" — the viewer sees Face A's mouth not moving while Face B's mouth moves in the background. This is the single most common production-quality failure in AI reframing.

**Face ID is the ground truth that breaks the tie.** When diarization and lip-sync disagree, the face ID signal (which tracked face has been most consistently correlated with the active audio) wins. The camera frames that face. The speaker label in the transcript gets corrected to that face's name.

This is the fundamental value: face ID ensures the right face is shown when someone is talking. Everything else — Status Play detection, reaction cam direction, cross-clip databases, niche routing — is a secondary benefit.

What named-identity input unlocks:

- **Speaker-attribution ground truth (PRIMARY)** — when diarization and lip-sync disagree about who's talking, face ID breaks the tie. The skill's cut brief can confidently say "frame Joe's face at 00:14:38 because he's the one producing audio, even though diarization thinks it's the guest." This is the camera-direction correctness layer.
- **Speaker-attributed cut briefs** — instead of "cold open on the guest's line," the brief says "cold open on Elon's line" and "frame Elon during the cold open, then cut to Joe's reaction at 00:14:38." Specific names → specific camera directions.
- **Reaction cam direction** — when the speaker is talking, the brief can recommend cutting to a named listener's reaction. "Cut to Joe Rogan's face at 00:14:38 — his expression sells the punchline." Only possible if the skill knows which face is Joe's.
- **Status Play detection (SECONDARY)** — the skill can flag clips where a named public figure's status is elevated or taken down. With names, the share-trigger score for drama/controversy and business niches jumps. But this is a bonus, not the reason to add face ID.
- **Cross-clip databases (SECONDARY)** — over multiple episodes, named identities let you track "this is the 3rd controversial thing Elon said this month" and surface recurring-guest patterns. Useful for editorial calendar planning.
- **Niche auto-routing (SECONDARY)** — if the speaker is a recognized comedian, route to the comedy playbook. If a recognized founder, route to business. Content cues do this by default, but face ID is a more reliable signal.

If your pipeline produces **anonymous** speaker IDs only (Speaker A, Speaker B), the skill still works — but you lose the camera-direction correctness layer. The cut brief can say "cut to the host's reaction" but cannot name the host. If your pipeline produces **no** diarization at all (just plain transcript), the skill works but loses speaker-attribution entirely. The skill degrades gracefully; each layer of enrichment adds specific capabilities.

See `references/integration-openshorts.md` for the full integration recipe (including the speaker-attribution ground-truth logic and T4 GPU recommendations) if you're wiring this into an existing pipeline.

## Output contract

Always return output in this structure. Output as Markdown unless the user asks for JSON.

```
# Viral Clip Report

**Source:** [name or filename]
**Niche detected:** [primary niche]
**Clips scanned:** [N candidate segments]
**Top picks:** [M clips returned]

---

## Clip #1 — [score]/100

| Field | Value |
|---|---|
| Start | 00:14:32 |
| End | 00:14:58 |
| Duration | 26s |
| Speaker | Guest |
| Niche | Motivation / Self-improvement |
| Primary pattern | Pop-the-Balloon |
| Secondary patterns | Concrete Specificity, Stakes Escalation |
| Platform fit | TikTok ⬤⬤⬤⬤○ · Reels ⬤⬤⬤⬤○ · Shorts ⬤⬤⬤⬤○ · X ⬤⬤⬤○○ |

**Transcript excerpt:**
> [00:14:32] Guest: You're not busy. You're just bad at prioritizing. Everyone has the same 24 hours. The difference is some people spend them on decisions that compound, and some people spend them on emails that disappear.

**Why it hits:**
[2-3 sentences explaining the viral mechanic — what loop it opens, what payoff it delivers, why the platform algorithms will reward it.]

**Cut instructions:**
- **Cold open line** (use as the on-screen hook text or first 1.5s audio): "You're not busy."
- **Trim suggestion:** Cut from 00:14:34 (drop the host's "what do you mean?" lead-in). Start hard on "You're not busy."
- **Caption text** (on-screen, 2-3 words max): "YOU'RE NOT BUSY."
- **B-roll cues:** Slow push-in on speaker's face; cut to b-roll of a cluttered inbox / phone notifications at "emails that disappear" (00:14:52).
- **Reaction cam** (only if enriched input with named identities is available): Cut to [host name]'s reaction at 00:14:38 — the visible surprise sells the reframe.
- **CTA placement:** End card at 00:14:58 — "Follow for part 2: how to actually prioritize."
- **Sound design:** Bass drop on "bad at prioritizing" (00:14:38). No music under the opening line — let it land dry.

**Risk flags:** None.

**Score breakdown:**
| Axis | Score | Max |
|---|---|---|
| Hook Strength | 14 | 15 |
| Payoff Density | 13 | 15 |
| Standalone Clarity | 14 | 15 |
| Emotional Spike | 8 | 10 |
| Share Trigger | 9 | 10 |
| Replay Value | 7 | 10 |
| Cut Quality | 9 | 10 |
| Cross-Platform Fit | 13 | 15 |
| **Total** | **87** | **100** |

---

## Clip #2 — ...
[repeat structure]

---

## Rejected candidates (with reasons)

| Timestamp | Reason rejected |
|---|---|
| 00:23:10 | Fake-Deep Quote — "Success is a journey" — no concrete payoff |
| 00:41:22 | Setup-Heavy, Payoff-Light — 22s of context for 3s of insight |
| 01:12:08 | Insider Baseball — references previous episode's in-joke |
```

The "Rejected candidates" section is mandatory. It shows the user the judgment is working and surfaces borderline calls they may want to override.

## The 15 named viral frameworks

Every clip you flag should be tagged with at least one primary pattern and optionally 1-2 secondary patterns. These names are how you reason about *why* a clip will work. Read `references/hook-library.md` for the full anatomy, examples, and platform notes for each — what follows is a quick reference.

| # | Pattern | One-line definition |
|---|---|---|
| 1 | **Open Loop** | Opens a question/puzzle the viewer can't close without finishing the clip |
| 2 | **Pattern Interrupt** | First sentence contradicts expectation; the contradiction IS the hook |
| 3 | **Status Play** | Speaker raises or lowers someone's public status (callout, praise, takedown) |
| 4 | **Pop-the-Balloon** | Punctures an inflated belief or ego; satisfaction of seeing the overinflated get deflated |
| 5 | **Curiosity Gap** | Speaker dangles information they have and the viewer doesn't |
| 6 | **Payoff Density** | Insight/punchline-per-second ratio is unusually high |
| 7 | **Share Trigger** | Expresses an identity/belief viewers will want to publicly endorse by sharing |
| 8 | **Stakes Escalation** | Perceived consequence of the topic ramps up across the clip |
| 9 | **Vulnerability Reveal** | Speaker shares something personal, counter-normative, or self-disclosing |
| 10 | **Contrarian Frame** | Speaker takes a position against consensus; generates debate in comments |
| 11 | **Concrete Specificity** | Uses specific numbers, names, scenarios — anchors abstract claims |
| 12 | **Sensory Hook** | Audio/visual moment (laughter, gasp, raised voice, prop) that grabs independent of content |
| 13 | **Tutorial Promise** | "Here's exactly how to X" — direct utility, clear takeaway |
| 14 | **Identity Threat** | Challenges something the viewer identifies with; compels engagement to defend/reinforce |
| 15 | **Myth Bust** | Explicitly counters a popular misconception with evidence |

A single clip often combines 2–3 patterns. The combination is what makes it strong — Pattern Interrupt + Concrete Specificity + Stakes Escalation is a near-formula for a viral motivation clip.

For each pattern's full anatomy (when to use, what makes it land, common failure modes, platform-specific notes), read `references/hook-library.md`.

## The 8-axis scoring rubric

Score every candidate clip on these 8 axes. Weights are baked into the max scores — Hook Strength and Payoff Density are weighted highest because they're the most reliable predictors of retention-driven virality. Full anchor descriptors (what 0 vs. 5 vs. 10 vs. 15 looks like for each axis, with examples) live in `references/scoring-rubric.md`.

| Axis | Max | What it measures |
|---|---|---|
| Hook Strength | 15 | Does the opening 1.5s demand attention? Pattern interrupt? Curiosity gap? |
| Payoff Density | 15 | Insight/punchline per second. Does it land hard and fast? |
| Standalone Clarity | 15 | Can a stranger with zero context follow it cold? |
| Emotional Spike | 10 | Does it produce a real emotion (laughter, anger, awe, validation, anxiety)? |
| Share Trigger | 10 | Does it express an identity/belief viewers will publicly endorse? |
| Replay Value | 10 | Worth rewatching? Density, nuance, or joke that benefits from re-hearing |
| Cut Quality | 10 | Is there a clean in/out point? No awkward mid-sentence entry? |
| Cross-Platform Fit | 15 | Fits 15–30s constraint for TikTok/Reels/Shorts without gutting it |
| **Total** | **100** | |

**Decision thresholds:**
- **85–100:** Tier A — cut immediately, this is a flagship clip.
- **70–84:** Tier B — solid, cut if you need volume. Worth A/B testing different cold-open framings.
- **55–69:** Tier C — has potential but needs editing work (reframe hook, trim setup, add context). Send to repair queue.
- **Below 55:** Drop. Even with editing, the underlying mechanic isn't there.

Don't return Tier C or below unless the user asks for a wider net. Default to returning only Tier A and B.

## Two clip length tiers

The skill supports two distinct clip-length tiers, each with its own rubric weighting and workflow. The user's pipeline may produce either or both. Always ask (or detect from context) which tier the user wants before scoring.

### Tier S — Short-form (15–30s, default)

The standard viral clip. Optimized for TikTok, Reels, Shorts, and X video. Hook must land in 1.5s. Payoff must arrive within the clip. No room for setup. Uses the 8-axis rubric above with the thresholds (85+ Tier A, 70–84 Tier B, etc.).

### Tier L — Long-form narrative (60–180s)

A full narrative clip — the kind of segment where the speaker tells a complete story, walks through a multi-step argument, or builds toward a payoff that needs context to land. Examples:
- A "Pop the Balloon" segment where a candidate's full experience with other candidates plays out over 90 seconds
- A founder's pivot story from realization → decision → fallout → resolution (120s)
- A comedian's full bit including setup, escalation, punchline, and callback (90s)
- An interview segment where a guest walks through a contrarian framework with evidence (150s)

These clips are typically distributed on YouTube (long-form), X (longer video), LinkedIn (professional long-form), or as multi-part TikTok/Reels series. They are NOT optimized for the 15-30s feed algorithm.

### Tier L — Modified 8-axis rubric

For Tier L, the rubric weights shift. Long-form rewards different things than short-form:

| Axis | Short-form max | Long-form max | What changes |
|---|---|---|---|
| Hook Strength | 15 | 10 | Less critical — long-form audiences tolerate slower hooks if the story earns it |
| Payoff Density | 15 | 10 | Less critical — you have time to develop ideas; density matters less than arc |
| Standalone Clarity | 15 | 15 | Same — cold viewers still need to follow |
| Emotional Spike | 10 | 15 | MORE critical — longer clips need multiple emotional beats to sustain attention |
| Share Trigger | 10 | 15 | MORE critical — long-form shares are deliberate, not impulse |
| Replay Value | 10 | 5 | Less critical — 1-3 min clips are rewatched less |
| Cut Quality | 10 | 15 | MORE critical — long clips need internal cuts; each cut is a failure point |
| Narrative Arc | — | 15 | NEW axis — does the clip tell a complete story with setup, tension, resolution? |
| **Total** | **100** | **100** | |

**Narrative Arc** is the new axis for Tier L. It measures whether the clip has:
- A clear setup (where are we, who's involved, what's the stakes)
- Rising tension (something happens that escalates)
- A climax (the moment of peak tension or insight)
- A resolution (what changed, what was learned, what's next)

A clip that's just "60 seconds of a person talking" with no arc scores low on Narrative Arc even if every sentence is interesting. The arc is what makes long-form watchable.

**Tier L thresholds:**
- **85–100:** Tier A — flagship long-form clip, publish as standalone.
- **70–84:** Tier B — solid, publish if you have a long-form distribution channel.
- **55–69:** Tier C — needs editing work, often to tighten the arc or remove digressions.
- **Below 55:** Drop. The story isn't strong enough to sustain 60+ seconds.

### Tier L — Workflow modifications

The 6-step workflow still applies, but with these modifications:

**Step 2 (Segment):** Instead of looking for 12–35s blocks, look for 45–200s blocks with a complete narrative arc. These are typically:
- A story the speaker tells (beginning, middle, end)
- A multi-step argument (premise, evidence, conclusion)
- A back-and-forth exchange that resolves
- A demonstration or tutorial that completes

Long-form candidates often contain short-form clips inside them. When you find a Tier L candidate, also note any Tier S sub-segments — the user may want to publish both.

**Step 3 (Tag patterns):** Long-form clips often layer 3–5 patterns across the arc. Tag the dominant pattern (the one that drives the arc) plus 2–4 secondary patterns that surface at specific moments.

**Step 6 (Generate cut brief):** The cut brief for Tier L is more complex. It includes:
- Cold open line (still critical — even long-form needs a strong opening)
- **Internal cut list** — timestamps of dead air, irrelevant tangents, or repetitive content to remove. The cuts must follow the seamless-cut rules in `references/seamless-cutting.md`.
- Speaker direction (which face to frame at each major beat)
- B-roll cues (more of them, since long-form needs visual variety)
- Caption text (longer — 4–8 words instead of 2–3)
- CTA placement (typically at the resolution, not the end)
- Sound design (multiple beats, not just one)
- Platform fit (rates YouTube long-form, X, LinkedIn, plus whether the clip can be split into a TikTok/Reels series)

For the internal cut rules — no overlap, no underlap, no mid-sentence cuts, natural flow — see `references/seamless-cutting.md`. This is the discipline that makes a 2-minute clip feel like it was recorded as one continuous take.

## Anti-pattern detector

Before scoring, run every candidate through the anti-pattern check. If a clip trips an anti-pattern, flag it explicitly in the risk field — don't silently suppress, because the user may want to override. But default to rejecting clips that hit any of the high-severity anti-patterns.

Read `references/anti-patterns.md` for the full list with examples and repair suggestions. Quick reference:

- **Fake-Deep Quote** — sounds profound, says nothing ("Success is a journey")
- **Vulnerable Out-of-Context** — personal reveal that needs more context to land
- **Insider Baseball** — references only meaningful to existing audience
- **Setup-Heavy, Payoff-Light** — long windup, short landing
- **Controversy Bait** — strong take that won't age well or carries brand risk
- **Insight Without Specificity** — abstract platitude without anchor
- **Premise Depends on Visual** — audio-only clip references something visible
- **Echo Chamber Take** — conventional wisdom dressed as contrarianism
- **Cliffhanger With No Resolution** — promises something the clip never delivers
- **Cringe Vulnerability** — oversharing that produces secondhand embarrassment
- **Stale Trend Reference** — riding a trend that's already peaked
- **In-Joke Cascade** — punchline only works if you've seen previous clip
- **Authority Without Substance** — "trust me, I've seen the data" with no data

## Workflow

Run this 6-step process for every transcript.

### 1. Orient

Read the first 2–3 minutes of the transcript to detect:
- **Niche** (podcast / comedy / motivation / education / lifestyle / sports / drama / business) — see `references/niches.md` for niche-specific patterns
- **Format** (interview, monologue, panel, conversation)
- **Tone** (casual, formal, heated, contemplative)
- **Speaker roles** (host, guest, co-host, caller)

This orientation tunes your pattern recognition. A motivation podcast has different viral mechanics than a comedy roundtable. Don't skip this.

### 2. Segment into candidate clips

Walk the transcript looking for **semantic boundaries** — places where a coherent thought, story, joke, or argument begins and ends. A candidate clip is a contiguous block that:
- Starts at a clean sentence opening (not mid-thought)
- Ends at a clean sentence close (not mid-thought)
- Falls in the 12–35 second range when read aloud at conversational pace (160–180 wpm)
- Contains at least one of: a claim, a story beat, a punchline, a tension point, a payoff

Use timestamps to verify length. If a candidate runs 35–50s, look for a tighter sub-segment. If it's under 12s, look for whether extending 2–3 seconds in either direction gives it a complete arc.

You should typically extract 15–40 candidates from an hour of long-form content. Fewer means you're being too selective; more means you're fragmenting thoughts.

### 3. Tag patterns

For each candidate, identify which of the 15 named frameworks are present. Tag a primary pattern (the dominant mechanic) and 0–2 secondary patterns. If you can't identify a pattern, that's a signal the clip is probably not viral — proceed to score low.

### 4. Run anti-pattern check

Pass each candidate through the anti-pattern list. If a high-severity anti-pattern is present (Fake-Deep Quote, Setup-Heavy Payoff-Light, Cliffhanger With No Resolution), default to reject. If a medium-severity one is present (Insider Baseball, Vulnerable Out-of-Context), flag but keep scoring.

### 5. Score on the 8-axis rubric

Apply each axis honestly. Anchor on the descriptors in `references/scoring-rubric.md`. Don't inflate scores — a clip with a weak hook is a weak clip even if the insight is great. The rubric is the discipline.

### 6. Generate cut instructions for top picks

For Tier A and B clips, generate the full cut brief:
- Cold-open line (the 1.5s that has to land)
- Trim suggestion (where to cut to make it tighter)
- Caption text (2-3 words for on-screen)
- B-roll cues (visual moments that reinforce the message)
- Reaction cam direction (only if enriched named-identity input is available — recommend cutting to a named listener's face when their reaction sells the line)
- CTA placement (where to insert the follow/subscribe/next-clip nudge)
- Sound design (when to add emphasis — bass drops, music swells, silence)
- Platform fit (rate TikTok / Reels / Shorts / X on a 5-dot scale, with rationale)

The cut brief is what makes this skill actually usable — without it, you're just giving the user a timestamp and a score, which they could get from any tool. The brief turns the skill's judgment into production-ready instructions.

## Niche routing

Different niches have different dominant viral mechanics. When you orient in Step 1, route to the matching section in `references/niches.md` for that niche's:

- **Dominant patterns** — which of the 15 frameworks show up most
- **Hook styles** — what openings tend to land
- **Niche-specific anti-patterns** — failure modes unique to the genre
- **Concrete "good" example** — what a top-tier clip in this niche looks like
- **Platform considerations** — where this niche's audience lives

The 8 niches with deep coverage:
1. Podcasts & long-form interviews
2. Comedy & entertainment
3. Motivation & self-improvement
4. Education & explainer
5. Lifestyle & vlog
6. Sports & gaming
7. Drama & controversy
8. Business & money

If the source material spans multiple niches (e.g. a business podcast with motivational moments), pull from each relevant section.

## Output discipline

A few rules that separate a useful clip report from a mediocre one:

**Be specific in the "why it hits" rationale.** Don't say "this is relatable and engaging." Say "this opens a curiosity gap at second 0 (the viewer needs to know what 'the wrong metric' is), delivers the payoff at second 18 (decisions, not hours), and the share trigger is identity — viewers who think of themselves as 'busy professionals' will share this to signal they're above that framing."

**Always give a cold-open line.** The hardest part of cutting a clip is choosing the first 1.5 seconds. Always specify exactly which sentence, phrase, or fragment should be the entry point. The default mistake is starting where the speaker started — usually you should start 2–3 seconds *later*, dropping the throat-clearing.

**Trim aggressively by default.** Most candidates have 3–8 seconds of fat. Identify the trim points explicitly. "Start at 00:14:34, not 00:14:30 — the host's question is setup, the guest's answer is the clip."

**Score honestly.** If a clip is a 62, give it a 62. The rubric is the discipline that makes this skill better than a vibes-based cut. Don't round up to make the user feel good — they're trusting you to be the filter.

**Show rejected candidates.** The "Rejected candidates" table at the bottom of the report is mandatory. It surfaces borderline calls and demonstrates the filter is working. Include at least 3–5 rejected clips per report.

## Examples

### Example 1: Motivation podcast clip

**Input transcript fragment:**
```
[00:14:30] Host: So why do most people fail at this?
[00:14:33] Guest: Honestly? It's not laziness. People aren't lazy. [PAUSE] They're just optimizing for the wrong metric. Everyone tracks hours. Nobody tracks decisions per hour. And decisions — not hours — are the only thing that actually compounds. If you spent an hour making one good decision, you'd be richer than if you spent ten hours answering emails.
```

**Expected output:**
- Score: ~85 (Tier A)
- Primary pattern: Pop-the-Balloon (deflates "I'm busy" identity)
- Secondary patterns: Concrete Specificity, Stakes Escalation
- Cold-open line: "People aren't lazy." (cut from 00:14:33, drop the host's question)
- Caption: "PEOPLE AREN'T LAZY"
- B-roll cue: at "answering emails" cut to b-roll of cluttered inbox
- Why it hits: opens a loop at second 0 (what's the wrong metric?), delivers the answer at second 12 (decisions, not hours), share trigger is identity defense for viewers labeled "lazy"

### Example 2: Comedy podcast clip

**Input transcript fragment:**
```
[00:42:15] Host: What's the worst date you've ever been on?
[00:42:18] Guest: Oh god. So I'm at this restaurant, right, and she pulls out a spreadsheet. An actual printed spreadsheet. Of her last twelve dates. With ratings. [LAUGHTER] And she's like, "I just want to be transparent about where you rank." [LAUGHTER] And I'm sitting there thinking — rank? Like I'm a Costco product?
```

**Expected output:**
- Score: ~78 (Tier B)
- Primary pattern: Pattern Interrupt (spreadsheet reveal)
- Secondary patterns: Concrete Specificity, Sensory Hook (laughter)
- Cold-open line: "So she pulls out a spreadsheet." (cut to 00:42:21, drop "Oh god" and the question)
- Caption: "THE SPREADSHEET DATE"
- B-roll cue: quick cut to stock image of an Excel spreadsheet at "actual printed spreadsheet"
- Why it hits: pattern interrupt at second 1 (a date with a spreadsheet?), concrete specificity anchors it (twelve dates, ratings, Costco product), sensory hook from the laughter signals to viewer "this is funny, you can laugh"

### Example 3: Education clip to REJECT

**Input transcript fragment:**
```
[00:08:30] Guest: And that's why success is really a journey, not a destination. You have to embrace the process. The destination is just a moment, but the journey is where you become who you are.
```

**Expected output:**
- Rejected: Fake-Deep Quote
- Reason: sounds profound, contains no specific claim, no actionable insight, no novelty. The phrases "success is a journey" and "embrace the process" are clichés that have been said ten thousand times. No share trigger — viewers won't share platitudes. Score would land around 35–45.

## When in doubt

- **If you can't identify a pattern, reject.** No pattern = no mechanic = no virality.
- **If the payoff is more than 8 seconds after the hook, reject.** Too slow for short-form.
- **If the clip requires context from earlier in the conversation, reject.** Cold clips must stand alone.
- **If the speaker is rambling, look for the densest 15-second sub-segment.** Often there's a clip buried inside a longer block.
- **If two clips have similar scores, prefer the one with the stronger Share Trigger.** Shares drive reach more than views.
- **If unsure between Tier A and Tier B, default to B.** Under-promising keeps the user's trust.

## Reference files

Read these as needed. Don't load them all at once — load the one relevant to the current step.

- `references/niches.md` — Deep playbooks for all 8 niches, with both short-form and long-form coverage per niche. Read the section(s) matching the source material's niche during Step 1 (Orient).
- `references/hook-library.md` — Full anatomy of each of the 15 named viral frameworks. Read during Step 3 (Tag patterns) when you need to verify a pattern or understand how a pattern combines with another.
- `references/anti-patterns.md` — Full list of anti-patterns with examples and repair suggestions. Read during Step 4 when a candidate feels off but you can't articulate why.
- `references/scoring-rubric.md` — Detailed anchor descriptors (0/5/10/15) for each of the 8 scoring axes, for both Tier S (short-form) and Tier L (long-form) rubrics. Read during Step 5 when scoring borderline candidates.
- `references/seamless-cutting.md` — The cut-to-fit-narrative methodology for Tier L clips. Rules for no overlap, no underlap, no mid-sentence cuts, natural flow. Read during Step 6 when generating cut briefs for long-form clips.
- `references/integration-openshorts.md` — How to wire this skill into an existing clip-generation pipeline (OpenShorts, OpusClip-style, or custom). Includes face-ID integration recipe with T4 GPU recommendations. Read this when integrating the skill into production code.
