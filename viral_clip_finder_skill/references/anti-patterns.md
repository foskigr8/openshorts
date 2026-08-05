# Anti-Patterns — The False-Positive Viral Moments

This is the deep reference for the anti-patterns the skill uses to filter out clips that *look* viral but aren't. Read this when a candidate clip feels off but you can't articulate why, or when scoring a clip and wanting to verify whether a concern should down-rank it.

The reason most "find viral clips" tools over-promise and under-deliver is that they only look for positive signals. They flag anything that sounds punchy. But the skill's value comes as much from what it rejects as from what it accepts. A mediocre clip finder returns 40 clips; a great one returns 8.

Each anti-pattern entry contains:
- **Definition** — what it is
- **Why it looks viral but isn't** — the trap
- **Example** — a clip that would trip this anti-pattern
- **Severity** — high (default reject), medium (flag and down-rank), low (note but keep scoring)
- **Repair** — when applicable, how to fix the clip so it can be rescued

## Table of contents

1. [Fake-Deep Quote](#1-fake-deep-quote) — *high*
2. [Vulnerable Out-of-Context](#2-vulnerable-out-of-context) — *medium*
3. [Insider Baseball](#3-insider-baseball) — *medium*
4. [Setup-Heavy, Payoff-Light](#4-setup-heavy-payoff-light) — *high*
5. [Controversy Bait](#5-controversy-bait) — *high*
6. [Insight Without Specificity](#6-insight-without-specificity) — *high*
7. [Premise Depends on Visual](#7-premise-depends-on-visual) — *medium*
8. [Echo Chamber Take](#8-echo-chamber-take) — *high*
9. [Cliffhanger With No Resolution](#9-cliffhanger-with-no-resolution) — *high*
10. [Cringe Vulnerability](#10-cringe-vulnerability) — *high*
11. [Stale Trend Reference](#11-stale-trend-reference) — *medium*
12. [In-Joke Cascade](#12-in-joke-cascade) — *medium*
13. [Authority Without Substance](#13-authority-without-substance) — *high*
14. [Cold Take Hot Packaging](#14-cold-take-hot-packaging) — *medium*
15. [Context-Dependent Punchline](#15-context-dependent-punchline) — *medium*
16. [Punchline Without Setup](#16-punchline-without-setup) — *high*
17. [Beautiful Sentences, Empty Claims](#17-beautiful-sentences-empty-claims) — *medium*
18. [Misleading Hook](#18-misleading-hook) — *high*

---

## 1. Fake-Deep Quote

**Definition:** A statement that sounds profound but contains no specific claim, no actionable insight, and no novelty.

**Why it looks viral but isn't:** The cadence and word choice mimic depth — short, declarative, slightly aphoristic. Brains pattern-match on "this is wisdom" and give a small dopamine hit. But there's nothing to remember, nothing to apply, nothing to share that adds to the viewer's identity expression. Watch time is fine, but saves/shares are zero, and the algorithm punishes clips with high view-to-share ratios.

**Example:**
> "Success is a journey, not a destination. You have to embrace the process. The destination is just a moment, but the journey is where you become who you are."

**Severity:** High. Default reject.

**Repair:** Almost never repairable. If there's a specific experience behind the platitude, ask: can we extract that story instead? If not, drop.

---

## 2. Vulnerable Out-of-Context

**Definition:** A personal reveal that needs more context than the clip provides to land properly. The vulnerability is real, but a cold viewer can't tell what's at stake.

**Why it looks viral but isn't:** Vulnerability triggers work because the viewer feels they're being let in on something private. But if the reveal depends on knowing the speaker's history, the relationship being discussed, or prior context, the cold viewer feels confused rather than moved. The vulnerability reads as random confession.

**Example:**
> "And I just — I couldn't keep doing it. I had to walk away. [PAUSE] It was the hardest thing I've ever done, but I knew if I stayed one more year, I'd lose myself entirely."

A regular viewer has no idea what "it" is. The emotion is real but the stakes are invisible.

**Severity:** Medium. Flag and down-rank. May be rescuable.

**Repair:** Extend the clip 5–10 seconds earlier to establish what "it" is. If the preceding context is too long/complex to fit in 30s, reject.

---

## 3. Insider Baseball

**Definition:** References that only make sense to an existing audience — names of previous guests, callbacks to past episodes, in-jokes, niche terminology.

**Why it looks viral but isn't:** The existing audience will love it. The cold viewer — who is the entire point of a viral clip — will feel excluded and scroll. Viral clips must work for strangers.

**Example:**
> "You know, it's like what Sarah was saying in episode 47 — when she talked about the mountain thing — this is the same pattern. And honestly, I think we're going to see the same kind of moment here that we saw with the launch."

A subscriber nods. A stranger has no idea who Sarah is, what "the mountain thing" is, what "the launch" was. They leave.

**Severity:** Medium. Flag and down-rank.

**Repair:** Cut the references. If the underlying point still works without them, rescue. If the point depends on the references, reject.

---

## 4. Setup-Heavy, Payoff-Light

**Definition:** A clip with a long, slow setup and a thin payoff. The ratio is wrong.

**Why it looks viral but isn't:** Short-form rewards density. A 25-second clip with 22 seconds of setup and 3 seconds of payoff will lose viewers before the payoff lands. The retention graph will show drop-off at second 8, and the algorithm will deprioritize.

**Example:**
> "So, I want to talk about something that's been on my mind for a while, and I think it's relevant to what we were discussing earlier, which is the question of how we think about productivity in the modern workplace. There's a lot of discussion about this, and I've read several books on it recently, and I think the consensus is missing something. The thing is, productivity isn't about output. It's about input quality."

22 seconds of throat-clearing for a 3-second payoff. The viewer is gone before the insight lands.

**Severity:** High. Default reject.

**Repair:** Aggressive trimming. Find the densest 5-second sub-segment (the payoff) and see if the clip can stand on that alone, or with a 3-second lead-in. If yes, rescue. If the setup is genuinely necessary, reject — the clip won't fit short-form constraints.

---

## 5. Controversy Bait

**Definition:** A strong, controversial take that will generate engagement but carries brand risk or won't age well.

**Why it looks viral but isn't:** It might *actually* go viral. But the cost is high: alienated segments of the audience, potential brand deals lost, possible platform moderation issues, and a clip that lives forever in search results tied to the creator's name. The short-term engagement spike isn't worth the long-term reputation drag.

**Example:**
> "Honestly, [ethnic group] [gender] [political affiliation] are the reason [industry] is failing, and if you can't see that, you're part of the problem."

This is the obvious case. More subtle versions: attacking a recently-deceased figure, taking sides in an active lawsuit, making claims about specific individuals that could trigger defamation issues, weighing in on contested scientific topics without credentials.

**Severity:** High. Default reject.

**Repair:** Sometimes the underlying point is salvageable by reframing from specific to general, or from "X group is bad" to "this pattern is worth examining." But often the controversy IS the appeal, and removing it guts the clip. When in doubt, reject.

---

## 6. Insight Without Specificity

**Definition:** A real insight, but expressed so abstractly that the viewer can't anchor it to anything.

**Why it looks viral but isn't:** The insight is technically there, but without specifics, it doesn't feel earned or memorable. The viewer's brain treats it as a platitude even if it isn't one.

**Example:**
> "The thing about leadership is that you have to be willing to make hard decisions. Most people avoid them. But the best leaders — they make the hard calls. And that's what separates them."

True? Sure. Specific? Not at all. Forgettable.

**Severity:** High. Default reject.

**Repair:** Look for specific examples in the surrounding transcript. Can the speaker's own story anchor the insight? If yes, pull that in. If not, reject.

---

## 7. Premise Depends on Visual

**Definition:** The clip references something visible — a prop, a chart, a gesture, a person's reaction — that won't be present or clear in the final short.

**Why it looks viral but isn't:** The transcript reads fine, but the audio-only or audio+subtitles experience won't carry the meaning. The viewer hears "and as you can see here..." with nothing to see.

**Example:**
> "And look at this — look at the size of this thing. [HOLDS UP OBJECT] I mean, you can't even — this is insane. And the weight? Forget about it."

A listener without the visual has no idea what "this" is.

**Severity:** Medium. Flag and down-rank.

**Repair:** If B-roll can be sourced to provide the visual, rescue (note in cut instructions that B-roll is mandatory). If the visual is one-time footage that can't be replicated, reject.

---

## 8. Echo Chamber Take

**Definition:** A take presented as contrarian but actually reflects consensus among the target audience.

**Why it looks viral but isn't:** The contrarian framing appeals to viewers who want to feel like they're in the know. But the take itself is something the audience already believes, so there's no real shift in perspective. The share trigger is weak — viewers don't share things they already believe as if they're revelations.

**Example (in a business podcast):**
> "Here's a hot take: remote work is actually good. Everyone's saying return to office, but the data shows remote teams are more productive. I know this is controversial, but someone had to say it."

This isn't controversial in 2024+ tech/business audiences. It's consensus dressed as contrarianism.

**Severity:** High. Default reject.

**Repair:** Find the actual contrarian version. What's the take that the same audience would disagree with? "Remote work is making us all worse at our jobs" — that's contrarian in this audience. If the speaker has a real contrarian take elsewhere in the transcript, find that instead.

---

## 9. Cliffhanger With No Resolution

**Definition:** The clip promises a payoff that never arrives within the clip.

**Why it looks viral but isn't:** Open loops work because they close. A clip that opens a loop and ends before closing it leaves the viewer unsatisfied — they feel manipulated. Watch time is fine but completion rate drops, comments are negative, and shares are zero.

**Example:**
> "I'm going to tell you the one question that changed my entire career. It came from a mentor I had when I was 24, and it completely reframed how I thought about success. The question is — well, actually, let me back up. The context matters here. So I was living in..."

The question never arrives. The viewer waits and is cheated.

**Severity:** High. Default reject.

**Repair:** Extend the clip to include the actual payoff. If the payoff is more than 30 seconds away from the hook, the structure won't work for short-form — reject.

---

## 10. Cringe Vulnerability

**Definition:** Oversharing that produces secondhand embarrassment rather than empathy.

**Why it looks viral but isn't:** Vulnerability triggers work when they create identification ("I've felt that too"). They fail when they create discomfort ("I would never admit that publicly"). The viewer cringes on the speaker's behalf and scrolls to escape the feeling.

**Example:**
> "I genuinely believed my ex was going to come back to me until like two months ago. I kept her toothbrush. I kept her shampoo. I would smell her pillow every night before bed. And I know that sounds bad, but — look, I'm just being honest."

The overshoot into specific, slightly-creepy detail triggers cringe, not empathy.

**Severity:** High. Default reject.

**Repair:** Sometimes rescueable by trimming to a less specific version. "I held onto hope my ex would come back for way too long" — that lands. The pillow detail kills it. If the speaker doesn't have a less-specific version, reject.

---

## 11. Stale Trend Reference

**Definition:** A clip that rides a trend, sound, or meme that has already peaked.

**Why it looks viral but isn't:** Trends have a half-life. A clip using a trend sound or trend format 2 weeks after peak will get buried by the algorithm, which has already moved on. The clip looks fine in isolation but won't get distribution.

**Example:** A transcript from 2024 referencing "the Roman Empire trend" or using "girl math" framing — both peaked and declined.

**Severity:** Medium. Flag and down-rank.

**Repair:** Strip the trend reference. If the underlying content works without it, rescue. If the trend IS the content, reject (by the time you cut and post, it'll be even staler).

---

## 12. In-Joke Cascade

**Definition:** The punchline depends on having seen previous clips from the same source.

**Why it looks viral but isn't:** Existing fans will laugh. Cold viewers won't. Since cold viewers are the entire point of a viral clip, this fails the standalone test.

**Example:**
> "And then he said 'you can't say that on a podcast' — and I was like, dude, I say that every week. Classic."

Only meaningful if you know the recurring bit about "you can't say that on a podcast."

**Severity:** Medium. Flag and down-rank.

**Repair:** If the joke has a standalone version, use that. If not, reject.

---

## 13. Authority Without Substance

**Definition:** The speaker cites authority — their own experience, "the data," "studies show" — without providing any actual evidence.

**Why it looks viral but isn't:** Authority claims feel compelling in the moment but don't survive reflection. The viewer's brain trusts the speaker for 10 seconds, then wonders "wait, what data?" and distrusts them. Saves and shares are low because the claim feels slippery.

**Example:**
> "Look, I've seen the data on this. I've worked with hundreds of founders. The numbers are clear — most VCs are fundamentally bad at their jobs. I can't get into specifics, but trust me, the pattern is unmistakable."

No data, no names, no specifics. Just "trust me."

**Severity:** High. Default reject.

**Repair:** Find the actual data or specific examples in the surrounding transcript. If the speaker actually cites a study or names a firm, pull that in. If they don't, reject — the clip is unsupported assertion.

---

## 14. Cold Take Hot Packaging

**Definition:** A mundane observation dressed up in viral-language patterns.

**Why it looks viral but isn't:** The packaging — pattern interrupt, curiosity gap, etc. — triggers the brain's "this is going to be a revelation" response. Then the actual content is mundane. The viewer feels cheated and remembers the cheat more than the content.

**Example:**
> "Here's the secret to productivity that nobody talks about. It's not what you think. It's not about apps. It's not about routines. It's this: just do the work. That's it. Just do the work."

Strong packaging, empty content.

**Severity:** Medium. Flag and down-rank.

**Repair:** Look for a more substantial version of the same point elsewhere in the transcript. If none exists, reject.

---

## 15. Context-Dependent Punchline

**Definition:** A punchline that requires the setup to be in the same clip, but the setup is too long to include.

**Why it looks viral but isn't:** The punchline alone doesn't land. The setup is necessary but bloated. There's no version of this clip that works at 30 seconds.

**Example:**
> "And that's when I realized — it was a tuna sandwich all along."

Without 60 seconds of mystery-story setup, this is meaningless.

**Severity:** Medium. Flag and down-rank.

**Repair:** Sometimes rescueable by adding 5 seconds of compressed setup. If the necessary setup is more than 10 seconds, reject — the clip won't fit short-form constraints.

---

## 16. Punchline Without Setup

**Definition:** The clip starts at the punchline. The setup happened earlier in the conversation and isn't included.

**Why it looks viral but isn't:** A punchline without setup is just a confusing statement. The viewer doesn't know what's being responded to.

**Example:**
> "I mean, that's what she said! [LAUGHTER]"

The laughter suggests this is funny. The cold viewer has no idea why.

**Severity:** High. Default reject.

**Repair:** Extend backward to include the setup. If setup + punchline together exceed 30s, look for a different clip.

---

## 17. Beautiful Sentences, Empty Claims

**Definition:** The clip is well-written — rhythmic, alliterative, quotable — but doesn't actually claim anything specific.

**Why it looks viral but isn't:** Beautiful language gives the brain pleasure. But if there's no claim, there's nothing to remember or share. Viewers will quote it back to themselves, then forget it.

**Example:**
> "Time is the only currency that compounds in silence. While we measure wealth in dollars and status in followers, the hours slip past us like water through open hands. And one day we wake up and realize — the river was the wealth all along."

Gorgeous. Says nothing actionable. Says nothing new.

**Severity:** Medium. Flag and down-rank.

**Repair:** Look for the actual claim or example behind the language. If there's a real point being made in plainer words elsewhere, use that. If not, reject.

---

## 18. Misleading Hook

**Definition:** The hook promises something the clip doesn't actually deliver.

**Why it looks viral but isn't:** Short-term retention is fine — the viewer is hooked. But completion-rate drop-off is sharp, comments are negative ("wait, that's not what the title said"), and the algorithm learns the clip disappoints.

**Example:**
> "Here's how I made a million dollars in 30 days."

[Then the clip is actually a 25-second story about how the speaker got lucky on a crypto trade their cousin tipped them off to — no method, no takeaway, just a story.]

The hook promised a method. The clip delivered a story. Viewer feels bait-and-switched.

**Severity:** High. Default reject.

**Repair:** Sometimes rescueable by re-cutting the hook to match the actual content. "Here's how I accidentally made a million dollars" — that matches the story. If the hook can be honestly reframed, rescue. If not, reject.

---

## How anti-patterns interact with the rubric

Anti-patterns aren't separate from scoring — they show up as low scores on specific axes:

- **Fake-Deep Quote** → low Payoff Density, low Share Trigger
- **Vulnerable Out-of-Context** → low Standalone Clarity
- **Setup-Heavy, Payoff-Light** → low Payoff Density, low Hook Strength (in the relevant portion)
- **Controversy Bait** → low Cross-Platform Fit (some platforms will demote)
- **Insight Without Specificity** → low Payoff Density
- **Echo Chamber Take** → low Share Trigger (no one shares things they already believe)
- **Cliffhanger With No Resolution** → low Payoff Density, low Replay Value
- **Authority Without Substance** → low Standalone Clarity, low Payoff Density

When a clip trips an anti-pattern, expect to see it reflected in the score breakdown. If the score is high but an anti-pattern is present, that's a signal the scoring was inflated — re-score honestly.

## The judgment call

Not every anti-pattern instance means automatic rejection. The skill's job is to flag, not to mechanically filter. A clip that trips a medium-severity anti-pattern might still score 75+ if the underlying mechanic is strong enough to overcome it. But:

- Default to rejecting high-severity anti-patterns
- Flag medium-severity ones in the "Risk flags" field of the output
- Let the user override

The "Rejected candidates" table at the bottom of every report should include at least one example of each high-severity anti-pattern that was triggered, so the user can see the filter is working.

## The meta-anti-pattern: looking for clips instead of moments

The biggest failure mode of clip-finding tools is that they search for "clippable moments" — sentences that sound punchy — rather than "viral moments" — structural combinations of hook + tension + payoff that retain and share. A clip can have great sentences and no mechanic. A clip can have mediocre sentences and a great mechanic.

The mechanic is what matters. The anti-patterns above all share a common failure: they look like clips but lack the underlying narrative mechanic that drives retention and sharing. When in doubt, ask: "What loop does this open? What payoff does it deliver? Why would a stranger share this?" If you can't answer all three, reject.
