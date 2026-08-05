# Hook Library — The 15 Named Viral Frameworks

This is the deep reference for each of the 15 viral frameworks the skill uses to tag clips. Read this when you need to verify a pattern, understand how patterns combine, or recognize failure modes of a specific pattern.

Each entry contains:
- **Definition** — what the pattern is, precisely
- **The mechanic** — why it works on the brain / platform algorithm
- **Anatomy** — the structural shape of a clip using this pattern
- **Example** — a concrete illustration
- **Combinations** — patterns it pairs well with
- **Failure modes** — how the pattern goes wrong
- **Platform notes** — where it over/under-performs

## Table of contents

1. [Open Loop](#1-open-loop)
2. [Pattern Interrupt](#2-pattern-interrupt)
3. [Status Play](#3-status-play)
4. [Pop-the-Balloon](#4-pop-the-balloon)
5. [Curiosity Gap](#5-curiosity-gap)
6. [Payoff Density](#6-payoff-density)
7. [Share Trigger](#7-share-trigger)
8. [Stakes Escalation](#8-stakes-escalation)
9. [Vulnerability Reveal](#9-vulnerability-reveal)
10. [Contrarian Frame](#10-contrarian-frame)
11. [Concrete Specificity](#11-concrete-specificity)
12. [Sensory Hook](#12-sensory-hook)
13. [Tutorial Promise](#13-tutorial-promise)
14. [Identity Threat](#14-identity-threat)
15. [Myth Bust](#15-myth-bust)

---

## 1. Open Loop

**Definition:** The clip opens a question, puzzle, or unresolved tension that the viewer can only close by finishing the clip.

**The mechanic:** The human brain has a powerful need to close open cognitive loops. When a question is raised, the brain holds it in working memory and cannot disengage until it's resolved. This is the Zeigarnik effect in action. The platform algorithms reward retention, and open loops are the most reliable retention mechanic. A clip that opens a loop in second 0 and closes it in second 25 will have dramatically better completion rate than a clip that just states information.

**Anatomy:**
- Second 0–3: Raise a question or unresolved tension (explicit question, or implicit "the reason X isn't what you think")
- Second 3–15: Build toward the answer (often with mini-reveals that delay full closure)
- Second 15–25: Deliver the payoff/answer
- Second 25–30: Optional kicker that recontextualizes the answer

**Example:**
> "The reason your morning routine is failing isn't what you think. It's not that you're waking up too late. It's not that you're not disciplined. It's that you're optimizing for the wrong thing entirely — and once you see it, you can't unsee it. Most people optimize for productivity. You should optimize for decisions."

Loop opens at second 0 ("isn't what you think"), stays open through the three "it's not" mini-reveals, closes at "you should optimize for decisions."

**Combinations:**
- Pairs with **Concrete Specificity** — anchor the answer in specifics
- Pairs with **Stakes Escalation** — raise the stakes during the open phase
- Pairs with **Curiosity Gap** — sometimes the loop IS the curiosity gap; the distinction is open loops are forward-looking ("what's the answer"), curiosity gaps are information-asymmetric ("you don't know what I know")

**Failure modes:**
- **Loop too long** — if the answer doesn't arrive within ~20s, viewers drop
- **Loop too vague** — "this will change your life" with no specificity
- **Loop closed too early** — answer arrives at second 5, then 20s of elaboration = boring
- **Loop never closed** — clip ends before the payoff (see Cliffhanger With No Resolution anti-pattern)

**Platform notes:** Strong on all four platforms. Particularly important for TikTok/Reels where the algorithm heavily weights early retention.

---

## 2. Pattern Interrupt

**Definition:** The clip's opening contradicts what the viewer expects to hear in this context. The contradiction itself is the hook.

**The mechanic:** The brain builds predictive models moment-by-moment based on context, speaker, setting. When the model is violated, attention snaps to resolve the discrepancy. This is why "X is wrong" openers work — the brain predicted "X is good" and now needs to update.

**Anatomy:**
- Second 0–2: Statement that contradicts the viewer's expectation OR contradicts the speaker's own previous framing
- Second 2–8: The reframe that explains the contradiction
- Second 8–20: Evidence/elaboration
- Second 20–25: Reaffirmation or kicker

**Example:**
> "Most successful people wake up at 5am. That's a lie. They wake up at 5am because they're already successful enough to control their schedule. The 5am thing is a symptom, not a cause. If you force yourself awake at 5am and you hate it, you'll just be tired and resentful — and that's not a recipe for success, that's a recipe for burnout."

The first sentence sets up the expected "5am is good" frame. The second contradicts it. The brain needs to keep watching to resolve.

**Combinations:**
- Pairs with **Contrarian Frame** — pattern interrupts that take a contrarian stance are doubly powerful
- Pairs with **Pop-the-Balloon** — puncturing an inflated belief IS a pattern interrupt
- Pairs with **Concrete Specificity** — anchor the reframe in specifics to prevent it feeling like contrarianism for its own sake

**Failure modes:**
- **Cheap contradiction** — "X is wrong" with no follow-through
- **Strawman interrupt** — contradicting a position nobody actually holds
- **Interrupt without reframe** — the contradiction has no payoff

**Platform notes:** Excellent on all platforms. Particularly strong on Twitter/X where quote-tweets amplifying "wait, this is wrong" drive the algorithm.

---

## 3. Status Play

**Definition:** The speaker raises or lowers someone's public status during the clip. Callouts, praise of high-status figures, takedowns of common beliefs held by aspirational groups.

**The mechanic:** Humans are social primates wired to track status. We pay rapt attention to status changes because they signal who matters, who's rising, who's falling. A clip where someone's status shifts is inherently engaging because the viewer is updating their own social map.

**Anatomy:**
- Second 0–3: Identify the target (named person, named group, archetype)
- Second 3–10: Make the status move (praise or takedown)
- Second 10–20: Justify the move with evidence
- Second 20–25: Implication for the viewer or the broader status landscape

**Example (takedown):**
> "Elon doesn't actually understand Tesla's engineering. I've talked to the engineers. He's a marketing operation that happens to own a car company. The real work happens despite him, not because of him. And if you don't believe me, go look at the actual patent filings — his name isn't on the ones that matter."

**Example (elevation):**
> "Everyone talks about Naval. Nobody talks about the guy who actually taught Naval everything he knows — and that guy is still running a $40M business out of a one-bedroom in Mountain View. He just doesn't want the attention. And that's exactly why he's worth listening to."

**Face-ID enhancement (secondary benefit, when enriched input is available):**
Status Play is one of the patterns that benefits from named-speaker identity, but this is a SECONDARY benefit of face ID — not the primary reason to add it. The primary reason for face ID is speaker-attribution correctness (see `references/integration-openshorts.md` section 8). With that said, when face ID is available and identifies a recognized public figure as the target of a Status Play, the skill unlocks:

- **Higher share-trigger scores for clips about recognized figures.** A takedown of Elon Musk is fundamentally more shareable than a takedown of a random founder — the audience knows who Elon is, which means the share carries social signal ("I'm the kind of person who agrees with this take on Elon"). Without face ID, the skill can't distinguish these.
- **Status magnitude weighting.** A status move on a head of state (very high status) is more viral than a status move on a mid-tier influencer (medium status). Face ID + a public-figure database lets the skill weight this.
- **Reaction cam cuts to the named target.** When the speaker takes down a named figure who is on-screen (e.g. a guest takes down the host, or vice versa), the cut brief can recommend "cut to [name]'s reaction at [timestamp]" — only possible if the skill knows which face is which.
- **Cross-clip status arc tracking.** Over multiple episodes, the skill can flag "this is the 3rd time [name] has been taken down on this show" — useful for editorial pattern detection.

When face-ID input is available and the target of a Status Play is a recognized public figure, boost the Share Trigger axis score by 1–2 points (max 10). When the target is on-screen during the clip, also boost Cut Quality by 1 point (the reaction cam option adds production value).

**Combinations:**
- Pairs with **Concrete Specificity** — name names, cite patents, point to specific evidence
- Pairs with **Contrarian Frame** — elevating a low-status figure or taking down a high-status one is inherently contrarian

**Failure modes:**
- **Status play without evidence** — "Elon is overrated" with no follow-up
- **Petty status play** — taking down someone much lower status than yourself (feels like bullying)
- **Status play that ages badly** — defending someone who later turns out to be wrong/criminal

**Platform notes:** Twitter/X gold. Strong on TikTok for takedowns, weaker for elevation. Reels audiences tend to prefer elevation over takedown.

---

## 4. Pop-the-Balloon

**Definition:** The speaker punctures an inflated belief, ego, or identity the viewer holds. The satisfaction comes from watching something overinflated get deflated — including the viewer's own self-image.

**The mechanic:** Most people walk around with inflated self-narratives ("I'm busy," "I'm a hard worker," "I'm self-aware"). A speaker who cleanly punctures one of these without cruelty triggers relief — the viewer secretly knew it was inflated, and the puncture lets them drop the pretense. This is one of the most powerful motivation-niche mechanics.

**Anatomy:**
- Second 0–3: Name the inflated belief ("You're not busy" / "You're not lazy, you're scared" / "You don't have a time problem, you have a priority problem")
- Second 3–12: Reframe what's actually true (usually harder and more uncomfortable than the inflation)
- Second 12–20: Offer the path forward (what to do instead)
- Second 20–25: Affirmation that the viewer can do this

**Example:**
> "You're not busy. I've watched your week. You spent four hours on TikTok. You spent two hours reorganizing your desk. You spent forty-five minutes picking what to wear. You're not busy — you're avoiding. And the reason you're avoiding is that the thing you should actually be doing scares you. So let's talk about that thing."

The puncture happens at second 0. The reframe at second 5 ("you're avoiding"). The path forward at second 15.

**Combinations:**
- Pairs with **Concrete Specificity** — list the actual behaviors that reveal the inflation
- Pairs with **Stakes Escalation** — what's at stake if you keep inflating?
- Pairs with **Vulnerability Reveal** — speaker shares their own moment of being punctured

**Failure modes:**
- **Cruel puncture** — feels like attack, not relief; viewer defends instead of relates
- **Puncture without path** — deflates without offering a way forward; leaves viewer worse
- **Puncture of belief nobody holds** — "You think you're a perfect parent" — nobody thinks that

**Platform notes:** Massive on TikTok and Reels for motivation/self-improvement. The dominant pattern in David Goggins, Chris Williamson, Hormozi clips. Weaker on LinkedIn where audiences prefer constructive framing.

---

## 5. Curiosity Gap

**Definition:** The speaker dangles information they possess and the viewer doesn't. The clip is structured around the viewer's need to find out.

**The mechanic:** Information asymmetry creates compulsion. When the viewer knows the speaker has a specific piece of knowledge, the brain treats it as an unresolved task — it can't disengage until the knowledge is acquired. This is distinct from Open Loop: open loops raise a question, curiosity gaps advertise that an answer exists and is being withheld briefly.

**Anatomy:**
- Second 0–3: Signal that you have specific knowledge ("There's one question..." / "I learned this from..." / "The data shows something weird...")
- Second 3–15: Build the asymmetry — make the viewer want the information
- Second 15–25: Deliver the information
- Second 25–30: Implication or call to action

**Example:**
> "There's one question every self-made millionaire I've interviewed asks themselves every morning. Just one. And it's not what you'd expect — it has nothing to do with goals, nothing to do with money, nothing to do with the day ahead. It's about something that happened yesterday. Want to know what it is? It's this: 'What did I avoid yesterday that I need to face today?'"

The gap opens at second 0 ("one question"), deepens through the "not what you'd expect" framing, closes at second 22 with the actual question.

**Combinations:**
- Pairs with **Concrete Specificity** — make the answer specific
- Pairs with **Stakes Escalation** — what happens if you don't know/act on this

**Failure modes:**
- **Gap too wide** — promises more than the payoff delivers (viewer feels cheated)
- **Gap never closes** — clip ends without delivering (Cliffhanger With No Resolution)
- **Gap with boring answer** — "What's the secret? Hard work." (anti-climax)

**Platform notes:** Strong across all platforms. Works especially well for educational and business niches.

---

## 6. Payoff Density

**Definition:** The clip has an unusually high ratio of insight, punchline, or specificity per second. There's no filler — every sentence delivers.

**The mechanic:** Short-form video rewards density. A 25-second clip with 5 distinct insights will outperform a 25-second clip with 1 insight and 24 seconds of elaboration, because the dense clip rewards rewatching and the sparse clip doesn't. Density is the underlying mechanic behind why certain "fast talker" creators go viral — they compress more signal per second.

**Anatomy:** No fixed structure — density is a quality of the writing/speaking, not a structural template. But dense clips tend to:
- Open with a claim (not a setup)
- Stack specifics rapidly
- Avoid hedging language
- Land a kicker in the final 2–3 seconds that recontextualizes

**Example:**
> "Stop networking. Start being useful. The people you want to meet don't need another contact. They need someone who's already done the work. So pick three people you want to know. Build something for each of them — a teardown, a research brief, a feature they're missing. Send it. Don't ask for anything. Six months of this and your network will be unrecognizable. Six weeks and you'll have your first mentor."

Every sentence delivers a distinct move: reframe → diagnose → tactic → tactic → tactic → outcome timeline.

**Combinations:**
- Pairs with **Tutorial Promise** — dense clips often have tutorial structure
- Pairs with **Concrete Specificity** — specifics ARE density
- Pairs with **Pattern Interrupt** — opening contradiction + dense follow-through

**Failure modes:**
- **Dense but incoherent** — too many ideas, viewer can't follow
- **Dense but generic** — fast-moving but saying nothing new
- **False density** — lots of specifics but no actual insight (the "life hack" trap)

**Platform notes:** Highest leverage on TikTok and Shorts where audience patience is shortest. On X, slightly longer dense clips (45–60s) work well.

---

## 7. Share Trigger

**Definition:** The clip expresses an identity, belief, or value that viewers will want to publicly endorse by sharing. The share is the goal, not just the view.

**The mechanic:** People share content that lets them say something about themselves. "I'm the kind of person who believes X." "I'm part of the group that gets Y." A clip with a strong share trigger functions as a social signal — the viewer uses it to communicate their identity to their network. Shares drive reach more than views because each share introduces the clip to a new network.

**Anatomy:**
- The clip must contain a clear, articulable position or value
- The position must be one the viewer is comfortable publicly endorsing
- The framing should be quotable — short, punchy, complete

**Example (business/values):**
> "If your company isn't paying you enough to live within five miles of the office, you don't have a job — they have a hostage. Remote work isn't a perk. It's the bare minimum acknowledgement that your time has value."

A remote-work advocate shares this to signal "I'm pro-worker, I'm not falling for office culture propaganda." The share IS the statement.

**Example (motivation/identity):**
> "Discipline is just self-respect at scale. You can't say you respect yourself and refuse to do the hard thing you promised yourself you'd do. That's not self-respect. That's self-betrayal with good PR."

**Combinations:**
- Pairs with **Pop-the-Balloon** — punctures a belief the viewer's outgroup holds
- Pairs with **Contrarian Frame** — contrarian takes are inherently shareable as identity signals
- Pairs with **Concrete Specificity** — make the position specific enough to be quotable

**Failure modes:**
- **Position too vague** — "Believe in yourself" — nobody will share because it adds nothing to their identity expression
- **Position too extreme** — viewers agree privately but won't share publicly
- **Position too partisan** — limits share audience to one tribe

**Platform notes:** Twitter/X and LinkedIn are the share-driven platforms. TikTok shares work for identity-heavy niches (motivation, fitness, business). Reels shares work for aesthetic/relatable content.

---

## 8. Stakes Escalation

**Definition:** The perceived consequence of the topic ramps up across the clip. What starts as a small observation ends as a high-stakes implication.

**The mechanic:** Viewer attention is sustained when the stakes rise. A clip that maintains the same stakes throughout feels flat; a clip where the stakes escalate feels like it's building toward something, which keeps the viewer watching. The escalation should be earned — not "this matters for your life, your family, your COUNTRY, your CIVILIZATION" — but real, logical escalation from specific to general, from immediate to long-term.

**Anatomy:**
- Second 0–5: Establish the topic at low stakes (a small observation, a specific case)
- Second 5–15: Reveal the broader implication
- Second 15–25: Show the highest-stakes consequence
- Second 25–30: Kicker that brings it back to the viewer

**Example:**
> "If you check your phone in the first ten minutes after waking up, you've lost the morning. If you've lost the morning, you've lost your peak cognitive window. If you've lost that, you're doing your best work when you're at 60%. And if you do that for a year, you don't have a bad year — you have a year of 60% work that compounds into a career that's 40% behind where it should be. All from the first ten minutes."

The escalation: morning → peak window → daily work → annual output → career trajectory.

**Combinations:**
- Pairs with **Open Loop** — open the loop with low stakes, close it with high stakes
- Pairs with **Concrete Specificity** — anchor each escalation level in a specific consequence
- Pairs with **Identity Threat** — the highest escalation is "this threatens who you are"

**Failure modes:**
- **Unearned escalation** — jumps from "morning routine" to "civilizational collapse" with no logic
- **Flat stakes** — claims escalation but stays at the same level
- **Escalation without payoff** — builds and builds but never lands

**Platform notes:** Works on all platforms. Especially strong for motivation and business content.

---

## 9. Vulnerability Reveal

**Definition:** The speaker shares something personal, counter-normative, or self-disclosing. Triggers parasocial bonding and viewer identification.

**The mechanic:** Vulnerability from a high-status speaker triggers a status equalization — the viewer feels closer to the speaker, which increases trust and engagement. Vulnerability also serves as an implicit tutorial — "if this person admits this, maybe I should too." The clip works because the viewer feels they're being let in on something private.

**Anatomy:**
- Second 0–3: Signal that something personal is coming ("I'll be honest..." / "This is hard to admit..." / "I never told anyone this but...")
- Second 3–15: The reveal
- Second 15–22: The reframe — what the speaker learned or how they changed
- Second 22–25: The takeaway for the viewer

**Example:**
> "I'll be honest — three years ago, I was making $400k a year and I cried in my car after work twice a week. I had everything I'd been told to want. And I was miserable. The thing nobody tells you about success is that if you achieve someone else's definition of it, you don't get the joy — you just get the obligations. And that's worse than failing on your own terms."

**Combinations:**
- Pairs with **Pop-the-Balloon** — puncture the inflated "success = happiness" belief
- Pairs with **Stakes Escalation** — escalate from personal anecdote to universal implication
- Pairs with **Share Trigger** — viewers share vulnerable content to signal their own emotional openness

**Failure modes:**
- **Cringe Vulnerability** (see anti-patterns) — oversharing that produces secondhand embarrassment rather than empathy
- **Vulnerability without insight** — confession with no takeaway
- **Performance vulnerability** — feels calculated, not genuine
- **Vulnerable Out-of-Context** (see anti-patterns) — needs more setup than the clip provides

**Platform notes:** Very strong on podcasts (Diary of a CEO, Steven Bartlett's whole format). Strong on TikTok for motivation/mental health. Weaker on Twitter/X where audiences prefer analysis over confession.

---

## 10. Contrarian Frame

**Definition:** The speaker takes a position against consensus. Generates debate in comments, which the algorithm rewards.

**The mechanic:** Contrarian takes provoke responses. Half the audience agrees and wants to amplify; half disagrees and wants to argue. Both groups comment. Comments signal engagement to the algorithm, which promotes the clip further. The contrarian frame is one of the most reliable virality mechanics — but it has the highest variance in brand safety.

**Anatomy:**
- Second 0–3: State the contrarian position clearly
- Second 3–12: Steelman the consensus view (so you don't sound uninformed)
- Second 12–22: Make your case with evidence
- Second 22–25: Kicker or implication

**Example:**
> "College is a better deal now than it's been in twenty years. I know that's not what you're hearing. Everyone's saying 'don't go, learn to code, trade school.' But here's the math: average grad makes $1.2M more over a career. Tuition is $80k. The ROI is still 15x. The reason everyone's against college is that the worst 20% of degrees are terrible. The best 80% are still incredible deals. Stop taking the headlines and start reading the data."

**Combinations:**
- Pairs with **Concrete Specificity** — contrarian claims need numbers to land
- Pairs with **Status Play** — taking down consensus figures is status play + contrarian
- Pairs with **Share Trigger** — contrarian takes are highly shareable as identity signals

**Failure modes:**
- **Contrarian without evidence** — feels like edge-lord posturing
- **Contrarian on settled issues** — "actually, smoking is fine" — won't age well
- **Echo Chamber Take** (see anti-patterns) — pretending to be contrarian on something everyone already believes
- **Contrarian that crosses into harm** — vaccine denial, climate denial — brand death

**Platform notes:** Twitter/X is the contrarian home. TikTok works if the contrarianism is counter-intuitive but not offensive. LinkedIn contrarians tend to be "industry hot takes" rather than culture-war stuff.

---

## 11. Concrete Specificity

**Definition:** The clip uses specific numbers, names, scenarios, and details rather than abstract generalities. Anchors claims in reality.

**The mechanic:** Specificity triggers the brain's concrete processing, which is more memorable and more believable than abstract processing. "I made $43,217" is more credible than "I made a lot of money." Specificity also creates the feeling of insider access — the viewer feels they're being told something real, not sold a line.

**Anatomy:** Specificity is a quality layered onto other patterns rather than a structural pattern itself. Look for:
- Exact numbers (not "a lot" but "$43,217")
- Named entities (not "a famous CEO" but "Tim Cook")
- Specific timeframes (not "recently" but "March 14th")
- Specific scenarios (not "a meeting" but "the Q3 planning offsite")
- Sensory details (not "it was bad" but "the air conditioning was broken and the CFO was sweating through his shirt")

**Example:**
> "I worked with a founder last year — 32 years old, Series B, $14M raised, 60 employees. He was working 80-hour weeks and his company was dying. Not because he wasn't smart. Because he was spending 47% of his calendar on meetings about meetings. We did a calendar audit. He had 23 hours a week of meetings where he wasn't a decision-maker. He just liked feeling important. Once we cut those, the company turned around in 90 days."

Every claim has a number. Every number earns the next claim's credibility.

**Combinations:**
- Pairs with **everything** — specificity makes every other pattern stronger
- Especially valuable with **Contrarian Frame** and **Pop-the-Balloon** — both need evidence

**Failure modes:**
- **Specific but irrelevant** — details that don't support the claim
- **Specific but unverifiable** — "I made $43,217" with no proof feels made up
- **Over-specific** — "On March 14th at 2:47pm Eastern, in conference room B..." — detail spillover

**Platform notes:** Universally valuable. Educational and business niches especially benefit.

---

## 12. Sensory Hook

**Definition:** An audio or visual moment that grabs attention independent of the content. Laughter, gasp, raised voice, prop, physical movement.

**The mechanic:** The brain prioritizes sensory salience — loud noises, sudden movements, emotional vocalizations all trigger orienting responses. A clip with a sensory hook in the first 1–2 seconds grabs attention before the viewer has even processed the content. This is why clips with [LAUGHTER], [LOUD], [GASP] markers in transcripts tend to perform — they're literal attention signals.

**Anatomy:** Sensory hooks are usually 0.5–2 seconds and sit at the very start of the clip. They can be:
- Vocal (laughter, gasp, raised voice, sudden whisper)
- Physical (gesture, prop reveal, costume change, slap, fall)
- Environmental (door slam, glass break, dog bark, alarm)
- Musical (sudden beat drop, key change, silence after noise)

**Example (vocal):**
> "[LAUGHTER] No — wait — let me finish — [LAUGHTER] — ok so the spreadsheet thing — she actually had columns. Columns! For personality, for height, for whether you texted back within 24 hours. And she scored me a 6.2."

The laughter at the start signals "this is going to be funny" before the joke lands.

**Combinations:**
- Pairs with **Pattern Interrupt** — sensory surprise + content surprise = double hook
- Pairs with **Vulnerability Reveal** — emotional vocalization signals the vulnerability is real

**Failure modes:**
- **Sensory hook with weak content** — grabs attention for 2 seconds, then loses it
- **Sensory hook that misleads** — laughter that suggests comedy when the clip is actually sad
- **Sensory hook mid-clip** — too late; viewer already scrolled

**Platform notes:** Critical on TikTok and Reels (sound-on default). Less critical on Shorts (many users watch muted) and X.

---

## 13. Tutorial Promise

**Definition:** The clip explicitly promises to teach the viewer how to do something. Direct utility, clear takeaway.

**The mechanic:** Tutorial content has high save and rewatch rates — viewers save it to reference later and rewatch to absorb the steps. Saves and rewatches are strong algorithmic signals. Tutorial clips also tend to be shared peer-to-peer ("you should try this") which extends reach.

**Anatomy:**
- Second 0–3: Promise the outcome ("Here's exactly how to..." / "The three-step method for..." / "How I [specific outcome] in [specific time]")
- Second 3–8: Frame the problem (why most people fail)
- Second 8–22: Deliver the method (numbered steps, specific tactics)
- Second 22–25: Result or call to action

**Example:**
> "Here's exactly how I doubled my newsletter open rate in 14 days. Most people A/B test subject lines. That's a trap — you're optimizing 10% of the funnel. The 90% is sender name and preview text. So: step one, change your sender name to a human name, not a brand. Step two, write the preview text as an incomplete sentence that creates an open loop. Step three, send at 6:42am — not 7, not 6:30. 6:42. The weird time outperforms round numbers by 11%. I went from 31% open rate to 67% in two weeks."

**Combinations:**
- Pairs with **Concrete Specificity** — tutorials live and die on specificity
- Pairs with **Payoff Density** — pack more steps per second
- Pairs with **Open Loop** — open with the outcome, deliver the method

**Failure modes:**
- **Tutorial without specifics** — "just be consistent and you'll grow" — not a tutorial
- **Tutorial for trivial outcome** — "how to tie your shoes" — nobody cares
- **Tutorial too long for short-form** — needs 60s+ to deliver; doesn't fit 30s constraint

**Platform notes:** Strong on TikTok and Shorts. Especially powerful for business, education, and lifestyle niches.

---

## 14. Identity Threat

**Definition:** The clip challenges something the viewer identifies with. Compels engagement to defend or reinforce.

**The mechanic:** When identity is threatened, the viewer cannot disengage — they must either defend (comment/quote-tweet) or reinforce (share to their in-group). Both behaviors signal engagement to the algorithm. Identity threats are powerful but risky — go too far and the viewer blocks you, not engages.

**Anatomy:**
- Second 0–3: Identify the identity ("If you call yourself an entrepreneur..." / "Parents who..." / "Anyone who works in tech...")
- Second 3–10: Challenge a belief or behavior common to that identity
- Second 10–20: Reframe — what the identity should mean instead
- Second 20–25: Affirmation of the version of the identity worth keeping

**Example:**
> "If you call yourself a 'self-made entrepreneur' and your parents paid your rent until you were 28, you're not self-made — you're a trust fund kid with a YouTube channel. And that's fine. But stop pretending the game was fair. The actual self-made people I know — the ones who really did start with nothing — they don't use the phrase. They're too busy working. The phrase is for people who need the identity more than the outcome."

**Combinations:**
- Pairs with **Status Play** — taking down a sub-group's self-mythology
- Pairs with **Contrarian Frame** — challenging the in-group's consensus
- Pairs with **Pop-the-Balloon** — puncturing the inflated self-image

**Failure modes:**
- **Cruel threat** — feels like attack, viewer blocks
- **Threat to identity nobody holds** — "if you call yourself a left-handed vegan..." — too narrow
- **Threat without reframe** — just criticism, no path forward

**Platform notes:** Twitter/X gold. Works on TikTok if the identity is broad enough ("entrepreneurs," "creatives," "parents"). Risky on LinkedIn (professional context punishes overt aggression).

---

## 15. Myth Bust

**Definition:** The clip explicitly counters a popular misconception with evidence. Educational mainstay.

**The mechanic:** Myth-busting works because it gives the viewer a small status boost — "I knew this and most people don't." It's inherently shareable as a correction signal ("see, I told you"). Myth busts also tend to have clear before/after structure: belief → evidence → correction, which is highly compressible for short-form.

**Anatomy:**
- Second 0–3: State the myth ("Everyone thinks X..." / "The common wisdom is..." / "You've been told that...")
- Second 3–10: Acknowledge why it sounds true (steelman)
- Second 10–22: Deliver the evidence that busts it
- Second 22–25: Implication — what to do instead

**Example:**
> "Everyone thinks you need 10,000 hours to master a skill. That's a misread of the research. Ericsson never said that — Malcolm Gladwell did, and he was extrapolating from violinists. The actual data shows you can hit 90th-percentile competence in most skills in about 50 hours of deliberate practice. 50. Not 10,000. The 10,000-hour rule is the most successful misinterpretation in pop science history — and it's keeping people from starting things they'd actually be decent at in a month."

**Combinations:**
- Pairs with **Contrarian Frame** — myth busts are inherently contrarian
- Pairs with **Concrete Specificity** — name the source, cite the actual data
- Pairs with **Status Play** — take down the figure who popularized the myth

**Failure modes:**
- **Myth nobody actually believes** — busting a strawman
- **Myth bust with weak evidence** — "actually, that's wrong because I think so"
- **Myth bust on settled issues** — busting things that aren't actually myths
- **Myth bust that's actually wrong** — embarrassing if the original myth turns out to be true

**Platform notes:** Strong on TikTok and Shorts for education niche. Strong on Twitter/X. Works on LinkedIn for industry-specific myths.

---

## How patterns combine

Most strong clips layer 2–3 patterns. Some reliable combos:

- **Pattern Interrupt + Concrete Specificity + Stakes Escalation** — the motivation template (open with a reframe, anchor it with specifics, escalate to high stakes)
- **Pop-the-Balloon + Vulnerability Reveal + Share Trigger** — the parasocial mentor template (puncture a belief, share your own experience with it, give the viewer language to share)
- **Contrarian Frame + Concrete Specificity + Myth Bust** — the thought-leader template (take a contrarian stance, bust the myth with data)
- **Open Loop + Curiosity Gap + Tutorial Promise** — the educational template (open a loop, deepen with information asymmetry, deliver as a tutorial)
- **Sensory Hook + Pattern Interrupt + Concrete Specificity** — the comedy template (laughter/sound hook, surprising reveal, specific anchors)
- **Status Play + Contrarian Frame + Share Trigger** — the commentary template (status move on a named figure, contrarian framing, shareable identity signal)

When tagging a clip, look for the dominant mechanic first (primary pattern), then identify 1–2 supporting patterns. The combinations explain *why* a clip will work, which is what the "why it hits" section of the output needs to articulate.
