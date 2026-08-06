"""DeepSeek narrative-arc-aware clip selection.

Single-pass, full-transcript analysis: DeepSeek sees every segment of the
whole video in one call, so it can trace a narrative across what the old
windowed Gemini scoring passes (main.py's original get_viral_clips) silently
truncated at ~90s window edges — there was no narrative-arc-completeness
concept anywhere in the pipeline before this. See NARRATIVE_PROMPT_TEMPLATE
for the actual rules. Falls back to windowed chunking
(clip_selection.build_transcript_windows) only for exceptionally long
transcripts that would risk the model's context budget — see
_MAX_SINGLE_PASS_ESTIMATED_TOKENS below for the conservative threshold this
is checked against.

DeepSeek's API is OpenAI-compatible chat completions with json_object mode,
not Gemini's native response_schema, so the model must be told the exact
shape in the prompt and the response is validated (not just trusted) with
Pydantic after parsing — reuses gemini_worker.py's provider-agnostic JSON
recovery helpers (code-fence stripping, malformed-\\u escape repair) rather
than duplicating them.
"""
import json
import math
import os
import time
from typing import List, Optional

import httpx
from pydantic import BaseModel, ValidationError

from clip_selection import build_transcript_windows, lookup_deepseek_model_prices
from gemini_worker import _parse_json_response_text

DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"

# Gemini's OpenAI-compatibility endpoint (documented by Google) — same
# request/response shape as DeepSeek's own API (chat/completions, Bearer
# auth, {"type": "json_object"} response_format, choices[0].message.content,
# usage.{prompt,completion}_tokens), so _run_deepseek_stage below can drive
# either provider unchanged; only the base_url/key/model differ. Added
# 1-aug-2026 (user, DeepSeek tokens ran out) as a same-prompt substitute —
# the narrative-arc prompt engineering is unchanged, only the model
# underneath it. Verified working directly against the live endpoint before
# wiring this in (both the chat-completions call and the model name).
GEMINI_OPENAI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
# Matches the model name already standardized elsewhere in this codebase
# (main.py's GEMINI_MODEL default) rather than introducing a new default.
DEFAULT_NARRATIVE_GEMINI_MODEL = "gemini-3.1-flash-lite"

# Conservative: leaves generous room for the prompt scaffolding + a 1M
# context window, while still exercising the chunked fallback path on
# anything that would be an unreasonably long single video (~15hr+ of
# transcript text). 4 chars/token is a standard rough heuristic.
_MAX_SINGLE_PASS_ESTIMATED_TOKENS = 500_000
_CHARS_PER_TOKEN_ESTIMATE = 4

# Segments shorter than this get merged into a neighbor before serializing —
# fewer JSON objects (less per-object brace/key overhead) for the same
# amount of actual transcript content.
_MIN_SEGMENT_CHARS_BEFORE_MERGE = 12


class KeepSpan(BaseModel):
    start: float
    end: float


class NarrativeClipModel(BaseModel):
    start: float
    end: float
    predicted_score: int
    # "short" (punchy hook/payoff beat) or "long_context" (a full 1-3 min
    # grounded arc — setup -> tension -> resolution, dead air trimmed but the
    # story intact). Defaults to short; normalized defensively after parsing.
    clip_type: str = "short"
    hook_type: str
    narrative_summary: str      # what question/tension opens, what resolves it
    essential_span_note: str    # human-readable rationale for keep_spans below
    keep_spans: List[KeepSpan] = []  # machine-usable jump-cut plan, see prompt
    video_description_for_tiktok: str
    video_description_for_instagram: str
    video_title_for_youtube_short: str
    viral_hook_text: str


class TermCorrection(BaseModel):
    wrong: str    # exact mistranscribed text as it appears in the transcript
    correct: str  # the corrected term


class NarrativeResponse(BaseModel):
    clips: List[NarrativeClipModel]
    term_corrections: List[TermCorrection] = []


NARRATIVE_PROMPT_TEMPLATE = """
You are a senior short-form video strategist AND editor working directly on
the full transcript in one pass — there is no artificial window boundary
here, so you can and must trace a story across the whole video.

{long_context_directive}
{style_directive}

AD/SPONSOR SEGMENT EXCLUSION (read this FIRST, before anything else — a
segment that fails this check is disqualified regardless of how good its
hook/payoff structure looks):
- Some segments are a sponsor read, affiliate pitch, or direct-response ad
  embedded in the video, not the video's actual content — the creator (or a
  guest) pauses whatever is happening to pitch a product, app, survey,
  referral link, "check the description," discount code, or similar
  paid/promotional offer, then the real content resumes afterward.
- These segments can look deceptively strong by every other rule below —
  clear hook, a "the reader gets $X" payoff, high energy, even a sentiment
  spike — because ad copy is written to be engaging. That surface appeal is
  exactly why this check has to run first: judged only on hook/payoff
  structure, an ad read can outscore the video's real content.
- Identify these by SUBJECT MATTER, not tone: is this segment about getting
  the viewer to click a link, download something, take a survey, use a
  promo/referral code, or otherwise transact with a third party — as opposed
  to being part of the video's own story, argument, or entertainment? If
  yes, EXCLUDE it as a candidate entirely, even if it's the most "quotable"
  moment in the transcript. The clip must represent what the video is
  actually about, not an ad that happened to run during it.

ANTI-DEFAULT RULE (read this FIRST, it is the most common mistake):
- Every video — including a single continuous sketch, rant, or monologue —
  has an internal structure of smaller beats: a setup, a specific turn, a
  specific payoff, then a comedown or a new topic. "The whole video is one
  inseparable narrative" is almost never actually true; it is what it looks
  like when you have not yet found the ONE beat inside it that stands alone.
- If your first instinct is a candidate anywhere near the
  {max_duration_ceiling}-second ceiling, that is a strong signal you
  defaulted instead of finding the real sub-arc. Go back through
  SEGMENTS_JSON and identify: where does a NEW specific tension/question
  first get raised (not the video's overall topic — a specific line, turn,
  or beat), and where does THAT SPECIFIC thing resolve? That span is almost
  always much shorter than the whole video.
- If the transcript is a single comedic bit or sketch, the payoff is the
  punchline or the specific twist — not the sketch's ending line, and not
  every setup that led there. Cut in close to the turn, hold only as much
  setup as the punchline needs to land for a cold viewer.

SEARCH METHOD — work backward from the peak, not forward from the start:
- Don't scan the transcript in order looking for a good opening. Instead,
  first find the PEAK: the segment(s) with `hl: true` (AssemblyAI flagged
  it as a standout moment) or a sharp `sent` swing (e.g. neutral/positive
  segments suddenly turning "-", or vice versa) — these are your best proxy
  for a real spike (a shock, a comeback, a walkout, a sudden reaction).
- Treat that peak as the PAYOFF (or very close to it) and work backward
  from there through SEGMENTS_JSON to find the specific line or moment that
  triggered it — that becomes your `start`. This finds real dramatic arcs
  far more reliably than scanning forward and hoping a strong opening
  happens to lead somewhere.
- Then apply the TIGHTNESS RULE below: strip dead air, stutters, "um"s, and
  tangential side conversations between that hook and that payoff. This
  must never change what anyone actually meant — it only removes filler
  and inessential words, the same way a good editor cuts pauses without
  altering a person's actual statement.
- Prefer ending the clip right AT the payoff moment (the reaction, the
  comeback, the walkout) rather than lingering into whatever comes after —
  a clip that ends right on its climax reads as tighter and loops better
  than one that trails into a comedown.

HIGH-TENSION EDITORIAL PROFILE — apply this especially to dating, panel,
reality, and confrontation formats:
- A short is a self-contained micro-story, not a neutral recap. Its first
  0-3 seconds should lead with the clearest real friction: a blunt standard,
  rejection, challenge, confidence claim, shocked reaction, or decisive
  question. Do not spend those seconds on greetings, names, or procedural
  setup unless that information is the friction itself.
- Build BACKWARD from a visible/audible payoff: a balloon pop, decisive
  rejection, pointed comeback, contradiction, walkout, crowd reaction, or
  mutual spark. Then include only the minimum real setup that makes that
  payoff hit for a cold viewer.
- Relentlessly remove pauses, repeated phrasing, hedging, side commentary,
  and low-stakes back-and-forth. Keep escalation continuous: claim -> pushback
  -> reaction -> payoff. A clip must never feel like it is waiting for the
  next important thing to happen.
- For conflict formats, preferentially rank a concrete, defensible disagreement
  above a merely pleasant exchange. Strong comment-driving angles include
  clear standards, mismatched expectations, a real contradiction, or unusually
  specific honesty. The *presentation* may be provocative; the selected words
  must still support the angle without reversing their material meaning.
- Use `viral_hook_text`, titles, and descriptions to spotlight the most
  debate-worthy truthful interpretation of the exchange. Make the audience
  want to take a side or respond, but never attribute a claim, motive, or
  outcome that the clip itself does not establish.

EDITORIAL PRIORITY, NOT A VOLUME SCORE:
- `predicted_score` is diagnostic metadata only; it must not be treated as a
  ranking based on yelling, profanity, or raw audio intensity. Return clips
  in your actual editorial priority order, strongest first.
- Look for concrete TRIGGER BLOCKS in the transcript: absolute standards or
  refusals ("never", "absolutely not"), social-value disputes ("standards",
  "50/50", "broke", "height"), a specific criticism, an exposed contradiction,
  or a firm boundary. A quiet exchange containing one of these can outrank a
  loud but empty argument.
- A trigger block alone is not enough. It must lead to a visible or audible
  turn — pushback, disbelief, laughter, a pop, a comeback, a walkout, or a
  meaningful reaction. Prefer the smallest self-contained span that carries
  both the trigger and that turn.
- Do not select a clip merely because a keyword occurs. Read the surrounding
  exchange and choose it only when the human tension is real and legible.

NARRATIVE-ARC RULE:
- A clip is a complete narrative unit: it opens a hook, question, or tension,
  and it MUST include that same thread's payoff/resolution before it ends.
- Do not end a clip just because the topic drifts in the transcript — follow
  the SAME thread to its actual resolution.
- DURATION IS A REAL, ENFORCED RANGE: every clip must be between 15 and
  {max_duration_ceiling} seconds. This is not a fallback safety net — it is
  your actual working range. If the strongest version of a moment would run
  longer than {max_duration_ceiling}s, you have not found the tight version
  yet: apply the TIGHTNESS RULE below harder, or pick a narrower beat within
  it, rather than returning something over the limit.
- Once the payoff has landed, STOP. Do not keep going because the transcript
  keeps talking — a clip that runs past its own payoff reads as padded, not
  thorough, and directly hurts retention.
- If you cannot find the payoff for a strong hook within {max_duration_ceiling}
  seconds, DROP that candidate rather than returning it unresolved or over-length.

TIGHTNESS RULE (apply within the rules above, not instead of them) — real
edited shorts in this genre do this in EVERY single example studied
(6/6 real published clips analyzed had zero dead air via internal jump
cuts, not just trimmed start/end boundaries — this is not optional
polish, it's the single most consistent thing that separates a real edit
from a raw excerpt):
- Once you've found a hook-to-payoff span, go through SEGMENTS_JSON within
  [start, end] and classify each segment as ESSENTIAL (the hook itself,
  real turning points, the payoff, anything the story can't be understood
  without) or CUTTABLE (redundant restating, hedging, filler, a tangent
  that doesn't serve this specific arc, dead air between beats).
- Return the ESSENTIAL segments as `keep_spans`: a list of
  `{{"start": <number>, "end": <number>}}` sub-ranges, in chronological order, each one
  a contiguous run of essential segments. This is what actually gets kept —
  everything inside [start, end] NOT covered by a keep_span is cut as a
  jump cut, the same way a real editor removes a pause or a tangent
  mid-clip, not just off the ends.
- Every keep_span must be at least 1.5 seconds — don't fragment into
  choppy micro-cuts faster than a viewer can track. Merge adjacent
  essential segments into one keep_span rather than returning many tiny
  ones back to back.
- If the ENTIRE [start, end] span is essential with nothing to cut (a
  genuinely tight span already), return a single keep_span covering the
  whole thing — do not invent cuts that aren't needed.
- `keep_spans` must be non-overlapping, sorted by start time, and fully
  contained within [start, end].

REAL HOOK RULE (this is not optional — a clip that fails this is worthless
regardless of how tight its payoff is):
- The test is NOT "does this need the line before it." The test is: is the
  opening line a SELF-CONTAINED CLAIM, standard, accusation, or assertion —
  something a cold viewer understands and reacts to on its own — or is it a
  bare REPLY FRAGMENT that only means something next to the question or
  statement it's answering?
  - Self-contained, can open cold with ZERO setup: "If you don't understand
    the difference between a good wig and a bad wig, I can't help you."
    "I don't really have deal breakers, because I'm from the ghetto." A
    confident claim, standard, or callout works standing completely alone —
    the viewer doesn't need to have heard a question first.
  - A bare fragment, CANNOT open cold: "I'm not." "A 9? That's high."
    "Yeah, she's a 9." These are replies — half of an exchange — and mean
    nothing without the other half. This is the actual failure mode to
    catch, not "lacks a formal introduction."
- Before finalizing `start`, identify what KIND of line it is. If it's a
  self-contained claim, leave it — do not pad it with unneeded setup just
  because a stricter rule once said "always include what came before." If
  it's a bare reply fragment, move `start` EARLIER to the line it's replying
  to (or find a different beat entirely whose opening IS a real claim) —
  even if that costs some of the "tightest span" preference from the
  TIGHTNESS RULE below.
- Never open on a bland self-introduction ("Hi, I'm...") with nothing else
  happening yet — that's dead air, not a hook, even though it's technically
  full context. If a real introduction is needed for the hook to make
  sense, keep it as short as the transcript allows and get to the actual
  tension immediately after.
- THE PRONOUN TRAP (explicit, mechanical, and the most common way a
  grammatically complete opening still fails cold): an opening that leads
  with a bare pronoun — "He", "She", "It", "This", "That", "They", or a
  "We"/"You" that names nobody the viewer knows — is a reply-style opening
  even when the sentence itself is complete. "He just walked in and said it
  to her face" is a complete sentence and STILL a cold-viewer failure: the
  viewer has no idea who "he" is, who "her" is, or what "it" means. Before
  accepting any opening that leads with a pronoun, rewind `start` EARLIER
  to the line that establishes the referent (the name, role, or thing the
  pronoun points back to), keeping that addition as short as the transcript
  allows — and if no antecedent exists close enough to include, DROP the
  candidate rather than ship an opening a cold viewer cannot parse. The one
  exception is a clip whose own first two seconds NAME the referent ("She
  — and her name is literally said here — ..."); in practice that means the
  naming line is inside the clip, so rewind to it. Re-check the final
  opening through this lens before committing `start`: does a stranger
  understand who/what every leading pronoun refers to from inside the clip?

THE 2-SECOND TEST (opening only): the first 2 seconds must stop a cold
viewer — no prior context — from scrolling past.

DIVERSITY: never return two clips that make the same point, tell the same
story, or land the same joke. Keep the stronger, drop the other.

CANDIDATE COUNT: a later review pass watches the actual footage for each
candidate and drops any whose opening or payoff doesn't hold up on video —
that review has no fallback, so returning only your single favorite risks
this video producing NOTHING if that one candidate is rejected. Return every
distinct beat that independently passes the rules above.
{candidate_count_directive}
Do not pad the count with weak or duplicate beats just to hit a number —
DIVERSITY and the arc rules still apply to every candidate you include; a
genuinely thin source still only returns what it actually contains.

RECOGNIZABLE PATTERNS — if this transcript is a multi-person panel,
interview, or reality/dating-show format, these are common shapes real
drama takes there; use them as a lens for WHERE to look, not as license to
invent or exaggerate anything that isn't genuinely present:
- Reality Check: someone states extreme confidence or requirements, then is
  immediately, genuinely rejected/contradicted by the group.
- Hypocrisy / Double Standard: someone rejects a person for a trait, then
  later accepts (or is caught having) that same trait themselves — only
  flag this if the transcript actually shows both halves.
- Brutal Honesty: someone gives a specific, blunt real reason for a
  rejection or criticism — the impact comes from the actual specificity of
  what they really said, not from trimming it to sound harsher than it was.
- Instant Chemistry: a genuinely warm, smooth exchange where both people's
  actual words show real mutual interest.

CAPTION STAGING NOTE: `viral_hook_text` will be rendered as an opening
overlay. Make it concise, emotionally legible, and specific enough to frame
the real tension — ideally a sharp question, standard, or disputed claim.
Do not use a generic summary when the speaker's own real conflict supplies a
stronger angle.

HOOK PLAYBOOK — pick the strongest fitting pattern for `viral_hook_text` (max 10 words):
- Open question: "Why does everyone get this wrong?"
- Hot take / controversy: "Stop doing this. Seriously."
- Number / fact shock: "97% of people miss this."
- Story loop: "This one email almost ruined me."
- POV / pattern interrupt: "POV: you finally understand it."
(These are English PATTERNS — always write the actual hook in TRANSCRIPT_LANGUAGE.)

COPY RULES — ALL text fields (descriptions, title, hook) MUST be written in TRANSCRIPT_LANGUAGE ({language}):
- Descriptions (TikTok + Instagram): 1-2 punchy sentences that tease the payoff
  without spoiling it, then 3-5 topically relevant hashtags. No generic hashtag spam.
- `video_title_for_youtube_short`: max 100 chars, curiosity-driven, no fake claims.
- `predicted_score`: honest 0-100 estimate of viral potential.
- `hook_type`: which HOOK PLAYBOOK pattern you used (e.g. "open question").
- `narrative_summary`: one line — what SPECIFIC question/tension opens this
  clip, and what resolves it by the end. If what you'd write here is really
  just "the video's overall topic," you have defaulted — find the smaller
  beat instead. If you can't state the resolution in one line, the clip
  probably doesn't actually resolve — reconsider it.
- `essential_span_note`: one line on what part of [start, end] is essential
  vs padding, for a later review pass to sanity-check the tightness rule.

TRANSCRIPTION CORRECTIONS: ASR transcription frequently mishears proper
nouns, brand names, or other distinctive repeated terms as an ordinary
similar-sounding word (e.g. a product name transcribed phonetically as an
unrelated common word). Scan the full SEGMENTS_JSON for words that are
almost certainly ASR errors — judged by: the word reads oddly out of place
for its sentence, AND a specific, similar-sounding proper noun/brand term
that is clearly this video's actual subject (from repeated context, or from
a spelling of it that DOES appear correctly elsewhere) would fit far better.
List each as `{{"wrong": "<exact mistranscribed text as it literally appears
in the transcript>", "correct": "<the corrected term>"}}`. Be conservative:
only include corrections you are genuinely confident about from context —
do not "fix" slang, filler words, or ordinary ambiguous word choices. Return
an empty list if nothing needs correcting. These corrections will be applied
to the burned-in captions, so precision matters more than recall here.

INPUT FORMAT: SEGMENTS_JSON is a compact array of this video's transcript
segments in chronological order, absolute seconds from the start:
  {{"s": <start>, "e": <end>, "t": "<segment text>", "sent": "+"|"0"|"-", "hl": true|false}}
`sent` is a per-segment sentiment hint ("+"=positive, "0"=neutral,
"-"=negative) — a lightweight signal for where emotional beats or turning
points likely are, on top of reading the text itself. `hl` is true when
AssemblyAI's own algorithmic key-phrase detection flagged that segment as
a standout moment — an independent second signal for where something
important is happening; segments with `hl: true` are good candidates for
where the actual hook or payoff lives, but always confirm against the text
itself rather than trusting the flag blindly.
SPEAKER LABELS ("sp"): when present, each segment carries WHO said it
(diarized). Absent means the source is a single speaker. Use it as structure,
never as a topic:
- A fast alternation between two speakers (short turns, trading lines,
  interrupting, talking over each other, a question answered instantly with a
  sharp comeback) is one of the strongest short-form signals there is. Prefer a
  candidate built on a real EXCHANGE over one built on an equally interesting
  monologue.
- The reverse is also true: a long stretch with no speaker change is a
  monologue. It can still be excellent, but it must earn it on the strength of
  the claim, not on energy it doesn't have.
- For a long_context clip, the participants should stay CONSISTENT across the
  segment. A span whose speaker set changes completely partway through is
  usually two unrelated scenes glued together, not one arc — exactly what the
  coherence requirement rules out.
- Never cut in the middle of a turn where the reply is the payoff. If a
  speaker's line lands only because of what the next speaker says back, the
  clip has to include the reply.
{context_note}
TRANSCRIPT_LANGUAGE: {language}
VIDEO_DURATION_SECONDS: {video_duration}
SEGMENTS_JSON:
{segments_json}

Return only valid JSON, no markdown fences, no commentary:
{{
  "clips": [
    {{
      "start": <number>,
      "end": <number>,
      "predicted_score": <integer 0-100>,
      "clip_type": "<short | long_context>",
      "hook_type": "<hook playbook pattern used>",
      "narrative_summary": "<what opens, what resolves it>",
      "essential_span_note": "<what's essential vs padding in this span>",
      "keep_spans": [
        {{"start": <number>, "end": <number>}}
      ],
      "video_description_for_tiktok": "<description + hashtags>",
      "video_description_for_instagram": "<description + hashtags>",
      "video_title_for_youtube_short": "<title max 100 chars>",
      "viral_hook_text": "<short overlay max 10 words>"
    }}
  ],
  "term_corrections": [
    {{"wrong": "<exact mistranscribed text>", "correct": "<corrected term>"}}
  ]
}}
"""


def _compact_segment(segment):
    text = str(segment.get("text") or "").strip()
    sentiment = segment.get("sentiment")  # only present on AssemblyAI-sourced transcripts
    out = {
        "s": round(float(segment.get("start", 0)), 1),
        "e": round(float(segment.get("end", 0)), 1),
        "t": text,
        "sent": sentiment if sentiment in ("+", "0", "-") else "0",
        "hl": bool(segment.get("highlight")),  # AssemblyAI Auto-Highlights salience flag
    }
    # WHO is speaking (AssemblyAI diarization). One character per segment, so
    # the token cost is negligible, and it is the single most useful structural
    # signal the selector was missing: a transcript with no speaker labels reads
    # as one uninterrupted monologue, so the model cannot tell a rapid two-way
    # exchange (a strong short-form hook) from someone talking at length, and
    # cannot verify that a long-context segment stays with the same
    # participants instead of drifting across unrelated scenes.
    speaker = segment.get("speaker")
    if speaker:
        out["sp"] = str(speaker)
    return out


def _merge_short_segments(compact_segments):
    """Merge consecutive very-short segments into their neighbor — fewer JSON
    objects for the same content, which is pure token overhead otherwise."""
    merged = []
    for seg in compact_segments:
        # Never merge across a speaker change: collapsing two people's lines
        # into one object destroys exactly the turn-taking structure the
        # speaker label exists to expose, and would make a fast exchange look
        # like one long monologue.
        same_speaker = (not merged) or merged[-1].get("sp") == seg.get("sp")
        if (merged and same_speaker
                and len(seg["t"]) < _MIN_SEGMENT_CHARS_BEFORE_MERGE):
            prev = merged[-1]
            prev["e"] = seg["e"]
            prev["t"] = (prev["t"] + " " + seg["t"]).strip()
            # Keep whichever sentiment is non-neutral; arbitrary but harmless
            # tie-break when merging rarely matters for a short filler segment.
            if prev["sent"] == "0":
                prev["sent"] = seg["sent"]
            prev["hl"] = prev["hl"] or seg["hl"]
        else:
            merged.append(dict(seg))
    return merged


def build_compact_segments(transcript_result):
    """Transcript segments -> compact {s,e,t,sent} dicts, short-segment-merged.

    No word-level timestamps are included — word-level snapping happens later,
    locally, against the full word list main.py already holds, costing zero
    LLM tokens either way.
    """
    compact = [_compact_segment(seg) for seg in transcript_result.get("segments", [])
               if str(seg.get("text") or "").strip()]
    return _merge_short_segments(compact)


def _estimate_tokens(text):
    return len(text) // _CHARS_PER_TOKEN_ESTIMATE


# Minimum candidates to target for a video this long — one per ~2 minutes
# of runtime (user spec, 1-aug-2026: "if it's a one hour clip, nothing less
# than 30 clips are coming out... there's a lot of narrative to spin").
# Floored at 2 so a short clip isn't forced to pad with weak candidates.
CLIP_COUNT_TARGET_SECONDS_PER_CLIP = 120

# Long-form sources are forced through the chunked path (see
# deepseek_select_narrative_clips) rather than trusting a single JSON
# response to enumerate dozens of well-reasoned candidates at once — that's
# a lot to ask of one call, and splitting keeps quality high per chunk
# while making the count floor mechanical rather than hopeful.
LONG_FORM_SECONDS_THRESHOLD = 900  # 15 min


def _target_clip_count(video_duration, override=None):
    """Target number of candidates for a video.

    ``override`` (from the dashboard's clip-count slider) wins outright: it is
    a HARD TARGET, not a floor — the user explicitly asked for that many
    clips and wants narrative built around them. Without an override the
    duration-based auto floor from CLIP_COUNT_TARGET_SECONDS_PER_CLIP stands.
    """
    if override is not None:
        return max(2, int(override))
    return max(2, round(video_duration / CLIP_COUNT_TARGET_SECONDS_PER_CLIP))


def _candidate_count_directive(video_duration, chunk_target=None, chunk_index=None,
                               chunk_count=None, override=None):
    """Text dropped into NARRATIVE_PROMPT_TEMPLATE's {candidate_count_directive}
    — a HARD FLOOR phrased as a floor, not a suggestion, since "usually 2-5"
    reliably undercounted long-form sources with plenty of narrative to spin.
    With an explicit user override the phrasing escalates from "floor" to
    "hard target": actively hunt for that many distinct threads/angles.
    """
    overall_target = _target_clip_count(video_duration, override=override)
    if override is not None:
        if chunk_target is None:
            return (
                f"The user explicitly requested {overall_target} clips from this "
                f"{video_duration:.0f}s video — this is a HARD TARGET, not a "
                f"suggestion. Actively look for {overall_target} distinct "
                f"threads, angles, or beats to build them around; split a strong "
                f"arc into its component beats if needed, and only return fewer "
                f"if the source genuinely cannot support {overall_target} "
                f"distinct, independently-passing clips after applying the "
                f"ANTI-DEFAULT RULE.")
        return (
            f"The user explicitly requested {overall_target} clips total from this "
            f"{video_duration:.0f}s video — a HARD TARGET. This is chunk "
            f"{chunk_index + 1} of {chunk_count}: aim for at least "
            f"{chunk_target} distinct candidates from THIS chunk (fewer only if "
            f"it genuinely lacks that many independent arcs), so the full video "
            f"lands at the requested {overall_target}.")
    if chunk_target is None:
        return (
            f"For a video this long ({video_duration:.0f}s), that is a HARD FLOOR of "
            f"{overall_target} distinct candidates (roughly one per two minutes of "
            f"runtime) — return AT LEAST that many if the source supports it. Only "
            f"return fewer than {overall_target} if the transcript truly does not "
            f"contain that many distinct, independently-passing arcs after applying "
            f"the ANTI-DEFAULT RULE.")
    return (
        f"This is chunk {chunk_index + 1} of {chunk_count} from a {video_duration:.0f}s "
        f"video — return AT LEAST {chunk_target} distinct candidates from THIS chunk "
        f"alone if it supports that many (the full video's overall floor across all "
        f"chunks is {overall_target}). Only return fewer if this specific chunk truly "
        f"does not contain that many distinct, independently-passing arcs.")


def _long_context_directive(long_context_count, max_duration_ceiling):
    """Opt-in prompt block for the long-context clip category (plan item 4).

    Empty (no directive) unless the job requested long clips — always-on costs
    tokens on every job even when the source has no full arc to tell.
    """
    if not long_context_count or long_context_count <= 0:
        return ""
    return (
        f"LONG-CONTEXT CLIPS (the user explicitly asked for up to "
        f"{long_context_count} of these): alongside the tight shorts above, "
        f"also find a FULL-ARC category: 1-3 minute grounded segments that keep "
        f"an ENTIRE narrative arc — setup, tension, resolution (e.g. a "
        f"contestant's complete run through a challenge, dead air trimmed but "
        f"the story intact from entry to outcome). This is a DIFFERENT framing "
        f"from the tight-hook shorts: a long_context clip may run up to "
        f"{max_duration_ceiling}s and does NOT need a punchy first-2-seconds "
        f"hook — its job is a complete, watchable arc with a real endpoint "
        f"(the payoff lands, or the exchange genuinely ends). Mark those clips "
        f"\"clip_type\": \"long_context\" and apply the TIGHTNESS rules to "
        f"dead-air removal only, never to compressing the arc itself. Include "
        f"at most {long_context_count} long_context clip(s), and only when the "
        f"source actually contains a full arc worth telling — never pad a "
        f"short beat out to fake one. Every other clip keeps \"clip_type\": "
        f"\"short\".")


def _normalize_clip_types(clips):
    """Defensively normalize clip_type to the known set — a provider returning
    a made-up value must not fail the job's schema validation or confuse the
    renderer's category handling downstream."""
    for c in clips or []:
        if c.get("clip_type") not in ("short", "long_context"):
            c["clip_type"] = "short"


# AI Preferences -> narrative prompt variants (round 3, item 6). "balanced"
# is today's unchanged default (empty block); the others are real, named
# directive blocks spliced into the prompt — never decorative.
_STYLE_DIRECTIVES = {
    "balanced": "",
    "high_energy": (
        "STYLE VARIANT — HIGH-ENERGY HOOKS: favor openings with immediate "
        "conflict, strong opinions, or a punchy claim in the first words "
        "(max 8 words for viral_hook_text); prefer the sharpest, most "
        "confrontational framing of each beat while keeping it accurate to "
        "what is actually said. The other rules (arc, payoff, diversity) "
        "still apply unchanged."),
    "story_driven": (
        "STYLE VARIANT — STORY-DRIVEN: favor clips that carry a complete "
        "narrative arc (situation -> obstacle -> how it resolves), including "
        "fuller context windows where the arc needs it; prefer the long-"
        "context framing for any beat with a real resolution, even when it "
        "fits in a short clip, and give narrative_summary the story shape "
        "(setup/tension/resolution). The other rules (payoff, diversity, "
        "accuracy) still apply unchanged."),
}


def _calculate_deepseek_cost_analysis(usage, model_name):
    if not usage:
        return None
    prices = lookup_deepseek_model_prices(model_name)
    price_estimated = prices is None
    if prices is None:
        prices = (0.20, 0.40)  # conservative estimate for an unknown model
    input_price_per_million, output_price_per_million = prices
    prompt_tokens = usage.get("prompt_tokens") or 0
    output_tokens = usage.get("completion_tokens") or 0
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (output_tokens / 1_000_000) * output_price_per_million
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "model": model_name,
        "price_estimated": price_estimated,
    }


_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def _run_deepseek_stage(api_key, model_name, prompt, response_model, base_url=DEEPSEEK_API_BASE):
    """One JSON-mode chat completion with transient-error backoff — DeepSeek
    by default, or any OpenAI-compatible provider via base_url (e.g.
    GEMINI_OPENAI_COMPAT_BASE — see deepseek_select_narrative_clips).

    Structurally parallel to main.py's _run_gemini_stage (3 attempts,
    exponential backoff), adapted to DeepSeek's plain-HTTP-status transient
    signal (429/5xx) rather than Gemini's error-message-token matching, since
    that's what's actually documented for this API.
    Returns (parsed_dict, cost_analysis).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.4,  # precise-but-not-robotic: narrative judgment plus real copywriting
    }

    max_attempts = 3
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(f"{base_url}/chat/completions", headers=headers, json=body)
            if resp.status_code in _TRANSIENT_STATUS:
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}", request=resp.request, response=resp)
            resp.raise_for_status()
            payload = resp.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = _parse_json_response_text(content)
            response_model.model_validate(parsed)  # raises on shape mismatch
            cost = _calculate_deepseek_cost_analysis(payload.get("usage"), model_name)
            return parsed, cost
        except ValidationError as e:
            last_exc = e
            if attempt == max_attempts:
                raise
        except (httpx.HTTPStatusError, httpx.TransportError, ValueError) as e:
            last_exc = e
            if attempt == max_attempts:
                raise
        wait = 5 * (2 ** (attempt - 1))
        # Name the provider actually being called. This function is
        # provider-agnostic (base_url decides), but the message said "DeepSeek"
        # unconditionally — so a Gemini failure was reported as a DeepSeek one
        # and read like DeepSeek was still wired in when it has not been since
        # 1-aug-2026.
        _provider = "DeepSeek" if DEEPSEEK_API_BASE in (base_url or "") else "Gemini"
        print(f"⚠️ {_provider} transient error (attempt {attempt}/{max_attempts}), "
              f"retrying in {wait}s: {str(last_exc)[:150]}")
        time.sleep(wait)


def _narrative_provider_candidates(api_key, model_name):
    """Ordered (provider, api_key, model_name, base_url) candidates to try
    for the narrative-arc call.

    DeepSeek REMOVED from the equation per explicit user direction
    (1-aug-2026) — Gemini (NARRATIVE_GEMINI_API_KEY) is now the sole
    provider, not a fallback. The tuned NARRATIVE_PROMPT_TEMPLATE + parsing
    is unchanged and provider-agnostic (Gemini's OpenAI-compatible endpoint
    returns the identical response shape DeepSeek's own API does), so this
    is purely a transport swap — kept as a SEPARATE key from GEMINI_API_KEY
    so it doesn't compound quota pressure on the key vision/scene-context
    calls already use. `_run_deepseek_stage`/DEEPSEEK_API_BASE/
    DEFAULT_DEEPSEEK_MODEL are left in place (the function itself is
    provider-agnostic, just driven by base_url) in case DeepSeek is ever
    reinstated, but nothing here calls into it anymore.
    """
    candidates = []
    gemini_key = os.environ.get("NARRATIVE_GEMINI_API_KEY")
    if gemini_key:
        candidates.append(("gemini", gemini_key,
                           os.environ.get("NARRATIVE_GEMINI_MODEL") or DEFAULT_NARRATIVE_GEMINI_MODEL,
                           GEMINI_OPENAI_COMPAT_BASE))
    return candidates


def _run_narrative_selection_once(transcript_result, video_duration, max_duration_ceiling,
                                  provider, api_key, model_name, base_url, label,
                                  compact, segments_json, clip_count=None,
                                  long_context_count=0, style_variant="balanced"):
    """One full attempt (single-pass or chunked) against one provider.
    Raises on failure — the caller decides whether to try the next
    candidate. Returns {"clips": [...], "term_corrections": [...],
    "cost_analysis": {...} | absent}."""
    language = str(transcript_result.get("language") or "unknown")
    # Long-form sources are forced through the chunked path even when they'd
    # comfortably fit a single call (see LONG_FORM_SECONDS_THRESHOLD) — a
    # single JSON response reliably undercounted long videos ("usually 2-5"
    # regardless of actual runtime); splitting makes the count floor
    # mechanical (each chunk carries its own minimum) instead of hoping one
    # call enumerates 30+ well-reasoned candidates at once.
    single_pass_fits = _estimate_tokens(segments_json) <= _MAX_SINGLE_PASS_ESTIMATED_TOKENS
    if single_pass_fits and video_duration <= LONG_FORM_SECONDS_THRESHOLD:
        long_directive = _long_context_directive(
            long_context_count, max_duration_ceiling)
        style_directive = _STYLE_DIRECTIVES.get(style_variant, "")
        prompt = NARRATIVE_PROMPT_TEMPLATE.format(
            video_duration=video_duration, language=language,
            max_duration_ceiling=max_duration_ceiling,
            candidate_count_directive=_candidate_count_directive(
                video_duration, override=clip_count),
            long_context_directive=long_directive,
            style_directive=style_directive,
            context_note="", segments_json=segments_json)
        parsed, cost = _run_deepseek_stage(api_key, model_name, prompt, NarrativeResponse,
                                           base_url=base_url)
        clips = parsed.get("clips") or []
        _normalize_clip_types(clips)
        term_corrections = parsed.get("term_corrections") or []
        costs = [cost] if cost else []
    else:
        # Wide windows (build_transcript_windows, unchanged), carrying a
        # short "what's still open" note forward instead of duplicating raw
        # transcript text the way the old 30s window overlap did.
        clips, term_corrections, costs = _chunked_narrative_selection(
            transcript_result, video_duration, api_key, model_name,
            language, max_duration_ceiling, base_url=base_url,
            clip_count=clip_count, long_context_count=long_context_count,
            style_variant=style_variant)

    # Post-hoc hard-target guard: if the model over-delivered beyond an
    # explicit request, keep the strongest N by predicted_score, then restore
    # narrative (start-time) order. Fewer than N is fine — that means the
    # source genuinely couldn't support it.
    if clip_count is not None and len(clips) > clip_count:
        strongest = sorted(
            clips, key=lambda c: int(c.get("predicted_score") or 0), reverse=True)
        clips = sorted(strongest[:clip_count], key=lambda c: c.get("start", 0))

    if not clips:
        print(f"⚠️ {label} returned no clips.")
        return None

    if term_corrections:
        print(f"📝 {label} term corrections: " +
              ", ".join(f"{c.get('wrong')!r}->{c.get('correct')!r}" for c in term_corrections))

    cost_analysis = None
    if costs:
        cost_analysis = {
            "input_tokens": sum(c.get("input_tokens", 0) for c in costs),
            "output_tokens": sum(c.get("output_tokens", 0) for c in costs),
            "total_cost": sum(c.get("total_cost", 0) for c in costs),
            "model": model_name,
        }
        print(f"💰 {label} cost ({model_name}, {len(costs)} call(s)): "
              f"${cost_analysis['total_cost']:.6f}")

    result = {"clips": clips, "term_corrections": term_corrections}
    if cost_analysis:
        result["cost_analysis"] = cost_analysis
    return result


def deepseek_select_narrative_clips(transcript_result, video_duration,
                                    api_key=None, model_name=None,
                                    max_duration_ceiling=180.0, clip_count=None,
                                    long_context_count=0, style_variant="balanced"):
    """Narrative-arc-aware clip selection — replaces Gemini's 2-pass text
    scoring for the audio-present path. Tries each configured candidate in
    order (see _narrative_provider_candidates) and only gives up once every
    one has failed — a DeepSeek key that's still SET but out of tokens
    looks like any other failure, not "not configured", so it doesn't
    short-circuit the Gemini substitute. Returns {"clips": [...],
    "cost_analysis": {...}} or None.
    """
    candidates = _narrative_provider_candidates(api_key, model_name)
    if not candidates:
        return None

    compact = build_compact_segments(transcript_result)
    if not compact:
        return None
    segments_json = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))

    for i, (provider, cand_key, cand_model, base_url) in enumerate(candidates):
        label = "DeepSeek" if provider == "deepseek" else "Gemini (narrative)"
        print(f"🐋 {label}: model={cand_model} "
              f"segments={len(compact)} est_tokens={_estimate_tokens(segments_json)}")
        try:
            result = _run_narrative_selection_once(
                transcript_result, video_duration, max_duration_ceiling,
                provider, cand_key, cand_model, base_url, label,
                compact, segments_json, clip_count=clip_count,
                long_context_count=long_context_count,
                style_variant=style_variant)
            if result is not None:
                return result
        except Exception as e:
            print(f"❌ {label} error: {e}")
        if i < len(candidates) - 1:
            print("   ↳ falling back to the next configured narrative-selection provider…")
    return None


def _chunked_narrative_selection(transcript_result, video_duration, api_key, model_name,
                                 language, max_duration_ceiling, chunk_seconds=300,
                                 base_url=DEEPSEEK_API_BASE, clip_count=None,
                                 long_context_count=0, style_variant="balanced"):
    """Split path for transcripts too long for a single call, AND (see
    LONG_FORM_SECONDS_THRESHOLD) any long-form video regardless of size —
    a single JSON response reliably undercounted long videos, so the count
    floor is made mechanical here: each chunk carries its own minimum
    (overall target divided across however many chunks actually exist),
    rather than hoping one call enumerates dozens of candidates at once.

    5-minute windows (no overlap needed since we carry context forward
    explicitly) via the existing build_transcript_windows helper, with a
    short carry-over summary passed between sequential calls instead of
    duplicating raw transcript text the way the old windowed Gemini scoring
    pass's 30s overlap did — that was pure token waste.
    """
    windows = build_transcript_windows(
        transcript_result, video_duration, window_seconds=chunk_seconds, overlap_seconds=0)
    overall_target = _target_clip_count(video_duration, override=clip_count)
    chunk_target = max(1, math.ceil(overall_target / max(1, len(windows))))
    all_clips = []
    all_term_corrections = []
    costs = []
    carry_over = ""
    for i, w in enumerate(windows):
        chunk_segments = [
            _compact_segment(seg) for seg in transcript_result.get("segments", [])
            if seg.get("start", 0) >= w["start"] and seg.get("end", 0) <= w["end"]
            and str(seg.get("text") or "").strip()
        ]
        chunk_segments = _merge_short_segments(chunk_segments)
        if not chunk_segments:
            continue
        context_note = (
            f"CONTEXT FROM EARLIER IN THE VIDEO (a thread that may still need "
            f"its payoff in this chunk): {carry_over}\n" if carry_over else "")
        prompt = NARRATIVE_PROMPT_TEMPLATE.format(
            video_duration=video_duration, language=language,
            max_duration_ceiling=max_duration_ceiling, context_note=context_note,
            candidate_count_directive=_candidate_count_directive(
                video_duration, chunk_target=chunk_target, chunk_index=i,
                chunk_count=len(windows), override=clip_count),
            long_context_directive=_long_context_directive(
                long_context_count, max_duration_ceiling),
            style_directive=_STYLE_DIRECTIVES.get(style_variant, ""),
            segments_json=json.dumps(chunk_segments, ensure_ascii=False, separators=(",", ":")))
        parsed, cost = _run_deepseek_stage(api_key, model_name, prompt, NarrativeResponse,
                                           base_url=base_url)
        chunk_clips = parsed.get("clips") or []
        _normalize_clip_types(chunk_clips)
        all_clips.extend(chunk_clips)
        all_term_corrections.extend(parsed.get("term_corrections") or [])
        if cost:
            costs.append(cost)
        # Carry forward only a short signal, not the transcript itself.
        if chunk_clips:
            carry_over = (chunk_clips[-1].get("narrative_summary") or "")[:200]
    return all_clips, all_term_corrections, costs
