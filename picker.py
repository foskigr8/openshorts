"""Stage 3 unified picker — the planner.

One Gemini planning pass reads the WHOLE transcript (1M context — no
slicing, no windows) plus the pre-loaded context blob and returns the EXACT
requested number of distinct viral moments. The requested count is a hard
promise: if a pass comes back short, the picker asks again ("keep looking")
with the already-picked spans listed as forbidden until the count is met —
the puzzle mechanic, bounded so it can never spin forever. The only thing
that can stop it is physics: N clips × the 15s minimum longer than the
video itself, and that shortfall is surfaced loudly in the result, never
accepted silently.

No separate engines, no vision confirmation, no niche playbooks. The picker
is general-purpose by design: find the moment people would love — the gem —
whatever the video is about.
"""
import json
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from google.genai import types as genai_types
from pydantic import BaseModel

import gemini_pool
import gemini_worker

load_dotenv()

_DEFAULT_MODEL = "gemini-3.1-flash-lite"

MIN_SHORT_SECONDS = 15.0
MAX_SHORT_SECONDS = 120.0
MIN_LONG_SECONDS = 45.0
MAX_LONG_SECONDS = 240.0
MAX_PICKER_PASSES = 3
_MIN_SEGMENT_CHARS_BEFORE_MERGE = 12

# Transient Gemini failures worth retrying inside one picker call (same list
# the old 2-pass stage used). Policy blocks are handled separately and never
# retried — they are deterministic.
_TRANSIENT_TOKENS = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
    "500", "INTERNAL", "overloaded", "Deadline",
    "empty response body", "did not contain a JSON object",
    "Failed to parse Gemini JSON response",
)


class KeepSpan(BaseModel):
    start: float
    end: float


class PickerClipModel(BaseModel):
    start: float
    end: float
    predicted_score: int
    clip_type: str = "short"  # short | long_context
    hook_type: str
    narrative_summary: str      # what opens, what resolves it
    essential_span_note: str    # what's essential vs padding in this span
    keep_spans: List[KeepSpan] = []  # jump-cut plan (script-level trim)
    video_description_for_tiktok: str
    video_description_for_instagram: str
    video_title_for_youtube_short: str
    viral_hook_text: str


class TermCorrection(BaseModel):
    wrong: str
    correct: str


class PickerResponse(BaseModel):
    clips: List[PickerClipModel]
    term_corrections: List[TermCorrection] = []


PICKER_PROMPT_TEMPLATE = """
You are a senior short-form video strategist AND editor working directly on
the full transcript in one pass — there is no artificial window boundary,
so you can and must trace a story across the whole video.

{context_block}
{target_directive}
{forbidden_directive}
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
  spike — because ad copy is written to be engaging. Identify them by
  SUBJECT MATTER, not tone: is this segment about getting the viewer to
  click a link, download something, use a promo/referral code, or otherwise
  transact with a third party — as opposed to being part of the video's own
  story, argument, or entertainment? If yes, EXCLUDE it entirely, even if
  it is the most "quotable" moment in the transcript.

ANTI-DEFAULT RULE (read this FIRST, it is the most common mistake):
- Every video — including a single continuous sketch, rant, or monologue —
  has an internal structure of smaller beats: a setup, a specific turn, a
  specific payoff, then a comedown or a new topic. "The whole video is one
  inseparable narrative" is almost never actually true; it is what it looks
  like when you have not yet found the ONE beat inside it that stands alone.
- If your first instinct is a candidate near the 120-second ceiling (240s
  for long_context), that is a strong signal you defaulted instead of
  finding the real sub-arc. Go back through SEGMENTS_JSON and identify:
  where does a NEW specific tension/question first get raised (not the
  video's overall topic — a specific line, turn, or beat), and where does
  THAT SPECIFIC thing resolve? That span is almost always much shorter.
- If the transcript is a single comedic bit or sketch, the payoff is the
  punchline or the specific twist — not the sketch's ending line, and not
  every setup that led there. Cut in close to the turn, hold only as much
  setup as the punchline needs to land for a cold viewer.

SEARCH METHOD — work backward from the peak, not forward from the start:
- Don't scan the transcript in order looking for a good opening. Instead,
  first find the PEAK: the segment(s) with `hl: true` (AssemblyAI flagged
  it as a standout moment) or a sharp `sent` swing (e.g. neutral/positive
  segments suddenly turning "-", or vice versa) — your best proxy for a real
  spike (a shock, a comeback, a walkout, a sudden reaction). Cross-check
  against the context blob's HIGHLIGHTS and LOVABLE_MOMENTS.
- Treat that peak as the PAYOFF (or very close to it) and work BACKWARD
  through SEGMENTS_JSON to find the specific line or moment that triggered
  it — that becomes your `start`. This finds real dramatic arcs far more
  reliably than scanning forward and hoping a strong opening happens to
  lead somewhere.
- Prefer ending the clip right AT the payoff moment (the reaction, the
  comeback, the walkout) rather than lingering into whatever comes after —
  a clip that ends on its climax reads tighter and loops better.

EDITORIAL PRIORITY, NOT A VOLUME SCORE:
- `predicted_score` is diagnostic metadata only; it must not be a ranking
  based on yelling, profanity, or raw audio intensity. Return clips in your
  actual editorial priority order, strongest first.
- Look for concrete TRIGGER BLOCKS in the transcript: absolute standards or
  refusals ("never", "absolutely not"), social-value disputes, a specific
  criticism, an exposed contradiction, a firm boundary, a genuinely
  surprising reveal. A quiet exchange containing one of these can outrank a
  loud but empty argument.
- A trigger block alone is not enough. It must lead to a visible or audible
  turn — pushback, disbelief, laughter, a pop, a comeback, a walkout, or a
  meaningful reaction. Prefer the smallest self-contained span that carries
  both the trigger and that turn.
- Do not select a clip merely because a keyword occurs. Read the surrounding
  exchange and choose it only when the human tension is real and legible.

NARRATIVE-ARC RULE:
- A clip is a complete narrative unit: it opens a hook, question, or
  tension, and it MUST include that same thread's payoff/resolution before
  it ends.
- Do not end a clip just because the topic drifts in the transcript — follow
  the SAME thread to its actual resolution.
- DURATION IS A REAL, ENFORCED RANGE: a short clip must be between 15 and
  120 seconds (prefer 15-90); a long_context clip between 45 and 240
  seconds. If the strongest version of a moment runs over the ceiling, you
  have not found the tight version yet: apply the TIGHTNESS RULE harder, or
  pick a narrower beat within it, rather than returning something over the
  limit.
- Once the payoff has landed, STOP. Do not keep going because the transcript
  keeps talking — a clip that runs past its own payoff reads as padded.

TIGHTNESS RULE — real edited shorts cut dead air via internal jump cuts,
not just trimmed start/end boundaries (this is the single most consistent
thing that separates a real edit from a raw excerpt):
- Once you've found a hook-to-payoff span, go through SEGMENTS_JSON within
  [start, end] and classify each segment as ESSENTIAL (the hook itself, real
  turning points, the payoff, anything the story can't be understood
  without) or CUTTABLE (redundant restating, hedging, filler, a tangent
  that doesn't serve this specific arc, dead air between beats).
- Return the ESSENTIAL segments as `keep_spans`: a list of
  {{"start": <number>, "end": <number>}} sub-ranges, in chronological order,
  each a contiguous run of essential segments. This is what actually gets
  kept — everything inside [start, end] NOT covered by a keep_span is cut
  as a jump cut, the same way a real editor removes a pause or a tangent
  mid-clip, not just off the ends.
- Every keep_span must be at least 1.5 seconds — don't fragment into choppy
  micro-cuts. Merge adjacent essential segments into one keep_span.
- If the ENTIRE [start, end] span is essential with nothing to cut, return a
  single keep_span covering the whole thing — do not invent cuts.
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
    "I don't really have deal breakers, because I'm from the ghetto."
  - A bare fragment, CANNOT open cold: "I'm not." "A 9? That's high." —
    these are replies, half of an exchange, and mean nothing without the
    other half.
- Before finalizing `start`, identify what KIND of line it is. If it's a
  self-contained claim, leave it — do not pad it with unneeded setup. If
  it's a bare reply fragment, move `start` EARLIER to the line it's
  replying to (or find a different beat entirely whose opening IS a real
  claim).
- Never open on a bland self-introduction ("Hi, I'm...") with nothing else
  happening yet — that's dead air, not a hook.
- THE PRONOUN TRAP (explicit, mechanical, the most common way a
  grammatically complete opening still fails cold): an opening that leads
  with a bare pronoun — "He", "She", "It", "This", "That", "They", or a
  "We"/"You" that names nobody the viewer knows — is a reply-style opening
  even when the sentence is complete. "He just walked in and said it to her
  face" is a complete sentence and STILL a cold-viewer failure: the viewer
  has no idea who "he" is or what "it" means. Before accepting any opening
  that leads with a pronoun, rewind `start` EARLIER to the line that
  establishes the referent, keeping that addition as short as possible — and
  if no antecedent exists close enough, DROP the candidate rather than ship
  an opening a cold viewer cannot parse. Re-check the final opening through
  this lens before committing `start`.

THE 2-SECOND TEST (opening only): the first 2 seconds must stop a cold
viewer — no prior context — from scrolling past.

DIVERSITY: never return two clips that make the same point, tell the same
story, or land the same joke. Keep the stronger, drop the other. Overlapping
spans are duplicates — each clip must cover a DIFFERENT moment.

SPEAKER LABELS ("sp"): when present, each segment carries WHO said it
(diarized). Absent means a single speaker. Use it as structure, never as a
topic:
- A fast alternation between two speakers (short turns, trading lines,
  interrupting, a question answered instantly with a sharp comeback) is one
  of the strongest short-form signals there is. Prefer a candidate built on
  a real EXCHANGE over an equally interesting monologue.
- A long stretch with no speaker change is a monologue. It can still be
  excellent, but it must earn it on the strength of the claim.
- For a long_context clip, the participants should stay CONSISTENT across
  the segment — a span whose speaker set changes completely partway through
  is usually two unrelated scenes glued together.
- Never cut in the middle of a turn where the reply is the payoff. If a
  speaker's line lands only because of what the next speaker says back, the
  clip has to include the reply.

COPY RULES — ALL text fields (descriptions, title, hook) MUST be written in
TRANSCRIPT_LANGUAGE ({language}):
- Descriptions (TikTok + Instagram): 1-2 punchy sentences that tease the
  payoff without spoiling it, then 3-5 topically relevant hashtags. No
  generic hashtag spam.
- `video_title_for_youtube_short`: max 100 chars, curiosity-driven, no fake
  claims.
- `predicted_score`: honest 0-100 estimate of viral potential.
- `hook_type`: the pattern you used (e.g. "open question", "hot take").
- `narrative_summary`: one line — what SPECIFIC question/tension opens this
  clip, and what resolves it by the end. If what you'd write here is really
  just "the video's overall topic," you have defaulted — find the smaller
  beat. If you can't state the resolution in one line, the clip probably
  doesn't resolve — reconsider it.
- `essential_span_note`: one line on what part of [start, end] is essential
  vs padding.
- `viral_hook_text` (max 10 words): concise, emotionally legible, specific
  — ideally a sharp question, standard, or disputed claim the speaker's own
  words supply. Do not use a generic summary when the real conflict
  supplies a stronger angle.

TRANSCRIPTION CORRECTIONS: ASR frequently mishears proper nouns, brand
names, or other distinctive repeated terms as an ordinary similar-sounding
word. Scan the full SEGMENTS_JSON for words that are almost certainly ASR
errors — the word reads oddly out of place for its sentence, AND a specific,
similar-sounding proper noun/brand term that is clearly this video's actual
subject would fit far better. List each as
{{"wrong": "<exact mistranscribed text as it literally appears>",
"correct": "<the corrected term>"}}. Be conservative: only include
corrections you are genuinely confident about — do not "fix" slang, filler
words, or ordinary ambiguous word choices. Return an empty list if nothing
needs correcting. These corrections are applied to burned-in captions, so
precision matters more than recall.

INPUT FORMAT: SEGMENTS_JSON is a compact array of this video's transcript
segments in chronological order, absolute seconds from the start:
  {{"s": <start>, "e": <end>, "t": "<segment text>", "sent": "+"|"0"|"-",
    "hl": true|false, "sp": "<speaker label, when present>"}}
`sent` is a per-segment sentiment hint ("+"=positive, "0"=neutral,
"-"=negative) — a lightweight signal for where emotional beats or turning
points likely are, on top of reading the text itself. `hl` is true when
AssemblyAI's own algorithmic key-phrase detection flagged that segment as a
standout moment — an independent second signal; segments with `hl: true`
are good candidates for where the actual hook or payoff lives, but always
confirm against the text itself rather than trusting the flag blindly.

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
      "hook_type": "<hook pattern used>",
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


def _model_name():
    return os.environ.get("GEMINI_MODEL") or _DEFAULT_MODEL


def _compact_segment(segment):
    text = str(segment.get("text") or "").strip()
    sentiment = segment.get("sentiment")  # only present on AssemblyAI transcripts
    out = {
        "s": round(float(segment.get("start", 0)), 1),
        "e": round(float(segment.get("end", 0)), 1),
        "t": text,
        "sent": sentiment if sentiment in ("+", "0", "-") else "0",
        "hl": bool(segment.get("highlight")),
    }
    speaker = segment.get("speaker")
    if speaker:
        out["sp"] = str(speaker)
    return out


def _merge_short_segments(compact_segments):
    """Merge consecutive very-short segments into their neighbor — fewer JSON
    objects for the same content. Never merges across a speaker change."""
    merged = []
    for seg in compact_segments:
        same_speaker = (not merged) or merged[-1].get("sp") == seg.get("sp")
        if (merged and same_speaker
                and len(seg["t"]) < _MIN_SEGMENT_CHARS_BEFORE_MERGE):
            prev = merged[-1]
            prev["e"] = seg["e"]
            prev["t"] = (prev["t"] + " " + seg["t"]).strip()
            if seg.get("sent") != "0":
                prev["sent"] = seg["sent"]
            if seg.get("hl"):
                prev["hl"] = True
        else:
            merged.append(dict(seg))
    return merged


def _context_block(context_blob):
    """The pre-loaded brain, or a short note when it is unavailable."""
    if not context_blob:
        return ("PRE-LOADED CONTEXT: none (Gemini could not read this link "
                "ahead of time) — select purely from the transcript and its "
                "signals.")
    compact = {
        "summary": context_blob.get("summary", ""),
        "highlights": context_blob.get("highlights", []),
        "lovable_moments": context_blob.get("lovable_moments", []),
    }
    return (
        "PRE-LOADED CONTEXT (Gemini watched the video and built this brain "
        "BEFORE the transcript existed — it adds what the audio alone cannot: "
        "visuals, reactions, on-screen text, and an editorial map of where "
        "the moments are. Its timestamps are APPROXIMATE; your transcript "
        "timestamps are the source of truth, so always prefer the "
        "transcript's numbers and refine these approximate spans):\n"
        + json.dumps(compact, ensure_ascii=False)
    )


def _target_directive(target, video_duration):
    physical_max = max(1, int(video_duration // MIN_SHORT_SECONDS))
    if target > physical_max:
        note = (f" NOTE: the video is only {video_duration:.0f}s long, so "
                f"{physical_max} clips of minimum length is the physical "
                f"maximum — return as many distinct clips as it can hold; "
                f"the pipeline reports the gap loudly.")
    else:
        note = ""
    return (
        f"TARGET COUNT — {target} CLIPS (this is a HARD PROMISE, not a "
        f"suggestion):\n"
        f"- The user asked for exactly {target} distinct, non-overlapping "
        f"clips from this video. You MUST return exactly {target} of them.\n"
        f"- There IS a viral moment in this video — treat this like a "
        f"puzzle: keep looking until the candidate fits. Returning fewer "
        f"than {target} is a failure.\n"
        f"- To reach the count, split a strong arc into its component beats, "
        f"or find a different part of a strong exchange, or a smaller "
        f"self-contained moment — but every clip must still independently "
        f"pass every rule above.{note}"
    )


def _forbidden_directive(picked_spans, missing):
    if not picked_spans:
        return (
            f"No clips have been accepted yet. Returning fewer than {missing} "
            f"distinct clips is a FAILURE — every video contains viral "
            f"moments; search the transcript again, look harder, and return "
            f"exactly {missing} of them."
        )
    spans = [{"start": round(s, 2), "end": round(e, 2)} for s, e in picked_spans]
    return (
        f"ALREADY-PICKED SPANS — DO NOT REPEAT. An earlier pass already "
        f"found these moments; they are FORBIDDEN, and anything overlapping "
        f"them is a duplicate:\n{json.dumps(spans, ensure_ascii=False)}\n"
        f"You still owe exactly {missing} more distinct clip(s). Return "
        f"exactly {missing} clips from parts of the video NOT covered by "
        f"the spans above — keep looking until you find them."
    )


def _long_context_directive(long_context_count):
    if not long_context_count or long_context_count <= 0:
        return ""
    return (
        f"LONG-CONTEXT CLIPS (the user explicitly asked for up to "
        f"{long_context_count} of these): alongside the tight shorts above, "
        f"also find a FULL-ARC category: 1-4 minute grounded segments that "
        f"keep an ENTIRE narrative arc — setup, tension, resolution (e.g. a "
        f"contestant's complete run through a challenge, dead air trimmed "
        f"but the story intact from entry to outcome). This is a DIFFERENT "
        f"framing from the tight-hook shorts: a long_context clip may run "
        f"up to {MAX_LONG_SECONDS:.0f}s and does NOT need a punchy first-"
        f"2-seconds hook — its job is a complete, watchable arc with a real "
        f"endpoint. Mark those clips \"clip_type\": \"long_context\" and "
        f"apply the TIGHTNESS rules to dead-air removal only, never to "
        f"compressing the arc itself. Include at most {long_context_count} "
        f"long_context clip(s), and only when the source actually contains "
        f"a full arc worth telling — never pad a short beat out to fake "
        f"one. Every other clip keeps \"clip_type\": \"short\"."
    )


def _call_gemini(api_key, prompt):
    """One structured picker call. Returns (parsed_dict, cost_analysis).

    Rotates across every configured Gemini key (GEMINI_API_KEY + the
    GEMINI_API_KEYS pool) on transient failures — the picker used to hammer
    the single primary key and hit 429 RESOURCE_EXHAUSTED while the extra
    keys sat unused (confirmed in the Ep_115 run: 4 extra keys configured,
    context layer still got quota-exhausted on the primary).
    """
    keys = _api_keys(api_key)
    last_exc = None
    for i, key in enumerate(keys):
        try:
            return _call_with_key(key, prompt)
        except gemini_worker.GeminiBlockedError:
            raise  # deterministic policy block — never retry, surface the reason
        except Exception as e:
            last_exc = e
            if not any(tok in str(e) for tok in _TRANSIENT_TOKENS):
                raise
            if i < len(keys) - 1:
                print(f"⚠️ Gemini key {i + 1}/{len(keys)} hit a transient "
                      f"error ({str(e)[:120]}) — rotating to the next key")
    raise last_exc


def _api_keys(primary):
    """Primary key + every GEMINI_API_KEYS extra, deduplicated, in order."""
    keys = []
    if primary:
        keys.append(primary)
    for k in (os.environ.get("GEMINI_API_KEYS") or "").split(","):
        k = k.strip()
        if k and k not in keys:
            keys.append(k)
    return keys or [primary]


def _call_with_key(api_key, prompt):
    """One structured call against one key (per-key 3-attempt backoff)."""
    client = gemini_worker.make_client(api_key)
    model_name = _model_name()
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PickerResponse,
        safety_settings=gemini_worker.RELAXED_SAFETY_SETTINGS,
    )
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = gemini_pool.generate_with_fallback(
                client, model_name, prompt, config=config, max_attempts=1,
                log=lambda msg: print(msg))
            gemini_worker.raise_if_blocked(response)
            parsed_obj = getattr(response, "parsed", None)
            if parsed_obj is not None:
                parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
            else:
                raw_text = gemini_worker._get_response_text(response)
                parsed = gemini_worker._parse_json_response_text(raw_text)
            return parsed, gemini_worker._calculate_cost_analysis(response, model_name)
        except gemini_worker.GeminiBlockedError:
            raise  # deterministic policy block — never retry, surface the reason
        except Exception as e:
            msg = str(e)
            if attempt == max_attempts or not any(tok in msg for tok in _TRANSIENT_TOKENS):
                raise
            wait = 5 * (2 ** (attempt - 1))
            print(f"⚠️ Gemini transient error (attempt {attempt}/{max_attempts}), "
                  f"retrying in {wait}s: {msg[:150]}")
            time.sleep(wait)


def _clean_clips(parsed_clips, video_duration):
    """Clamp to real duration, drop degenerate spans, normalize clip_type."""
    clean = []
    for c in parsed_clips or []:
        try:
            s = max(0.0, float(c.get("start", 0)))
            e = min(float(video_duration), float(c.get("end", 0)))
        except (TypeError, ValueError):
            continue
        if e - s < 1.0:
            continue
        clip = dict(c)
        clip["start"], clip["end"] = s, e
        clip["clip_type"] = (clip.get("clip_type")
                             if clip.get("clip_type") in ("short", "long_context")
                             else "short")
        clean.append(clip)
    return clean


def _dedup_overlaps(clips):
    """Drop lower-scored clips when two picks overlap in time (same policy as
    main.py's tail dedup, applied at pick time so the keep-looking loop counts
    what will actually ship)."""
    if len(clips) < 2:
        return clips
    ranked = sorted(enumerate(clips), key=lambda t: (t[1].get("start", 0), t[0]))
    kept = []
    for idx, clip in ranked:
        prev = kept[-1][1] if kept else None
        if prev is not None and clip["start"] < prev["end"]:
            cur_score = clip.get("predicted_score") or 0
            prev_score = prev.get("predicted_score") or 0
            if cur_score > prev_score:
                kept[-1] = (idx, clip)
            continue
        kept.append((idx, clip))
    return [c for _, c in kept]


def _aggregate_cost(costs, model_name):
    if not costs:
        return None
    return {
        "input_tokens": sum(c.get("input_tokens", 0) for c in costs),
        "output_tokens": sum(c.get("output_tokens", 0) for c in costs),
        "total_cost": sum(c.get("total_cost", 0) for c in costs),
        "model": model_name,
    }


def select_viral_clips(transcript_result, video_duration, clip_count=None,
                       long_context_count=0, style_variant="balanced",
                       context_blob=None):
    """Find exactly the requested number of viral moments.

    ``clip_count`` is REQUIRED — there is no auto mode. The count is a hard
    promise: up to ``MAX_PICKER_PASSES`` calls, each forbidden-span pass
    asking for exactly what is still missing, until the total is met or the
    video physically cannot hold it (surfaced in the result as ``shortfall``,
    never silent).
    """
    if clip_count is None:
        raise ValueError("picker.select_viral_clips requires an explicit "
                         "clip_count — auto mode was removed")
    target = int(clip_count) + int(long_context_count or 0)
    target = max(1, target)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment variables.")

    compact = [_compact_segment(s)
               for s in transcript_result.get("segments", [])
               if str(s.get("text") or "").strip()]
    compact = _merge_short_segments(compact)
    segments_json = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    language = str(transcript_result.get("language") or "unknown")
    model_name = _model_name()

    physical_max = max(1, int(video_duration // MIN_SHORT_SECONDS))
    if target > physical_max:
        print(f"⚠️ Picker: requested {target} clip(s) but the video is only "
              f"{video_duration:.0f}s — the physical maximum is "
              f"{physical_max}. Returning as many distinct clips as fit; "
              f"the gap is surfaced in the job metadata.")

    picked = []
    term_corrections = []
    costs = []
    passes = 0
    while len(picked) < target and passes < MAX_PICKER_PASSES:
        passes += 1
        missing = target - len(picked)
        prompt = PICKER_PROMPT_TEMPLATE.format(
            context_block=_context_block(context_blob),
            target_directive=_target_directive(missing, video_duration),
            forbidden_directive=_forbidden_directive(
                [(c["start"], c["end"]) for c in picked], missing),
            long_context_directive=_long_context_directive(long_context_count),
            style_directive=_STYLE_DIRECTIVES.get(style_variant, ""),
            language=language,
            video_duration=f"{video_duration:.0f}",
            segments_json=segments_json,
        )
        print(f"🧩 Picker pass {passes}/{MAX_PICKER_PASSES} — "
              f"hunting for {missing} more clip(s)...")
        parsed, cost = _call_gemini(api_key, prompt)
        if cost:
            costs.append(cost)
        term_corrections.extend(parsed.get("term_corrections") or [])
        fresh = _clean_clips(parsed.get("clips") or [], video_duration)
        combined = _dedup_overlaps(picked + fresh)
        newly_added = len(combined) - len(picked)
        picked = combined
        if newly_added == 0:
            print("⚠️ Picker pass added no new clips — trying again with the "
                  "picked spans listed as forbidden.")

    shortfall = max(0, target - len(picked))
    if shortfall:
        print(f"⚠️ Picker delivered {len(picked)}/{target} clip(s) — "
              f"short by {shortfall}. Surfaced in metadata, never silent.")
    elif passes > 1:
        print(f"✅ Picker fulfilled the full target ({target} clips) "
              f"in {passes} pass(es).")
    else:
        print(f"✅ Picker returned all {len(picked)} requested clip(s) "
              f"in a single pass.")

    return {
        "shorts": picked,
        "term_corrections": term_corrections,
        "cost_analysis": _aggregate_cost(costs, model_name),
        "requested": target,
        "delivered": len(picked),
        "shortfall": shortfall,
    }
