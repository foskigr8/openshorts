import argparse
import json
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

# Bounded Gemini read timeout (6-aug-2026). The SDK default (~5 min) let a
# hung provider stall a whole job for 3m19s in the complaint run (ReadTimeout
# at 02:50:54, call started 02:47:35) — longer than any other stage. Capping
# it means one bad provider call can no longer be the job's longest stage;
# the provider ladder / fallback handles the failure. Env-overridable.
GEMINI_TIMEOUT_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "120000"))


def make_client(api_key):
    """genai.Client with a bounded read timeout (see GEMINI_TIMEOUT_MS)."""
    timeout_ms = int(os.environ.get("GEMINI_TIMEOUT_MS", GEMINI_TIMEOUT_MS))
    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(timeout=timeout_ms))
from pydantic import BaseModel

from clip_selection import lookup_model_prices
import gemini_pool

load_dotenv()


# --- Structured output schemas (passed as response_schema so the API
# --- guarantees the format instead of us repairing free-form JSON). ---

class ScoredWindowModel(BaseModel):
    id: str
    start: float
    end: float
    score: int
    reason: str


class ScoreResponse(BaseModel):
    windows: List[ScoredWindowModel]


class DetailClipModel(BaseModel):
    start: float
    end: float
    source_window_id: str
    predicted_score: int
    video_description_for_tiktok: str
    video_description_for_instagram: str
    video_title_for_youtube_short: str
    viral_hook_text: str


class DetailResponse(BaseModel):
    shorts: List[DetailClipModel]


# Visual (no-transcript) clip selection: Gemini watches a silent video and
# picks moments from the imagery. Same output shape as DetailClipModel minus
# the transcript-only source_window_id.
class VisualClipModel(BaseModel):
    start: float
    end: float
    predicted_score: int
    video_description_for_tiktok: str
    video_description_for_instagram: str
    video_title_for_youtube_short: str
    viral_hook_text: str


class VisualResponse(BaseModel):
    shorts: List[VisualClipModel]


# This pipeline's actual source material is mainstream broadcast-style
# reality/dating-show content (revealing outfits, blunt language, romantic/
# sexual innuendo) — legal, TV-safe content that Gemini's DEFAULT safety
# thresholds still block outright (confirmed in prod: a vision-confirmation
# call on real dating-show footage came back PROHIBITED_CONTENT, 31-jul-2026,
# silently dropping that candidate's context check). BLOCK_ONLY_HIGH keeps
# Google's non-configurable policy floor (CSAM etc. are never permitted
# regardless of any setting) while stopping the default MEDIUM threshold
# from false-positiving on content that's genuinely fine to analyze. Applied
# to every Gemini call in this pipeline that touches this content, not just
# the one that got caught — the same genre risk exists everywhere.
RELAXED_SAFETY_SETTINGS = [
    genai_types.SafetySetting(category=cat, threshold=genai_types.HarmBlockThreshold.BLOCK_ONLY_HIGH)
    for cat in (
        genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]


# Vision-confirmation of a DeepSeek-selected candidate clip: Gemini watches a
# short rough-cut (not the whole source video) and either approves it or
# suggests a small boundary nudge. Split into two specialized calls per
# candidate (run in parallel, across a key pool — see gemini_pool.py) rather
# than one do-everything call, per the user's own request: one focused on
# what the footage/audio itself look/sound like, one focused on cross-
# checking the actual transcript text against what's shown.
class VisionVisualCheckResponse(BaseModel):
    approved: bool
    suggested_start_delta: float  # seconds; negative = start earlier, positive = start later
    reason: str


class VisionContextCheckResponse(BaseModel):
    approved: bool
    narrative_resolved: bool
    has_real_hook: bool         # does the OPENING work for a viewer with zero prior context?
    suggested_start_delta: float  # seconds; only ever used to move the start EARLIER (<=0)
    suggested_end_delta: float  # seconds; only ever used to EXTEND, never shrink
    reason: str
    # 6-aug-2026 (PART 3.3): boundary-bleed checks. All optional with defaults
    # so older model outputs (and the long-context template, which doesn't ask
    # for them) still parse — a null here must never throw the response away
    # the way reaction_cam:null did.
    different_speaker_open: Optional[bool] = None  # another face owns frame 0
    speaker_on_screen_at: Optional[float] = None   # sec into clip the intended speaker first appears
    end_scene_bleed: Optional[bool] = None         # next scene's person already on screen at the end
    end_bleed_pullback: Optional[float] = None     # seconds to trim from the END (>=0)


# --- scene context / focus direction (third verification layer) -------------
# Runs AFTER selection + the two vision checks, on the already-streamlined
# clip (post jump-cut), so timestamps map 1:1 onto the final render. The two
# checks above judge WHETHER to keep a clip; this one decides WHO the camera
# should be on, moment to moment, and is the only stage in the pipeline that
# understands *why* something is happening rather than just detecting faces
# and motion.
class FocusDirective(BaseModel):
    start: float          # clip-relative seconds
    end: float
    subject: str          # short visual description, e.g. "man in cream tee, arms up"
    x_position: float     # subject's horizontal centre, 0.0 = left edge .. 1.0 = right edge
    reason: str           # speaking | causing_reaction | referenced | reacting
    intensity: float      # 0..1 — how strongly this moment wants a tight shot


class SceneContextResponse(BaseModel):
    primary_subject: str
    # Horizontal centre (0..1) of the clip's ONE key subject — the person the
    # whole clip is really about (e.g. the contestant). The reframe engine
    # uses this to keep the return-bias and split-screen cells anchored on
    # the right person even when a later directive's subject label is vague.
    primary_subject_x: Optional[float] = None
    summary: str
    directives: List[FocusDirective]


VISION_SCENE_CONTEXT_PROMPT_TEMPLATE = """
You are the DIRECTOR of a vertical short. The clip below is already cut and
locked — you are NOT judging whether to keep it. Your only job is to decide
WHO THE CAMERA SHOULD BE ON at each moment, and to explain what is actually
happening between the people on screen.

A downstream program will crop a tall 3:4 window out of this wide footage and
move that window to follow your directives. It can only see faces and motion;
it has no idea what anything MEANS. You are the only stage that does.

Transcript for this clip (times are clip-relative seconds):
{transcript_excerpt}

THE RULE THAT MATTERS MOST — SHOW THE CAUSE, NOT THE REACTION:
When people laugh, gasp, point, recoil, or crack up, the interesting shot is
almost never the people reacting — it is WHOEVER OR WHATEVER CAUSED IT. If
someone is clowning, mugging, grunting, posing, acting out a role, or doing
something outrageous, and others react to it, the camera belongs ON THAT
PERSON for as long as they are driving the moment (`reason:
"causing_reaction"`). A clip that shows a room laughing without ever showing
what they are laughing at has failed. Reaction shots are punctuation — brief,
and only after the cause has been seen.

Likewise, when someone points at, gestures toward, describes, or calls out
another person, cut to the PERSON BEING REFERRED TO (`reason: "referenced"`),
not the person doing the pointing. The reference is the setup; the person
referred to is the payoff.

THE KEY SUBJECT — WHO THIS CLIP IS REALLY ABOUT:
Most of these clips are built around ONE person: the contestant, the star,
the one being interviewed, the one doing the bit. Before anything else,
identify that person and keep the camera on them by default. Transcript
speaker labels (the `[12.3s A]` / `[12.3s B]` prefixes) are real audio
diarization — match the talking voice to the actual face/position in the
video, and when the KEY SUBJECT is the one talking, mark that directive
`reason: "speaking"` and frame THEM, not whoever else happens to be audible
or most centred. A talking head that is NOT the key subject is usually a
host setting up the bit — brief, then back to the key subject.

GESTURES AND PERFORMING ARE THE ACTION:
When a person is making gestures with their hands, pulling faces, posing,
acting out, or performing WHILE someone else is talking, the camera belongs
on the performer — they are the thing happening, the talker is just audio.
Mark these `reason: "causing_reaction"` when others react, and keep the
camera on the performer for as long as they drive the moment.

Otherwise, default to whoever is speaking (`reason: "speaking"`).
Use `reason: "reacting"` only for a short, deliberate cutaway to a strong
individual reaction that is worth seeing in its own right.

OUTPUT:
Return an ordered, NON-OVERLAPPING list of directives covering the clip from
0 to about {clip_duration:.1f} seconds. Aim for roughly {target_count} of
them — a real edit changes shot every 1.5-3 seconds, so do not return one
directive for the whole clip, and do not return dozens of sub-second ones.

For each directive:
- `start` / `end`: clip-relative seconds. Contiguous, in order, no overlaps.
- `subject`: a SHORT visual description that identifies the person by how
  they LOOK (clothing colour, headwear, position in the line), not by name —
  the program matches on position, and your description is for logging.
- `x_position`: that person's horizontal CENTRE in the frame you can see,
  as a fraction: 0.0 = far left edge, 0.5 = dead centre, 1.0 = far right
  edge. Be as precise as you can — this is the single most important number
  you return, because it is how the camera actually finds them. If the
  subject moves during the directive, give their average position.
- `reason`: one of speaking | causing_reaction | referenced | reacting.
- `intensity`: 0..1. High (0.7-1.0) for a punchy single-person beat that
  deserves a tight push-in; low (0.0-0.3) when several people matter at once
  and the shot should stay wide.

Also return `primary_subject` (the one person this clip is really about),
`primary_subject_x` (their horizontal CENTRE as a fraction, 0..1 — the same
number you would put in an x_position for them), and a one-line `summary` of
what actually happens. The key subject's x_position must stay CONSISTENT
across every directive that frames them.

Return only valid JSON, no markdown fences, no commentary:
{{"primary_subject": "<short>", "primary_subject_x": <0..1>,
  "summary": "<one line>",
  "directives": [{{"start": <number>, "end": <number>, "subject": "<short>",
                  "x_position": <number>, "reason": "<one of the four>",
                  "intensity": <number>}}]}}
"""

VISION_VISUAL_CHECK_PROMPT_TEMPLATE = """
You are reviewing a ROUGH CUT of a short-form video candidate before final
export. This clip was chosen by a separate narrative-analysis pass; you are
NOT re-deciding whether the story is good — you are checking whether the
FOOTAGE ITSELF is clean.

The proposed final clip begins {candidate_start_offset:.1f} seconds after the
start of the uploaded review video. The preceding footage is deliberate
pre-roll for boundary review. Judge the proposed boundary and the frames
immediately around it — NOT the first frame of the uploaded file.

The candidate's intended narrative (for context only): {narrative_summary}

Check:
1. OPENING FRAME: is the frame at the proposed boundary clean — not
   mid-blink, not a jump-cut artifact, not mid-motion, not an awkward
   transition? If not, suggest a small `suggested_start_delta` in seconds
   (typically -3 to +3) to land the open on a cleaner moment nearby. The clip
   was extracted with a couple of seconds of padding before/after the actual
   proposed boundaries specifically so you have room to suggest a shift.
   For confrontation, dating, panel, or reality footage, look specifically at
   the 0.5–1.0 seconds immediately BEFORE the first meaningful line: an inhale,
   eye-roll, stare, laugh, disbelief, or look up-and-down can be a stronger
   cold open than starting exactly on the first word. If that real reaction
   improves the opening without adding dead air, use a small NEGATIVE start
   delta to retain it.
2. AUDIO/VISUAL MATCH: does the audio track (tone of voice, any background
   sound or music) match what the footage is showing? Flag if they feel
   mismatched or disjointed.
3. General shot quality: is this composed/framed reasonably, or is it
   awkward, blurry, or off-putting in a way that would hurt retention?

Return only valid JSON, no markdown fences, no commentary:
{{"approved": <bool>, "suggested_start_delta": <number>, "reason": "<short>"}}
"""

VISION_CONTEXT_CHECK_PROMPT_TEMPLATE = """
You are the COLD-OPEN and narrative-context editor reviewing a short-form
video candidate before final export. You have been given a CONTEXT REVIEW
cut: it deliberately includes footage before and after the proposed clip so
you can judge whether the proposed opening gives a new viewer enough setup.

The proposed clip begins {candidate_start_offset:.1f} seconds after the
start of the uploaded context-review video. Do NOT judge the first frame of
the uploaded file as the clip opening; judge the frame/audio at that proposed
boundary and the first 2-3 seconds AFTER it. The footage before that boundary
is evidence you may use to find the minimum setup required for the clip.

The candidate's intended narrative (what should open, what should resolve):
{narrative_summary}

Transcript for the context-review window (the proposed clip starts at
`[PROPOSED CLIP START]`):
{transcript_excerpt}

Check:
1. Does what's shown on screen actually match what's being said in the
   transcript? Flag any clear mismatch (e.g. the speaker describes an action
   or object that the footage doesn't show).
2. NARRATIVE CLOSURE: judging by both the transcript AND what you can see,
   does this clip's story/question/tension actually resolve by the end — not
   just technically stop, but land a payoff? If it does not resolve, but a
   short extension would plausibly capture the resolution, suggest a
   `suggested_end_delta` in seconds (positive only — this can extend the
   clip, never shrink it) up to about 60 seconds. If no plausible extension
   would fix it, set `approved` and `narrative_resolved` to false instead —
   the clip should be dropped, not stretched indefinitely.
3. REAL HOOK / CONTEXT RESCUE: judging ONLY by the first few seconds AFTER
   the proposed boundary, with zero knowledge of anything before this clip
   starts — is the opening line a SELF-CONTAINED CLAIM, standard,
   accusation, or assertion that a cold viewer understands and reacts to on
   its own, or is it a bare REPLY FRAGMENT that only means something next to
   the question or statement it's answering?
   - Self-contained, fine to open cold with NO setup: a confident claim,
     standard, or callout — e.g. "I don't really have deal breakers, because
     I'm from the ghetto." The viewer doesn't need to have heard a question
     first; the line carries its own point.
   - A bare fragment, NOT a working hook even if it's a clean cut and even
     if it's entertaining once you already understand the premise — e.g.
     "I'm not." / "A 9? That's high." / "Get 'em, boy!" These are one half
     of an exchange and mean nothing alone. THIS is the actual failure mode
     to catch — not "lacks a preceding question," but "is a reply with no
     claim of its own."
   - Identify which kind of line the opening actually is before deciding.
     Do not flag a real, self-contained claim just because something was
     technically said before it in the source video — most claims don't
     need their setup, only fragments do.
   If the opening doesn't work standalone, inspect the pre-roll you were given and
   find the LATEST earlier boundary that includes the minimum setup. Set
   `has_real_hook` to false and suggest `suggested_start_delta` — a NEGATIVE
   number of seconds — for how much earlier the final clip should start. Do
   not pull in an entire conversation when one question, challenge, or visual
   reveal is enough. If no reasonable amount of earlier start would fix it
   (the setup needed is too far back / too long), set `has_real_hook` false
   AND leave `suggested_start_delta` at 0 — the clip should be dropped.
   Total clip duration must never be pushed past {max_duration_ceiling}
   seconds regardless of what you suggest.
4. DIFFERENT-SPEAKER OPEN: is a face OTHER than the intended speaker
   dominating the frame AT the proposed clip boundary? If the intended
   speaker is not yet on screen at the boundary but appears shortly after
   (within ~3 seconds), set `different_speaker_open` true and
   `speaker_on_screen_at` to that moment. Do NOT move the start earlier for
   this — the fix is to start AT the moment the intended speaker is on
   screen. If the intended speaker never appears within the first ~5
   seconds, leave `different_speaker_open` false and let check 3 decide.
5. END-SCENE BLEED: does the final ~1 second of the proposed clip show a
   person who belongs to the NEXT scene (a different face clearly beginning
   their own turn) already on screen? If so, set `end_scene_bleed` true and
   `end_bleed_pullback` to the number of seconds to trim from the END so the
   clip stops at a clean moment (never 0 when bleeding). If the ending is
   clean, set both to null/false.

Return only valid JSON, no markdown fences, no commentary:
{{"approved": <bool>, "narrative_resolved": <bool>, "has_real_hook": <bool>,
  "suggested_start_delta": <number>, "suggested_end_delta": <number>,
  "reason": "<short>", "different_speaker_open": <bool|null>,
  "speaker_on_screen_at": <number|null>, "end_scene_bleed": <bool|null>,
  "end_bleed_pullback": <number|null>}}
"""


VISION_LONG_CONTEXT_CHECK_PROMPT_TEMPLATE = """
You are the NARRATIVE editor reviewing a LONG-CONTEXT segment before final
export. This is NOT a tight hook-and-payoff short and must not be judged as
one. Its job is to carry a COMPLETE, self-contained story arc — setup,
tension, resolution — that someone watches the whole way through.

You have been given a CONTEXT REVIEW cut that deliberately includes footage
before and after the proposed segment. The proposed segment begins
{candidate_start_offset:.1f} seconds into the uploaded file. Judge the
proposed boundary, not the first frame of the upload.

The candidate's intended narrative:
{narrative_summary}

Transcript for the review window (segment starts at `[PROPOSED CLIP START]`):
{transcript_excerpt}

Check:
1. AUDIO/VISUAL MATCH: does what's on screen match what's being said? Flag a
   clear mismatch only.

2. ARC COMPLETENESS (the primary test). Does this segment contain a WHOLE
   arc — a clear entry point, something developing, and an outcome that
   actually lands? A long-context segment that stops before the outcome is a
   failure. If a longer end would capture the resolution, set
   `suggested_end_delta` (positive seconds, up to about 90) to reach it.
   Prefer extending to a real endpoint over cutting the story short.

3. CLEAN ENTRY — NOT a punchy hook. This segment is allowed, and expected,
   to open at a natural beginning: a person arriving, a question being posed,
   a round starting, a topic being introduced. Do NOT require a self-contained
   punchy claim in the first two seconds, and do NOT pull the start later to
   manufacture one. The only failure here is an entry that drops the viewer
   mid-thought with no idea what situation they are in. If that happens,
   suggest a NEGATIVE `suggested_start_delta` to begin at the natural start of
   the situation (the entrance, the question, the beginning of the round).
   Set `has_real_hook` true whenever the entry is simply CLEAR — clarity is
   the standard here, not punchiness.

4. INTERNAL COHERENCE: the parts must connect. This is one continuous
   situation, not a montage of unrelated beats. If the segment stitches
   together things that don't belong to the same thread, set `approved` false.

Dead air, filler and irrelevant tangents should be trimmed, but NEVER at the
cost of the arc — do not compress this into a short. Total duration must not
exceed {max_duration_ceiling} seconds.

Return only valid JSON, no markdown fences, no commentary:
{{"approved": <bool>, "narrative_resolved": <bool>, "has_real_hook": <bool>,
  "suggested_start_delta": <number>, "suggested_end_delta": <number>,
  "reason": "<short>"}}
"""


VISUAL_PROMPT_TEMPLATE = """
You are a senior short-form video editor. This video has NO speech/audio — judge
it purely by what you SEE. Watch the whole thing and pick the 3–15 MOST engaging
visual moments for TikTok / Reels / Shorts (action, reveals, transformations,
striking or funny shots, satisfying payoffs, dramatic movement).

TIME CONTRACT — STRICT:
- Timestamps in ABSOLUTE SECONDS from the start (usable with ffmpeg -ss/-to).
- Only numbers with up to 3 decimals (e.g. 0, 12.5, 47.250).
- 0 <= start < end <= {video_duration}.
- Each clip 15 to 60 seconds long. If the whole video is shorter than 15s,
  return one clip spanning the full video.
- Cut on visual scene changes, never mid-motion.

For each clip write catchy copy in {language} (a scroll-stopping hook, a TikTok
and an Instagram description, and a YouTube title ≤100 chars). Order clips best
to worst by how likely they are to stop a viewer scrolling.
"""


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if not stream or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _log(message: str) -> None:
    stream = sys.stdout
    text = str(message)
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        stream.write(safe_text + "\n")
    stream.flush()

SCORE_PROMPT_TEMPLATE = """
You are a senior short-form video strategist.
Select the MOST viral candidate windows from this batch.

Rules:
- Return only valid JSON.
- Choose up to 3 windows from this batch.
- `score` must be an integer from 0 to 100.
- THE 2-SECOND TEST is the main criterion: would the first 2 seconds of this
  moment force a cold viewer (no context) to keep watching? Windows that only
  work with prior context score low.
- Prefer windows with strong hooks, conflict, surprise, outrage, emotion,
  novelty, big numbers, or a clear payoff.
- Ignore weak filler, housekeeping, outros, rambling transitions, and
  low-signal padding unless there is an obvious hook or payoff.

TRANSCRIPT_LANGUAGE: {language}
VIDEO_DURATION_SECONDS: {video_duration}
WINDOWS_JSON:
{windows_json}

Return only:
{{
  "windows": [
    {{
      "id": "<window id>",
      "start": <number>,
      "end": <number>,
      "score": <integer 0-100>,
      "reason": "<very short reason>"
    }}
  ]
}}
"""

DETAIL_PROMPT_TEMPLATE = """
You are a senior short-form video editor and viral copywriter.
Choose the BEST short clips from these shortlisted candidate windows.

CLIP RULES:
- Return only valid JSON.
- Each clip must be 15 to 60 seconds long, in absolute seconds from the start of the source video.
- Stay within the candidate window boundaries.
- THE 2-SECOND RULE: the clip MUST open on its strongest moment. If the first
  2 seconds would not stop a cold viewer from scrolling, move the start or skip the clip.
- Start slightly before the hook and end slightly after the payoff when possible.
- Do not cut in the middle of a word or phrase.
- No generic intros/outros unless they are the hook.
- Prefer one great clip per candidate window. Maximum 2 clips per window only if clearly justified.
- DIVERSITY: never return two clips that make the same point, tell the same
  story, or land the same joke — even across different windows. Pick the
  stronger one and drop the other.

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

TRANSCRIPT_LANGUAGE: {language}
VIDEO_DURATION_SECONDS: {video_duration}
CANDIDATE_WINDOWS_JSON:
{windows_json}

Return only:
{{
  "shorts": [
    {{
      "start": <number>,
      "end": <number>,
      "source_window_id": "<window id>",
      "predicted_score": <integer 0-100>,
      "video_description_for_tiktok": "<description + hashtags>",
      "video_description_for_instagram": "<description + hashtags>",
      "video_title_for_youtube_short": "<title max 100 chars>",
      "viral_hook_text": "<short overlay max 10 words>"
    }}
  ]
}}
"""


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_json_candidate(text: str) -> str:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]
    return cleaned


def _escape_invalid_unicode_escapes(text: str) -> str:
    chars = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] == "u":
            hex_digits = text[i + 2:i + 6]
            if len(hex_digits) < 4 or any(ch not in "0123456789abcdefABCDEF" for ch in hex_digits):
                chars.append("\\\\u")
                i += 2
                continue
        chars.append(text[i])
        i += 1
    return "".join(chars)


def _parse_json_response_text(text: str) -> dict:
    if not text:
        raise ValueError("Gemini returned an empty response body.")
    candidate = _extract_json_candidate(text).replace("\x00", "").strip()
    if not candidate:
        raise ValueError("Gemini response did not contain a JSON object.")
    parse_attempts = [candidate]
    sanitized_candidate = _escape_invalid_unicode_escapes(candidate)
    if sanitized_candidate != candidate:
        parse_attempts.append(sanitized_candidate)
    last_error: Optional[Exception] = None
    for parse_candidate in parse_attempts:
        try:
            return json.loads(parse_candidate)
        except json.JSONDecodeError as e:
            last_error = e
    raise ValueError(f"Failed to parse Gemini JSON response: {last_error}")


class GeminiBlockedError(ValueError):
    """The API refused the request for content-policy reasons.

    Deterministic: the same payload is rejected every time (verified in prod,
    23-jul-2026 — a stand-up video came back PROHIBITED_CONTENT in ~300ms on
    every attempt), and BLOCK_NONE safety settings do NOT lift it. Retrying is
    pointless, so callers must fail fast with a message that tells the user the
    video's content is the problem, not the service."""


_BLOCKED_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST",
                           "SPII", "IMAGE_SAFETY", "RECITATION"}


def raise_if_blocked(response):
    """Raise GeminiBlockedError when the API refused to answer on policy grounds."""
    pf = getattr(response, "prompt_feedback", None)
    reason = getattr(pf, "block_reason", None)
    if reason:
        name = getattr(reason, "name", None) or str(reason)
        raise GeminiBlockedError(
            f"Gemini blocked this video's content ({name}). The AI provider's "
            "usage policies reject this material, so it can't be analyzed.")
    for c in (getattr(response, "candidates", None) or []):
        fr = getattr(c, "finish_reason", None)
        name = (getattr(fr, "name", None) or str(fr or "")).upper()
        if name in _BLOCKED_FINISH_REASONS:
            raise GeminiBlockedError(
                f"Gemini blocked its answer for this video ({name}). The AI "
                "provider's usage policies reject this material, so it can't be analyzed.")


def _get_response_text(response) -> str:
    try:
        text = response.text
        if text:
            return text
    except Exception:
        pass

    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "\n".join(parts).strip()


def _calculate_cost_analysis(response, model_name: str) -> Optional[dict]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return None
    prices = lookup_model_prices(model_name)
    price_estimated = prices is None
    if prices is None:
        # Unknown model: conservative estimate so the UI shows something sane.
        prices = (0.50, 3.00)
    input_price_per_million, output_price_per_million = prices
    prompt_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    # Thinking tokens bill at the output rate even though they are invisible.
    thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = ((output_tokens + thinking_tokens) / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model": model_name,
        "price_estimated": price_estimated,
    }


def _thinking_config_from_env(model_name: str):
    """GEMINI_THINKING_SCORE: off (default) | low | high | <token budget>.

    Applied only to the scoring stage. Gemini 3 models take thinking_level,
    Gemini 2.5 takes thinking_budget; returns None (= model default) if the
    setting is off or the SDK rejects the config."""
    raw = (os.getenv("GEMINI_THINKING_SCORE") or "off").strip().lower()
    if raw in ("", "off", "0", "none", "false"):
        return None
    try:
        if raw.isdigit():
            return genai_types.ThinkingConfig(thinking_budget=int(raw))
        if raw in ("low", "high"):
            if model_name.startswith("gemini-3"):
                return genai_types.ThinkingConfig(thinking_level=raw)
            return genai_types.ThinkingConfig(thinking_budget=2048 if raw == "low" else 8192)
    except Exception as e:
        _log(f"⚠️ Ignoring GEMINI_THINKING_SCORE={raw!r}: {e}")
    return None


def _config_for_strategy(strategy: str, mode: str, model_name: str) -> genai_types.GenerateContentConfig:
    # The detail stage writes creative copy (hooks/descriptions) — it gets a
    # high temperature; timestamps are validated and word-snapped afterwards.
    # The score stage stays precise. Fallback strategies get conservative.
    creative = mode == "detail"
    kwargs = {
        "response_mime_type": "application/json",
        "candidate_count": 1,
        "safety_settings": RELAXED_SAFETY_SETTINGS,
    }
    if strategy == "strict-json":
        kwargs["temperature"] = 0.7 if creative else 0.1
    elif strategy == "json-text-recovery":
        kwargs["temperature"] = 0.2 if creative else 0.0
    else:  # structured-schema: schema-enforced output, primary strategy
        kwargs["temperature"] = 0.9 if creative else 0.2
        kwargs["response_schema"] = DetailResponse if mode == "detail" else ScoreResponse
        if mode == "score":
            thinking = _thinking_config_from_env(model_name)
            if thinking is not None:
                kwargs["thinking_config"] = thinking
    return genai_types.GenerateContentConfig(**kwargs)


def main() -> int:
    _configure_stdio()

    parser = argparse.ArgumentParser(description="Run a single Gemini request for clip scoring/detailing.")
    parser.add_argument("--mode", choices=["score", "detail"], required=True)
    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output", dest="output_path", required=True)
    parser.add_argument("--strategy", default="structured-schema")
    parser.add_argument("--model", default="gemini-2.5-flash")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY.")

    with open(args.input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    model_name = args.model
    client = make_client(api_key)
    config = _config_for_strategy(args.strategy, args.mode, model_name)
    language = str(payload.get("language") or "unknown")

    template = SCORE_PROMPT_TEMPLATE if args.mode == "score" else DETAIL_PROMPT_TEMPLATE
    prompt = template.format(
        video_duration=payload["video_duration"],
        language=language,
        windows_json=json.dumps(payload["windows"], ensure_ascii=False),
    )

    _log(f"🤖 Gemini worker request: mode={args.mode} strategy={args.strategy} model={model_name} items={len(payload.get('windows', []))}")
    response = gemini_pool.generate_with_fallback(
        client, model_name, prompt, config=config, max_attempts=1,
        log=lambda msg: print(msg, file=sys.stderr))


    raw_text = _get_response_text(response)
    # With response_schema the SDK returns an already-validated object; fall
    # back to the text-repair path only when that is unavailable.
    parsed_obj = getattr(response, "parsed", None)
    if parsed_obj is not None:
        parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
    else:
        parsed = _parse_json_response_text(raw_text)
    result = {
        "mode": args.mode,
        "payload": parsed,
        "cost_analysis": _calculate_cost_analysis(response, model_name),
        "raw_text": raw_text,
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    _log(f"✅ Gemini worker success: mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
