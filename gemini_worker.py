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


