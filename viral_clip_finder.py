"""Viral Clip Finder — the skill's judgment layer as a structured engine.

This module turns the packaged ``viral_clip_finder_skill`` (the judgment-layer
methodology: 15 named frameworks, 8-axis rubric, 18 anti-patterns, 8 niche
playbooks, seamless-cut rules) into an executable Stage 3 for OpenShorts.

Architecture
------------
1. ``format_transcript_for_skill`` converts OpenShorts' internal transcript
   (segments + word timestamps + optional diarized speaker labels) into the
   timecoded ``[HH:MM:SS] Speaker: text`` contract the skill expects.
2. ``build_system_prompt`` assembles the system prompt from SKILL.md plus the
   reference docs (progressive disclosure: the seamless-cutting reference is
   only included when long-form clips are requested).
3. The LLM answers in **JSON matching the skill's output contract** (scored
   clips + rejected candidates with reasons) — the report is structured, not
   Markdown, so nothing is regex-parsed out of prose.
4. ``normalize_clip`` maps the skill's fields onto OpenShorts' clip shape:
   ``start``/``end`` (absolute seconds, clamped), ``keep_spans`` derived from
   the cut brief (trim suggestion for Tier S, inverted cut list for Tier L),
   ``clip_type`` (short/long_context), platform copy, and the skill-only
   fields (score, patterns, risk flags, cut brief) that flow into
   metadata.json for the dashboard and editors.

The transport is provider-agnostic and identical to the existing narrative
engine: ``deepseek_worker._run_deepseek_stage`` against the OpenAI-compatible
endpoint of the configured provider (NARRATIVE_GEMINI_API_KEY by default,
matching the current Stage 3; DeepSeek is opt-in via VCF_ALLOW_DEEPSEEK=1).

Environment
-----------
VIRAL_ENGINE=auto|skill|narrative   select the Stage 3 engine (see main.py)
VCF_ALLOW_DEEPSEEK=1                allow DeepSeek's own API as a provider
VCF_LONG_FORM_REFS=1                always include seamless-cutting.md
VCF_REFERENCES=lean|full|a.md,b.md  which references to load (default: full)
"""

import json
import os
import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from deepseek_worker import (
    DEEPSEEK_API_BASE,
    DEFAULT_DEEPSEEK_MODEL,
    GEMINI_OPENAI_COMPAT_BASE,
    DEFAULT_NARRATIVE_GEMINI_MODEL,
    _run_deepseek_stage,
)

# --- Bundled skill package -------------------------------------------------

_SKILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "viral_clip_finder_skill")
_SKILL_MD_PATH = os.path.join(_SKILL_DIR, "SKILL.md")
_REF_DIR = os.path.join(_SKILL_DIR, "references")

# Progressive disclosure: the judgment references are always loaded; the
# cutting reference is long-form-only (see SKILL.md's own "load on demand"
# guidance). Niche routing is inside niches.md and the prompt tells the model
# to read the section matching the source material.
_ALWAYS_REFERENCES = (
    "niches.md",
    "hook-library.md",
    "anti-patterns.md",
    "scoring-rubric.md",
)
_LONG_FORM_REFERENCES = ("seamless-cutting.md",)

_FFMPEG_TIME_CONTRACT = """
⚠️ FFMPEG TIME CONTRACT — STRICT REQUIREMENTS:
- Return timestamps in ABSOLUTE SECONDS from the start of the video
  (usable in: ffmpeg -ss <start> -to <end> -i <input> ...).
- Only NUMBERS with decimal point, up to 3 decimals (examples: 0, 1.250, 17.350).
- Ensure 0 <= start < end <= VIDEO_DURATION_SECONDS.
- Tier S clips: 15-30s (hard 15s minimum, 60s absolute maximum).
- Tier L clips: 60-180s.
- Use silence moments for natural cuts; never cut in the middle of a word.
"""


# --- Output contract (skill report, JSON-native) ---------------------------


class _SkillModel(BaseModel):
    """Base for every skill response model: an explicit JSON ``null`` is treated
    as "not provided" rather than a type error.

    The prompt asks for `"narrative_arc": null` in the score breakdown, so the
    model quite reasonably emits null for other optional fields too —
    reaction_cam, cta_placement, b_roll_cues. A bare `field: str = ""` default
    does NOT accept null, so the whole response was rejected over fields nobody
    reads for the cut. Observed in production 6-aug-2026: five clips returned,
    every one discarded on `reaction_cam`, three full retries burned (each a
    ~150KB prompt), then a silent fall back to the narrative engine.

    Dropping the null keys lets each field's own default apply. Required fields
    (start/end) are untouched and still fail loudly if they arrive null.
    """

    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data):
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data


class CutEntry(_SkillModel):
    """One internal cut for a Tier L clip (seamless-cutting.md §8)."""

    cut_from: float = Field(description="end timestamp of the retained segment A")
    cut_to: float = Field(description="start timestamp of retained segment B (> cut_from + 0.010)")
    reason: str = Field(default="", description="why this content is removed")
    masking: str = Field(default="",
                         description="same-frame | reaction shot | b-roll | zoom")


class ScoreBreakdown(_SkillModel):
    """One axis of the 8-axis rubric. Key is the axis name, value the score."""

    hook_strength: int = 0
    payoff_density: int = 0
    standalone_clarity: int = 0
    emotional_spike: int = 0
    share_trigger: int = 0
    replay_value: int = 0
    cut_quality: int = 0
    cross_platform_fit: int = 0
    narrative_arc: Optional[int] = None  # Tier L only


class SkillClipModel(_SkillModel):
    """A scored clip with the full cut brief — the skill's per-clip report."""

    start: float
    end: float
    tier: str = "S"                     # "S" (15-30s) or "L" (60-180s)
    tier_class: str = ""                # "A" | "B" | "C"
    niche: str = ""
    score: int = 0                      # 0-100
    primary_pattern: str = ""
    secondary_patterns: List[str] = []
    why_it_hits: str = ""
    cold_open_line: str = ""
    trim_suggestion: str = ""
    cut_list: List[CutEntry] = []
    caption_text: str = ""
    b_roll_cues: List[str] = []
    reaction_cam: str = ""
    cta_placement: str = ""
    sound_design: str = ""
    platform_fit: str = ""
    risk_flags: List[str] = []
    score_breakdown: Optional[ScoreBreakdown] = None
    video_description_for_tiktok: str = ""
    video_description_for_instagram: str = ""
    video_title_for_youtube_short: str = ""


class RejectedClipModel(_SkillModel):
    """A candidate that tripped the anti-pattern detector (mandatory output)."""

    start: float
    end: float
    anti_pattern: str = ""
    reason: str = ""


class TermCorrectionModel(_SkillModel):
    """An ASR mishearing to repair before captions/word-snapping run.

    Same contract as deepseek_worker.TermCorrection — the narrative engine has
    always returned these and main.py applies them to the transcript before
    building the word list, so captions show the fixed spelling. The skill path
    has to supply them too or switching engines silently loses the feature.
    """

    wrong: str    # exact mistranscribed text as it appears in the transcript
    correct: str  # the corrected term


class SkillResponse(_SkillModel):
    clips: List[SkillClipModel] = []
    rejected: List[RejectedClipModel] = []
    term_corrections: List[TermCorrectionModel] = []


# --- Input formatting -------------------------------------------------------


def skill_available() -> bool:
    """True when the packaged skill files are present."""
    return os.path.exists(_SKILL_MD_PATH)


def format_timestamp(seconds: float) -> str:
    """1234.567 -> '00:20:34' (the skill's input contract)."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_transcript_for_skill(transcript_result: dict,
                                face_identities: Optional[dict] = None) -> str:
    """Convert OpenShorts' transcript to the skill's timecoded input contract.

    OpenShorts transcripts are segment dicts with start/end/text and optional
    word-level timestamps; when diarization is present, segments carry a
    ``speaker`` label. ``face_identities`` (from face_id.py) maps anonymous
    speaker labels to names and upgrades the input to the skill's Tier 3
    (named-speaker) contract.
    """
    face_identities = face_identities or {}
    lines = []
    for segment in transcript_result.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        speaker = segment.get("speaker") or "Speaker"
        if speaker in face_identities:
            speaker = face_identities[speaker]
        ts = format_timestamp(segment.get("start", 0))
        lines.append(f"[{ts}] {speaker}: {text}")
    return "\n".join(lines)


# --- Prompt assembly --------------------------------------------------------


_REFERENCE_CACHE: dict = {}


def _read_reference(name: str) -> str:
    if name in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[name]
    path = os.path.join(_REF_DIR, name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        content = ""
    _REFERENCE_CACHE[name] = content
    return content


def _selected_reference_names(include_long_form: bool) -> Tuple[str, ...]:
    """Which references this run pays for.

    The full set is ~124KB (~35k tokens) prepended to every Stage 3 call. That
    is the intended default — the judgment quality IS the references — but
    VCF_REFERENCES lets an operator trade breadth for cost/latency:

      VCF_REFERENCES=lean   scoring-rubric + anti-patterns only (the scoring
                            discipline and the reject filters; drops the
                            pattern/niche catalogues, ~-69KB)
      VCF_REFERENCES=a.md,b.md   an explicit list
      unset / anything else      the full always-on set

    seamless-cutting.md is added on top whenever long-form clips are requested
    (or VCF_LONG_FORM_REFS=1), since only Tier L uses internal cut lists.
    """
    setting = os.environ.get("VCF_REFERENCES", "").strip().lower()
    if setting == "lean":
        names = ("scoring-rubric.md", "anti-patterns.md")
    elif setting and setting not in ("full", "all"):
        names = tuple(n.strip() for n in setting.split(",") if n.strip())
    else:
        names = _ALWAYS_REFERENCES
    if include_long_form:
        names = names + tuple(n for n in _LONG_FORM_REFERENCES if n not in names)
    return names


def _load_references(include_long_form: bool) -> dict:
    return {name: _read_reference(name)
            for name in _selected_reference_names(include_long_form)}


def build_system_prompt(include_long_form: bool = False,
                        style_variant: str = "balanced") -> str:
    """Assemble the skill system prompt from the bundled package.

    SKILL.md is the core; the reference docs are appended as named sections so
    the model can consult them while executing the 6-step workflow. This is
    the same load-everything approach the integration guide's own wrapper uses
    (references are progressive in the sense that only long-form runs pay for
    seamless-cutting.md).
    """
    if "SKILL.md" not in _REFERENCE_CACHE:
        with open(_SKILL_MD_PATH, "r", encoding="utf-8") as f:
            _REFERENCE_CACHE["SKILL.md"] = f.read()
    skill_md = _REFERENCE_CACHE["SKILL.md"]
    parts = [
        "You are executing the viral-clip-finder methodology against a "
        "timecoded transcript. Follow the skill's 6-step workflow exactly: "
        "Orient -> Segment -> Tag patterns -> Anti-pattern check -> Score -> "
        "Generate cut briefs.",
        "",
        "=== SKILL ===",
        skill_md,
    ]
    for name, content in _load_references(include_long_form).items():
        if content:
            parts.append(f"\n=== REFERENCE: {name.replace('.md', '').upper()} ===")
            parts.append(content)
    if style_variant != "balanced":
        parts.append(_STYLE_DIRECTIVE.format(variant=style_variant))
    return "\n\n".join(parts)


_STYLE_DIRECTIVE = """
STYLE VARIANT ({variant}): apply the skill's style preferences:
- high_energy: favor openings with immediate conflict, strong opinions, or a
  punchy claim in the first words; prefer the sharpest accurate framing.
- story_driven: favor clips that carry a complete narrative arc; prefer
  long-form (Tier L) framing for any beat with a real resolution.
The skill's scoring discipline and anti-pattern filters still apply unchanged.
"""


def _candidate_count_directive(video_duration: float, override: Optional[int]) -> str:
    """Scale the requested clip count with duration, like the narrative engine."""
    if override:
        return (f"Return exactly up to {override} top clips (the user asked "
                f"for this count explicitly).")
    target = max(3, min(15, int(video_duration // 90) + 2))
    return (f"Return the {target} strongest Tier A/B clips (3-15 total). "
            f"Fewer is fine when the source genuinely cannot support them.")


def build_user_prompt(transcript_text: str, video_duration: float,
                      clip_count: Optional[int] = None,
                      long_context_count: int = 0,
                      style_variant: str = "balanced",
                      target_platforms: str = "TikTok, Instagram Reels, YouTube Shorts, X") -> str:
    """The task message: transcript + tier/count/platform parameters.

    Long-form (Tier L) clips are requested alongside short-form when
    ``long_context_count`` > 0, matching the pipeline's existing flag.
    """
    long_directive = ""
    if long_context_count and long_context_count > 0:
        long_directive = (
            f"Also produce up to {long_context_count} Tier L long-form clips "
            "(60-180s, full narrative arc with setup -> tension -> climax -> "
            "resolution). Give each a complete internal cut list (cut_from/"
            "cut_to/reason/masking) following the seamless-cutting rules, and "
            "flag any Tier S sub-segments worth publishing separately."
        )
    return f"""
Find the top viral clips in this transcript using the skill methodology.

VIDEO_DURATION_SECONDS: {video_duration}
TARGET PLATFORMS: {target_platforms}
STYLE VARIANT: {style_variant}
{_candidate_count_directive(video_duration, clip_count)}
{long_directive}

{_FFMPEG_TIME_CONTRACT}

OUTPUT — RETURN ONLY VALID JSON (no markdown, no prose) matching this exact
shape:
{{
  "clips": [
    {{
      "start": 12.340,
      "end": 37.900,
      "tier": "S",
      "tier_class": "A",
      "niche": "Motivation",
      "score": 87,
      "primary_pattern": "Pop-the-Balloon",
      "secondary_patterns": ["Concrete Specificity", "Stakes Escalation"],
      "why_it_hits": "2-3 sentences: the loop it opens, the payoff, the share trigger",
      "cold_open_line": "You're not busy.",
      "trim_suggestion": "Cut from 00:14:34, drop the host's question lead-in",
      "cut_list": [{{"cut_from": 15.200, "cut_to": 18.400, "reason": "filler", "masking": "same-frame"}}],
      "caption_text": "YOU'RE NOT BUSY",
      "b_roll_cues": ["cluttered inbox at 00:14:52"],
      "reaction_cam": "cut to Host's reaction at 00:14:38",
      "cta_placement": "end card at 00:14:58",
      "sound_design": "bass drop on the reframe; no music under the open",
      "platform_fit": "TikTok 4/5 Reels 4/5 Shorts 4/5 X 3/5",
      "risk_flags": [],
      "score_breakdown": {{
        "hook_strength": 14, "payoff_density": 13, "standalone_clarity": 14,
        "emotional_spike": 8, "share_trigger": 9, "replay_value": 7,
        "cut_quality": 9, "cross_platform_fit": 13, "narrative_arc": null
      }},
      "video_description_for_tiktok": "TikTok caption",
      "video_description_for_instagram": "Instagram caption",
      "video_title_for_youtube_short": "title <= 100 chars"
    }}
  ],
  "rejected": [
    {{"start": 120.5, "end": 148.0, "anti_pattern": "Fake-Deep Quote", "reason": "..."}}
  ],
  "term_corrections": [
    {{"wrong": "exact mistranscribed text", "correct": "corrected term"}}
  ]
}}

Rules:
- The "rejected" array is MANDATORY — list every candidate that tripped an
  anti-pattern (at least 3), with the anti-pattern name and why.
- "term_corrections": names, brands and jargon the transcriber clearly got
  wrong (empty list if none). "wrong" must be the EXACT text as it appears in
  the transcript so it can be string-replaced; do not "fix" ordinary wording,
  grammar or punctuation — only genuine mishearings.
- Score every kept clip honestly on the 8-axis rubric (Tier L adds
  narrative_arc). Default to returning only Tier A (85+) and Tier B (70-84).
- score_breakdown axis names are snake_case as shown; narrative_arc is null
  for Tier S clips.
- cut_list is for Tier L clips only (empty for Tier S; use trim_suggestion).
- Descriptions must include a CTA. All text fields in the SAME LANGUAGE as
  the transcript.

TRANSCRIPT:
{transcript_text}
""".strip()


# --- Output normalization ---------------------------------------------------


def _span(start: float, end: float) -> dict:
    """One keep_span in the pipeline's contract shape (see derive_keep_spans)."""
    return {"start": round(float(start), 3), "end": round(float(end), 3)}


def _parse_trim_timestamp(text: str) -> Optional[float]:
    """Extract the first HH:MM:SS (or plain seconds) from a trim string."""
    if not text:
        return None
    m = re.search(r"(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?", text)
    if m:
        h, mm, s = (int(g) for g in m.groups())
        return h * 3600 + mm * 60 + s
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*s(?:econds)?\b", text)
    if m:
        return float(m.group(1))
    return None


def derive_keep_spans(clip: dict, clip_start: float, clip_end: float) -> List[dict]:
    """Convert the skill's cut brief into OpenShorts' keep_spans contract.

    Tier S: single keep span from the trim suggestion ("Cut from HH:MM:SS" /
    "Start at HH:MM:SS" / "End at HH:MM:SS"), defaulting to the whole clip.
    Tier L: complement of the cut list within [clip_start, clip_end] — every
    range NOT removed becomes a keep span (seamless-cutting.md §8 inversion).

    Spans are ``{"start": float, "end": float}`` dicts, NOT [start, end]
    pairs: that is the contract every consumer in main.py reads with
    ``span.get("start")`` (deepseek_worker.KeepSpan is the same shape). Pairs
    here raise AttributeError inside _extend_keep_spans_to_cover_boundaries
    and _snap_keep_spans_to_words — see tests/test_viral_clip_finder.py's
    cross-module contract test.
    """
    if clip_end <= clip_start:
        return [_span(clip_start, clip_end)]

    cut_list = clip.get("cut_list") or []
    if len(cut_list) > 1 or (cut_list and clip.get("tier") == "L"):
        spans = []
        cursor = clip_start
        for cut in sorted(cut_list, key=lambda c: c.get("cut_from", 0)):
            cut_from = max(clip_start, float(cut.get("cut_from", 0)))
            cut_to = min(clip_end, float(cut.get("cut_to", 0)))
            if cut_to <= cut_from:
                continue
            if cut_from - cursor >= 1.0:
                spans.append(_span(cursor, cut_from))
            cursor = max(cursor, cut_to)
        if clip_end - cursor >= 1.0:
            spans.append(_span(cursor, clip_end))
        return spans or [_span(clip_start, clip_end)]

    trim = clip.get("trim_suggestion") or ""
    keep_start, keep_end = clip_start, clip_end
    start_ts = _parse_trim_timestamp(trim)
    if start_ts is not None:
        # Only treat it as a start trim if the suggestion actually references
        # a cut-in ("Cut from"/"Start at"/"Start hard on ... at").
        if re.search(r"(?:cut from|start (?:hard on .*? )?at|start at|trim (?:from|to))",
                     trim, re.IGNORECASE):
            keep_start = max(clip_start, min(clip_end, start_ts))
    end_ts = None
    for token in (r"end at", r"cut to", r"stop at", r"trim to"):
        m = re.search(token + r"\s+(\d{2}:\d{2}:\d{2})", trim, re.IGNORECASE)
        if m:
            h, mm, s = (int(g) for g in m.groups())
            end_ts = h * 3600 + mm * 60 + s
            break
    if end_ts is not None:
        keep_end = min(clip_end, max(clip_start, end_ts))
    if keep_end - keep_start < 1.0:
        return [_span(clip_start, clip_end)]
    return [_span(keep_start, keep_end)]


def _clamp_time(value: float, video_duration: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.0
    return min(max(v, 0.0), video_duration)


def _shorten(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text[:limit] if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def normalize_clip(raw: dict, video_duration: float) -> Optional[dict]:
    """Map one skill clip onto OpenShorts' internal clip shape.

    Downstream Stage 4+ consumes start/end/keep_spans/clip_type; everything
    else is preserved in metadata.json for the dashboard and editors.
    """
    start = _clamp_time(raw.get("start"), video_duration)
    end = _clamp_time(raw.get("end"), video_duration)
    if end - start < 1.0:
        return None

    tier = str(raw.get("tier") or "S").strip().upper()[:1]
    tier = "L" if tier == "L" else "S"
    clip_type = "long_context" if tier == "L" else "short"

    # Enforce the same duration floors the rest of the pipeline assumes.
    min_duration = 15.0 if tier == "S" else 45.0
    if end - start < min_duration:
        # Prefer extending backward (keeps the cold open intact); fall back to
        # extending forward, then to keeping the span as-is (snap will fix it).
        pad = min_duration - (end - start)
        if start - pad >= 0:
            start = round(start - pad, 3)
        elif end + pad <= video_duration:
            end = round(end + pad, 3)

    hook = (raw.get("caption_text") or raw.get("cold_open_line") or "").strip()
    hook = _shorten(hook, 60)
    if len(hook.split()) > 10:
        hook = " ".join(hook.split()[:10]) + "…"

    clip = {
        "start": round(start, 3),
        "end": round(end, 3),
        "clip_type": clip_type,
        "tier": tier,
        "tier_class": str(raw.get("tier_class") or "").upper()[:1],
        "score": int(raw.get("score") or 0),
        "predicted_score": int(raw.get("score") or 0),
        "primary_pattern": raw.get("primary_pattern") or "",
        "secondary_patterns": list(raw.get("secondary_patterns") or []),
        "why_it_hits": raw.get("why_it_hits") or "",
        # Aliases onto the narrative engine's field names. The vision
        # confirmation pass builds its prompt from narrative_summary
        # (main.py's confirm_clip_with_vision) — without this the judge is
        # told nothing about what the clip is supposed to resolve and reviews
        # it blind. hook_type is the same idea for anything reading the
        # narrative engine's shape.
        "narrative_summary": raw.get("why_it_hits") or "",
        "hook_type": raw.get("primary_pattern") or "",
        "cold_open_line": raw.get("cold_open_line") or "",
        "caption_text": raw.get("caption_text") or "",
        "b_roll_cues": list(raw.get("b_roll_cues") or []),
        "reaction_cam": raw.get("reaction_cam") or "",
        "cta_placement": raw.get("cta_placement") or "",
        "sound_design": raw.get("sound_design") or "",
        "platform_fit": raw.get("platform_fit") or "",
        "risk_flags": list(raw.get("risk_flags") or []),
        "niche": raw.get("niche") or "",
        "keep_spans": [],
        "viral_hook_text": hook,
        "video_description_for_tiktok": _shorten(
            raw.get("video_description_for_tiktok")
            or f"{hook} — {raw.get('why_it_hits', '')}", 2200),
        "video_description_for_instagram": _shorten(
            raw.get("video_description_for_instagram")
            or raw.get("video_description_for_tiktok")
            or f"{hook} — {raw.get('why_it_hits', '')}", 2200),
        "video_title_for_youtube_short": _shorten(
            raw.get("video_title_for_youtube_short") or hook, 100),
    }
    if raw.get("score_breakdown"):
        clip["score_breakdown"] = raw["score_breakdown"]
    if raw.get("cut_list"):
        clip["cut_list"] = raw["cut_list"]
    if raw.get("trim_suggestion"):
        clip["trim_suggestion"] = raw["trim_suggestion"]
    clip["keep_spans"] = derive_keep_spans(clip, clip["start"], clip["end"])
    return clip


def normalize_response(parsed: dict, video_duration: float,
                       clip_count: Optional[int]) -> Tuple[List[dict], List[dict]]:
    """Normalize a parsed SkillResponse into OpenShorts clips + reject list."""
    clips = []
    for raw in parsed.get("clips") or []:
        clip = normalize_clip(raw, video_duration)
        if clip is not None:
            clips.append(clip)
    rejected = []
    for raw in parsed.get("rejected") or []:
        rejected.append({
            "start": round(_clamp_time(raw.get("start"), video_duration), 3),
            "end": round(_clamp_time(raw.get("end"), video_duration), 3),
            "anti_pattern": raw.get("anti_pattern") or "",
            "reason": raw.get("reason") or "",
        })

    # Skill discipline: default to Tier A/B only, but never return zero clips
    # when borderline candidates exist (the user can always reject them).
    strong = [c for c in clips if (c["tier_class"] in ("A", "B")) or c["score"] >= 70]
    if strong:
        clips = strong
    elif clips:
        print(f"⚠️ Viral Clip Finder: no Tier A/B candidates; keeping "
              f"{len(clips)} highest-scoring borderline clip(s).")

    if clip_count is not None and len(clips) > clip_count:
        clips = sorted(clips, key=lambda c: c["score"], reverse=True)[:clip_count]
    clips.sort(key=lambda c: c["start"])
    clips = _drop_overlapping(clips)
    return clips, rejected


def _drop_overlapping(clips: List[dict], min_gap: float = 0.0) -> List[dict]:
    """Keep the higher-scoring clip when two candidates overlap in time.

    The model occasionally returns the same moment twice at different
    boundaries, and normalize_clip's duration padding can push a short
    candidate backward into its neighbour. Rendering both spends GPU on two
    near-identical clips, so the lower score loses. Input must be
    start-sorted; output stays start-sorted.
    """
    kept: List[dict] = []
    for clip in clips:
        prev = kept[-1] if kept else None
        if prev is not None and clip["start"] < prev["end"] - min_gap:
            if clip["score"] > prev["score"]:
                print(f"   ⚠️ overlapping candidates "
                      f"[{prev['start']:.1f}-{prev['end']:.1f}] / "
                      f"[{clip['start']:.1f}-{clip['end']:.1f}] — keeping the "
                      f"higher-scoring one ({clip['score']} > {prev['score']})")
                kept[-1] = clip
            continue
        kept.append(clip)
    return kept


# --- Provider selection -----------------------------------------------------


def _provider_candidates() -> List[Tuple[str, str, str, str]]:
    """Ordered (provider, api_key, model, base_url) candidates.

    Mirrors the current Stage 3 chain: NARRATIVE_GEMINI_API_KEY (the existing
    narrative provider) first, then DeepSeek only when explicitly allowed,
    then the main GEMINI_API_KEY. Each candidate is a full attempt; the first
    success wins.
    """
    candidates = []
    narrative_key = os.environ.get("NARRATIVE_GEMINI_API_KEY")
    if narrative_key:
        candidates.append((
            "gemini", narrative_key,
            os.environ.get("NARRATIVE_GEMINI_MODEL") or DEFAULT_NARRATIVE_GEMINI_MODEL,
            GEMINI_OPENAI_COMPAT_BASE,
        ))
    if os.environ.get("VCF_ALLOW_DEEPSEEK", "").strip().lower() in ("1", "true", "yes"):
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        if deepseek_key:
            candidates.append((
                "deepseek", deepseek_key,
                os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL,
                DEEPSEEK_API_BASE,
            ))
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        candidates.append((
            "gemini", gemini_key,
            os.environ.get("GEMINI_MODEL") or DEFAULT_NARRATIVE_GEMINI_MODEL,
            GEMINI_OPENAI_COMPAT_BASE,
        ))
    return candidates


# --- Main entry -------------------------------------------------------------


def select_viral_clips(transcript_result: dict, video_duration: float,
                       clip_count: Optional[int] = None,
                       long_context_count: int = 0,
                       style_variant: str = "balanced",
                       face_identities: Optional[dict] = None) -> Optional[dict]:
    """Run the viral-clip-finder judgment layer on a transcript.

    Returns OpenShorts-shaped {"clips": [...], "rejected": [...],
    "cost_analysis": {...} | absent} or None when no provider is configured
    or every provider failed — the caller decides how to fall back.
    """
    if not skill_available():
        print("⚠️ Viral Clip Finder: bundled skill package missing "
              "(viral_clip_finder_skill/SKILL.md).")
        return None
    candidates = _provider_candidates()
    if not candidates:
        print("⚠️ Viral Clip Finder: no provider key configured "
              "(NARRATIVE_GEMINI_API_KEY, GEMINI_API_KEY, or DEEPSEEK_API_KEY "
              "with VCF_ALLOW_DEEPSEEK=1).")
        return None

    transcript_text = format_transcript_for_skill(
        transcript_result, face_identities=face_identities)
    if not transcript_text.strip():
        print("⚠️ Viral Clip Finder: empty formatted transcript.")
        return None

    include_long_form = bool(
        long_context_count and long_context_count > 0) or os.environ.get(
        "VCF_LONG_FORM_REFS", "").strip().lower() in ("1", "true", "yes")
    system_prompt = build_system_prompt(
        include_long_form=include_long_form, style_variant=style_variant)
    user_prompt = build_user_prompt(
        transcript_text, video_duration, clip_count=clip_count,
        long_context_count=long_context_count, style_variant=style_variant)
    # The transport needs the request in a single message slot; system
    # instructions ride in the user message, exactly like the existing
    # narrative engine (NARRATIVE_PROMPT_TEMPLATE is one big prompt).
    prompt = system_prompt + "\n\n" + user_prompt

    for i, (provider, api_key, model_name, base_url) in enumerate(candidates):
        label = "Viral Clip Finder (gemini)" if provider == "gemini" else "Viral Clip Finder (deepseek)"
        print(f"🐋 {label}: model={model_name} "
              f"segments={len(transcript_result.get('segments', []))} "
              f"prompt={len(prompt) // 1024}KB "
              f"(refs: {', '.join(_selected_reference_names(include_long_form))})")
        try:
            parsed, cost = _run_deepseek_stage(
                api_key, model_name, prompt, SkillResponse, base_url=base_url)
            clips, rejected = normalize_response(parsed, video_duration, clip_count)
            if not clips:
                print(f"⚠️ {label} returned no usable clips.")
                if i < len(candidates) - 1:
                    continue
                return None
            print(f"🔥 {label}: {len(clips)} clip(s), "
                  f"{len(rejected)} rejected candidate(s) — "
                  + ", ".join(f"#{j+1} {c['score']}/100 {c['primary_pattern'] or ''}"
                              for j, c in enumerate(clips[:5])))
            result = {"clips": clips, "rejected": rejected,
                      "term_corrections": parsed.get("term_corrections") or []}
            if cost:
                result["cost_analysis"] = {
                    "input_tokens": cost.get("input_tokens", 0),
                    "output_tokens": cost.get("output_tokens", 0),
                    "total_cost": cost.get("total_cost", 0),
                    "model": model_name,
                    "engine": "viral-clip-finder",
                }
                print(f"💰 {label} cost ({model_name}): "
                      f"${result['cost_analysis']['total_cost']:.6f}")
            return result
        except Exception as e:
            print(f"❌ {label} error: {type(e).__name__}: {e}")
            if i < len(candidates) - 1:
                print("   ↳ falling back to the next configured provider…")
    return None
