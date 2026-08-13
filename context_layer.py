"""Pre-download Gemini context layer — the Stage 3 "brain".

When a job starts from a YouTube URL, main.py sends the LINK to Gemini in a
background thread while yt-dlp downloads the file. Gemini reads the video on
Google's side (``Part.from_uri`` — no file upload, no waiting on an ACTIVE
state) and returns a 3-part, niche-agnostic summary:

  1. SUMMARY — what the video is about, audiovisually (premise, people,
     what is on screen, stakes, tone).
  2. HIGHLIGHTS — everything notable that happens, with approximate
     timestamps and a beat type.
  3. LOVABLE_MOMENTS — the parts people would love / would keep watching,
     with approximate durations and why they land.

The picker (picker.py) consumes this as pre-loaded context on top of the
transcript. Timestamps here are approximations by design — the transcript
refines them. The call uses its own key (``CONTEXT_GEMINI_API_KEY``) so it
never contends with the picker's calls for the same rate limit.
"""
import json
import os
import threading
import time
from typing import List, Optional

from dotenv import load_dotenv
from google.genai import types as genai_types
from pydantic import BaseModel

import gemini_pool
import gemini_worker
from security_utils import assert_public_url

load_dotenv()

CONTEXT_BLOB_FILENAME = "gemini_context.json"
_DEFAULT_MODEL = "gemini-3.1-flash-lite"

TRANSCRIPT_CONTEXT_PROMPT_TEMPLATE = """
You are the FIRST PASS of a short-form clip pipeline. You do not have the
video — you have its full DIARIZED TRANSCRIPT (speaker labels + timestamps).
Build the same 3-part "brain" a clip picker uses, inferring from the
dialogue what is actually happening: roles, the premise, format, structure,
stakes, tone, and the beats where people react, laugh, argue, or pop.
Never assume a niche or genre — adapt to what the dialogue shows.

Return EXACTLY three sections:
1. SUMMARY — what the video is about, who is involved (roles, not just
   names), the setting/format, the stakes/throughline, the tone.
2. HIGHLIGHTS — an ENUMERATION of the notable beats, each with an
   approximate time range. Be dense: for a long video list many.
3. LOVABLE_MOMENTS — the moments people would love/share/clip, each with an
   approximate time range.

The transcript follows, then respond with JSON:
summary: str, highlights: [{start_s, end_s, description}],
lovable_moments: [{start_s, end_s, description}], video_duration_s: float.
"""

_AUTH_FAILURE_TOKENS = (
    "unauthenticated", "access_token_type_unsupported",
    "invalid authentication", "api key not valid", "invalid_api_key",
    "api key not found", "permission_denied",
)

_TRANSIENT_TOKENS = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
    "500", "INTERNAL", "overloaded", "Deadline",
    "empty response body", "did not contain a JSON object",
    "Failed to parse Gemini JSON response",
) + _AUTH_FAILURE_TOKENS


class ContextHighlight(BaseModel):
    start_s: float
    end_s: float
    description: str
    type: str  # hook | reaction | joke | tension | payoff | twist | stake | context


class ContextLovableMoment(BaseModel):
    start_s: float
    end_s: float
    description: str
    why_it_lands: str


class ContextBlobResponse(BaseModel):
    summary: str
    highlights: List[ContextHighlight]
    lovable_moments: List[ContextLovableMoment]
    video_duration_s: Optional[float] = None


CONTEXT_PROMPT_TEMPLATE = """
You are the FIRST PASS of a short-form clip pipeline. You are watching a
YouTube video FROM ITS LINK — you see the actual footage and hear the audio.
Your summary becomes the "brain" a clip picker uses later, alongside a
transcript. The video can be about anything (podcast, dating show, tutorial,
vlog, sport, news, comedy sketch, ASMR...): never assume a niche, genre, or
format — adapt to what is actually there.

Return EXACTLY three sections:

1. SUMMARY — what the video is about. A plain-language explanation covering:
   - the premise and what is actually happening;
   - who is involved (roles, not just names — "the host", "two contestants",
     "the guest");
   - the setting, format, and structure;
   - the stakes, throughline, or tension;
   - the tone;
   - AND the visuals: what is on screen, what people are doing, reactions,
     on-screen text. This audiovisual context is exactly what a transcript
     cannot provide, so be specific and concrete.

2. HIGHLIGHTS — EVERY notable beat that happens, in chronological order.
   This is an ENUMERATION, not a summary: do not merge beats, do not condense
   the video down to a handful of items. Aim for roughly one highlight per
   60-90 seconds of runtime — a 60-minute episode should produce 40-80+
   highlights. Each entry has an approximate time range {{start_s, end_s}},
   a short concrete description, and a type from: hook, reaction, joke,
   tension, payoff, twist, stake, context. Cover the WHOLE video from start
   to finish. Timestamps are approximations on purpose — a transcript will
   refine them later, so do not stress about precision.

3. LOVABLE_MOMENTS — the parts people would love / would keep watching /
   would share / would clip themselves: the viral moments. List MANY of
   them, scaled to the runtime — at least one per 3-5 minutes, and never
   fewer than 8 for a full-length episode (30-60+ minutes). Each with an
   approximate time range and a one-line "why it lands" (surprise, conflict,
   payoff, humor, relatable stakes, a strong reaction, a satisfying twist,
   a quotable line, high energy...). These are candidates only — the picker
   decides which become clips.

RULES:
- Be specific: names, numbers, actual claims — never "the video has
  interesting parts".
- Use absolute seconds from the start of the video for every timestamp.
- If a timestamp is uncertain, give your best estimate — the transcript
  will fix it later.
- DENSITY CHECK before you finish: if your HIGHLIGHTS list feels short for
  the runtime (e.g. under 30 items for a long episode), you missed beats —
  go back through the whole video and keep enumerating. A thin list is a
  failure; the picker needs every candidate moment visible.
"""


def _model_name():
    return os.environ.get("GEMINI_MODEL") or _DEFAULT_MODEL


def resolve_api_key():
    """The context layer's own key (CONTEXT_GEMINI_API_KEY), so its calls
    never share a rate budget with the picker. Falls back to the pipeline
    primary only when the dedicated key is unset."""
    return (os.environ.get("CONTEXT_GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEY"))


def _call_gemini(api_key, url, prompt):
    """One Part.from_uri generate call. Returns (parsed_dict, cost) or
    raises — the caller decides how to treat failure.

    Rotates across every configured Gemini key on transient failures (the
    context layer used to hammer the single primary key and hit 429
    RESOURCE_EXHAUSTED while GEMINI_API_KEYS extras sat unused)."""
    keys = _api_keys(api_key)
    last_exc = None
    for i, key in enumerate(keys):
        try:
            return _call_with_key(key, url, prompt)
        except gemini_worker.GeminiBlockedError:
            raise  # deterministic policy block — never retry, surface the reason
        except Exception as e:
            last_exc = e
            if not any(tok in str(e).lower() for tok in _TRANSIENT_TOKENS):
                raise
            if i < len(keys) - 1:
                print(f"⚠️ Context-layer Gemini key {i + 1}/{len(keys)} hit a "
                      f"transient error ({str(e)[:120]}) — rotating to the "
                      f"next key")
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


def _call_with_key(api_key, url, prompt):
    """One link call against one key (per-key 3-attempt backoff)."""
    client = gemini_worker.make_client(api_key)
    model_name = _model_name()
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ContextBlobResponse,
        safety_settings=gemini_worker.RELAXED_SAFETY_SETTINGS,
    )
    video_part = genai_types.Part.from_uri(file_uri=url, mime_type="video/mp4")
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = gemini_pool.generate_with_fallback(
                client, model_name, [video_part, prompt], config=config,
                max_attempts=1, log=lambda msg: print(msg))
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
            if any(tok in msg.lower() for tok in _AUTH_FAILURE_TOKENS):
                # A bad key will never succeed on retry or model fallback —
                # raise immediately so _call_gemini rotates to the next key.
                raise
            if attempt == max_attempts or not any(tok in msg for tok in _TRANSIENT_TOKENS):
                raise
            wait = 5 * (2 ** (attempt - 1))
            print(f"⚠️ Context-layer transient error (attempt {attempt}/{max_attempts}), "
                  f"retrying in {wait}s: {msg[:150]}")
            time.sleep(wait)


def _call_with_text(api_key, transcript_text, prompt, model=None):
    """One TEXT-only call (no video URI — the from_uri video tier is
    frequently quota-exhausted even when the text tier is healthy). Uses the
    stable Part(text=...) field constructor: Part.from_text() is broken on
    the installed google-genai (version drift)."""
    client = gemini_worker.make_client(api_key)
    model_name = model or _model_name()
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ContextBlobResponse,
        safety_settings=gemini_worker.RELAXED_SAFETY_SETTINGS,
    )
    content = [genai_types.Part(text=transcript_text),
               genai_types.Part(text=prompt)]
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            response = gemini_pool.generate_with_fallback(
                client, model_name, content, config=config, max_attempts=1,
                log=lambda msg: print(msg))
            gemini_worker.raise_if_blocked(response)
            parsed_obj = getattr(response, "parsed", None)
            if parsed_obj is not None:
                parsed = (parsed_obj.model_dump()
                          if hasattr(parsed_obj, "model_dump")
                          else parsed_obj)
            else:
                raw_text = gemini_worker._get_response_text(response)
                parsed = gemini_worker._parse_json_response_text(raw_text)
            return parsed, gemini_worker._calculate_cost_analysis(
                response, model_name)
        except gemini_worker.GeminiBlockedError:
            raise
        except Exception as e:
            msg = str(e)
            if any(tok in msg.lower() for tok in _AUTH_FAILURE_TOKENS):
                raise  # bad key — rotate at the caller
            if attempt == max_attempts or not any(
                    tok in msg for tok in _TRANSIENT_TOKENS):
                raise
            wait = 5 * (2 ** (attempt - 1))
            print(f"⚠️ Context-layer transient error (attempt "
                  f"{attempt}/{max_attempts}), retrying in {wait}s: "
                  f"{msg[:150]}")
            time.sleep(wait)


def _call_gemini_text(api_key, transcript_text, prompt, model=None):
    """Structured TEXT context call with the same per-key rotation as the
    video-link call."""
    keys = _api_keys(api_key)
    last_exc = None
    for i, key in enumerate(keys):
        try:
            return _call_with_text(key, transcript_text, prompt, model=model)
        except gemini_worker.GeminiBlockedError:
            raise
        except Exception as e:
            last_exc = e
            if not any(tok in str(e).lower() for tok in _TRANSIENT_TOKENS):
                raise
            if i < len(keys) - 1:
                print(f"⚠️ Context-layer Gemini key {i + 1}/{len(keys)} hit a "
                      f"transient error ({str(e)[:120]}) — rotating to the "
                      f"next key")
    raise last_exc


def _compact_transcript_segments(segments, max_chars=20000) -> str:
    """Flatten diarized segments to '[start-end]s speaker X: text' lines —
    the transcript-only context brain's input. Skips empty segments."""
    lines = []
    for seg in segments or []:
        speaker = seg.get("speaker")
        words = " ".join((w.get("text") or w.get("word") or "")
                         for w in (seg.get("words") or []))
        text = (words or str(seg.get("text") or "")).strip()
        if not text:
            continue
        s = float(seg.get("start") or 0)
        e = float(seg.get("end") or 0)
        lines.append(f"[{s:.1f}-{e:.1f}s]"
                     + (f" speaker {speaker}:" if speaker else ":")
                     + f" {text[:300]}")
    return "\n".join(lines)[:max_chars]


def build_context_from_transcript(transcript, source_url="", source_title="",
                                  model=None):
    """Transcript-only context brain: when the video-link (from_uri) call is
    quota-exhausted — the video tier dies long before the text tier — build
    the same 3-part blob from the diarized transcript with a plain-text call.
    Returns the blob dict or None (fail-open)."""
    if os.environ.get("CONTEXT_FROM_TRANSCRIPT", "1").strip().lower() in (
            "0", "false", "no", "off"):
        return None
    segments = (transcript or {}).get("segments") or []
    if not segments:
        return None
    compact = _compact_transcript_segments(segments)
    if not compact:
        return None
    api_key = resolve_api_key()
    if not api_key:
        return None
    print(f"🧠 Context layer (transcript fallback): building the brain from "
          f"the transcript (model={model or _model_name()})...")
    try:
        parsed, _cost = _call_gemini_text(
            api_key, compact, TRANSCRIPT_CONTEXT_PROMPT_TEMPLATE, model=model)
    except Exception as e:
        print(f"   ⚠️ Transcript context failed ({type(e).__name__}: {e})")
        return None
    blob = {
        "source_url": source_url,
        "source_title": source_title,
        "summary": str(parsed.get("summary", "")),
        "highlights": parsed.get("highlights") or [],
        "lovable_moments": parsed.get("lovable_moments") or [],
        "video_duration_s": parsed.get("video_duration_s"),
        "context_source": "transcript",
    }
    print(f"✅ Context layer (transcript) ready: "
          f"{len(blob['highlights'])} highlight(s), "
          f"{len(blob['lovable_moments'])} lovable moment(s)")
    return blob


def analyze_url(url, output_path=None, source_title=""):
    """Blocking: ask Gemini to read the video at ``url`` and write the 3-part
    context blob to ``output_path``. Returns the blob dict on success, None
    on any failure (logged loudly — the picker runs transcript-only then).
    """
    try:
        assert_public_url(url)
        api_key = resolve_api_key()
        if not api_key:
            print("⚠️ Context layer: no GEMINI_API_KEY / CONTEXT_GEMINI_API_KEY "
                  "— proceeding transcript-only.")
            return None
        print(f"🧠 Context layer: asking Gemini to read the link "
              f"(model={_model_name()}) while the video downloads...")
        prompt = CONTEXT_PROMPT_TEMPLATE
        parsed, cost = _call_gemini(api_key, url, prompt)
        blob = {
            "source_url": url,
            "source_title": source_title,
            "summary": str(parsed.get("summary", "")),
            "highlights": parsed.get("highlights") or [],
            "lovable_moments": parsed.get("lovable_moments") or [],
            "video_duration_s": parsed.get("video_duration_s"),
        }
        if cost:
            blob["cost_analysis"] = cost
        if output_path:
            _tmp = output_path + ".tmp"
            with open(_tmp, "w", encoding="utf-8") as f:
                json.dump(blob, f, ensure_ascii=False, indent=2)
            os.replace(_tmp, output_path)
        print(f"✅ Context layer ready: "
              f"{len(blob['highlights'])} highlight(s), "
              f"{len(blob['lovable_moments'])} lovable moment(s)")
        return blob
    except gemini_worker.GeminiBlockedError as e:
        print(f"🚫 Context layer blocked: {e} — proceeding transcript-only.")
    except Exception as e:
        print(f"⚠️ Context layer could not read the link ({type(e).__name__}: {e}) "
              f"— proceeding transcript-only; the picker still fulfills its count.")
    return None


def analyze_url_async(url, output_path=None, source_title=""):
    """Start the context call on a daemon thread (runs in parallel with the
    download). Join with a short timeout in main.py right before Stage 3."""
    thread = threading.Thread(
        target=analyze_url,
        args=(url, output_path, source_title),
        daemon=True,
        name="gemini-context-layer",
    )
    thread.start()
    return thread


def load_context(output_path):
    """Read a previously written context blob, or None."""
    if not output_path or not os.path.exists(output_path):
        return None
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def wait_budget(elapsed: float, target: float = 35.0, floor: float = 5.0) -> float:
    """How much longer to wait for the async context blob before proceeding
    transcript-only.

    On a fresh run the thread gets the whole download+transcribe runway, so
    by picker time it has usually landed and a short join is enough. A cached
    re-run (cached transcript, no download) gives the thread ~zero head
    start, which is why the old fixed 5s cap made the context effectively
    unusable there. This waits out the remainder of `target` seconds of total
    runway (bounded below by `floor`). join() returns the instant the thread
    finishes, so a stuck 504 thread never blocks the job beyond `target`.
    """
    return max(floor, min(target, target - elapsed))
