"""Transcription backends: NVIDIA Parakeet (onnx-asr) with faster-whisper fallback.

Every caller goes through transcribe_media(), which returns the transcript
contract the whole pipeline depends on:

    {
      "text": str,          # full punctuated transcript
      "language": str,      # whisper-style short code ("es", "en", ...)
      "segments": [
        {"start": float, "end": float, "text": str,
         "words": [{"word": str, "start": float, "end": float}, ...]},
      ],
    }

Invariants the consumers rely on (clip cutting, karaoke subtitles, Remotion):
  - word["word"] carries a LEADING SPACE on true word starts; continuation
    fragments are merged into their base word (merge_continuation_words).
  - all numerics are native Python floats (json.dump of the transcript).
  - words sorted by start, segments chronological, absolute file timestamps.

TRANSCRIBE_BACKEND env: "whisper" (default) | "parakeet" | "assemblyai".
The parakeet path falls back to whisper automatically when the model errors,
produces no usable words, or the detected language is outside its 25
supported European languages (e.g. Japanese/Chinese/Arabic uploads).
The assemblyai path (a real external paid API, not just a different local
model) falls back to whisper on ANY error — bad/missing key, timeout,
network failure. It additionally requests speaker-label utterances and
per-sentence sentiment analysis, which the narrative clip-selection stage
(deepseek_worker.py) consumes as extra signal on top of the base contract.
GPU whisper in turn falls back to CPU whisper on CUDA errors (VRAM is shared
with other models on the host, so loads can OOM under load).
"""
import os
import subprocess
import tempfile
import threading
import time

import httpx

from subtitles import (
    get_whisper_config,
    WHISPER_TRANSCRIBE_PARAMS,
    merge_continuation_words,
)

PARAKEET_MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"

# The 25 European languages parakeet-tdt-0.6b-v3 supports (ISO 639-1).
PARAKEET_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru",
    "uk",
}

# Serializes GPU transcription across concurrent jobs so N jobs can't stack
# N model contexts / decode batches in VRAM. CPU whisper stays ungated
# (CTranslate2 models are thread-safe and that matches the old behavior).
_ASR_GATE = threading.Semaphore(int(os.environ.get("ASR_GPU_CONCURRENCY", "1")))


class _NullGate:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_GATE = _NullGate()


class _TranscribeProgress:
    """Emits '🎙️ Transcribing… NN% (Xs)' lines at 25% steps.

    These are the only transcription lines cloud users see (log_view keeps
    them), so they must stay free of technical detail.
    """

    def __init__(self, total_seconds):
        self.total = max(float(total_seconds or 0), 0.0)
        self.started = time.time()
        self.next_pct = 25

    def update(self, position_seconds):
        if self.total <= 0:
            return
        pct = min(int(position_seconds / self.total * 100), 100)
        while pct >= self.next_pct and self.next_pct <= 100:
            elapsed = int(time.time() - self.started)
            print(f"🎙️ Transcribing… {self.next_pct}% ({elapsed}s)", flush=True)
            self.next_pct += 25

# --- whisper singleton ------------------------------------------------------

_whisper_model = None
_whisper_key = None
_whisper_lock = threading.Lock()
# Set after a CUDA failure (e.g. VRAM exhausted by other models on the GPU)
# so every later transcription goes straight to CPU instead of re-failing.
_whisper_force_cpu = False


def _get_whisper_model():
    """Process-wide WhisperModel singleton, rebuilt if the env config changes.

    Keeping the model resident avoids a full reload per transcription (which
    on GPU would also mean re-allocating a couple of GB of VRAM per job).
    """
    global _whisper_model, _whisper_key
    cfg = get_whisper_config()
    if _whisper_force_cpu:
        cfg["device"] = "cpu"
        cfg["compute_type"] = "int8"
    key = (cfg["model_size"], cfg["device"], cfg["compute_type"])
    with _whisper_lock:
        if _whisper_model is None or _whisper_key != key:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(key[0], device=key[1], compute_type=key[2])
            _whisper_key = key
    return _whisper_model, cfg["device"]


def _run_whisper_once(media_path, **params):
    model, device = _get_whisper_model()
    gate = _ASR_GATE if device != "cpu" else _NULL_GATE
    with gate:
        segments, info = model.transcribe(media_path, **params)
        progress = _TranscribeProgress(getattr(info, "duration", 0))
        materialized = []
        for segment in segments:
            materialized.append(segment)
            progress.update(segment.end)
        # VAD trims trailing silence, so the last segment can end short of the
        # media duration — force the 100% line.
        progress.update(progress.total)
        return materialized, info


def run_whisper_transcription(media_path, **params):
    """Transcribe and FULLY materialize the segments inside the GPU gate.

    faster-whisper returns a lazy generator — decoding happens while
    iterating, so the gate must wrap list(segments), not just transcribe().
    Returns (segments_list, info).

    A CUDA failure (model load OOM or mid-decode) retries once on CPU and
    pins CPU for the rest of the process — the GPU is shared with other
    models, so a job must degrade instead of dying when VRAM runs out.
    """
    global _whisper_model, _whisper_force_cpu
    try:
        return _run_whisper_once(media_path, **params)
    except RuntimeError as e:
        if _whisper_force_cpu or "cuda" not in str(e).lower():
            raise
        print(f"⚠️ [ASR] whisper GPU failed ({e}) — retrying on CPU", flush=True)
        _whisper_force_cpu = True
        with _whisper_lock:
            _whisper_model = None  # drop the GPU model to release its VRAM
        return _run_whisper_once(media_path, **params)


def _transcribe_with_whisper(media_path):
    segments, info = run_whisper_transcription(media_path, **WHISPER_TRANSCRIBE_PARAMS)

    out_segments = []
    text_parts = []
    for segment in segments:
        words = [
            {"word": w.word, "start": float(w.start), "end": float(w.end)}
            for w in (segment.words or [])
        ]
        out_segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text,
            "words": merge_continuation_words(words),
        })
        text_parts.append(segment.text.strip())

    return {
        "text": " ".join(part for part in text_parts if part),
        "language": info.language,
        "segments": out_segments,
    }


# --- parakeet ---------------------------------------------------------------

_parakeet_model = None
_parakeet_lock = threading.Lock()


def _get_parakeet_model():
    global _parakeet_model
    with _parakeet_lock:
        if _parakeet_model is None:
            import onnx_asr
            model = onnx_asr.load_model(
                PARAKEET_MODEL_ID,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            vad = onnx_asr.load_vad("silero")
            _parakeet_model = model.with_vad(vad).with_timestamps()
    return _parakeet_model


def _extract_wav(media_path):
    """Parakeet wants 16kHz mono PCM wav; ffmpeg-extract to a temp file."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="asr_")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", media_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=1800)
    return wav_path


def _words_from_tokens(tokens, timestamps, seg_start, seg_end):
    """Group parakeet BPE tokens into words with absolute timestamps.

    Verified on the prod model: tokens already carry the leading-space
    word-start convention (" T", "odo", " el", ...) and timestamps are token
    START times in seconds relative to the VAD segment. A token without a
    leading space (subword continuations, punctuation like ",") belongs to
    the previous word — same semantics merge_continuation_words expects.
    Word end is inferred: next word's start, capped near the word's last
    token so a long inter-word silence doesn't stretch the highlight.
    """
    words = []
    last_token_ts = []
    for token, ts in zip(tokens, timestamps):
        if not token:
            continue
        abs_ts = float(ts) + seg_start
        if token.startswith(" ") or not words:
            words.append({
                "word": token if token.startswith(" ") else " " + token,
                "start": abs_ts,
            })
            last_token_ts.append(abs_ts)
        else:
            words[-1]["word"] += token
            last_token_ts[-1] = abs_ts

    for i, word in enumerate(words):
        next_start = words[i + 1]["start"] if i + 1 < len(words) else seg_end
        cap = last_token_ts[i] + 0.6
        word["end"] = float(max(word["start"] + 0.05, min(next_start, cap)))

    return words


def _transcribe_with_parakeet(media_path):
    model = _get_parakeet_model()
    wav_path = _extract_wav(media_path)
    try:
        # 16kHz mono s16le wav -> 32000 bytes per second of audio.
        try:
            duration = os.path.getsize(wav_path) / 32000.0
        except OSError:
            duration = 0.0
        with _ASR_GATE:
            progress = _TranscribeProgress(duration)
            results = []
            for seg in model.recognize(wav_path):
                results.append(seg)
                progress.update(float(seg.end))
            progress.update(progress.total)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    out_segments = []
    text_parts = []
    for seg in results:
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        seg_text = str(seg.text or "").strip()
        if not seg_text:
            continue
        out_segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "words": _words_from_tokens(
                list(seg.tokens or []), list(seg.timestamps or []),
                seg_start, seg_end,
            ),
        })
        text_parts.append(seg_text)

    text = " ".join(text_parts)
    return {
        "text": text,
        "language": _detect_language(text),
        "segments": out_segments,
    }


def _detect_language(text):
    """Parakeet doesn't report a language; classify the transcribed text.

    py3langid is pure-Python and returns ISO 639-1 codes compatible with the
    whisper codes the pipeline expects (thumbnail titles, Gemini prompts).
    """
    sample = (text or "").strip()
    if len(sample) < 20:
        return "en"
    try:
        import py3langid
        lang, _score = py3langid.classify(sample[:4000])
        return lang
    except Exception:
        return "en"


def _parakeet_fallback_reason(transcript, duration_hint=None):
    """Return why the parakeet result is untrustworthy, or None if it's fine."""
    segments = transcript.get("segments") or []
    total_words = sum(len(s.get("words") or []) for s in segments)
    if total_words == 0:
        return "no words recognized"
    language = transcript.get("language")
    if language not in PARAKEET_LANGS:
        return f"language '{language}' outside parakeet's supported set"
    duration = duration_hint or (segments[-1]["end"] if segments else 0)
    # Real speech averages >100 wpm; under ~12 wpm on a long video means the
    # audio was mostly not recognized (e.g. unsupported language or music).
    if duration > 60 and total_words < duration * 0.2:
        return f"only {total_words} words in {duration:.0f}s of audio"
    return None


# --- assemblyai --------------------------------------------------------------

_ASSEMBLYAI_API_BASE = "https://api.assemblyai.com"
_ASSEMBLYAI_POLL_DEADLINE_S = 600  # a real network dependency — must not hang a job forever
_ASSEMBLYAI_POLL_INTERVAL_S = 3


def _word(w, speaker=None):
    """AssemblyAI word -> the repo's word dict: seconds, whisper-style leading
    space (AssemblyAI has no such convention on its own). `speaker` is a
    transient tag (stripped before the word dict is returned to callers) used
    only to derive each resulting segment's speaker after chunking — see
    _assemblyai_words_to_segments."""
    text = str(w.get("text", "")).strip()
    d = {
        "word": (" " + text) if text else "",
        "start": float(w.get("start", 0)) / 1000.0,
        "end": float(w.get("end", 0)) / 1000.0,
    }
    if speaker is not None:
        d["_speaker"] = speaker
    return d


def _assemblyai_sentiment_lookup(sentiment_results):
    """Returns a fast (seg_start, seg_end) -> "+"/"0"/"-" lookup function.

    Sentiment analysis is per-*sentence*, not per-segment, so segments are
    tagged by whichever sentiment span covers their midpoint — good enough
    for "is this an emotional beat" without needing exact overlap logic.
    """
    spans = []
    for s in (sentiment_results or []):
        label = str(s.get("sentiment", "NEUTRAL")).upper()
        tag = "+" if label == "POSITIVE" else "-" if label == "NEGATIVE" else "0"
        spans.append((float(s.get("start", 0)) / 1000.0, float(s.get("end", 0)) / 1000.0, tag))

    def lookup(seg_start, seg_end):
        mid = (seg_start + seg_end) / 2.0
        for s, e, tag in spans:
            if s <= mid <= e:
                return tag
        return "0"

    return lookup


_MAX_SEGMENT_GAP_S = 1.2
_MAX_SEGMENT_WORDS = 20


def _chunk_words_into_segments(word_dicts, force_boundary_after=None):
    """Groups a flat, chronological list of {'word','start','end'} dicts into
    segments, splitting on whichever comes first: a >1.2s silence gap, a
    20-word ceiling, or an explicit forced boundary (an AssemblyAI
    speaker/utterance change, when known). force_boundary_after: a set of
    indices into word_dicts after which a split is forced regardless of
    gap/word-count — this is what lets continuous single-speaker narration
    (one giant AssemblyAI utterance spanning an entire short video) still get
    proper internal structure instead of collapsing into one segment.
    """
    force_boundary_after = force_boundary_after or set()
    segments = []
    current = []

    def _flush():
        if not current:
            return
        segments.append({
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "text": "".join(w["word"] for w in current).strip(),
            "words": merge_continuation_words(list(current)),
        })

    for i, word in enumerate(word_dicts):
        if current:
            gap = word["start"] - current[-1]["end"]
            if gap > _MAX_SEGMENT_GAP_S or len(current) >= _MAX_SEGMENT_WORDS or (i - 1) in force_boundary_after:
                _flush()
                current = []
        current.append(word)
    _flush()
    return segments


def _assemblyai_highlight_lookup(highlights_result):
    """Returns a (seg_start, seg_end) -> bool function: True if any
    AssemblyAI Auto-Highlights key-phrase occurrence overlaps the segment.
    Auto Highlights is AssemblyAI's own algorithmic salience detection
    (statistically/linguistically standout phrases, not sentiment) — a
    second, independent signal for where the "important" moments are."""
    spans = []
    for h in (highlights_result or []):
        for ts in (h.get("timestamps") or []):
            spans.append((float(ts.get("start", 0)) / 1000.0, float(ts.get("end", 0)) / 1000.0))

    def overlaps(seg_start, seg_end):
        return any(s < seg_end and e > seg_start for s, e in spans)

    return overlaps


def _assemblyai_words_to_segments(words, utterances=None, sentiment_results=None, highlights_result=None):
    """Group AssemblyAI's flat word list into the repo's segment contract.

    Utterance (speaker-turn) boundaries, when available, are treated as
    FORCED splits — not as the sole segmentation unit. A single continuous
    narrator with no speaker changes can otherwise come back as ONE
    AssemblyAI utterance spanning an entire video, which leaves nothing for
    the narrative-selection stage to reason about (confirmed in prod,
    30-jul-2026: a 208s solo-narration video produced exactly one utterance,
    and the clip-selection pipeline downstream had no internal structure to
    find a tighter span, so it just handed back almost the whole video).
    The same pause-gap/word-count chunker now runs regardless of whether
    utterances are present, so long stretches always get real internal
    structure; utterance boundaries just add extra forced splits on top.

    Each segment additionally carries a compact "sentiment" tag ("+"/"0"/"-")
    and a "highlight" bool (AssemblyAI's own Auto-Highlights salience
    detection) — extra metadata beyond the base transcript contract,
    consumed by the narrative clip-selection stage as free signal instead of
    having to re-derive tone/importance from raw text.
    """
    sentiment_at = _assemblyai_sentiment_lookup(sentiment_results)
    highlight_at = _assemblyai_highlight_lookup(highlights_result)

    if utterances:
        flat = []
        force_boundary_after = set()
        for utt in utterances:
            speaker = utt.get("speaker")
            utt_words = [_word(w, speaker) for w in (utt.get("words") or []) if str(w.get("text", "")).strip()]
            flat.extend(utt_words)
            if utt_words:
                force_boundary_after.add(len(flat) - 1)
    else:
        flat = [_word(w) for w in (words or []) if str(w.get("text", "")).strip()]
        force_boundary_after = set()

    segments = _chunk_words_into_segments(flat, force_boundary_after)
    for seg in segments:
        seg["sentiment"] = sentiment_at(seg["start"], seg["end"])
        seg["highlight"] = highlight_at(seg["start"], seg["end"])
        # Speaker (AssemblyAI diarization, when available): every word in a
        # chunked segment shares one speaker, since an utterance/speaker
        # change is always a forced segmentation boundary above. Derive from
        # the first word's transient tag, then strip it — word dicts must
        # keep the documented {'word','start','end'} contract everywhere else
        # in the pipeline (subtitles.py, Remotion) that reads them.
        seg_words = seg.get("words") or []
        seg["speaker"] = seg_words[0].get("_speaker") if seg_words else None
        for w in seg_words:
            w.pop("_speaker", None)
    return segments


def _transcribe_with_assemblyai(media_path):
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("ASSEMBLYAI_API_KEY not set")
    headers = {"authorization": api_key}

    with open(media_path, "rb") as f:
        with httpx.Client(timeout=300.0) as client:
            upload_resp = client.post(
                f"{_ASSEMBLYAI_API_BASE}/v2/upload", headers=headers, content=f.read())
    upload_resp.raise_for_status()
    upload_url = upload_resp.json()["upload_url"]

    submit_body = {
        "audio_url": upload_url,
        "speaker_labels": True,
        "sentiment_analysis": True,
        "language_detection": True,
        # Auto Highlights: AssemblyAI's own algorithmic detection of
        # standout key phrases (statistical/linguistic salience, not an
        # LLM call — unlike Auto Chapters, this is a plain flag on the same
        # job, no extra cost). Extra editorial signal for DeepSeek on top of
        # sentiment: "these specific phrases stood out" vs. just "this
        # segment's overall tone."
        "auto_highlights": True,
    }
    with httpx.Client(timeout=30.0) as client:
        submit_resp = client.post(
            f"{_ASSEMBLYAI_API_BASE}/v2/transcript",
            headers={**headers, "content-type": "application/json"},
            json=submit_body,
        )
    submit_resp.raise_for_status()
    transcript_id = submit_resp.json()["id"]

    deadline = time.time() + _ASSEMBLYAI_POLL_DEADLINE_S
    data = None
    with httpx.Client(timeout=30.0) as client:
        while True:
            poll_resp = client.get(
                f"{_ASSEMBLYAI_API_BASE}/v2/transcript/{transcript_id}", headers=headers)
            poll_resp.raise_for_status()
            data = poll_resp.json()
            status = data.get("status")
            if status == "completed":
                break
            if status == "error":
                raise RuntimeError(f"AssemblyAI transcription error: {data.get('error')}")
            if time.time() > deadline:
                raise TimeoutError("AssemblyAI transcription timed out after 10 minutes")
            time.sleep(_ASSEMBLYAI_POLL_INTERVAL_S)

    segments = _assemblyai_words_to_segments(
        data.get("words") or [], data.get("utterances"), data.get("sentiment_analysis_results"),
        (data.get("auto_highlights_result") or {}).get("results"))

    return {
        "text": str(data.get("text", "")).strip(),
        "language": data.get("language_code") or "en",
        "segments": segments,
    }


# --- public entry point -----------------------------------------------------

class NoAudioError(Exception):
    """The media has no audio track — nothing to transcribe."""


def _has_audio_stream(media_path) -> bool:
    """True if the file has at least one audio stream (ffprobe)."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", media_path],
            capture_output=True, text=True, timeout=60,
        )
        return bool(out.stdout.strip())
    except Exception:
        return True  # probe failed — don't block, let the backend try


def _select_backend():
    """Which ASR backend to use.

    TRANSCRIBE_BACKEND wins when set. When it is NOT set but an AssemblyAI key
    is configured, prefer assemblyai over the historical "whisper" default:
    supplying the key and getting local whisper anyway is never what anyone
    means by it, and the cost of the silent default is not just speed. Whisper
    produces no diarization, and diarization is the speaker-binding pipeline's
    diarized-tier evidence — the signal that stops the camera sitting on the
    wrong person in multi-speaker footage. Measured on Kaggle 5-aug-2026 (job
    b86b8c5a): 254s of a ~420s job spent in local whisper, with the framing
    evidence line reading "lip-sync 97%, directed 3%" and diarized at 0%.

    The fallback chain below is unchanged, so an AssemblyAI outage still lands
    on whisper.
    """
    configured = os.environ.get("TRANSCRIBE_BACKEND", "").strip().lower()
    if configured:
        return configured
    if os.environ.get("ASSEMBLYAI_API_KEY", "").strip():
        print("🎙️ [ASR] ASSEMBLYAI_API_KEY is set and TRANSCRIBE_BACKEND is not "
              "— using assemblyai (API + diarization). Set "
              "TRANSCRIBE_BACKEND=whisper to force local transcription.")
        return "assemblyai"
    return "whisper"


def transcribe_media(media_path):
    """Transcribe with the configured backend, falling back to whisper."""
    # Silent videos (AI-generated clips, muted screen recordings) have no audio
    # stream; every ASR backend then crashes deep inside libav with an opaque
    # "tuple index out of range". Detect it up front and fail with a clear,
    # actionable reason instead.
    if not _has_audio_stream(media_path):
        raise NoAudioError(
            "This video has no audio track. OpenShorts finds viral moments from "
            "speech, so it needs a video with audio.")

    backend = _select_backend()

    if backend == "parakeet":
        try:
            transcript = _transcribe_with_parakeet(media_path)
            reason = _parakeet_fallback_reason(transcript)
            if reason is None:
                print(f"🎙️ [ASR] parakeet ok: lang={transcript['language']} "
                      f"segments={len(transcript['segments'])}")
                return transcript
            print(f"⚠️ DEGRADED OUTPUT: [ASR] parakeet result rejected ({reason}) — "
                  f"falling back to whisper")
        except Exception as e:
            print(f"⚠️ DEGRADED OUTPUT: [ASR] parakeet failed ({type(e).__name__}: {e}) — "
                  f"falling back to whisper")

    if backend == "assemblyai":
        try:
            transcript = _transcribe_with_assemblyai(media_path)
            print(f"🎙️ [ASR] assemblyai ok: lang={transcript['language']} "
                  f"segments={len(transcript['segments'])}")
            return transcript
        except Exception as e:
            print(f"⚠️ DEGRADED OUTPUT: [ASR] assemblyai failed ({type(e).__name__}: {e}) — "
                  f"falling back to whisper (slow, no diarization)")

    return _transcribe_with_whisper(media_path)
