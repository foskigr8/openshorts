"""
Pure helpers for the Gemini clip-selection pipeline.

Standard-library only so both main.py and gemini_worker.py can import it and
the logic stays unit-testable without the heavy video dependencies.
"""
import re


# USD per 1M tokens (input, output incl. thinking), from ai.google.dev pricing.
MODEL_PRICES = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),  # deprecated (shut down 2026-06-01)
}

# USD per 1M tokens (input cache-miss, output), from
# api-docs.deepseek.com/quick_start/pricing (checked 2026-07-30). Cache-hit
# input tokens are far cheaper (~50x) but we conservatively cost every run at
# the cache-miss rate since hit/miss depends on DeepSeek's own cache state,
# not something this pipeline controls run-to-run.
DEEPSEEK_MODEL_PRICES = {
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (0.435, 0.87),
}


def _lookup_prices(table, model_name):
    """Longest-prefix match against a {model_prefix: (input, output)} table."""
    name = str(model_name or "").lower()
    best_key = None
    for key in table:
        if name.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return table[best_key] if best_key else None


def lookup_model_prices(model_name):
    """Longest-prefix match against MODEL_PRICES; None if unknown."""
    return _lookup_prices(MODEL_PRICES, model_name)


def lookup_deepseek_model_prices(model_name):
    """Longest-prefix match against DEEPSEEK_MODEL_PRICES; None if unknown."""
    return _lookup_prices(DEEPSEEK_MODEL_PRICES, model_name)


def compact_words(words, precision=2):
    """Round word timestamps for prompts — full float precision wastes tokens."""
    return [
        {
            "w": w.get("w", ""),
            "s": round(float(w.get("s", 0)), precision),
            "e": round(float(w.get("e", 0)), precision),
        }
        for w in words
    ]


def build_transcript_windows(transcript_result, video_duration,
                             window_seconds=90, overlap_seconds=30):
    """
    Build scoring windows aligned to Whisper segment boundaries, so a sentence
    (and usually a viral moment) is never cut in half mid-window. Windows grow
    segment by segment to roughly window_seconds (up to 1.25x for the closing
    segment) and the next window starts ~overlap_seconds before the previous
    end, also snapped to a segment start.
    """
    segments = []
    for segment in transcript_result.get("segments", []):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        segments.append((float(segment.get("start", 0)), float(segment.get("end", 0)), text))

    windows = []
    window_index = 1
    i = 0
    n = len(segments)
    while i < n:
        w_start = segments[i][0]
        j = i
        # Extend while the NEXT segment still fits within a tolerant cap, so the
        # window closes on a segment boundary near window_seconds.
        while j + 1 < n and segments[j + 1][1] - w_start <= window_seconds * 1.25:
            j += 1
            if segments[j][1] - w_start >= window_seconds:
                break
        w_end = segments[j][1]
        windows.append({
            "id": f"window_{window_index:03d}",
            "start": round(w_start, 3),
            "end": round(w_end, 3),
            "text": " ".join(seg[2] for seg in segments[i:j + 1]),
        })
        window_index += 1

        if j >= n - 1:
            break
        # Next window starts at the first segment beginning after (end - overlap),
        # but always makes progress.
        target = w_end - overlap_seconds
        k = i + 1
        while k <= j and segments[k][0] < target:
            k += 1
        i = max(k, i + 1)

    if not windows:
        windows.append({
            "id": "window_001",
            "start": 0.0,
            "end": round(float(video_duration), 3),
            "text": str(transcript_result.get("text", "") or ""),
        })
    return windows


_SENTENCE_END_RE = re.compile(r"[.?!]+[\"')\]]*\s*$")


def sentence_boundaries(words):
    """(sentence_start_times, sentence_end_times) from punctuated word tokens.

    Word-level snapping alone lands cuts on a clean word edge, but a clean word
    edge is usually still the MIDDLE of a sentence — which is why clips could
    open on a half-thought or stop before one finished. The transcript already
    carries terminal punctuation (measured: ~1.8k sentence-enders in an 8.4k
    word transcript), so real sentence edges are available for free; nothing
    was reading them.

    A sentence STARTS at the word following a terminal-punctuation word, and
    the first word overall. A sentence ENDS at any word carrying terminal
    punctuation.
    """
    starts, ends = [], []
    prev_terminal = True  # the first word opens a sentence
    for w in words:
        text = str(w.get("w", "")).strip()
        if prev_terminal and text:
            starts.append(float(w.get("s", 0)))
        if text and _SENTENCE_END_RE.search(text):
            ends.append(float(w.get("e", 0)))
            prev_terminal = True
        elif text:
            prev_terminal = False
    return starts, ends


def _nearest_within(values, target, window, prefer_after=False, prefer_before=False):
    """Closest value to `target` within `window`, or None.

    `prefer_after` biases toward a boundary at/after the target, so an end can
    complete the sentence in progress rather than truncating it — but only when
    the later option isn't dramatically further away than an earlier one.
    `prefer_before` is the mirror for START boundaries: moving a start later can
    only REMOVE context (round-5 spec 1.2 — a question pulled back into the
    clip was being re-truncated because the answer's sentence start was 0.7s
    away while the question's was ~4.4s away, and nearest-distance picked the
    later one), so bias toward the nearest boundary at/before the target and
    accept a later one only when it is dramatically closer (the same
    ``* 1.6 + 0.35`` tolerance shape as prefer_after, inverted).
    """
    near = [v for v in values if abs(v - target) <= window]
    if not near:
        return None
    if prefer_before:
        before = [v for v in near if v <= target]
        if before:
            best_before = max(before)
            after = [v for v in near if v > target]
            # Take the later boundary only if it is clearly closer.
            if not after or (target - best_before) <= (min(after) - target) * 1.6 + 0.35:
                return best_before
    if prefer_after:
        after = [v for v in near if v >= target]
        if after:
            best_after = min(after)
            before = [v for v in near if v < target]
            # Take the earlier boundary only if it is clearly closer.
            if not before or (best_after - target) <= (target - max(before)) * 1.6 + 0.35:
                return best_after
    return min(near, key=lambda v: abs(v - target))


def snap_clip_to_words(start, end, words, video_duration,
                       min_duration=15.0, max_duration=180.0,
                       search_window=1.5, max_lead=0.35, max_tail=0.45,
                       sentence_window=2.5, context_start=None):
    """
    Snap LLM-proposed clip boundaries onto real word boundaries plus a bit
    of the surrounding silence. LLMs are bad at millisecond arithmetic; the
    word-level timestamps are ground truth, so cuts land in pauses instead of
    mid-word.

    min_duration=15 / max_duration=180 are real working bounds, not just
    crash protection — an earlier "no target length, just guidance" design
    let DeepSeek's narrative-arc selection default to near-full-video spans
    in practice (confirmed in prod, 30-jul-2026: identical ~300s "whole
    video" candidate returned even after strengthening the prompt's
    tightness guidance). A comparable open-source tool
    (github.com/NaufalRizqullah/opensource-clipping) hard-caps at 20-179s
    with no exception, and its output was noticeably tighter — bounds that
    are actually enforced beat guidance the model is free to ignore.

    words: [{'w','s','e'}, ...] for the whole video, sorted by start.
    Returns (start, end); falls back to the input if no words are nearby or
    snapping cannot satisfy the duration bounds.
    """
    original = (round(float(start), 3), round(float(end), 3))
    if not words:
        return original

    starts = [float(w.get("s", 0)) for w in words]
    ends = [float(w.get("e", 0)) for w in words]

    # Prefer a real sentence edge when one is close enough, so a clip opens on
    # a complete thought and closes on a finished one. Falls through to
    # word-level snapping below whenever no sentence edge is in range, so this
    # can only ever improve a boundary, never strand one.
    sent_starts, sent_ends = sentence_boundaries(words)
    snapped_start = _nearest_within(sent_starts, float(start), sentence_window,
                                    prefer_before=True)
    if context_start is not None and snapped_start is not None:
        # Context lock (round-5 spec 1.2): when the question backstop pulled
        # the start back to include the question, snapping must never move it
        # forward past that point again — later is always a re-truncation.
        snapped_start = min(snapped_start, context_start)
    snapped_end = _nearest_within(sent_ends, float(end), sentence_window,
                                  prefer_after=True)

    if snapped_start is not None and snapped_end is not None:
        cand_start = max(0.0, snapped_start - min(max_lead, 0.2))
        cand_end = min(float(video_duration), snapped_end + min(max_tail, 0.3))
        # Only accept the sentence-aligned pair if it still satisfies the
        # duration contract; otherwise fall back to word snapping.
        if min_duration <= (cand_end - cand_start) <= max_duration:
            return (round(cand_start, 3), round(cand_end, 3))

    # START: snap to the nearest word start, then lead into the silence before it.
    new_start = float(start)
    candidates = [s for s in starts if abs(s - new_start) <= search_window]
    if candidates:
        word_start = min(candidates, key=lambda s: abs(s - new_start))
        prev_ends = [e for e in ends if e <= word_start]
        if prev_ends:
            gap = max(0.0, word_start - max(prev_ends))
            lead = min(max_lead, gap / 2)
        else:
            lead = max_lead
        new_start = max(0.0, word_start - lead)
    if context_start is not None:
        new_start = min(new_start, context_start)

    # END: snap to the nearest word end, then trail into the silence after it.
    new_end = float(end)
    candidates = [e for e in ends if abs(e - new_end) <= search_window]
    if candidates:
        word_end = min(candidates, key=lambda e: abs(e - new_end))
        next_starts = [s for s in starts if s >= word_end]
        if next_starts:
            gap = max(0.0, min(next_starts) - word_end)
            tail = min(max_tail, gap / 2)
        else:
            tail = max_tail
        new_end = min(float(video_duration), word_end + tail)

    # Repair duration bounds while staying on word boundaries.
    if new_end - new_start < min_duration:
        target = new_start + min_duration
        later = sorted(e for e in ends if e >= target)
        if later and later[0] - new_start <= max_duration:
            new_end = min(float(video_duration), later[0] + 0.2)
        else:
            return original
    if new_end - new_start > max_duration:
        target = new_start + max_duration
        earlier = [e for e in ends if new_start < e <= target]
        new_end = (max(earlier) + 0.2) if earlier else target
        new_end = min(new_end, new_start + max_duration, float(video_duration))

    if new_end <= new_start or new_end - new_start < min_duration:
        return original
    return (round(new_start, 3), round(new_end, 3))
