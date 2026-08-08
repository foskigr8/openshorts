"""Ground-truth labels for the framing eval harness.

The plan's constraint: ground truth must be cheap to produce (~8 clicks per
video, not per second) or nobody will ever build a labeled set. Diarization
already segments the video into speaker turns; the only manual step is
mapping each diarized label to a real identity ONCE. Per-second ground truth
is then derived automatically from the turns.

File format (JSON), hand-authored per video:

    {
      "video": "ptb_raw.mp4",
      "speaker_names": {"A": "host", "B": "guest_1", "C": "guest_2"},
      "notes": "optional free text"
    }

`speaker_names` only needs entries for labels that actually speak in the
span being evaluated. A label with no entry is left as its diarization
letter (still useful for matching, just less readable in reports).
"""
from __future__ import annotations

import json
import math


def load_speaker_names(path):
    """{"A": "host", ...} from a ground-truth JSON file. Missing file or
    missing key -> {} (fails open, matching the rest of the pipeline's
    convention: a missing label just means "compare by diarization letter").
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data.get("speaker_names") or {})


def per_second_speaker_from_transcript(segments, clip_start, clip_end, speaker_names=None):
    """Derive per-second ground truth from diarized transcript segments.

    segments: transcript_result["segments"], each with 'start', 'end',
        'speaker' (assemblyai backend; see transcribe_backends.py) — the
        SAME segments the rest of the pipeline reads, so this is not a
        parallel transcript, it is a read of the existing one.
    clip_start/clip_end: absolute seconds into the source video.
    speaker_names: optional {"A": "host", ...} to resolve labels to names
        (see load_speaker_names above); unresolved labels pass through as-is.

    Returns a list, index = second offset from clip_start, value = the
    speaker identity active that second, or None if no segment covers it
    (silence, cross-talk gap — not scored by speaker_on_screen_pct).
    """
    speaker_names = speaker_names or {}
    duration = int(clip_end - clip_start)
    result = [None] * max(0, duration)
    for seg in segments or []:
        s, e = float(seg.get("start", 0)), float(seg.get("end", 0))
        speaker = seg.get("speaker")
        if speaker is None or e <= clip_start or s >= clip_end:
            continue
        label = speaker_names.get(speaker, speaker)
        lo = max(0, int(s - clip_start))
        hi = min(duration, math.ceil(e - clip_start))
        for i in range(lo, hi):
            result[i] = label
    return result
