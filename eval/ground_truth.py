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
      "track_names": {"0": "host", "1": "guest_1"},
      "notes": "optional free text"
    }

`speaker_names` only needs entries for labels that actually speak in the
span being evaluated. A label with no entry is left as its diarization
letter (still useful for matching, just less readable in reports).

`track_names` is the SAME kind of one-time label, for the ASD bake-off
(eval/asd_bakeoff.py): after running face_spine.build_face_spine on a span,
watch it once and note which face_spine track id is which person — using
the SAME names as speaker_names so the two map together. JSON object keys
are always strings, so track ids are written as "0", "1", ... here even
though face_spine's own dict keys are ints; load_track_names converts back.
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


def load_track_names(path):
    """{0: "host", ...} from a ground-truth JSON file's "track_names" key,
    with string keys converted back to the ints face_spine.py uses. Same
    fail-open behavior as load_speaker_names: missing file/key -> {}, and a
    non-integer key is skipped (not raised) since a hand-edited JSON file is
    the likeliest source of a typo here, and a bake-off run failing outright
    on one bad key is worse than ignoring it and scoring what is labeled.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    raw = data.get("track_names") or {}
    result = {}
    for k, v in raw.items():
        try:
            result[int(k)] = v
        except (TypeError, ValueError):
            continue
    return result


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
