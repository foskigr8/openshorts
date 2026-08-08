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

# The transcript-segments -> per-second-label conversion is a production
# concern (Phase 3's fusion needs it too, to know when each diarized speaker
# is talking), so it lives in speaker_fusion.py and this eval module reuses
# it rather than keeping a second copy that could drift.
from speaker_fusion import per_second_speaker_label as per_second_speaker_from_transcript  # noqa: F401,E501


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
