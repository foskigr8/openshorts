"""Ground-truth labels for the Stage 3 selection eval harness.

Same cheap-to-produce constraint eval/ground_truth.py established for
framing: this must be cheap enough to label that someone actually builds a
set, not an exhaustive/adversarial-proof corpus. For one source video, watch
it once and note the spans that should become clips (and, optionally, spans
that clearly should NOT) — a handful of entries, not a per-second label.

File format (JSON), hand-authored per video:

    {
      "video": "some_source_id",
      "transcript_metadata": "eval/selection/fixtures/some_source_metadata.json",
      "expected_clips": [
        {"start": 120.5, "end": 165.2, "note": "the balloon-pop reaction"}
      ],
      "must_not_include": [
        {"start": 300.0, "end": 340.0, "note": "dead air, nothing happens"}
      ],
      "notes": "optional free text"
    }

`transcript_metadata` points at a job's metadata.json (or a standalone
transcript fixture in the same {"segments": [...]} shape) — the same file
shape eval/run.py's `_load_transcript` already reads for the framing
harness, reused here rather than inventing a second transcript format.

`expected_clips`/`must_not_include` are in SOURCE-VIDEO seconds, matching the
`start`/`end` fields Stage 3 itself produces — no unit conversion needed
anywhere downstream.
"""
from __future__ import annotations

import json


def load_ground_truth(path):
    """Ground-truth dict from a JSON file, or an empty-but-valid shape on any
    read/parse failure — fails open like eval/ground_truth.py's loaders, so a
    missing/malformed fixture degrades to "nothing to score against" rather
    than crashing the whole harness run.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"video": None, "expected_clips": [], "must_not_include": []}
    return {
        "video": data.get("video"),
        "transcript_metadata": data.get("transcript_metadata"),
        "expected_clips": [_as_span(c) for c in (data.get("expected_clips") or [])],
        "must_not_include": [_as_span(c) for c in (data.get("must_not_include") or [])],
        "notes": data.get("notes", ""),
    }


def _as_span(entry):
    return {
        "start": float(entry.get("start", 0.0)),
        "end": float(entry.get("end", 0.0)),
        "note": str(entry.get("note", "")),
    }


def load_transcript_fixture(path):
    """{"segments": [...]} from a job metadata.json or a standalone
    transcript fixture — same shape/tolerance as eval/run.py's
    _load_transcript, duplicated here (not imported) because that function
    is scoped to the framing harness's CLI and this package intentionally
    has no dependency on it.
    """
    with open(path) as f:
        meta = json.load(f)
    transcript = meta.get("transcript", meta)
    return transcript if isinstance(transcript, dict) else {"segments": transcript}
