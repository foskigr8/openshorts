"""Gemini face-identity confirmation (director v2, per clip).

The diarized transcript (AssemblyAI) knows WHO talks and when; the face
spine knows WHICH faces exist. The missing link is the label->face map.
This module asks Gemini to build it once per clip: it receives labeled face
crops plus the clip's diarized transcript and returns
``{speaker_label: track_id}``. The ASR-first binding
(``speaker_fusion.active_from_identity_map``) then frames whoever the
transcript says is talking, with no per-second LR-ASD voting.

Fail-open by contract: any error, key problem, or low confidence returns
None and the caller keeps the LR-ASD fusion — identity confirmation can
only ever IMPROVE the mapping, never break the render.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional


IDENTITY_CONFIRM_MODEL = (
    os.environ.get("IDENTITY_CONFIRM_MODEL")
    or os.environ.get("GEMINI_MODEL")
    or "gemini-3.1-flash-lite")
MAX_TRACKS = 8
MAX_CROPS_PER_TRACK = 2
CROP_SIZE = 256


def _face_crop_bytes(frame, box, size=CROP_SIZE):
    """Crop a face box (with a little padding) from a BGR frame, return
    PNG bytes for the Gemini request."""
    import cv2
    fh, fw = frame.shape[:2]
    x, y, w, h = [float(v) for v in box]
    pad_x, pad_y = w * 0.15, h * 0.15
    x0 = max(0, int(x - pad_x)); y0 = max(0, int(y - pad_y))
    x1 = min(fw, int(x + w + pad_x)); y1 = min(fh, int(y + h + pad_y))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, (size, size),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", resized)
    return buf.tobytes() if ok else None


def extract_face_crops(video_path, tracks, max_tracks=MAX_TRACKS,
                       max_crops=MAX_CROPS_PER_TRACK) -> Dict[int, List[bytes]]:
    """A few representative face crops per track, from the clip itself.
    Sparse by design (1-2 per track): enough for Gemini to see each face,
    cheap to send. Skips tracks with no sampled crop."""
    import cv2
    ranked = sorted(
        ((tid, t) for tid, t in tracks.items()
         if t.get("frames") and t.get("boxes")),
        key=lambda kv: -len(kv[1]["frames"]))[:max_tracks]
    if not ranked:
        return {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    out: Dict[int, List[bytes]] = {}
    try:
        for track_id, track in ranked:
            frames = track["frames"]
            boxes = track["boxes"]
            n = min(max_crops, len(frames))
            idxs = sorted({round(i * (len(frames) - 1) / max(1, n - 1))
                           for i in range(n)})
            for i in idxs:
                t = float(frames[i])
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    continue
                png = _face_crop_bytes(frame, boxes[i])
                if png:
                    out.setdefault(track_id, []).append(png)
    finally:
        cap.release()
    return out


def _compact_transcript(segments, clip_start, clip_end,
                        max_chars=6000) -> str:
    """Clip-relative diarized lines: 'A (2.1-5.8s): text…' — the semantic
    context (roles, self-introductions, who says what) Gemini uses to match
    speakers to faces."""
    lines = []
    for seg in segments or []:
        speaker = seg.get("speaker")
        if speaker is None:
            continue
        s = float(seg.get("start") or 0)
        e = float(seg.get("end") or 0)
        if e <= clip_start or s >= clip_end:
            continue
        text = " ".join((w.get("text") or w.get("word") or "")
                        for w in (seg.get("words") or [])).strip()
        if not text:
            text = str(seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"speaker {speaker} ({max(0, s - clip_start):.1f}-"
                     f"{e - clip_start:.1f}s): {text[:160]}")
    joined = "\n".join(lines)
    return joined[:max_chars]


def _parse_mapping(text: str, tracks) -> Dict[str, int]:
    """Parse the Gemini JSON response into {speaker_label: track_id}.
    Only labels pointing at known tracks are kept — anything else is
    unconfirmed and must stay out of the map (no guessing)."""
    import gemini_worker
    try:
        parsed = gemini_worker._parse_json_response_text(text)
    except Exception:
        return {}
    mapping = (parsed or {}).get("mapping") or {}
    out = {}
    for label, face_label in mapping.items():
        face_label = str(face_label or "").strip()
        if face_label.startswith("F") and face_label[1:].isdigit():
            track_id = int(face_label[1:])
        elif face_label.isdigit():
            track_id = int(face_label)
        else:
            continue
        if track_id in tracks:
            out[str(label)] = track_id
    return out


def confirm_clip_identities(video_path, tracks, segments,
                            clip_start=0.0, clip_end=None,
                            model: Optional[str] = None) -> Optional[Dict[str, int]]:
    """One Gemini call: labeled face crops + the diarized transcript ->
    {speaker_label: track_id}. None on any failure (fail-open)."""
    # Default OFF: the owner fell back to base. Opt in with IDENTITY_CONFIRM=1.
    if os.environ.get("IDENTITY_CONFIRM", "0").strip().lower() in (
            "0", "false", "no", "off"):
        return None
    if not tracks or not segments:
        return None
    crops = extract_face_crops(video_path, tracks)
    if not crops:
        return None
    compact = _compact_transcript(segments, clip_start, clip_end or 1e18)
    if not compact:
        return None

    from google.genai import types as genai_types
    face_labels = []
    parts = []
    for track_id in sorted(crops):
        label = f"F{track_id}"
        face_labels.append(label)
        for png in crops[track_id]:
            parts.append(genai_types.Part.from_bytes(
                data=png, mime_type="image/png"))
    prompt = (
        "You are confirming WHO is WHO in a short video clip. Below are "
        f"face crops labeled {', '.join(face_labels)} (each label is one "
        "person, possibly repeated). The clip's diarized transcript follows "
        "with speaker labels and times.\n\n"
        f"{compact}\n\n"
        "Match each transcript speaker label to a face label using the "
        "dialogue (self-introductions, roles like the host, who says what) "
        "and visual consistency. Respond ONLY with JSON:\n"
        '{"mapping": {"A": "F1", "B": "F2"}}\n'
        "Rules: map a speaker only when you are reasonably confident; leave "
        "a speaker OUT of the mapping when you cannot tell which face is "
        "them. Never invent face labels.")
    # Part(text=...) (the dataclass field) instead of Part.from_text(): the
    # installed google-genai on Kaggle rejects from_text() with
    # 'takes 1 positional argument but 2 were given' (version drift) — the
    # field constructor is stable across versions.
    parts.append(genai_types.Part(text=prompt))

    model = model or IDENTITY_CONFIRM_MODEL
    import gemini_worker
    import gemini_pool

    pool = gemini_pool.pool_from_env()
    last_exc = None
    for _ in range(3):
        key = pool.acquire()
        if not key:
            break
        try:
            client = gemini_worker.make_client(key)
            response = gemini_pool.generate_with_fallback(
                client, model, parts, max_attempts=1,
                log=lambda msg: print(f"   ⚠️ Identity-confirm: {msg}"))
            gemini_worker.raise_if_blocked(response)
            text = gemini_worker._get_response_text(response)
            mapping = _parse_mapping(text, tracks)
            if mapping:
                print(f"   🪪 Gemini identity confirm: "
                      f"{len(mapping)} speaker(s) mapped "
                      f"({', '.join(f'{k}->{v}' for k, v in mapping.items())})")
                return mapping
        except Exception as e:
            last_exc = e
            pool.mark_bad(key)
            continue
    print(f"   ⚠️ Identity confirmation unavailable "
          f"({type(last_exc).__name__}: {last_exc}) — using LR-ASD fusion")
    return None
