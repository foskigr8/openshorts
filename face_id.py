"""Optional named-identity enrichment layer (face ID).

The integration guide's biggest recommendation is speaker-attribution
correctness: OpenShorts already knows *which tracked face* is speaking
(LR-ASD) and *which anonymous speaker label* is talking (diarization), but it
does not know *who* either of them is. This module adds names where confident,
using InsightFace (ArcFace / buffalo_l) against a small known-faces database.

It is deliberately optional and fails open:
- Nothing activates unless ``FACE_ID_DB`` points at a directory of
  ``{person_name}.jpg`` headshots.
- Any import, model, or inference failure logs a warning and leaves the
  transcript untouched — the pipeline never depends on this layer.

Named on-screen ranges are matched against the diarized speaker turns to
produce a ``{speaker_label: name}`` mapping. That mapping renames the
transcript's segments in place AND is passed to
viral_clip_finder.select_viral_clips as ``face_identities``, upgrading the
transcript to the skill's Tier 3 (named-speaker) input contract so cut briefs
can say "frame Joe" and recommend reaction cuts by name.

Environment
-----------
FACE_ID_DB            path to a folder of {name}.jpg headshots (activates)
FACE_ID_MODEL         insightface model name (default buffalo_l)
FACE_ID_CTX           device id (default 0)
FACE_ID_SAMPLE_FPS    frames sampled per second (default 2)
FACE_ID_THRESHOLD     embedding similarity threshold (default 0.4)
FACE_ID_MIN_COVERAGE  min on-screen coverage to rename a speaker (default 0.6)
FACE_ID_MIN_TURNS_S   min total speaking seconds before renaming (default 3)
FACE_ID_MAX_SECONDS   stop scanning after this much source (default 900; 0 =
                      no limit). ArcFace over a full 2-hour stream is tens of
                      minutes of GPU before Stage 3 even starts, and speaker
                      identity is established in the first few minutes.

Known limitation: LR-ASD is deliberately clip-scoped in this codebase (whole-
source scoring is tens of minutes of compute), so the speaking ranges in the
face trajectory are left empty here. On-screen ranges still provide a
conservative speaker-rename signal; the pipeline's per-clip LR-ASD pass keeps
doing the precise who-is-speaking work during reframing.
"""

import os
from typing import Dict, List, Optional, Tuple

# One model + one DB load per process: enrich_if_configured (source-level)
# and the per-clip framing pass both need identifications, and InsightFace
# model load is the expensive part. Keyed by the exact configuration so
# different DBs/models never collide.
_DB_CACHE: Dict[tuple, "KnownFacesDB"] = {}


def default_ctx_id() -> int:
    """Which GPU InsightFace runs on.

    Defaults to the LAST visible device, not device 0. On the 2×T4 target that
    puts face ID on GPU 1 while the render pipeline (SCRFD/ArcFace, LR-ASD,
    ffmpeg) keeps GPU 0 — the integration guide's explicit recommendation, and
    it matters because face ID runs BEFORE clip selection, so sharing device 0
    would stall the stage everything else waits on. Single-GPU and CPU hosts
    get 0, which is the old behaviour. FACE_ID_CTX overrides.
    """
    configured = os.environ.get("FACE_ID_CTX", "").strip()
    if configured:
        try:
            return int(configured)
        except ValueError:
            print(f"⚠️ Face ID: FACE_ID_CTX={configured!r} is not an integer — using 0")
            return 0
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            return torch.cuda.device_count() - 1
    except Exception:
        pass
    return 0


def available() -> bool:
    """True when a known-faces DB is configured and the library imports."""
    if not os.environ.get("FACE_ID_DB"):
        return False
    db_path = os.environ.get("FACE_ID_DB")
    if not db_path or not os.path.isdir(db_path):
        return False
    try:
        import insightface  # noqa: F401
    except ImportError:
        print("⚠️ Face ID: InsightFace is not installed — install with "
              "`pip install insightface onnxruntime-gpu` to enable named "
              "speaker enrichment.")
        return False
    return True


def get_known_faces_db(db_path: str = None, model_name: str = None,
                       ctx_id: int = None):
    """Cached KnownFacesDB instance (see _DB_CACHE)."""
    db_path = db_path or os.environ.get("FACE_ID_DB")
    if not db_path or not os.path.isdir(db_path):
        return None
    model_name = model_name or os.environ.get("FACE_ID_MODEL", "buffalo_l")
    if ctx_id is None:
        ctx_id = default_ctx_id()
    key = (db_path, model_name, ctx_id)
    if key not in _DB_CACHE:
        _DB_CACHE[key] = KnownFacesDB(db_path, model_name=model_name,
                                      ctx_id=ctx_id)
    return _DB_CACHE[key]


class KnownFacesDB:
    """Compare detected faces against a folder of {name}.jpg headshots."""

    def __init__(self, db_path: str, model_name: str = "buffalo_l",
                 ctx_id: int = 0):
        import insightface
        self.app = insightface.app.FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.embeddings: Dict[str, "object"] = {}
        self._load_db(db_path)

    def _load_db(self, db_path: str) -> None:
        import cv2
        for fname in sorted(os.listdir(db_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            name = os.path.splitext(fname)[0]
            img = cv2.imread(os.path.join(db_path, fname))
            if img is None:
                continue
            faces = self.app.get(img)
            if not faces:
                continue
            largest = max(faces, key=lambda f: (
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
            self.embeddings[name] = largest.normed_embedding

    def identify(self, embedding, threshold: float = 0.4) -> Tuple[Optional[str], float]:
        """Return (name, similarity) or (None, 0) below the threshold."""
        import numpy as np
        best_name, best_sim = None, 0.0
        for name, db_emb in self.embeddings.items():
            sim = float(np.dot(embedding, db_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= threshold:
            return best_name, best_sim
        return None, 0.0


def identify_faces_in_video(video_path: str, db: KnownFacesDB,
                            sample_fps: float = 2.0,
                            threshold: float = 0.4,
                            max_seconds: float = 900.0) -> List[dict]:
    """Sample frames and return [{timestamp, name, confidence, bbox}, ...].

    Sampling stops at ``max_seconds`` of source (0 disables the limit) and
    seeks between samples instead of decoding every frame — a 2fps scan of a
    long stream otherwise decodes ~200k frames it immediately discards.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(fps / max(sample_fps, 0.1))))
    identifications = []
    frame_idx = 0
    while True:
        if max_seconds and frame_idx / fps > max_seconds:
            print(f"🪪  Face ID: reached the {max_seconds:.0f}s scan limit "
                  "(FACE_ID_MAX_SECONDS) — identities found so far are used.")
            break
        # Seek straight to the next sample instead of decoding (and throwing
        # away) the interval-1 frames in between. If the container refuses to
        # seek, cap.set returns False and grab() skips forward instead.
        if interval > 1 and not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
            for _ in range(interval - 1):
                if not cap.grab():
                    break
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_idx / fps
        try:
            faces = db.app.get(frame)
        except Exception:
            faces = []
        for face in faces:
            name, conf = db.identify(face.normed_embedding, threshold)
            if name:
                identifications.append({
                    "timestamp": round(timestamp, 3),
                    "name": name,
                    "confidence": round(conf, 4),
                    "bbox": [float(v) for v in face.bbox],
                })
        frame_idx += interval
    cap.release()
    return identifications


def _ranges_from_samples(identifications: List[dict],
                         gap_tolerance: float = 2.0) -> List[List[float]]:
    """Group consecutive samples of one name into [start, end] ranges."""
    if not identifications:
        return []
    ranges, range_start = [], identifications[0]["timestamp"]
    prev = identifications[0]["timestamp"]
    for ident in identifications[1:]:
        t = ident["timestamp"]
        if t - prev > gap_tolerance:
            ranges.append([range_start, prev])
            range_start = t
        prev = t
    ranges.append([range_start, prev])
    return ranges


def build_face_trajectory(identifications: List[dict],
                          gap_tolerance: float = 2.0) -> dict:
    """Collapse identifications into the skill's face-trajectory sidecar.

    ``gap_tolerance`` must exceed the sampling period or every sample becomes
    its own range — the caller derives it from FACE_ID_SAMPLE_FPS.
    """
    by_name: Dict[str, List[dict]] = {}
    for ident in identifications:
        by_name.setdefault(ident["name"], []).append(ident)
    identities = []
    for name, samples in sorted(by_name.items()):
        samples.sort(key=lambda s: s["timestamp"])
        identities.append({
            "name": name,
            "confidence": round(sum(s["confidence"] for s in samples) / len(samples), 4),
            "on_screen": _ranges_from_samples(samples, gap_tolerance),
            # Speaking ranges are left empty: LR-ASD is clip-scoped by design
            # (see module docstring). The per-clip reframe pass owns that signal.
            "speaking": [],
        })
    return {"identities": identities}


def _iou(a, b):
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _xywh(bbox):
    """Normalize an InsightFace bbox [x1, y1, x2, y2] to (x, y, w, h)."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def find_track_at_timestamp(tracks, timestamp, bbox, iou_threshold=0.3):
    """Which track is active at this timestamp with bbox overlap (Step 3d).

    ``tracks``: list of {track_id, frames: [{timestamp, bbox}]} — the
    shape the integration guide specifies. ``bbox`` may be InsightFace
    [x1,y1,x2,y2] or (x,y,w,h); both are normalized internally.
    Returns the matching track_id or None.
    """
    ident_box = _xywh(bbox)
    for track in tracks:
        for frame in track.get("frames", []):
            if abs(float(frame["timestamp"]) - timestamp) < 0.5:
                if _iou(_xywh(frame["bbox"]), ident_box) >= iou_threshold:
                    return track["track_id"]
    return None


def merge_face_id_with_tracker(tracker_tracks, face_identifications,
                               min_agreement=0.6):
    """Map anonymous track IDs to named identities (Step 3d, integration doc).

    tracker_tracks: list of {track_id, frames: [{timestamp, bbox}]}
    face_identifications: list of {timestamp, name, confidence, bbox}

    Returns {track_id -> {"name", "confidence", "on_screen": [[s, e], ...]}}.
    A track is only named when >= min_agreement of its matched
    identifications vote for the same name, so a face that flickers between
    two people never gets committed.
    """
    track_to_name: Dict[int, Dict[str, int]] = {}
    track_to_samples: Dict[int, List[dict]] = {}
    for ident in face_identifications:
        track_id = find_track_at_timestamp(
            tracker_tracks, ident["timestamp"], ident["bbox"])
        if track_id is None:
            continue
        track_to_name.setdefault(track_id, {})
        track_to_name[track_id][ident["name"]] = \
            track_to_name[track_id].get(ident["name"], 0) + 1
        track_to_samples.setdefault(track_id, []).append(ident)

    result = {}
    for track_id, votes in track_to_name.items():
        if not votes:
            continue
        best_name = max(votes, key=votes.get)
        total = sum(votes.values())
        confidence = votes[best_name] / total
        if confidence >= min_agreement:
            samples = sorted(track_to_samples[track_id],
                             key=lambda s: s["timestamp"])
            result[track_id] = {
                "name": best_name,
                "confidence": round(confidence, 4),
                "on_screen": _ranges_from_samples(samples),
            }
    return result


def _overlap_seconds(turns: List[List[float]], ranges: List[List[float]]) -> float:
    total = 0.0
    for s1, e1 in turns:
        for s2, e2 in ranges:
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                total += e - s
    return total


def enrich_transcript_speakers(transcript_result: dict, face_trajectory: dict,
                               min_coverage: float = 0.6,
                               min_turn_seconds: float = 3.0) -> Tuple[dict, dict]:
    """Conservatively rename anonymous diarized speaker labels to names.

    For each anonymous label, the name whose on-screen ranges cover the most
    of the label's speaking time wins — but only when that coverage clears
    ``min_coverage`` AND the label has at least ``min_turn_seconds`` of speech
    (a 2-second blip is too little evidence). Everything else stays
    anonymous. Returns (transcript_result, {label: name}) — the mapping is
    exactly what viral_clip_finder.format_transcript_for_skill consumes.
    """
    identities = face_trajectory.get("identities") or []
    ranges_by_name = {i["name"]: i.get("on_screen") or [] for i in identities}
    if not ranges_by_name:
        return transcript_result, {}

    turns_by_label: Dict[str, List[List[float]]] = {}
    for segment in transcript_result.get("segments", []):
        speaker = segment.get("speaker")
        if not speaker:
            continue
        turns_by_label.setdefault(speaker, []).append(
            [float(segment.get("start", 0)), float(segment.get("end", 0))])

    mapping: Dict[str, str] = {}
    for label, turns in turns_by_label.items():
        speaking_seconds = sum(e - s for s, e in turns)
        if speaking_seconds < min_turn_seconds:
            continue
        best_name, best_cover = None, 0.0
        for name, ranges in ranges_by_name.items():
            cover = _overlap_seconds(turns, ranges)
            if cover > best_cover:
                best_cover, best_name = cover, name
        if best_name and (best_cover / speaking_seconds) >= min_coverage:
            mapping[label] = best_name

    if not mapping:
        return transcript_result, {}
    for segment in transcript_result.get("segments", []):
        speaker = segment.get("speaker")
        if speaker in mapping:
            segment["speaker"] = mapping[speaker]
            segment["speaker_named"] = True
    print(f"🪪  Face ID: renamed {len(mapping)} speaker label(s) → "
          + ", ".join(f"{k}->{v}" for k, v in mapping.items()))
    return transcript_result, mapping


def enrich_if_configured(transcript_result: dict,
                         video_path: str) -> Tuple[dict, Optional[dict]]:
    """Top-level entry: run face ID when configured; otherwise no-op.

    Returns ``(transcript_result, {speaker_label: name} | None)``. The second
    value is the LABEL MAPPING, not the face trajectory: it is fed straight to
    viral_clip_finder.format_transcript_for_skill, which looks speakers up in
    it with ``speaker in face_identities``. Returning the trajectory dict here
    made that lookup silently impossible to satisfy. The full trajectory is
    still reachable via build_face_trajectory for callers that want the
    on-screen ranges. Never raises — failures degrade to the original
    transcript.
    """
    if not available() or not video_path or not os.path.exists(video_path):
        return transcript_result, None
    try:
        ctx_id = default_ctx_id()
        db = get_known_faces_db(ctx_id=ctx_id)
        if db is None:
            return transcript_result, None
        if not db.embeddings:
            print("⚠️ Face ID: known-faces DB is empty — no names to match.")
            return transcript_result, None
        sample_fps = float(os.environ.get("FACE_ID_SAMPLE_FPS", "2"))
        identifications = identify_faces_in_video(
            video_path, db,
            sample_fps=sample_fps,
            threshold=float(os.environ.get("FACE_ID_THRESHOLD", "0.4")),
            max_seconds=float(os.environ.get("FACE_ID_MAX_SECONDS", "900")))
        if not identifications:
            print("⚠️ Face ID: no faces matched the known-faces DB.")
            return transcript_result, None
        trajectory = build_face_trajectory(
            identifications, gap_tolerance=max(2.0, 2.0 / max(sample_fps, 0.1)))
        named = ", ".join(i["name"] for i in trajectory["identities"])
        print(f"🪪  Face ID: identified {named} over "
              f"{len(identifications)} sample(s) on GPU {ctx_id}.")
        enriched, mapping = enrich_transcript_speakers(
            transcript_result, trajectory,
            min_coverage=float(os.environ.get("FACE_ID_MIN_COVERAGE", "0.6")),
            min_turn_seconds=float(os.environ.get("FACE_ID_MIN_TURNS_S", "3")))
        return enriched, (mapping or None)
    except Exception as e:
        print(f"⚠️ Face ID enrichment failed ({type(e).__name__}: {e}) — "
              "continuing with anonymous speakers.")
        return transcript_result, None
