"""Phase 1 of the framing-engine rebuild: one shared face-track spine.

WHY THIS EXISTS
---------------
Today there are TWO independent face trackers in this pipeline: LR-ASD builds
its own ByteTrack instance over its own face crops (asd_worker.py), and the
renderer builds a SECOND, unrelated one over MediaPipe/YOLO boxes
(reframe_v2.render). Their track-id spaces have nothing to do with each other,
so matching "the face LR-ASD says is speaking" to "the box the renderer is
about to frame" has to happen by PIXEL POSITION every frame
(reframe_v2._apply_asd_speaker_boost), with a decisiveness gate that silently
discards the evidence when two candidates look similar. That discarded
evidence falls through to "hold whoever the camera already had" — which is the
actual mechanism behind the wrong-person framing this rebuild exists to fix,
not a tuning problem in either tracker.

This module is the fix: ONE face-track spine, built once, that both speaker
resolution (Phase 2/3) and shot planning (Phase 4) read from. No more
positional matching between two id spaces, because there is only one.

WHAT IT DOES
------------
1. SCRFD face detection (via InsightFace, already a project dependency for
   face_id.py's named-identity layer — buffalo_l bundles SCRFD + ArcFace) at
   GPU-batched, onnxruntime speed, in place of MediaPipe's CPU-only BlazeFace.
2. A short-term tracker (identity_tracker.IdentityTracker, the same
   ByteTrack/BoT-SORT adapter the renderer already uses — no new tracking
   dependency) to link detections frame-to-frame within one continuous shot.
3. ArcFace embeddings on every detection, used AFTER the short-term tracking
   pass to merge tracks that the short-term tracker could never bridge on its
   own: a scene cut resets ByteTrack's motion model entirely (there is no
   continuity to track across a hard cut), so the same person reappearing in
   the next shot gets a brand-new raw track id. Embedding similarity is what
   recognises "this is the same face as raw track 3" and merges them.

Deliberately NOT included: YOLO/body boxes. Per the tool-proposal in the
rebuild plan, a body box's centre sits far from a face box's centre for the
same person — mixing them into one aim target is the root of the
"speaker_on_screen" instability this module is meant to remove. Body boxes
stay a separate, later-stage occlusion fallback (extending a track through a
face-occluded stretch), never a framing target in their own right.

NO GLOBAL LOCK
---------------
main.py's MediaPipe/YOLO detection is serialized behind one process-wide
DETECT_LOCK because the MediaPipe graph and the YOLO model instance are not
thread-safe to call concurrently. onnxruntime sessions (what InsightFace runs
on) do not have that restriction — concurrent Run() calls on a session are
safe, and each device gets its own analyzer instance here (see
_get_analyzer's cache key), so parallel clip/GPU workers do not need to queue
behind a shared lock the way today's detection does.
"""
import os
import threading
from typing import Dict, List, Optional

import numpy as np

import identity_tracker


# Cached per (model_name, ctx_id): building an InsightFace FaceAnalysis is the
# expensive part (loads SCRFD + ArcFace onnx sessions). Keyed by device so
# concurrent workers on different GPUs never share (and therefore never
# contend for) one analyzer instance.
_ANALYZER_CACHE: Dict[tuple, "object"] = {}
_ANALYZER_LOCK = threading.Lock()

DEFAULT_MODEL = os.environ.get("FACE_SPINE_MODEL", "buffalo_l")
DEFAULT_DET_SIZE = (640, 640)

# Detections per second of source. Lower than face_id.py's FACE_ID_SAMPLE_FPS
# default (2) because this spine feeds tracking, which needs enough temporal
# density for ByteTrack's motion model to hold a track across ordinary
# movement — too sparse and every shot looks like a series of new people.
DEFAULT_SAMPLE_FPS = float(os.environ.get("FACE_SPINE_SAMPLE_FPS", "5"))

# Cosine similarity floor for two raw tracks to be merged as the same person.
# ArcFace's own convention (see face_id.KnownFacesDB.identify) uses 0.4 for
# comparing a live face against a single reference headshot; merging two
# TRACKS (each backed by several detections, hence a less noisy mean
# embedding) can afford to be stricter without losing real matches, and a
# stricter floor costs less than a false merge (two different people folded
# into one track silently breaks every consumer keyed on track identity).
DEFAULT_SIMILARITY_THRESHOLD = float(
    os.environ.get("FACE_SPINE_SIMILARITY_THRESHOLD", "0.5"))


def _resolve_ctx_id(device: Optional[str]) -> int:
    """GPU index for InsightFace's ctx_id, from a "cuda:N" device string or
    gpu_affinity's current-thread assignment. Mirrors face_id.default_ctx_id
    but takes an explicit device instead of always picking the last GPU —
    face_spine runs once per source video, not per clip, so it should follow
    whatever device the caller assigns it rather than assuming face_id's
    "stay off the render GPU" placement.
    """
    if device:
        s = str(device)
        if ":" in s:
            try:
                return int(s.split(":", 1)[1])
            except ValueError:
                pass
    try:
        import gpu_affinity
        current = gpu_affinity.current_device()
        if current:
            return int(current.split(":", 1)[1])
    except Exception:
        pass
    return 0


def _get_analyzer(model_name: str = DEFAULT_MODEL, ctx_id: int = 0,
                  det_size=DEFAULT_DET_SIZE):
    """Cached InsightFace FaceAnalysis (SCRFD detector + ArcFace recognizer),
    one per (model, device). Thread-safe to call concurrently once built —
    see the module docstring's "NO GLOBAL LOCK" section.
    """
    key = (model_name, ctx_id, det_size)
    with _ANALYZER_LOCK:
        if key not in _ANALYZER_CACHE:
            import insightface
            app = insightface.app.FaceAnalysis(name=model_name)
            app.prepare(ctx_id=ctx_id, det_size=det_size)
            _ANALYZER_CACHE[key] = app
        return _ANALYZER_CACHE[key]


def _xywh_from_bbox(bbox) -> tuple:
    """InsightFace bbox [x1, y1, x2, y2] -> (x, y, w, h)."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def extract_raw_tracks(video_path: str, device: Optional[str] = None,
                       sample_fps: float = DEFAULT_SAMPLE_FPS,
                       max_seconds: float = 0.0,
                       model_name: str = DEFAULT_MODEL) -> Dict[int, dict]:
    """Pass 1: SCRFD detection + short-term ByteTrack over the whole video.

    Seeks between samples (like face_id.identify_faces_in_video) rather than
    decoding every frame — this runs once per source video at a sparse rate,
    so decoding frames it will not detect on is pure waste on anything longer
    than a couple of minutes.

    Returns {raw_track_id: {"frames": [t, ...], "boxes": [(x,y,w,h), ...],
    "landmarks": [5x2 array or None, ...], "embeddings": [np.ndarray, ...],
    "det_scores": [float, ...]}} — one entry per person ByteTrack could
    follow WITHOUT a scene cut breaking it. merge_tracks_by_identity (pass 2)
    is what recognises the same person across a cut.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(fps / max(sample_fps, 0.1))))

    ctx_id = _resolve_ctx_id(device)
    analyzer = _get_analyzer(model_name, ctx_id)
    tracker = identity_tracker.IdentityTracker(detection_fps=sample_fps)

    tracks: Dict[int, dict] = {}
    frame_idx = 0
    detection_index = 0
    while True:
        if max_seconds and frame_idx / fps > max_seconds:
            break
        if interval > 1 and not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
            for _ in range(interval - 1):
                if not cap.grab():
                    break
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_idx / fps

        try:
            faces = analyzer.get(frame)
        except Exception:
            faces = []

        candidates = []
        for face in faces:
            x, y, w, h = _xywh_from_bbox(face.bbox)
            candidates.append({
                "box": (x, y, w, h),
                "score": float(getattr(face, "det_score", 1.0)),
                "_embedding": getattr(face, "normed_embedding", None),
                "_kps": getattr(face, "kps", None),
            })

        if candidates:
            tracker.update(candidates, frame=frame, frame_number=detection_index)
            for c in candidates:
                track_id = c["id"]
                if track_id < 0:
                    # Unconfirmed by the short-term tracker (see
                    # identity_tracker's provisional-id note). A one-off
                    # unconfirmed detection contributes little to an identity
                    # merge and would otherwise mint a throwaway track per
                    # flicker, so it is skipped here rather than recorded.
                    continue
                rec = tracks.setdefault(track_id, {
                    "frames": [], "boxes": [], "landmarks": [],
                    "embeddings": [], "det_scores": [],
                })
                rec["frames"].append(timestamp)
                rec["boxes"].append(c["box"])
                rec["landmarks"].append(c["_kps"])
                rec["det_scores"].append(c["score"])
                if c["_embedding"] is not None:
                    rec["embeddings"].append(c["_embedding"])

        frame_idx += interval
        detection_index += 1

    cap.release()
    return tracks


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mean_embedding(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    if not embeddings:
        return None
    mean = np.mean(np.stack(embeddings), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def _time_range(track: dict) -> tuple:
    frames = track.get("frames") or [0.0]
    return min(frames), max(frames)


def _ranges_overlap(a: tuple, b: tuple) -> bool:
    return a[0] < b[1] and b[0] < a[1]


class _UnionFind:
    """Standard disjoint-set with path compression, used to group raw track
    ids into identities. Kept tiny and local rather than pulling in a dep —
    this is ~10 lines of well-understood algorithm, not novel logic.
    """

    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def merge_tracks_by_identity(raw_tracks: Dict[int, dict],
                             similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
                             ) -> Dict[int, dict]:
    """Pass 2: merge raw (short-term) tracks that are the same person.

    Two raw tracks merge when their mean ArcFace embeddings are similar
    enough AND their time ranges do not overlap — two simultaneously-visible
    tracks can never be the same person no matter how similar their
    embeddings read (identical twins, a printed photo of the same person),
    so temporal overlap is a hard veto, not just a tiebreaker.

    This is deliberately a single-pass pairwise comparison with union-find
    (not a proper clustering algorithm): a similarity edge between A-B and
    B-C merges all three transitively even without a direct A-C edge, which
    is usually correct (the same person, sampled at different times, still
    looks like the same person) but can over-merge on a large cast with
    similar-looking members. Fine for the two/three-person format this
    rebuild targets; revisit if it is ever pointed at a large-cast show.

    Returns {merged_id: {"frames": [...], "boxes": [...], "landmarks": [...],
    "embeddings": [...], "det_scores": [...], "raw_track_ids": [...]}},
    densely renumbered from 0, sorted by first-appearance time.
    """
    ids = list(raw_tracks.keys())
    means = {i: _mean_embedding(raw_tracks[i].get("embeddings") or []) for i in ids}
    ranges = {i: _time_range(raw_tracks[i]) for i in ids}

    uf = _UnionFind(ids)
    for a_idx in range(len(ids)):
        for b_idx in range(a_idx + 1, len(ids)):
            a, b = ids[a_idx], ids[b_idx]
            if means[a] is None or means[b] is None:
                continue
            if _ranges_overlap(ranges[a], ranges[b]):
                continue
            if _cosine_similarity(means[a], means[b]) >= similarity_threshold:
                uf.union(a, b)

    groups: Dict[int, List[int]] = {}
    for i in ids:
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    merged = []
    for members in groups.values():
        rec = {"frames": [], "boxes": [], "landmarks": [], "embeddings": [],
               "det_scores": [], "raw_track_ids": sorted(members)}
        for m in sorted(members, key=lambda i: _time_range(raw_tracks[i])[0]):
            src = raw_tracks[m]
            rec["frames"].extend(src["frames"])
            rec["boxes"].extend(src["boxes"])
            rec["landmarks"].extend(src["landmarks"])
            rec["embeddings"].extend(src["embeddings"])
            rec["det_scores"].extend(src["det_scores"])
        merged.append(rec)

    merged.sort(key=lambda r: min(r["frames"]) if r["frames"] else 0.0)
    return {i: rec for i, rec in enumerate(merged)}


def build_face_spine(video_path: str, device: Optional[str] = None,
                     sample_fps: float = DEFAULT_SAMPLE_FPS,
                     max_seconds: float = 0.0,
                     model_name: str = DEFAULT_MODEL,
                     similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
                     ) -> Dict[int, dict]:
    """Top-level entry point: video in, one merged track per real person out.

    This is the Phase 1 deliverable per the rebuild plan — the single spine
    that speaker resolution (Phase 2/3) and shot planning (Phase 4) both read
    from, replacing the two independent trackers described in the module
    docstring.
    """
    raw = extract_raw_tracks(video_path, device=device, sample_fps=sample_fps,
                             max_seconds=max_seconds, model_name=model_name)
    return merge_tracks_by_identity(raw, similarity_threshold=similarity_threshold)
