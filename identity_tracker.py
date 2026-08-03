"""Stable per-person identity for the reframing pipeline (ByteTrack / BoT-SORT).

WHY THIS EXISTS
---------------
Identity used to come from ``SpeakerTracker.assign_ids``: for each detection,
scan remembered faces and take the nearest one by **horizontal centre alone**,
greedily, within 15% of frame width, forgetting anyone unseen for 30 frames.

That has four failure modes, all of which show up as the camera "changing its
mind" about who is who:

1. **x-only matching.** Two people at similar x but different depth/height are
   indistinguishable, so their ids swap. The renderer's own comments record
   this ("in a crowded lineup shot people get reassigned ids").
2. **No motion model.** Nothing predicts where a person will be next, so any
   real movement between detections looks like a new person.
3. **Greedy, order-dependent assignment.** The first candidate to claim a
   remembered face wins it, even when a later candidate is a far better match.
   There is no global optimum.
4. **Hard forgetting.** A person occluded for slightly over a second comes back
   as a brand new identity, resetting every id-keyed policy above it
   (primary-subject bias, speaker binding, split-cell assignment).

ByteTrack replaces all four with a Kalman motion model, Hungarian (globally
optimal) assignment over IoU, and — its actual contribution — a *second*
association pass that recovers low-confidence detections instead of discarding
them, which is exactly the occluded/turned-away face this pipeline keeps
losing. BoT-SORT adds camera-motion compensation on top, which matters on
handheld/panning source footage.

Both ship inside the ``ultralytics`` package this project already depends on,
so this is an adapter, not a new dependency.

DESIGN NOTES
------------
* The tracker is fed once per DETECTION, not once per rendered frame, so its
  frame rate is the detection rate (fps / stride). Getting this wrong makes the
  Kalman velocities and ``track_buffer`` wrong by the stride factor.
* ``update`` stamps ids onto the caller's dicts in place and returns them, so it
  is a drop-in for ``assign_ids``.
* A detection the tracker declines to confirm still gets an id (a negative,
  provisional one) rather than none — callers index by id and must never see
  a candidate without one.
* Everything degrades to the legacy matcher if ultralytics/lap is unavailable,
  so a self-hoster missing the extra never loses tracking entirely.
"""
from types import SimpleNamespace

import numpy as np


# Tracker defaults. Deliberately more forgiving than ultralytics' stock values:
# these are people in conversation, not fast-moving objects, and the cost of
# dropping a track (identity reset, camera re-decides who to frame) is far
# higher here than the cost of holding a stale one for an extra beat.
DEFAULTS = dict(
    tracker_type="botsort",
    track_high_thresh=0.30,
    track_low_thresh=0.08,   # keep weak detections for the recovery pass
    new_track_thresh=0.45,   # slower to invent a new person
    track_buffer=90,         # ~3s of detections; survives long occlusions
    match_thresh=0.75,
    fuse_score=True,
    # BoT-SORT extras
    gmc_method="sparseOptFlow",  # camera-motion compensation
    proximity_thresh=0.5,
    appearance_thresh=0.25,
    with_reid=False,         # ReID weights would be a new model download
)


class _Detections:
    """Duck-typed stand-in for an ultralytics ``Results`` boxes object.

    BYTETracker only touches ``.conf``, ``.xywh``, ``.cls``, ``len()`` and mask
    indexing, so a real Results object (which would drag in the whole
    prediction pipeline) is unnecessary.
    """

    __slots__ = ("xywh", "conf", "cls")

    def __init__(self, xywh, conf, cls):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return _Detections(self.xywh[mask], self.conf[mask], self.cls[mask])


def available():
    """True when the tracker backend can actually be constructed."""
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: F401
        from ultralytics.trackers.bot_sort import BOTSORT  # noqa: F401
        return True
    except Exception:
        return False


class IdentityTracker:
    """Assigns stable person ids across detections for one clip."""

    def __init__(self, detection_fps=7.5, tracker_type=None, **overrides):
        cfg = dict(DEFAULTS)
        if tracker_type:
            cfg["tracker_type"] = tracker_type
        cfg.update(overrides)
        self.cfg = cfg
        self._provisional = -1
        # Maps the backend's track_id -> the id we hand to callers. Kept
        # separate so ids stay small, dense and stable even if the backend
        # renumbers internally.
        self._ids = {}
        self._next_id = 0
        # Per-frame memo. The renderer calls assign_ids several times for the
        # SAME frame (once before the score boosts, again after), and a Kalman
        # tracker advanced three times per detection would triple its predicted
        # velocities and drift off the subjects. So the backend is stepped once
        # per frame_number and later calls re-stamp from this cache.
        self._memo_frame = None
        self._memo = {}

        args = SimpleNamespace(**cfg)
        frame_rate = max(1, int(round(detection_fps)))
        if cfg["tracker_type"] == "botsort":
            from ultralytics.trackers.bot_sort import BOTSORT
            self.backend = BOTSORT(args, frame_rate=frame_rate)
        else:
            from ultralytics.trackers.byte_tracker import BYTETracker
            self.backend = BYTETracker(args, frame_rate=frame_rate)

    def _public_id(self, track_id):
        key = int(track_id)
        if key not in self._ids:
            self._ids[key] = self._next_id
            self._next_id += 1
        return self._ids[key]

    def update(self, candidates, frame=None, frame_number=None):
        """Stamp ``candidates`` with a stable ``'id'`` and return them.

        ``candidates`` are the pipeline's dicts: ``{'box': (x, y, w, h),
        'score': float, ...}``. ``frame`` is the BGR image, used only by
        BoT-SORT's camera-motion compensation; ByteTrack ignores it.

        Passing ``frame_number`` makes repeat calls for the same frame free and
        side-effect-free (see the memo note in __init__) — the renderer relies
        on this, calling once before its score boosts and again after.
        """
        if not candidates:
            return candidates

        if frame_number is not None and frame_number == self._memo_frame:
            for c in candidates:
                cached = self._memo.get(tuple(c["box"]))
                if cached is not None:
                    c["id"] = cached
                elif "id" not in c:
                    c["id"] = self._provisional
                    self._provisional -= 1
            return candidates

        xywh = np.array(
            [[c["box"][0] + c["box"][2] / 2.0,
              c["box"][1] + c["box"][3] / 2.0,
              c["box"][2], c["box"][3]] for c in candidates],
            dtype=np.float32,
        )
        # Detector scores here are relevance weights, not detection
        # confidences, and can exceed 1 after the pipeline's boosts. ByteTrack
        # splits high/low confidence on absolute thresholds, so they are
        # clamped into (0, 1] to keep that split meaningful.
        conf = np.array(
            [min(max(float(c.get("score", 1.0)), 0.01), 1.0) for c in candidates],
            dtype=np.float32,
        )
        cls = np.zeros(len(candidates), dtype=np.float32)

        try:
            tracks = self.backend.update(_Detections(xywh, conf, cls), frame)
        except Exception:
            # Never let a tracker failure cost the caller its ids.
            tracks = np.empty((0, 8), dtype=np.float32)

        # result rows are [x1, y1, x2, y2, track_id, score, cls, det_index]
        for row in tracks:
            det_index = int(row[7])
            if 0 <= det_index < len(candidates):
                candidates[det_index]["id"] = self._public_id(row[4])

        # A detection the tracker hasn't confirmed yet still needs an id —
        # callers key policy decisions off it. Provisional ids are negative so
        # they can never collide with a confirmed track.
        for c in candidates:
            if "id" not in c:
                c["id"] = self._provisional
                self._provisional -= 1

        if frame_number is not None:
            self._memo_frame = frame_number
            self._memo = {tuple(c["box"]): c["id"] for c in candidates}
        return candidates
