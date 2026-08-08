"""Identity must survive the things that used to reset it.

The legacy matcher (v1/v2's SpeakerTracker.assign_ids) matched on
horizontal centre alone, greedily, forgetting anyone unseen for 30 frames.
These tests pin the behaviours that replaced it: crossings, occlusion, and
per-frame idempotence.
"""
import numpy as np
import pytest

identity_tracker = pytest.importorskip("identity_tracker")

pytestmark = pytest.mark.skipif(
    not identity_tracker.available(),
    reason="ultralytics tracker backend (or lap) not installed",
)

W, H = 1920, 1080


def _frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


def _tracker(**kw):
    kw.setdefault("detection_fps", 7.5)
    kw.setdefault("tracker_type", "bytetrack")
    return identity_tracker.IdentityTracker(**kw)


def _cand(x, y=300, w=160, h=200, score=0.9):
    return {"box": (x, y, w, h), "score": score}


def test_every_candidate_gets_an_id():
    t = _tracker()
    cands = [_cand(400), _cand(1200)]
    t.update(cands, _frame(), frame_number=0)
    assert all("id" in c for c in cands)


def test_identity_survives_two_people_crossing():
    """The case x-only matching gets wrong: paths cross, ids swap."""
    t = _tracker()
    a_ids, b_ids = [], []
    for i in range(20):
        cands = [_cand(400 + i * 40), _cand(1400 - i * 40, y=320)]
        t.update(cands, _frame(), frame_number=i)
        a_ids.append(cands[0]["id"])
        b_ids.append(cands[1]["id"])
    assert len(set(a_ids)) == 1, f"subject A changed id: {a_ids}"
    assert len(set(b_ids)) == 1, f"subject B changed id: {b_ids}"
    assert a_ids[0] != b_ids[0]


def test_identity_survives_a_long_occlusion():
    """Legacy forgot anyone unseen for 30 frames; track_buffer holds them."""
    t = _tracker()
    before = after = None
    for i in range(30):
        cands = [_cand(500)]
        visible = not (10 <= i < 21)          # gone for 11 detections
        if visible:
            cands.append(_cand(1200, y=310))
        t.update(cands, _frame(), frame_number=i)
        if i == 9:
            before = cands[1]["id"]
        if i == 29:
            after = cands[1]["id"]
    assert before is not None and after == before


def test_repeat_calls_for_one_frame_are_idempotent():
    """The renderer calls assign_ids twice per frame — before and after its
    score boosts. Stepping a Kalman tracker twice per detection would double
    its velocities, so repeat calls must be memoised, not re-run."""
    t = _tracker()
    for i in range(5):
        t.update([_cand(500 + i * 30), _cand(1300)], _frame(), frame_number=i)

    first = [_cand(650), _cand(1300)]
    t.update(first, _frame(), frame_number=5)
    ids_first = [c["id"] for c in first]

    repeat = [_cand(650), _cand(1300)]
    t.update(repeat, _frame(), frame_number=5)
    assert [c["id"] for c in repeat] == ids_first


def test_repeat_call_does_not_advance_the_backend():
    t = _tracker()
    t.update([_cand(500)], _frame(), frame_number=0)
    fid = t.backend.frame_id
    t.update([_cand(500)], _frame(), frame_number=0)
    assert t.backend.frame_id == fid


def test_empty_detections_are_safe():
    t = _tracker()
    assert t.update([], _frame(), frame_number=0) == []


def test_unconfirmed_detections_still_get_an_id():
    """Callers key policy off 'id' — a candidate must never lack one."""
    t = _tracker()
    cands = [_cand(500, score=0.02)]   # below every confirmation threshold
    t.update(cands, _frame(), frame_number=0)
    assert isinstance(cands[0]["id"], int)


def test_backend_failure_still_returns_ids():
    t = _tracker()

    class _Boom:
        def update(self, *a, **kw):
            raise RuntimeError("backend exploded")

    t.backend = _Boom()
    cands = [_cand(500), _cand(1200)]
    t.update(cands, _frame(), frame_number=0)
    assert all(isinstance(c["id"], int) for c in cands)


def test_scores_above_one_do_not_break_confidence_split():
    """Pipeline 'score' is a relevance weight (box area, boosted) and routinely
    exceeds 1; ByteTrack splits high/low confidence on absolute thresholds."""
    t = _tracker()
    cands = [_cand(500, score=48000.0), _cand(1200, score=31000.0)]
    t.update(cands, _frame(), frame_number=0)
    assert len({c["id"] for c in cands}) == 2


def test_botsort_backend_also_works():
    t = _tracker(tracker_type="botsort")
    cands = [_cand(400), _cand(1300)]
    t.update(cands, _frame(), frame_number=0)
    assert len({c["id"] for c in cands}) == 2


def test_provisional_ids_are_stable_across_frames():
    """An unconfirmed detection that has not moved keeps its id. Minting a
    fresh negative every frame made one motionless person read as a stream
    of different people to every id-keyed consumer downstream."""
    import identity_tracker as it

    class _Stub(it.IdentityTracker):
        def __init__(self):
            # Bypass the ultralytics backend: this behaviour is in the
            # provisional-id path, which runs regardless of the backend.
            self._provisional = -1
            self._prev_provisional = []
            self._ids = {}
            self._next_id = 0
            self._memo_frame = None
            self._memo = {}
            self.backend = None

        def _step(self, boxes):
            cands = [{"box": list(b), "score": 1.0} for b in boxes]
            taken = set()
            unclaimed = cands
            for c in unclaimed:
                inherited, best = None, it.PROVISIONAL_MATCH_IOU
                for pb, pid in self._prev_provisional:
                    if pid in taken:
                        continue
                    v = it._iou(c["box"], pb)
                    if v >= best:
                        inherited, best = pid, v
                if inherited is None:
                    inherited = self._provisional
                    self._provisional -= 1
                c["id"] = inherited
                taken.add(inherited)
            self._prev_provisional = [(tuple(c["box"]), c["id"]) for c in cands]
            return cands

    t = _Stub()
    a = t._step([(100, 100, 50, 50)])
    b = t._step([(102, 101, 50, 50)])   # same person, barely moved
    assert a[0]["id"] == b[0]["id"]
    c = t._step([(800, 400, 50, 50)])   # somewhere else entirely
    assert c[0]["id"] != b[0]["id"]
