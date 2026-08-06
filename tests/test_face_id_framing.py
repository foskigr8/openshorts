"""PART 4 (6-aug-2026): face ID -> camera binding."""
import face_id
import reframe_v2


def _tracks():
    return [
        {"track_id": 0, "frames": [
            {"timestamp": 1.0, "bbox": [100, 100, 200, 240]},
            {"timestamp": 1.4, "bbox": [102, 100, 202, 240]},
        ]},
        {"track_id": 1, "frames": [
            {"timestamp": 1.1, "bbox": [400, 120, 520, 280]},
        ]},
    ]


def test_find_track_at_timestamp_matches_by_iou():
    tracks = _tracks()
    # InsightFace-style [x1,y1,x2,y2] box overlapping track 0's frame at 1.0.
    assert face_id.find_track_at_timestamp(
        tracks, 1.05, [110, 110, 190, 230]) == 0
    assert face_id.find_track_at_timestamp(
        tracks, 1.1, [410, 130, 510, 270]) == 1


def test_find_track_at_timestamp_ignores_distant_frames():
    assert face_id.find_track_at_timestamp(
        _tracks(), 5.0, [110, 110, 190, 230]) is None


def test_merge_face_id_with_tracker_votes_and_ranges():
    idents = [
        {"timestamp": 1.0, "name": "Joe", "confidence": 0.9,
         "bbox": [110, 110, 190, 230]},
        {"timestamp": 1.4, "name": "Joe", "confidence": 0.88,
         "bbox": [112, 110, 192, 230]},
        {"timestamp": 1.1, "name": "Ann", "confidence": 0.7,
         "bbox": [410, 130, 510, 270]},
    ]
    merged = face_id.merge_face_id_with_tracker(_tracks(), idents)
    assert merged[0]["name"] == "Joe"
    assert merged[0]["on_screen"] == [[1.0, 1.4]]
    assert merged[1]["name"] == "Ann"


def test_merge_face_id_with_tracker_requires_agreement():
    # Two votes for different names on the same track -> nothing committed.
    idents = [
        {"timestamp": 1.0, "name": "Joe", "confidence": 0.9,
         "bbox": [110, 110, 190, 230]},
        {"timestamp": 1.4, "name": "Ann", "confidence": 0.9,
         "bbox": [112, 110, 192, 230]},
    ]
    assert face_id.merge_face_id_with_tracker(_tracks(), idents) == {}


def test_ident_bbox_normalization():
    assert reframe_v2._ident_bbox_xywh([100, 200, 300, 500]) == (100, 200, 200, 300)


def test_accumulate_face_id_votes():
    candidates = [{"id": 7, "box": (110, 110, 80, 120)}]
    idents = [
        {"timestamp": 1.0, "name": "Joe", "bbox": [110, 110, 190, 230]},
        {"timestamp": 1.4, "name": "Ann", "bbox": [110, 110, 190, 230]},
    ]
    votes = reframe_v2._accumulate_face_id_votes(
        candidates, idents, frame_number=30, fps=30.0, votes={})
    assert votes[7] == {"Joe": 1, "Ann": 1}


def test_face_id_binding_fresh_identification():
    candidates = [{"id": 7, "box": (110, 110, 80, 120)}]
    idents = [{"timestamp": 1.0, "name": "Joe", "bbox": [110, 110, 190, 230]}]
    assert reframe_v2._face_id_binding(
        candidates, {}, "Joe", frame_number=30, fps=30.0, face_identifications=idents) == 7


def test_face_id_binding_requires_name_match():
    candidates = [{"id": 7, "box": (110, 110, 80, 120)}]
    idents = [{"timestamp": 1.0, "name": "Ann", "bbox": [110, 110, 190, 230]}]
    assert reframe_v2._face_id_binding(
        candidates, {}, "Joe", frame_number=30, fps=30.0, face_identifications=idents) is None


def test_face_id_binding_vote_path():
    candidates = [{"id": 7, "box": (110, 110, 80, 120)}]
    votes = {7: {"Joe": 4, "Ann": 1}}
    assert reframe_v2._face_id_binding(
        candidates, votes, "Joe", frame_number=30, fps=30.0, face_identifications=None) == 7


def test_face_id_binding_ignores_weak_votes():
    candidates = [{"id": 7, "box": (110, 110, 80, 120)}]
    votes = {7: {"Joe": 1}}
    assert reframe_v2._face_id_binding(
        candidates, votes, "Joe", frame_number=30, fps=30.0, face_identifications=None) is None
