"""eval/asd_bakeoff.py's model-agnostic scoring — pure Python, no ASD model,
no video, no GPU. What actually runs LR-ASD/LoCoNet/etc against real footage
can only happen on Kaggle; this pins the logic that judges whichever boxes
those runs produce.
"""
from eval import asd_bakeoff as bo


def _spine(tracks):
    """tracks: {track_id: [(t, box), ...]} -> face_spine-shaped dict."""
    return {
        tid: {"frames": [t for t, _ in samples],
              "boxes": [b for _, b in samples]}
        for tid, samples in tracks.items()
    }


def test_box_iou_identical_is_one():
    b = (0.0, 0.0, 10.0, 10.0)
    assert bo._box_iou(b, b) == 1.0


def test_box_iou_disjoint_is_zero():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (100.0, 100.0, 10.0, 10.0)
    assert bo._box_iou(a, b) == 0.0


def test_nearest_box_at_within_tolerance():
    track = {"frames": [0.0, 1.0, 2.0], "boxes": [(0, 0, 1, 1), (10, 10, 1, 1), (20, 20, 1, 1)]}
    assert bo._nearest_box_at(track, 1.1, time_tolerance=0.5) == (10, 10, 1, 1)


def test_nearest_box_at_outside_tolerance_is_none():
    track = {"frames": [0.0], "boxes": [(0, 0, 1, 1)]}
    assert bo._nearest_box_at(track, 5.0, time_tolerance=0.5) is None


def test_match_box_to_track_picks_best_overlap():
    spine = _spine({
        0: [(1.0, (0.0, 0.0, 10.0, 10.0))],
        1: [(1.0, (100.0, 100.0, 10.0, 10.0))],
    })
    # Predicted box overlaps track 0 heavily, track 1 not at all.
    result = bo.match_box_to_track(spine, 1.0, (1.0, 1.0, 10.0, 10.0))
    assert result == 0


def test_match_box_to_track_below_threshold_is_none():
    spine = _spine({0: [(1.0, (0.0, 0.0, 10.0, 10.0))]})
    # Barely-overlapping box, IoU well under the default 0.3 threshold.
    result = bo.match_box_to_track(spine, 1.0, (9.0, 9.0, 10.0, 10.0))
    assert result is None


def test_asd_predicted_track_per_second_handles_no_call():
    spine = _spine({0: [(0.0, (0.0, 0.0, 10.0, 10.0))]})
    boxes = [(0.0, 0.0, 10.0, 10.0), None]
    result = bo.asd_predicted_track_per_second(boxes, spine)
    assert result == [0, None]


def test_expected_track_per_second_basic():
    track_names = {0: "host", 1: "guest"}
    per_second_name = ["host", "host", "guest", None]
    result = bo.expected_track_per_second(track_names, per_second_name)
    assert result == [0, 0, 1, None]


def test_expected_track_per_second_unknown_name_is_none():
    result = bo.expected_track_per_second({0: "host"}, ["someone_else"])
    assert result == [None]


def test_score_model_perfect_match():
    spine = _spine({0: [(0.0, (0, 0, 10, 10)), (1.0, (0, 0, 10, 10))]})
    boxes = [(0, 0, 10, 10), (0, 0, 10, 10)]
    expected = [0, 0]
    result = bo.score_model("perfect", boxes, spine, expected)
    assert result["pct"] == 100.0
    assert result["matched"] == 2
    assert result["scored"] == 2


def test_score_model_wrong_person():
    spine = _spine({
        0: [(0.0, (0, 0, 10, 10))],
        1: [(0.0, (100, 100, 10, 10))],
    })
    # Model predicts track 1's box, but ground truth expects track 0.
    boxes = [(100, 100, 10, 10)]
    expected = [0]
    result = bo.score_model("wrong", boxes, spine, expected)
    assert result["pct"] == 0.0


def test_compare_models_ranks_best_first():
    spine = _spine({
        0: [(0.0, (0, 0, 10, 10))],
        1: [(0.0, (100, 100, 10, 10))],
    })
    expected = [0]
    models = {
        "good": [(0, 0, 10, 10)],       # matches expected track 0
        "bad": [(100, 100, 10, 10)],    # matches track 1, wrong
    }
    results = bo.compare_models(models, spine, expected)
    assert results[0]["name"] == "good"
    assert results[0]["pct"] == 100.0
    assert results[1]["name"] == "bad"
    assert results[1]["pct"] == 0.0


def test_format_comparison_names_the_winner():
    spine = _spine({0: [(0.0, (0, 0, 10, 10))]})
    expected = [0]
    results = bo.compare_models({"only_model": [(0, 0, 10, 10)]}, spine, expected)
    report = bo.format_comparison(results)
    assert "winner: only_model" in report
