"""eval/selection/metrics.py and eval/selection/ground_truth.py are pure
Python (no LLM/network/video dependency), so unlike main.py's Stage 3 engines
they run and are testable anywhere. These tests pin the arithmetic the Stage
3 rebuild's Phase 0/5 gates rely on — see the plan's "ships with a metric
that moved" rule.
"""
import json

import pytest

from eval.selection import ground_truth, metrics


# ─── span_overlap_score ──────────────────────────────────────────────────────


def test_perfect_match_scores_full_iou():
    selected = [{"start": 100.0, "end": 150.0}]
    expected = [{"start": 100.0, "end": 150.0, "note": "exact"}]
    result = metrics.span_overlap_score(selected, expected)
    assert result.match_rate == 1.0
    assert result.mean_best_iou == pytest.approx(1.0)
    assert result.matched_count == 1


def test_no_overlap_scores_zero():
    selected = [{"start": 0.0, "end": 10.0}]
    expected = [{"start": 500.0, "end": 550.0, "note": "missed entirely"}]
    result = metrics.span_overlap_score(selected, expected)
    assert result.match_rate == 0.0
    assert result.mean_best_iou == 0.0
    assert result.matches[0].matched_start is None


def test_partial_overlap_below_threshold_does_not_count_as_matched():
    # 10s overlap out of a 100s union = 0.1 IoU, well under the 0.5 default
    selected = [{"start": 0.0, "end": 50.0}]
    expected = [{"start": 40.0, "end": 90.0, "note": "barely touches"}]
    result = metrics.span_overlap_score(selected, expected)
    assert result.matched_count == 0
    assert 0.0 < result.mean_best_iou < 0.5


def test_picks_the_best_of_multiple_candidates():
    selected = [
        {"start": 0.0, "end": 10.0},       # no overlap
        {"start": 95.0, "end": 155.0},     # good overlap
    ]
    expected = [{"start": 100.0, "end": 150.0, "note": "the real moment"}]
    result = metrics.span_overlap_score(selected, expected)
    assert result.matches[0].matched_start == 95.0
    assert result.matched_count == 1


def test_extra_selected_clips_beyond_ground_truth_are_not_penalized():
    """Scored per EXPECTED clip on purpose — ground truth is intentionally
    not exhaustive, so additional reasonable picks must not look like noise."""
    selected = [
        {"start": 100.0, "end": 150.0},
        {"start": 300.0, "end": 340.0},   # not in ground truth at all
        {"start": 500.0, "end": 520.0},
    ]
    expected = [{"start": 100.0, "end": 150.0, "note": "the labeled one"}]
    result = metrics.span_overlap_score(selected, expected)
    assert result.match_rate == 1.0


def test_no_expected_clips_is_vacuously_perfect_not_a_crash():
    result = metrics.span_overlap_score([{"start": 0.0, "end": 10.0}], [])
    assert result.expected_count == 0
    assert result.match_rate == 1.0  # nothing to have missed


def test_no_selected_clips_scores_zero_against_real_expectations():
    result = metrics.span_overlap_score([], [{"start": 10.0, "end": 20.0, "note": "x"}])
    assert result.match_rate == 0.0
    assert result.matches[0].best_iou == 0.0


# ─── must_not_include_violations ─────────────────────────────────────────────


def test_clean_selection_has_no_violations():
    selected = [{"start": 100.0, "end": 150.0}]
    forbidden = [{"start": 300.0, "end": 340.0, "note": "boring"}]
    assert metrics.must_not_include_violations(selected, forbidden) == []


def test_overlapping_forbidden_span_is_flagged():
    selected = [{"start": 290.0, "end": 350.0}]
    forbidden = [{"start": 300.0, "end": 340.0, "note": "dead air"}]
    violations = metrics.must_not_include_violations(selected, forbidden)
    assert len(violations) == 1
    assert violations[0].overlap_seconds == pytest.approx(40.0)
    assert violations[0].note == "dead air"


def test_touching_but_not_overlapping_is_not_a_violation():
    selected = [{"start": 100.0, "end": 300.0}]
    forbidden = [{"start": 300.0, "end": 340.0, "note": "adjacent, not inside"}]
    assert metrics.must_not_include_violations(selected, forbidden) == []


# ─── sentence_boundary_hit_rate ───────────────────────────────────────────────


def test_clip_on_sentence_boundaries_is_clean():
    selected = [{"start": 10.0, "end": 20.0}]
    result = metrics.sentence_boundary_hit_rate(selected, [10.0, 50.0], [20.0, 60.0])
    assert result.hit_rate == 1.0
    assert result.checks[0].start_on_sentence
    assert result.checks[0].end_on_sentence


def test_mid_sentence_cut_is_flagged_by_delta_not_just_boolean():
    """The whole point of this metric — 'still cuts sentence sometimes'
    needs a number, not a vibe."""
    selected = [{"start": 10.0, "end": 23.7}]  # end lands mid-sentence
    result = metrics.sentence_boundary_hit_rate(selected, [10.0], [30.0])
    assert result.hit_rate == 0.0
    assert result.checks[0].start_on_sentence
    assert not result.checks[0].end_on_sentence
    assert result.checks[0].end_delta == pytest.approx(6.3)


def test_tolerance_absorbs_float_rounding_not_real_slack():
    selected = [{"start": 10.001, "end": 20.0}]
    result = metrics.sentence_boundary_hit_rate(
        selected, [10.0], [20.0], tolerance=0.05)
    assert result.hit_rate == 1.0


def test_no_clips_is_vacuously_clean():
    result = metrics.sentence_boundary_hit_rate([], [1.0], [2.0])
    assert result.clip_count == 0
    assert result.hit_rate == 1.0


def test_empty_boundary_lists_never_falsely_match():
    selected = [{"start": 10.0, "end": 20.0}]
    result = metrics.sentence_boundary_hit_rate(selected, [], [])
    assert result.hit_rate == 0.0


# ─── summarize_cost_analysis ──────────────────────────────────────────────────


def test_sums_multiple_cost_entries():
    entries = [{"total_cost": 0.01}, {"total_cost": 0.02}, None]
    assert metrics.summarize_cost_analysis(entries) == pytest.approx(0.03)


def test_no_entries_returns_none_not_zero():
    """None means 'no data', distinct from a confirmed $0.00 run."""
    assert metrics.summarize_cost_analysis([]) is None
    assert metrics.summarize_cost_analysis([None, None]) is None


# ─── ground_truth loading ─────────────────────────────────────────────────────


def test_load_ground_truth_round_trips(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({
        "video": "test_vid",
        "transcript_metadata": "fixtures/test_vid_metadata.json",
        "expected_clips": [{"start": 1.0, "end": 2.0, "note": "a"}],
        "must_not_include": [{"start": 3.0, "end": 4.0, "note": "b"}],
        "notes": "hello",
    }))
    result = ground_truth.load_ground_truth(str(path))
    assert result["video"] == "test_vid"
    assert result["expected_clips"] == [{"start": 1.0, "end": 2.0, "note": "a"}]
    assert result["must_not_include"] == [{"start": 3.0, "end": 4.0, "note": "b"}]


def test_load_ground_truth_missing_file_fails_open():
    result = ground_truth.load_ground_truth("/nonexistent/path.json")
    assert result["expected_clips"] == []
    assert result["must_not_include"] == []
    assert result["video"] is None


def test_load_ground_truth_malformed_json_fails_open(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    result = ground_truth.load_ground_truth(str(path))
    assert result["expected_clips"] == []


def test_load_transcript_fixture_unwraps_metadata_json(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"transcript": {"segments": [{"start": 0, "end": 1, "text": "hi"}]}}))
    transcript = ground_truth.load_transcript_fixture(str(path))
    assert transcript == {"segments": [{"start": 0, "end": 1, "text": "hi"}]}


def test_load_transcript_fixture_accepts_bare_transcript(tmp_path):
    path = tmp_path / "bare.json"
    path.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "hi"}]}))
    transcript = ground_truth.load_transcript_fixture(str(path))
    assert transcript == {"segments": [{"start": 0, "end": 1, "text": "hi"}]}
