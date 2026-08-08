"""eval/metrics.py and eval/ground_truth.py are pure Python (no cv2/torch/
ffmpeg), so unlike the rest of the framing stack they run and are testable
outside the Docker/Kaggle environment. These tests pin the arithmetic the
Phase 0 gates rely on.
"""
from eval import ground_truth, metrics


def test_crop_motion_zero_when_static():
    rects = [(100, 100, 500, 700)] * 10
    result = metrics.crop_motion_within_shot(rects, [(0, 10)])
    assert result[0].max_delta_px == 0.0
    assert result[0].mean_delta_px == 0.0
    assert result[0].frame_count == 10


def test_crop_motion_detects_drift():
    rects = [(100 + i * 3, 100, 500, 700) for i in range(10)]
    result = metrics.crop_motion_within_shot(rects, [(0, 10)])
    assert result[0].max_delta_px == 3.0
    assert result[0].mean_delta_px == 3.0


def test_crop_motion_multiple_shots_isolated():
    # shot 0 static, shot 1 drifts — the drift must not leak into shot 0's numbers
    rects = [(0, 0, 100, 100)] * 5 + [(0 + i * 10, 0, 100, 100) for i in range(5)]
    result = metrics.crop_motion_within_shot(rects, [(0, 5), (5, 10)])
    assert result[0].max_delta_px == 0.0
    assert result[1].max_delta_px == 10.0


def test_shot_count_and_durations():
    count, durations = metrics.shot_count_and_durations([(0, 30), (30, 90)], fps=30)
    assert count == 2
    assert durations == [1.0, 2.0]


def test_speaker_on_screen_pct_excludes_none():
    framed = ["A", "A", "B", "B", "A"]
    truth = ["A", "A", "B", None, "A"]
    pct, matched, scored = metrics.speaker_on_screen_pct(framed, truth)
    assert scored == 4  # the None second is excluded
    assert matched == 4
    assert pct == 100.0


def test_speaker_on_screen_pct_wrong_person():
    framed = ["A", "A", "A", "A"]
    truth = ["A", "B", "B", "A"]
    pct, matched, scored = metrics.speaker_on_screen_pct(framed, truth)
    assert scored == 4
    assert matched == 2
    assert pct == 50.0


def test_speaker_on_screen_pct_all_none_is_zero_not_crash():
    pct, matched, scored = metrics.speaker_on_screen_pct(["A"], [None])
    assert scored == 0
    assert matched == 0
    assert pct == 0.0


def test_boundary_sentence_completeness_true_on_exact_edges():
    words = [
        {"w": "Hello", "s": 0.0, "e": 0.5},
        {"w": "world.", "s": 0.5, "e": 1.0},
        {"w": "Next", "s": 1.2, "e": 1.5},
        {"w": "sentence!", "s": 1.5, "e": 2.0},
    ]
    starts_ok, ends_ok = metrics.boundary_sentence_completeness(1.2, 2.0, words)
    assert starts_ok is True
    assert ends_ok is True


def test_boundary_sentence_completeness_false_mid_sentence():
    words = [
        {"w": "Hello", "s": 0.0, "e": 0.5},
        {"w": "world.", "s": 0.5, "e": 1.0},
        {"w": "Next", "s": 1.2, "e": 1.5},
        {"w": "sentence!", "s": 1.5, "e": 2.0},
    ]
    # starts at 1.35, well outside the 0.35s tolerance around the nearest
    # sentence start (1.2) -> genuinely mid-sentence, not just lead-in slack
    starts_ok, ends_ok = metrics.boundary_sentence_completeness(1.35, 1.7, words, tolerance=0.05)
    assert starts_ok is False
    assert ends_ok is False


def test_boundary_sentence_completeness_respects_tolerance():
    words = [{"w": "Hi.", "s": 0.0, "e": 0.5}]
    starts_ok, ends_ok = metrics.boundary_sentence_completeness(0.3, 0.5, words, tolerance=0.35)
    assert starts_ok is True  # within 0.35s of the only sentence start (0.0)


def test_format_report_includes_headline_number():
    motions = metrics.crop_motion_within_shot([(0, 0, 10, 10)] * 3, [(0, 3)])
    report = metrics.format_report("test", motions, (87.5, 7, 8))
    assert "87.5%" in report
    assert "7/8" in report


def test_load_speaker_names_missing_file_fails_open(tmp_path):
    names = ground_truth.load_speaker_names(str(tmp_path / "does_not_exist.json"))
    assert names == {}


def test_load_speaker_names_reads_mapping(tmp_path):
    p = tmp_path / "gt.json"
    p.write_text('{"speaker_names": {"A": "host", "B": "guest"}}')
    names = ground_truth.load_speaker_names(str(p))
    assert names == {"A": "host", "B": "guest"}


def test_per_second_speaker_from_transcript_basic():
    segments = [
        {"start": 0.0, "end": 2.5, "speaker": "A"},
        {"start": 2.5, "end": 5.0, "speaker": "B"},
    ]
    result = ground_truth.per_second_speaker_from_transcript(segments, 0.0, 5.0)
    assert result[0] == "A"
    assert result[1] == "A"
    assert result[4] == "B"
    assert len(result) == 5


def test_per_second_speaker_from_transcript_resolves_names():
    segments = [{"start": 0.0, "end": 2.0, "speaker": "A"}]
    result = ground_truth.per_second_speaker_from_transcript(
        segments, 0.0, 2.0, speaker_names={"A": "host"})
    assert result[0] == "host"


def test_per_second_speaker_from_transcript_gap_is_none():
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 3.0, "end": 4.0, "speaker": "B"},
    ]
    result = ground_truth.per_second_speaker_from_transcript(segments, 0.0, 4.0)
    assert result[0] == "A"
    assert result[1] is None
    assert result[2] is None
    assert result[3] == "B"


def test_per_second_speaker_from_transcript_clips_outside_span_ignored():
    segments = [
        {"start": -10.0, "end": -5.0, "speaker": "A"},
        {"start": 0.0, "end": 1.0, "speaker": "B"},
        {"start": 100.0, "end": 101.0, "speaker": "C"},
    ]
    result = ground_truth.per_second_speaker_from_transcript(segments, 0.0, 2.0)
    assert result == ["B", None]
