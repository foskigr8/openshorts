"""Tests for the pure clip-selection helpers (windows, snapping, pricing)."""
from clip_selection import (
    build_transcript_windows,
    snap_clip_to_words,
    sentence_boundaries,
    compact_words,
    lookup_model_prices,
    lookup_deepseek_model_prices,
)


def _seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def _word(w, s, e):
    return {"w": w, "s": s, "e": e}


class TestBuildTranscriptWindows:
    def test_windows_align_to_segment_boundaries(self):
        transcript = {"segments": [
            _seg(0, 40, "a"), _seg(40, 80, "b"), _seg(80, 100, "c"), _seg(100, 150, "d"),
        ]}
        windows = build_transcript_windows(transcript, 150, window_seconds=90, overlap_seconds=30)
        segment_edges = {0, 40, 80, 100, 150}
        for w in windows:
            assert w["start"] in segment_edges
            assert w["end"] in segment_edges

    def test_windows_overlap(self):
        transcript = {"segments": [_seg(i * 10, (i + 1) * 10, f"s{i}") for i in range(30)]}
        windows = build_transcript_windows(transcript, 300, window_seconds=90, overlap_seconds=30)
        assert len(windows) >= 3
        for prev, nxt in zip(windows, windows[1:]):
            # next window starts before the previous one ends (overlap)
            assert nxt["start"] < prev["end"]
        # full coverage to the end
        assert windows[-1]["end"] == 300

    def test_empty_transcript_falls_back_to_full_video(self):
        windows = build_transcript_windows({"segments": []}, 120)
        assert len(windows) == 1
        assert windows[0]["start"] == 0.0
        assert windows[0]["end"] == 120

    def test_always_progresses(self):
        # One giant segment must not loop forever
        transcript = {"segments": [_seg(0, 500, "long monolog")]}
        windows = build_transcript_windows(transcript, 500, window_seconds=90, overlap_seconds=30)
        assert len(windows) == 1


class TestSnapClipToWords:
    def _words(self):
        # words every ~2s with 0.4s gaps: [0,1.6], [2,3.6], [4,5.6], ...
        return [_word(f"w{i}", i * 2.0, i * 2.0 + 1.6) for i in range(40)]

    def test_start_snaps_into_silence_before_word(self):
        words = self._words()
        # Gemini proposes 10.3 — nearest word start is 10.0, gap before is 9.6->10.0
        start, end = snap_clip_to_words(10.3, 30.1, words, 80.0)
        assert 9.8 <= start <= 10.0  # word start minus half-gap lead
        # end 30.1 -> nearest word end 29.6 plus tail
        assert 29.6 <= end <= 30.05

    def test_no_words_nearby_keeps_original(self):
        words = [_word("far", 200.0, 201.0)]
        assert snap_clip_to_words(10.0, 40.0, words, 300.0) == (10.0, 40.0)

    def test_empty_words_keeps_original(self):
        assert snap_clip_to_words(5.0, 25.0, [], 100.0) == (5.0, 25.0)

    def test_duration_repaired_to_minimum(self):
        words = self._words()
        # snapping would yield ~14.4s; must be extended to >= 15s on a word end
        start, end = snap_clip_to_words(10.0, 24.5, words, 80.0)
        assert end - start >= 15.0

    def test_duration_capped_at_maximum(self):
        # Explicit max_duration=60 here to test the capping mechanism itself,
        # independent of whatever the default ceiling is set to.
        words = self._words()
        start, end = snap_clip_to_words(0.0, 59.9, words, 80.0, max_duration=60.0)
        assert end - start <= 60.0

    def test_default_max_duration_no_longer_60(self):
        # Narrative-driven clips are allowed to run well past the old 60s
        # window cap — length is dictated by narrative closure, not a fixed
        # target. A ~100s proposal must survive uncapped under the default.
        words = [_word(f"w{i}", i * 2.0, i * 2.0 + 1.6) for i in range(60)]  # up to ~120s
        start, end = snap_clip_to_words(0.0, 100.0, words, 150.0)
        assert end - start > 60.0

    def test_runaway_proposal_still_capped_at_safety_ceiling(self):
        # A degenerate/runaway proposal (e.g. a hallucinated end time) must
        # still be clamped to the real working max_duration bound (180s).
        words = [_word(f"w{i}", i * 5.0, i * 5.0 + 4.0) for i in range(200)]  # spans ~1000s
        start, end = snap_clip_to_words(0.0, 900.0, words, 1000.0)
        assert end - start <= 180.0


class TestSentenceSnappingContextLock:
    """Round-5 spec 1.2: the question backstop pulls a start back, but word/
    sentence snapping re-truncated it because the ANSWER's sentence start was
    nearest. Start selection must bias earlier (prefer_before) and a recorded
    context_start must lock the boundary so it can never move later again."""

    def _question_answer_words(self):
        # Real failing shape (Blind_Dating_Girls_By_Celebrity_Lookalikes):
        # previous sentence ends ~834, the question "…is it weekly?" occupies
        # 838.14-838.70, the answer "Daily…" starts 838.86. Model start 839.10.
        words = [
            _word(" old", 833.9, 834.4),
            _word("is", 838.14, 838.34),
            _word("it", 838.34, 838.50),
            _word("weekly?", 838.50, 838.70),
            _word("Daily,", 838.86, 839.20),
            _word("I'd", 839.20, 839.36),
            _word("say", 839.36, 839.52),
            _word("minimum", 839.52, 839.78),
            _word("3", 839.78, 839.92),
            _word("times", 839.92, 840.18),
            _word("a", 840.18, 840.30),
            _word("day", 840.30, 840.55),
        ]
        return words

    def test_context_lock_keeps_question_included(self):
        words = self._question_answer_words()
        start, end = snap_clip_to_words(
            839.10, 900.0, words, 950.0, context_start=838.14)
        # The question must stay included: never later than the locked start.
        assert start <= 838.14 + 1e-6
        # And it must actually be a clean boundary (not floating mid-word).
        assert start >= 838.14 - 0.4

    def test_prefer_before_biases_start_earlier_without_lock(self):
        # No lock: proposal at 8.0 with a sentence start at 7.8 (answer) and an
        # earlier one at 6.0 — prefer_before must take 7.8 (at/before), and the
        # later 8.6 boundary must NOT win even though it is 0.6s away (the
        # answer's own start is only 0.2s before the proposal).
        words = [_word("prev.", 5.6, 6.0), _word("A.", 7.8, 8.0),
                 _word("B.", 8.6, 9.0)]
        starts, _ends = sentence_boundaries(words)
        assert starts == [5.6, 7.8, 8.6]
        start, _end = snap_clip_to_words(8.0, 20.0, words, 60.0)
        assert start <= 7.8 + 0.4

    def test_start_snaps_without_lock_when_clean_boundary_near(self):
        words = [_word("Question?", 9.0, 9.6), _word(" Answer", 10.0, 10.3),
                 _word(" here.", 10.3, 10.7)]
        start, end = snap_clip_to_words(10.1, 20.0, words, 60.0)
        assert 9.7 <= start <= 10.1

    def test_end_still_biases_later(self):
        # Existing end behaviour must be unchanged: prefer the sentence end
        # AFTER the proposal when it's not dramatically farther.
        words = [_word("one.", 0.0, 0.5), _word(" two.", 28.0, 29.6),
                 _word(" three.", 30.6, 31.2), _word(" four.", 40.0, 40.5)]
        start, end = snap_clip_to_words(0.0, 30.1, words, 60.0)
        assert end >= 31.0  # completes the sentence in progress

    def test_sentence_boundaries_unchanged(self):
        words = [_word(" One", 0.0, 0.4), _word("two.", 0.4, 0.9),
                 _word("Three", 1.0, 1.4), _word("four?", 1.4, 1.9)]
        assert sentence_boundaries(words) == ([0.0, 1.0], [0.9, 1.9])


class TestPricing:
    def test_known_models(self):
        assert lookup_model_prices("gemini-2.5-flash") == (0.30, 2.50)
        assert lookup_model_prices("gemini-3-flash-preview") == (0.50, 3.00)

    def test_prefix_match_with_suffix(self):
        assert lookup_model_prices("gemini-2.5-flash-002") == (0.30, 2.50)

    def test_unknown_model_returns_none(self):
        assert lookup_model_prices("gpt-9-mega") is None
        assert lookup_model_prices(None) is None


class TestDeepseekPricing:
    def test_known_models(self):
        assert lookup_deepseek_model_prices("deepseek-v4-flash") == (0.14, 0.28)
        assert lookup_deepseek_model_prices("deepseek-v4-pro") == (0.435, 0.87)

    def test_prefix_match_with_suffix(self):
        assert lookup_deepseek_model_prices("deepseek-v4-flash-2026-07-01") == (0.14, 0.28)

    def test_unknown_model_returns_none(self):
        assert lookup_deepseek_model_prices("gpt-9-mega") is None
        assert lookup_deepseek_model_prices(None) is None

    def test_independent_from_gemini_price_table(self):
        # Regression guard for the _lookup_prices(table, ...) refactor: the
        # two tables must never bleed into each other.
        assert lookup_model_prices("deepseek-v4-flash") is None
        assert lookup_deepseek_model_prices("gemini-2.5-flash") is None


class TestCompactWords:
    def test_rounds_timestamps(self):
        words = [{"w": " hi", "s": 17.240000000000002, "e": 17.899999999999999}]
        assert compact_words(words) == [{"w": " hi", "s": 17.24, "e": 17.9}]
