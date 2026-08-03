"""Mid-sentence jump-cutting: DeepSeek's keep_spans mark the sub-ranges of
a candidate that are actually essential — everything else in [start, end]
gets cut as a jump cut, not just trimmed off the boundaries. Confirmed as
the single most consistent pattern across 6 real published shorts studied
(RESEARCH_viral_clip_patterns.md §4) and not something we built before.
"""
import pytest

main = pytest.importorskip("main")


def _word(w, s, e):
    return {'w': w, 's': s, 'e': e}


class TestSnapKeepSpansToWords:
    def test_empty_input_returns_empty(self):
        assert main._snap_keep_spans_to_words([], [], 0.0, 10.0) == []

    def test_snaps_to_nearest_word_boundaries(self):
        words = [_word('a', 0.0, 1.0), _word('b', 1.0, 2.0), _word('c', 2.0, 3.0),
                 _word('d', 3.0, 4.0), _word('e', 4.0, 5.0)]
        # proposed span 1.1-3.9 should snap out to the nearest word edges (1.0, 4.0)
        spans = [{'start': 1.1, 'end': 3.9}]
        result = main._snap_keep_spans_to_words(spans, words, 0.0, 5.0)
        assert result == [[1.0, 4.0]]

    def test_clamps_to_clip_bounds(self):
        words = [_word('a', -5.0, 0.5), _word('b', 0.5, 1.0), _word('c', 8.5, 9.5), _word('d', 9.5, 20.0)]
        spans = [{'start': -3.0, 'end': 25.0}]
        result = main._snap_keep_spans_to_words(spans, words, 0.0, 10.0)
        assert result[0][0] >= 0.0
        assert result[0][1] <= 10.0

    def test_drops_spans_below_minimum_duration(self):
        words = [_word('a', 0.0, 0.3), _word('b', 0.3, 0.6)]
        spans = [{'start': 0.0, 'end': 0.4}]  # snaps to [0.0, 0.6] = 0.6s < min 1.0
        result = main._snap_keep_spans_to_words(spans, words, 0.0, 10.0, min_span_duration=1.0)
        assert result == []

    def test_merges_touching_spans_after_snapping(self):
        words = [_word('a', 0.0, 2.0), _word('b', 2.0, 4.0), _word('c', 4.0, 6.0)]
        # two spans that both snap to cover [0,4] and [4,6] -> touching -> merge
        spans = [{'start': 0.1, 'end': 3.9}, {'start': 4.1, 'end': 5.9}]
        result = main._snap_keep_spans_to_words(spans, words, 0.0, 6.0)
        assert result == [[0.0, 6.0]]

    def test_keeps_distinct_nonoverlapping_spans_separate(self):
        words = [_word('a', 0.0, 1.0), _word('b', 5.0, 6.0), _word('c', 9.0, 10.0)]
        spans = [{'start': 0.0, 'end': 1.0}, {'start': 9.0, 'end': 10.0}]
        result = main._snap_keep_spans_to_words(spans, words, 0.0, 10.0)
        assert result == [[0.0, 1.0], [9.0, 10.0]]

    def test_falls_back_to_clamped_bounds_with_no_nearby_words(self):
        spans = [{'start': 2.0, 'end': 5.0}]
        result = main._snap_keep_spans_to_words(spans, [], 0.0, 10.0)
        assert result == [[2.0, 5.0]]

    def test_zero_or_negative_length_span_is_dropped(self):
        words = [_word('a', 0.0, 5.0)]
        spans = [{'start': 3.0, 'end': 3.0}, {'start': 5.0, 'end': 2.0}]
        result = main._snap_keep_spans_to_words(spans, words, 0.0, 10.0)
        assert result == []

    def test_results_sorted_even_if_input_unordered(self):
        words = [_word('a', 0.0, 1.0), _word('b', 5.0, 6.0)]
        spans = [{'start': 5.0, 'end': 6.0}, {'start': 0.0, 'end': 1.0}]
        result = main._snap_keep_spans_to_words(spans, words, 0.0, 10.0)
        assert result == [[0.0, 1.0], [5.0, 6.0]]


class TestRemapTranscriptOntoJumpCut:
    def _transcript(self, segments):
        return {"language": "en", "segments": segments}

    def test_single_span_shifts_onto_zero_based_timeline(self):
        transcript = self._transcript([
            {"start": 10.0, "end": 12.0, "text": "hello there",
             "words": [{"word": "hello", "start": 10.0, "end": 10.5},
                       {"word": "there", "start": 10.5, "end": 12.0}]},
        ])
        remapped, duration = main._remap_transcript_onto_jump_cut(transcript, [[10.0, 12.0]])
        assert duration == 2.0
        seg = remapped["segments"][0]
        assert seg["start"] == 0.0 and seg["end"] == 2.0
        assert seg["words"][0]["start"] == 0.0
        assert seg["words"][1]["end"] == 2.0

    def test_content_outside_keep_spans_is_dropped(self):
        transcript = self._transcript([
            {"start": 0.0, "end": 2.0, "text": "cut this", "words": []},
            {"start": 10.0, "end": 12.0, "text": "keep this", "words": []},
            {"start": 20.0, "end": 22.0, "text": "cut this too", "words": []},
        ])
        remapped, duration = main._remap_transcript_onto_jump_cut(transcript, [[10.0, 12.0]])
        assert len(remapped["segments"]) == 1
        assert remapped["segments"][0]["text"] == "keep this"

    def test_second_span_offset_by_first_spans_duration(self):
        transcript = self._transcript([
            {"start": 0.0, "end": 3.0, "text": "first", "words": []},
            {"start": 50.0, "end": 52.0, "text": "second", "words": []},
        ])
        # first span is 3s long (0-3), so second span's content should start at t=3.0
        remapped, duration = main._remap_transcript_onto_jump_cut(
            transcript, [[0.0, 3.0], [50.0, 52.0]])
        assert duration == 5.0
        segs = sorted(remapped["segments"], key=lambda s: s["start"])
        assert segs[0]["start"] == 0.0 and segs[0]["end"] == 3.0
        assert segs[1]["start"] == 3.0 and segs[1]["end"] == 5.0

    def test_segment_partially_overlapping_a_span_gets_clipped(self):
        transcript = self._transcript([
            {"start": 8.0, "end": 14.0, "text": "spans the boundary",
             "words": [{"word": "spans", "start": 8.0, "end": 9.0},
                       {"word": "boundary", "start": 13.0, "end": 14.0}]},
        ])
        remapped, duration = main._remap_transcript_onto_jump_cut(transcript, [[10.0, 12.0]])
        seg = remapped["segments"][0]
        # clipped to the keep_span's own bounds, then shifted to start at 0
        assert seg["start"] == 0.0
        assert seg["end"] == 2.0
        # the word entirely before the span (8.0-9.0) is dropped
        assert len(seg["words"]) == 0

    def test_empty_keep_spans_yields_empty_transcript(self):
        transcript = self._transcript([{"start": 0.0, "end": 5.0, "text": "x", "words": []}])
        remapped, duration = main._remap_transcript_onto_jump_cut(transcript, [])
        assert remapped["segments"] == []
        assert duration == 0.0


class TestBuildJumpCutSource:
    def test_cuts_each_span_and_concatenates(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            # simulate ffmpeg producing the output file at the last arg
            out_path = cmd[-1]
            with open(out_path, 'wb') as f:
                f.write(b'fake')
            return __import__('subprocess').CompletedProcess(cmd, 0)

        monkeypatch.setattr(main.subprocess, 'run', fake_run)
        workdir = str(tmp_path)
        result = main._build_jump_cut_source('source.mp4', [[1.0, 3.0], [10.0, 12.0]], workdir)

        # one ffmpeg call per span, plus one concat call
        cut_calls = [c for c in calls if '-ss' in c]
        concat_calls = [c for c in calls if '-f' in c and 'concat' in c]
        assert len(cut_calls) == 2
        assert len(concat_calls) == 1
        assert cut_calls[0][cut_calls[0].index('-ss') + 1] == '1.000'
        assert cut_calls[1][cut_calls[1].index('-to') + 1] == '12.000'
        assert result.endswith('jump_cut_combined.mp4')
