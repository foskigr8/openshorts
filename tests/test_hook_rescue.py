"""_extend_start_for_preceding_question() — deterministic backstop for the
"reply-only opening" pattern (e.g. "Deal breakers? I don't have any" /
"I'm not.") that DeepSeek's own prompt rule doesn't reliably avoid on its
own — confirmed recurring on real content (30-jul-2026) even with that rule
in the prompt, so this doesn't depend on the model applying it correctly.
"""
import pytest

main = pytest.importorskip("main")


def _transcript(segments):
    return {"segments": segments}


def _seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def test_pulls_start_back_to_include_preceding_question():
    transcript = _transcript([
        _seg(2873.90, 2876.06, "What are you looking for in a relationship right now?"),
        _seg(2876.06, 2877.26, "I'm be honest. I'm not."),
    ])
    candidate = {"start": 2876.06, "end": 2964.87}
    main._extend_start_for_preceding_question(candidate, transcript)
    assert candidate["start"] == 2873.90


def test_leaves_start_alone_when_preceding_segment_is_not_a_question():
    transcript = _transcript([
        _seg(10.0, 12.0, "So anyway, that's what happened."),
        _seg(12.0, 14.0, "Right, exactly."),
    ])
    candidate = {"start": 12.0, "end": 40.0}
    main._extend_start_for_preceding_question(candidate, transcript)
    assert candidate["start"] == 12.0


def test_leaves_start_alone_when_gap_is_too_large():
    transcript = _transcript([
        _seg(0.0, 2.0, "What are you looking for?"),
        _seg(30.0, 32.0, "I'm not sure."),  # 28s gap — unrelated, not a direct reply
    ])
    candidate = {"start": 30.0, "end": 60.0}
    main._extend_start_for_preceding_question(candidate, transcript)
    assert candidate["start"] == 30.0


def test_no_preceding_segment_leaves_start_alone():
    transcript = _transcript([_seg(5.0, 10.0, "Hello there.")])
    candidate = {"start": 5.0, "end": 20.0}
    main._extend_start_for_preceding_question(candidate, transcript)
    assert candidate["start"] == 5.0


def test_ignores_segments_that_overlap_or_start_after_candidate():
    # A segment starting AFTER the candidate start shouldn't be treated as "preceding".
    transcript = _transcript([
        _seg(10.0, 20.0, "Why do you say that?"),
        _seg(15.0, 25.0, "Because I do."),  # overlaps candidate start, not a clean predecessor
    ])
    candidate = {"start": 15.0, "end": 40.0}
    main._extend_start_for_preceding_question(candidate, transcript)
    # The only segment fully ending at/before start=15.0 is none (first ends at 20 > 15),
    # so nothing should move.
    assert candidate["start"] == 15.0


def test_empty_transcript_is_a_no_op():
    candidate = {"start": 10.0, "end": 20.0}
    main._extend_start_for_preceding_question(candidate, {"segments": []})
    assert candidate["start"] == 10.0


class TestQuestionRewindLandsOnSentenceStart:
    """Rewinding must reach the whole question, not its tail.

    Segments are utterance chunks and routinely split one sentence in half, so
    "the segment ending in '?'" is often only the question's last few words.
    Ground truth from a shipped clip: the question is a single sentence
    ("When you say ... how often do you like it weekly?") split across two
    segments. Rewinding to the segment start opened the clip on "like it
    weekly?" — which tells a cold viewer nothing about what is weekly.
    """

    def _transcript(self):
        # One question sentence split across TWO segments, then the answer.
        def words(text, t0, step=0.25):
            out, t = [], t0
            for tok in text.split():
                out.append({"word": " " + tok, "start": round(t, 2),
                            "end": round(t + step, 2)})
                t += step
            return out

        seg_a_text = "When you say that, just so he knows,"
        seg_b_text = "how often do you like it weekly?"
        seg_c_text = "Daily, I'd say minimum 3 times a day."
        # Timings must not overlap: segment B has to END before the candidate
        # start, otherwise it isn't a "preceding" segment at all.
        a = words(seg_a_text, 833.72, step=0.20)
        b = words(seg_b_text, 837.20, step=0.20)
        c = words(seg_c_text, 839.10)
        return {"segments": [
            {"start": 833.72, "end": a[-1]["end"], "text": seg_a_text, "words": a},
            {"start": 837.20, "end": b[-1]["end"], "text": seg_b_text, "words": b},
            {"start": 839.10, "end": c[-1]["end"], "text": seg_c_text, "words": c},
        ]}

    def test_rewinds_past_the_segment_split_to_the_sentence_start(self):
        tr = self._transcript()
        cand = {"start": 839.10, "end": 862.0}
        main._extend_start_for_preceding_question(cand, tr)
        # 837.20 would be the segment start = "how often do you like it weekly?"
        # only. The whole question begins at 833.72.
        assert cand["start"] == pytest.approx(833.72, abs=0.05)
        assert cand["_context_start"] == pytest.approx(833.72, abs=0.05)

    def test_sentence_helper_only_moves_earlier(self):
        tr = self._transcript()
        t = 839.10
        assert main._sentence_start_at_or_before(tr, t) <= t

    def test_sentence_helper_falls_back_when_nothing_close(self):
        tr = {"segments": []}
        assert main._sentence_start_at_or_before(tr, 500.0) == 500.0

    def test_respects_the_prepend_cap(self):
        """A question far enough back to blow the cap must not be dragged in."""
        tr = self._transcript()
        cand = {"start": 839.10, "end": 862.0}
        main._extend_start_for_preceding_question(cand, tr, max_prepend=1.0)
        assert cand["start"] == pytest.approx(839.10, abs=0.01)
