"""Round-5 spec 1.3 — the deterministic context backstop.

main.py's _extend_start_for_preceding_question rewinds a clip start to include
the question its opening answers. Table-driven: dependent openings (leading
conjunction, bare pronoun, dangling numeric answer), question two segments
back, unpunctuated spoken questions — plus regression guards that
self-contained claims are NEVER rewound.
"""
import pytest

# main pulls in torch/ultralytics/cv2 — the container has the full stack (the
# spec's test command runs there); lighter hosts skip these gracefully.
main = pytest.importorskip("main")


def _transcript(*segs):
    return {"segments": [
        {"start": s, "end": e, "text": t} for s, e, t in segs
    ]}


def _run(transcript, start):
    cand = {"start": start, "end": start + 20.0}
    main._extend_start_for_preceding_question(cand, transcript)
    return cand


CASES = [
    # (label, segments before the opening, opening text, expected rewind)
    ("leading conjunction", [
        (836.0, 839.2, "So is this the one you wanted?"),
    ], "And then she just said no", True),
    ("bare pronoun", [
        (836.0, 839.2, "Where did she go after that?"),
    ], "She walked out the door", True),
    ("dangling numeric answer", [
        (834.0, 838.7, "…is it weekly?"),
    ], "Daily, I'd say minimum 3 times a day", True),
    ("question two segments back", [
        (834.0, 836.0, "…just so he knows, h"),
        (836.0, 839.2, "it weekly?"),
    ], "Daily, minimum three times", True),
    ("unpunctuated spoken question", [
        (836.0, 839.2, "Tell me how often you go"),
    ], "Every single morning", True),
    ("self-contained claim not rewound", [
        (830.0, 834.0, "Why does everyone get this wrong?"),
    ], "If you don't understand the difference, I can't help you", False),
    ("imperative opening not rewound", [
        (830.0, 834.0, "Did you hear that?"),
    ], "Stop doing this", False),
]


@pytest.mark.parametrize("label,segs,opening,expected", CASES,
                         ids=[c[0] for c in CASES])
def test_backstop_rewinds_dependent_openings(label, segs, opening, expected):
    transcript = _transcript(*segs, (840.0, 844.0, opening))
    # Candidate starts just after the question/opening boundary.
    cand = _run(transcript, 839.5)
    question_start = segs[-1][0]
    if expected:
        assert cand["start"] == question_start, label
        assert cand["_context_start"] == question_start, label
    else:
        assert cand["start"] == 839.5, label
        assert "_context_start" not in cand, label


def test_backstop_respects_max_prepend_cap():
    # A question 30s earlier must NOT drag 30s of setup into the clip.
    transcript = _transcript((800.0, 804.0, "Is that the one?"),
                             (835.0, 839.0, "Daily, minimum three times"))
    cand = _run(transcript, 838.5)
    assert cand["start"] == 838.5
    assert "_context_start" not in cand


def test_question_like_detection():
    assert main._is_question_like("Is it weekly?") is True
    assert main._is_question_like("Tell me how often") is True
    assert main._is_question_like("So walk me through it") is True
    assert main._is_question_like("He just left") is False
    assert main._is_question_like("") is False


def test_dependent_opening_detection():
    assert main._is_dependent_opening("And then she said no") is True
    assert main._is_dependent_opening("She walked out") is True
    assert main._is_dependent_opening("Daily, minimum three times") is True
    assert main._is_dependent_opening("I'm not") is True
    assert main._is_dependent_opening("If you don't understand, I can't help you") is False
    assert main._is_dependent_opening("Stop doing this") is False
