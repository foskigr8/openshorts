"""Stage 3 boundary fixes: scene clamp (vision bleed rules were removed with
the vision-confirmation stage)."""
import pytest

main = pytest.importorskip("main")


SCENES = [(0.0, 100.0), (100.0, 200.0), (200.0, 300.0)]


def _candidate(start=95.0, end=205.0, clip_type=None):
    c = {"start": start, "end": end}
    if clip_type:
        c["clip_type"] = clip_type
    return c


def test_clamp_pulls_end_back_to_last_scene_boundary():
    c = _candidate(end=250.0)
    main._clamp_candidate_end_to_scene(c, SCENES)
    assert c["end"] == 200.0


def test_clamp_never_extends():
    c = _candidate(start=100.0, end=140.0)
    main._clamp_candidate_end_to_scene(c, SCENES)
    assert c["end"] == 140.0  # no boundary in range -> untouched


def test_clamp_respects_min_duration():
    # End at 105: the 100s boundary would shrink the clip below 15s.
    c = _candidate(start=95.0, end=105.0)
    main._clamp_candidate_end_to_scene(c, SCENES)
    assert c["end"] == 105.0


def test_clamp_respects_long_context_min_duration():
    # Long-context floor is 45s: 100 would leave only 5s -> untouched.
    c = _candidate(start=95.0, end=105.0, clip_type="long_context")
    main._clamp_candidate_end_to_scene(c, SCENES)
    assert c["end"] == 105.0
