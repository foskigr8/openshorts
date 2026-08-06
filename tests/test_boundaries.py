"""PART 3 boundary fixes (6-aug-2026): scene clamp + vision bleed rules."""
import main


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


def test_different_speaker_open_moves_start_later():
    c = _candidate()
    main._apply_boundary_bleed_fixes(c, {
        "different_speaker_open": True, "speaker_on_screen_at": 1.2})
    assert c["start"] == 96.2


def test_different_speaker_open_ignored_when_speaker_never_appears():
    c = _candidate()
    main._apply_boundary_bleed_fixes(c, {
        "different_speaker_open": True, "speaker_on_screen_at": 9.0})
    assert c["start"] == 95.0


def test_end_scene_bleed_trims_the_end():
    c = _candidate(end=205.0)
    main._apply_boundary_bleed_fixes(c, {
        "end_scene_bleed": True, "end_bleed_pullback": 2.0})
    assert c["end"] == 203.0


def test_end_scene_bleed_never_shrinks_below_minimum():
    c = _candidate(start=200.0, end=205.0)
    main._apply_boundary_bleed_fixes(c, {
        "end_scene_bleed": True, "end_bleed_pullback": 4.0})
    assert c["end"] == 205.0  # 201-200 = 1s < 8s -> untouched


def test_bleed_fixes_are_noops_for_clean_boundaries():
    c = _candidate()
    main._apply_boundary_bleed_fixes(c, {
        "different_speaker_open": False, "end_scene_bleed": False})
    assert c == _candidate()
