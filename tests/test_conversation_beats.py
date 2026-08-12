"""Conversation framing: exchange windows, vsplit splicing, ASD fallback,
panel containment."""
from shot_planner import (
    find_exchange_windows,
    plan_conversation_beats,
    Shot,
    SHOT_SINGLE,
    SHOT_TWO_SHOT,
    SHOT_VSPLIT,
)
from speaker_fusion import per_second_active_track


def _shots():
    return [
        Shot(0.0, 5.0, SHOT_SINGLE, [1], (0, 0, 100, 100)),
        Shot(5.0, 10.0, SHOT_SINGLE, [2], (0, 0, 100, 100)),
        Shot(10.0, 15.0, SHOT_SINGLE, [1], (0, 0, 100, 100)),
    ]


class TestExchangeWindows:
    def test_back_and_forth_yields_one_window(self):
        active = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        windows = find_exchange_windows(active, min_exchange_s=2.5, min_span_s=2.0)
        assert len(windows) == 1
        w_start, w_end, a, b = windows[0]
        assert {a, b} == {1, 2}
        assert w_end - w_start >= 8.0

    def test_monologue_yields_nothing(self):
        assert find_exchange_windows([1, 1, 1, 1, 1], 2.5, 2.0) == []

    def test_short_gap_between_tracks_still_counts(self):
        active = [1, None, 2, None, 1, None, 2]
        assert len(find_exchange_windows(active, min_exchange_s=2.5, min_span_s=2.0)) == 1

    def test_third_track_closes_the_window(self):
        active = [1, 2, 1, 2, 3, 3, 3, 3]
        windows = find_exchange_windows(active, min_exchange_s=2.5, min_span_s=2.0)
        assert len(windows) == 1
        assert windows[0][1] <= 4.5  # closes before the third track takes over


class TestPlanConversationBeats:
    def test_vsplit_spliced_into_single_shots(self):
        active = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        tracks = {1: {"frames": [0, 2, 4], "boxes": [(10, 10, 50, 60)] * 3},
                  2: {"frames": [1, 3, 5], "boxes": [(200, 10, 50, 60)] * 3}}
        out = plan_conversation_beats(_shots(), active, tracks)
        assert any(s.shot_type == SHOT_VSPLIT for s in out)
        # Splicing preserves coverage and order.
        assert abs(sum(s.duration for s in out) - 15.0) < 1e-6

    def test_no_exchange_leaves_shots_untouched(self):
        active = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        out = plan_conversation_beats(_shots(), active, {})
        assert [s.shot_type for s in out] == [SHOT_SINGLE] * 3


class TestTwoShotVsSplit:
    def test_close_pair_becomes_two_shot(self):
        # Two people working together in one frame → TWO_SHOT, not a split.
        active = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        tracks = {1: {"frames": [0, 2, 4, 6, 8], "boxes": [(10, 10, 50, 60)] * 5},
                  2: {"frames": [1, 3, 5, 7, 9], "boxes": [(200, 10, 50, 60)] * 5}}
        out = plan_conversation_beats(_shots(), active, tracks,
                                      frame_w=1920, frame_h=1080,
                                      aspect=16 / 9)
        assert any(s.shot_type == SHOT_TWO_SHOT for s in out)
        assert not any(s.shot_type == SHOT_VSPLIT for s in out)

    def test_far_pair_stays_vsplit(self):
        # Genuine exchange between people too far apart for one crop → split.
        active = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        tracks = {1: {"frames": [0, 2, 4, 6, 8], "boxes": [(10, 10, 50, 60)] * 5},
                  2: {"frames": [1, 3, 5, 7, 9], "boxes": [(1700, 10, 50, 60)] * 5}}
        out = plan_conversation_beats(_shots(), active, tracks,
                                      frame_w=1920, frame_h=1080,
                                      aspect=16 / 9)
        assert any(s.shot_type == SHOT_VSPLIT for s in out)
        assert not any(s.shot_type == SHOT_TWO_SHOT for s in out)

    def test_participant_not_on_screen_keeps_base_shots(self):
        # "If there is no subject to focus on, show the normal shot" — an
        # exchange with a track that has no face in the window is skipped.
        active = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        tracks = {1: {"frames": [0, 2, 4, 6, 8], "boxes": [(10, 10, 50, 60)] * 5},
                  2: {"frames": [12, 14], "boxes": [(200, 10, 50, 60)] * 2}}
        out = plan_conversation_beats(_shots(), active, tracks,
                                      frame_w=1920, frame_h=1080,
                                      aspect=16 / 9)
        assert [s.shot_type for s in out] == [SHOT_SINGLE] * 3


class TestAsdFallback:
    def test_unbound_speaker_uses_asd_track(self):
        active = per_second_active_track(["A", "B", None, "C"],
                                         {"A": 1}, [2, 2, 2, 2])
        assert active == [1, 2, 2, 2]

    def test_bound_speaker_wins(self):
        active = per_second_active_track(["A", "A", "B"],
                                         {"A": 1, "B": 3}, [2, 2, 2])
        assert active == [1, 1, 3]

    def test_no_signals_is_none(self):
        assert per_second_active_track([None, None], {}, [None, None]) == [None, None]


class TestPanelContainment:
    def test_vsplit_panels_contain_their_subjects(self):
        import reframe_v3
        subjects = [(100.0, 100.0, 120.0, 150.0), (900.0, 120.0, 110.0, 140.0)]
        aspect = 16.0 / 9.0
        for subject in subjects:
            crop = reframe_v3.crop_rect_containing(subject, 1920, 1080, aspect)
            assert reframe_v3.contains(crop, subject), crop
