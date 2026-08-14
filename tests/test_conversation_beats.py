"""Conversation framing: exchange windows, vsplit splicing, ASD fallback,
panel containment."""
import pytest

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
        # Both tracks are on screen for the WHOLE window — which is what a
        # real two-person exchange looks like, and what a panel commits to.
        tracks = {1: {"frames": [0, 2, 4, 6, 8], "boxes": [(10, 10, 50, 60)] * 5},
                  2: {"frames": [1, 3, 5, 7, 9], "boxes": [(200, 10, 50, 60)] * 5}}
        out = plan_conversation_beats(_shots(), active, tracks)
        assert any(s.shot_type == SHOT_VSPLIT for s in out)
        # Splicing preserves coverage and order.
        assert abs(sum(s.duration for s in out) - 15.0) < 1e-6

    def test_a_participant_who_leaves_the_frame_gets_no_panel(self):
        """`_track_present_in_span` used to be `any()`: ONE detection anywhere
        in the window bought a participant half the frame for its entire
        duration. Track 2 here is on screen for the first two seconds and then
        gone — a panel held on them would be an empty chair."""
        active = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        tracks = {1: {"frames": [0, 2, 4, 6, 8], "boxes": [(10, 10, 50, 60)] * 5},
                  2: {"frames": [1], "boxes": [(200, 10, 50, 60)]}}
        out = plan_conversation_beats(_shots(), active, tracks)
        assert not any(s.shot_type == SHOT_VSPLIT for s in out)

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
        # Faces are 150px tall: a real mid-shot in a 1080p source, and big
        # enough that a panel can frame them to the contract. See
        # test_far_pair_too_small_to_panel_falls_back for the other case.
        active = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        tracks = {1: {"frames": [0, 2, 4, 6, 8], "boxes": [(10, 300, 120, 150)] * 5},
                  2: {"frames": [1, 3, 5, 7, 9], "boxes": [(1700, 300, 120, 150)] * 5}}
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
    def test_no_label_falls_back_to_asd_but_an_unmapped_speaker_goes_wide(self):
        """The no-guess rule (plan §5.6), and the distinction it rests on.

        Second 1 and 3: we KNOW who is talking ("B", "C") and cannot find
        their face. Falling back to ASD there shows whoever the model liked
        that second — a laughing listener, the nearest torso — which is the
        "host audio over a picture of the group" failure. None means WIDE.

        Second 2: nobody told us who is talking at all, so LR-ASD is the best
        answer available and still wins.
        """
        active = per_second_active_track(["A", "B", None, "C"],
                                         {"A": 1}, [2, 2, 2, 2])
        assert active == [1, None, 2, None]

    def test_the_old_guessing_behaviour_is_still_reachable(self):
        active = per_second_active_track(["A", "B", None, "C"],
                                         {"A": 1}, [2, 2, 2, 2],
                                         unmapped_policy="asd")
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


class TestSplitAdmission:
    """The gate has to push in BOTH directions (plan §5.4). If tightening it
    only ever produces fewer splits, the far-apart test is mis-wired."""

    def _tracks(self, x_a, x_b, h=150):
        w = int(0.8 * h)
        return {1: {"frames": [0, 2, 4, 6, 8],
                    "boxes": [(x_a, 300, w, h)] * 5},
                2: {"frames": [1, 3, 5, 7, 9],
                    "boxes": [(x_b, 300, w, h)] * 5}}

    def _plan(self, tracks):
        active = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        return plan_conversation_beats(_shots(), active, tracks,
                                       frame_w=1920, frame_h=1080,
                                       aspect=16 / 9)

    def test_side_by_side_prefers_the_two_shot(self):
        out = self._plan(self._tracks(700, 1000))
        assert any(s.shot_type == SHOT_TWO_SHOT for s in out)
        assert not any(s.shot_type == SHOT_VSPLIT for s in out)

    def test_far_apart_prefers_the_split(self):
        out = self._plan(self._tracks(40, 1760))
        assert any(s.shot_type == SHOT_VSPLIT for s in out)

    def test_far_pair_too_small_to_panel_falls_back(self):
        """The resolution budget (plan §5.7). At 60px a face cannot reach the
        contract's subject size inside a panel, so the split is refused rather
        than shipped as two mushy upscales."""
        out = self._plan(self._tracks(40, 1760, h=60))
        assert not any(s.shot_type == SHOT_VSPLIT for s in out)

    def test_a_split_that_is_admitted_has_workable_panel_geometry(self):
        import framing_contract as fc
        tracks = self._tracks(40, 1760)
        out = self._plan(tracks)
        assert any(s.shot_type == SHOT_VSPLIT for s in out)
        panel_aspect = (9.0 / 16.0) * 2.0
        a, b = tracks[1]["boxes"][0], tracks[2]["boxes"][0]
        panels = (fc.frame_panel(a, 1920, 1080, panel_aspect),
                  fc.frame_panel(b, 1920, 1080, panel_aspect))
        assert fc.check_panels(panels, [a, b], 1920, 1080,
                               panel_aspect=panel_aspect) == []


class TestAdmissionThresholds:
    """The three §5.4 rows that were still missing: switch count, legibility,
    and the split-share diagnostic."""

    def _tracks(self, x_a=40, x_b=1760, h=150):
        w = int(0.8 * h)
        return {1: {"frames": [0, 2, 4, 6, 8], "boxes": [(x_a, 300, w, h)] * 5},
                2: {"frames": [1, 3, 5, 7, 9], "boxes": [(x_b, 300, w, h)] * 5}}

    def test_a_single_handover_is_not_an_exchange(self):
        """A -> B once is an ordinary speaker change, and reads better as a
        cut than as a split. Two switches used to be enough."""
        assert find_exchange_windows([1, 1, 1, 2, 2, 2], 2.5, 2.5) == []

    def test_three_switches_qualify(self):
        assert len(find_exchange_windows([1, 2, 1, 2, 1, 2], 2.5, 2.5)) == 1

    def test_a_window_without_two_full_sentences_is_not_split(self):
        out = plan_conversation_beats(
            _shots(), [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], self._tracks(),
            frame_w=1920, frame_h=1080, aspect=16 / 9,
            sentence_ends=[3.0])          # only one sentence closes in-window
        assert not any(s.shot_type == SHOT_VSPLIT for s in out)

    def test_two_full_sentences_admit_the_split(self):
        out = plan_conversation_beats(
            _shots(), [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], self._tracks(),
            frame_w=1920, frame_h=1080, aspect=16 / 9,
            sentence_ends=[3.0, 6.5])
        assert any(s.shot_type == SHOT_VSPLIT for s in out)

    def test_the_sentence_gate_fails_open_without_a_transcript(self):
        """Callers with no word timings keep the previous behaviour rather
        than losing every split."""
        out = plan_conversation_beats(
            _shots(), [1, 2, 1, 2, 1, 2, 1, 2, 1, 2], self._tracks(),
            frame_w=1920, frame_h=1080, aspect=16 / 9)
        assert any(s.shot_type == SHOT_VSPLIT for s in out)

    def test_split_share_is_reported_not_enforced(self, capsys):
        import shot_planner as sp
        shots = [Shot(0.0, 9.0, SHOT_VSPLIT, [1, 2]),
                 Shot(9.0, 10.0, SHOT_SINGLE, [1])]
        assert sp._warn_on_split_share(shots) == pytest.approx(0.9)
        assert "split screen on 90%" in capsys.readouterr().out
        # ...and the shots themselves are untouched: no silent downgrade.
        assert shots[0].shot_type == SHOT_VSPLIT

    def test_a_normal_split_share_says_nothing(self, capsys):
        import shot_planner as sp
        shots = [Shot(0.0, 3.0, SHOT_VSPLIT, [1, 2]),
                 Shot(3.0, 10.0, SHOT_SINGLE, [1])]
        sp._warn_on_split_share(shots)
        assert capsys.readouterr().out == ""
