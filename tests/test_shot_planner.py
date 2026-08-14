"""shot_planner.py is the actual jitter fix and the actual "no re-centering
after a cut" fix — the two most specific, most repeated complaints in this
rebuild. Pure Python, no video/GPU, so every claim here is checked directly.
"""
import pytest

import shot_planner as sp


def _spine(tracks):
    """tracks: {track_id: [(t, box), ...]} -> face_spine-shaped dict."""
    return {
        tid: {"frames": [t for t, _ in samples],
              "boxes": [b for _, b in samples]}
        for tid, samples in tracks.items()
    }


# ---------------------------------------------------------------------------
# hold_fill
# ---------------------------------------------------------------------------

def test_hold_fill_carries_forward():
    samples = [(0.0, 0), (1.0, None), (2.0, None), (3.0, 1)]
    assert sp.hold_fill(samples) == [(0.0, 0), (1.0, 0), (2.0, 0), (3.0, 1)]


def test_hold_fill_leading_none_stays_none():
    samples = [(0.0, None), (1.0, None), (2.0, 0)]
    assert sp.hold_fill(samples) == [(0.0, None), (1.0, None), (2.0, 0)]


# ---------------------------------------------------------------------------
# raw_runs
# ---------------------------------------------------------------------------

def test_raw_runs_basic_segmentation():
    filled = [(0.0, 0), (1.0, 0), (2.0, 1), (3.0, 1)]
    runs = sp.raw_runs(filled, total_duration=4.0)
    assert runs == [(0.0, 2.0, 0), (2.0, 4.0, 1)]


def test_raw_runs_coalesces_non_adjacent_equal_values():
    filled = [(0.0, 0), (1.0, 1), (2.0, 0)]
    runs = sp.raw_runs(filled, total_duration=3.0)
    # 0 at [0,1), 1 at [1,2), 0 at [2,3) -- NOT coalesced across the 1 in between
    assert runs == [(0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 3.0, 0)]


def test_raw_runs_empty_input():
    assert sp.raw_runs([], total_duration=10.0) == []


# ---------------------------------------------------------------------------
# split_at_forced_boundaries — the "half a person after a cut" fix
# ---------------------------------------------------------------------------

def test_split_at_forced_boundary_inside_a_run():
    runs = [(0.0, 10.0, 0)]
    result = sp.split_at_forced_boundaries(runs, [5.0])
    assert result == [(0.0, 5.0, 0), (5.0, 10.0, 0)]


def test_split_at_forced_boundary_ignores_boundary_outside_any_run():
    runs = [(0.0, 5.0, 0), (5.0, 10.0, 1)]
    # 5.0 sits exactly ON the existing boundary already -- no-op, not double-split.
    result = sp.split_at_forced_boundaries(runs, [5.0])
    assert result == runs


def test_split_at_multiple_forced_boundaries_in_one_run():
    runs = [(0.0, 10.0, 0)]
    result = sp.split_at_forced_boundaries(runs, [3.0, 7.0])
    assert result == [(0.0, 3.0, 0), (3.0, 7.0, 0), (7.0, 10.0, 0)]


def test_split_at_forced_boundaries_with_none_is_noop():
    runs = [(0.0, 10.0, 0)]
    assert sp.split_at_forced_boundaries(runs, None) == runs


# ---------------------------------------------------------------------------
# merge_short_runs — the min-shot-duration / no-jitter guarantee
# ---------------------------------------------------------------------------

def test_short_run_reabsorbed_into_preceding():
    # A 0.3s flicker to track 1 in the middle of a long track-0 hold.
    runs = [(0.0, 5.0, 0), (5.0, 5.3, 1), (5.3, 10.0, 0)]
    result = sp.merge_short_runs(runs, min_shot_seconds=1.2, forced_boundaries=None)
    assert result == [(0.0, 10.0, 0)]  # fully reabsorbed, one continuous shot


def test_run_meeting_minimum_survives():
    runs = [(0.0, 5.0, 0), (5.0, 7.0, 1), (7.0, 10.0, 0)]
    result = sp.merge_short_runs(runs, min_shot_seconds=1.2, forced_boundaries=None)
    assert result == [(0.0, 5.0, 0), (5.0, 7.0, 1), (7.0, 10.0, 0)]


def test_first_short_run_merges_forward():
    # A sub-minimum opening blip has nothing preceding it to reabsorb into —
    # it merges FORWARD into the next run instead of shipping a sub-minimum
    # shot (which used to fail validate_composition and kill the whole clip:
    # 'shot 0 [0.00-1.00s]: duration 1.00s < min 1.2s').
    runs = [(0.0, 0.5, 0), (0.5, 10.0, 1)]
    result = sp.merge_short_runs(runs, min_shot_seconds=1.2, forced_boundaries=None)
    assert result == [(0.0, 10.0, 1)]


def test_first_run_meeting_minimum_is_kept():
    runs = [(0.0, 2.0, 0), (2.0, 10.0, 1)]
    result = sp.merge_short_runs(runs, min_shot_seconds=1.2, forced_boundaries=None)
    assert result == [(0.0, 2.0, 0), (2.0, 10.0, 1)]


def test_a_forced_boundary_still_earns_a_cut_when_the_shot_is_watchable():
    # A splice point forces a real cut -- the video is physically
    # discontinuous there, it cannot be smoothed away -- provided the shot it
    # produces is long enough to register.
    runs = [(0.0, 5.0, 0), (5.0, 6.2, 0)]  # same target both sides, but forced
    result = sp.merge_short_runs(runs, min_shot_seconds=1.2, forced_boundaries=[5.0])
    assert result == [(0.0, 5.0, 0), (5.0, 6.2, 0)]


def test_a_forced_boundary_below_the_floor_is_absorbed():
    """The flash frame. `starts_at_forced` used to mean "never reabsorb", full
    stop, so a splice landing 0.3s before the next one emitted a 0.3s shot.
    The video really is discontinuous there, but a shot that brief is a glitch
    rather than a cut -- the surrounding shot carries the splice instead."""
    runs = [(0.0, 5.0, 0), (5.0, 5.3, 0)]
    result = sp.merge_short_runs(runs, min_shot_seconds=1.2, forced_boundaries=[5.0])
    assert result == [(0.0, 5.3, 0)]
    assert all(end - start >= sp.MIN_FORCED_SHOT_SECONDS
               for start, end, _ in result)


def test_adjacent_equal_targets_coalesce_after_reabsorption():
    # X, then a short Y blip, then X again with a normal duration -- after Y
    # reabsorbs into the first X, the trailing X run should merge in too.
    runs = [(0.0, 5.0, 0), (5.0, 5.3, 1), (5.3, 10.0, 0)]
    result = sp.merge_short_runs(runs, min_shot_seconds=1.2, forced_boundaries=None)
    assert result == [(0.0, 10.0, 0)]


def test_cascading_reabsorption_across_three_short_runs():
    runs = [(0.0, 5.0, 0), (5.0, 5.2, 1), (5.2, 5.4, 2), (5.4, 10.0, 0)]
    result = sp.merge_short_runs(runs, min_shot_seconds=1.2, forced_boundaries=None)
    assert result == [(0.0, 10.0, 0)]


def test_cap_run_durations_splits_long_runs():
    # A 29s static shot cannot follow a moving subject (the cutout + "not
    # framed" failure from the first rendered run) — cap it into chunks so
    # each chunk re-anchors on the subject's current position.
    runs = [(0.0, 29.0, 0)]
    out = sp._cap_run_durations(runs, max_shot_seconds=8.0,
                                forced_boundaries=None)
    assert len(out) == 4  # ceil(29/8) = 4 chunks
    assert abs(sum(e - s for s, e, _ in out) - 29.0) < 1e-6
    assert all(e - s <= 8.0 + 1e-6 for s, e, _ in out)


def test_cap_run_durations_keeps_short_runs_and_forced_boundaries():
    runs = [(0.0, 5.0, 0), (5.0, 20.0, 1)]
    out = sp._cap_run_durations(runs, max_shot_seconds=8.0,
                                forced_boundaries=[10.0])
    starts = [s for s, _, _ in out]
    assert 10.0 in starts  # a forced boundary is never crossed by a chunk
    assert (0.0, 5.0, 0) in out  # short run untouched


# ---------------------------------------------------------------------------
# crop_rect_for_track — the static-box computation
# ---------------------------------------------------------------------------

def test_crop_rect_is_median_not_mean_or_union():
    spine = _spine({0: [(0.0, (0, 0, 10, 10)), (1.0, (0, 0, 10, 10)),
                        (2.0, (1000, 1000, 10, 10))]})  # one wild outlier
    rect = sp.crop_rect_for_track(spine, 0, 0.0, 3.0)
    # median of [0,0,1000] is 0 -- the outlier does not drag the box
    assert rect == (0.0, 0.0, 10.0, 10.0)


def test_crop_rect_falls_back_to_nearest_when_span_has_no_detections():
    spine = _spine({0: [(100.0, (5, 5, 10, 10))]})
    rect = sp.crop_rect_for_track(spine, 0, 0.0, 1.0)  # no detections in [0,1)
    assert rect == (5, 5, 10, 10)


def test_crop_rect_none_for_unknown_track():
    spine = _spine({0: [(0.0, (0, 0, 10, 10))]})
    assert sp.crop_rect_for_track(spine, 99, 0.0, 1.0) is None


# ---------------------------------------------------------------------------
# plan_shots / plan_shots_from_samples — the full pipeline
# ---------------------------------------------------------------------------

def test_plan_shots_produces_stable_static_shots():
    spine = _spine({
        0: [(float(i), (0.0, 0.0, 10.0, 10.0)) for i in range(6)],
        1: [(float(i), (100.0, 0.0, 10.0, 10.0)) for i in range(6, 12)],
    })
    active = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    shots = sp.plan_shots(active, spine, min_shot_seconds=1.0)
    assert len(shots) == 2
    assert shots[0].shot_type == sp.SHOT_SINGLE
    assert shots[0].track_ids == [0]
    assert shots[0].start == 0.0 and shots[0].end == 6.0
    assert shots[1].track_ids == [1]
    assert shots[1].start == 6.0 and shots[1].end == 12.0


def test_plan_shots_wide_when_no_confident_speaker():
    spine = _spine({})
    active = [None, None, None]
    shots = sp.plan_shots(active, spine, default_wide_rect=(0, 0, 100, 100))
    assert len(shots) == 1
    assert shots[0].shot_type == sp.SHOT_WIDE
    assert shots[0].track_ids == []
    assert shots[0].crop_rect == (0, 0, 100, 100)


def test_plan_shots_respects_forced_boundary_even_across_a_splice():
    # Same speaker (track 0) the whole time, but a jump-cut splice sits at
    # t=5 -- must still produce two shots, not one, because the video itself
    # is discontinuous there (this is the actual "half a person" bug fix).
    spine = _spine({0: [(float(i), (0.0, 0.0, 10.0, 10.0)) for i in range(10)]})
    active = [0] * 10
    shots = sp.plan_shots(active, spine, forced_boundaries=[5.0], min_shot_seconds=1.0)
    assert len(shots) == 2
    assert shots[0].end == 5.0
    assert shots[1].start == 5.0


def test_plan_shots_end_to_end_with_realistic_flicker():
    # Speaker A (track 0) talks the whole clip; ASD briefly mis-points at
    # track 1 for one second mid-clip (the exact scenario Phase 3 tests
    # already proved binds correctly at the fusion level) -- the SHOT list
    # must also show zero visible effect from it.
    spine = _spine({
        0: [(float(i), (0.0, 0.0, 10.0, 10.0)) for i in range(10)],
        1: [(5.0, (200.0, 0.0, 10.0, 10.0))],
    })
    active = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    shots = sp.plan_shots(active, spine, min_shot_seconds=1.2)
    # The 10s single-speaker run is capped into <=8s chunks (each chunk
    # re-anchors on the current position) — but the ASD flicker still has
    # zero visible effect: every shot is track 0, covering 0-10.
    assert len(shots) == 2
    assert all(s.track_ids == [0] for s in shots)
    assert shots[0].start == 0.0 and shots[-1].end == 10.0
    assert all(s.duration <= 8.0 + 1e-6 for s in shots)


# ---------------------------------------------------------------------------
# _splice_window — the shared, conservative insertion primitive
# ---------------------------------------------------------------------------

def test_splice_window_inserts_cleanly_inside_a_shot():
    shots = [sp.Shot(0.0, 10.0, sp.SHOT_SINGLE, [0], (0, 0, 10, 10))]
    result = sp._splice_window(shots, 4.0, 6.0, sp.SHOT_REACTION, [1],
                               (1, 1, 1, 1), min_shot_seconds=1.0)
    assert len(result) == 3
    assert result[0] == sp.Shot(0.0, 4.0, sp.SHOT_SINGLE, [0], (0, 0, 10, 10))
    assert result[1] == sp.Shot(4.0, 6.0, sp.SHOT_REACTION, [1], (1, 1, 1, 1))
    assert result[2] == sp.Shot(6.0, 10.0, sp.SHOT_SINGLE, [0], (0, 0, 10, 10))


def test_splice_window_skips_when_it_would_leave_a_sub_minimum_sliver():
    shots = [sp.Shot(0.0, 10.0, sp.SHOT_SINGLE, [0], None)]
    # Left remainder would be 0.5s, under min_shot_seconds=1.0.
    result = sp._splice_window(shots, 0.5, 6.0, sp.SHOT_REACTION, [1], None,
                               min_shot_seconds=1.0)
    assert result == shots  # unchanged


def test_splice_window_skips_when_already_framing_that_track():
    shots = [sp.Shot(0.0, 10.0, sp.SHOT_SINGLE, [0], None)]
    result = sp._splice_window(shots, 4.0, 6.0, sp.SHOT_REACTION, [0], None,
                               min_shot_seconds=1.0)
    assert result == shots


def test_splice_window_skips_when_straddling_a_boundary():
    shots = [sp.Shot(0.0, 5.0, sp.SHOT_SINGLE, [0], None),
            sp.Shot(5.0, 10.0, sp.SHOT_SINGLE, [1], None)]
    result = sp._splice_window(shots, 4.0, 6.0, sp.SHOT_REACTION, [2], None,
                               min_shot_seconds=1.0)
    assert result == shots  # window crosses the 5.0 boundary -- skipped


def test_splice_window_no_left_remainder_when_window_starts_at_shot_start():
    shots = [sp.Shot(0.0, 10.0, sp.SHOT_SINGLE, [0], None)]
    result = sp._splice_window(shots, 0.0, 4.0, sp.SHOT_REACTION, [1], None,
                               min_shot_seconds=1.0)
    assert len(result) == 2
    assert result[0].shot_type == sp.SHOT_REACTION
    assert result[1] == sp.Shot(4.0, 10.0, sp.SHOT_SINGLE, [0], None)


# ---------------------------------------------------------------------------
# resolve_directive_track + insert_reaction_shots
# ---------------------------------------------------------------------------

def test_resolve_directive_track_picks_nearest_x():
    spine = _spine({
        0: [(1.0, (0.0, 0.0, 100.0, 100.0))],    # centre x = 50
        1: [(1.0, (900.0, 0.0, 100.0, 100.0))],  # centre x = 950
    })
    # directive x_position=0.9 of a 1000px frame -> target_x=900, track 1 wins
    track = sp.resolve_directive_track(0.9, spine, frame_width=1000.0, at_time=1.0)
    assert track == 1


def test_resolve_directive_track_none_when_nothing_close_enough():
    spine = _spine({0: [(1.0, (0.0, 0.0, 10.0, 10.0))]})
    track = sp.resolve_directive_track(0.9, spine, frame_width=1000.0, at_time=1.0,
                                       x_tolerance_frac=0.05)
    assert track is None


def test_insert_reaction_shot_for_causing_reaction_directive():
    # require_corroboration=False here: these fixtures test the SPLICE
    # mechanics (resolve -> bound -> insert), not the corroboration gate --
    # that gate has its own tests below, including on this exact scenario.
    spine = _spine({
        0: [(float(i), (0.0, 0.0, 100.0, 100.0)) for i in range(10)],
        1: [(float(i), (900.0, 0.0, 100.0, 100.0)) for i in range(10)],
    })
    shots = [sp.Shot(0.0, 10.0, sp.SHOT_SINGLE, [0], (0, 0, 100, 100))]
    directives = [{"start": 4.0, "end": 6.0, "x_position": 0.95, "reason": "causing_reaction"}]
    result = sp.insert_reaction_shots(shots, directives, spine, frame_width=1000.0,
                                      min_shot_seconds=1.0, require_corroboration=False)
    assert len(result) == 3
    assert result[1].shot_type == sp.SHOT_REACTION
    assert result[1].track_ids == [1]


def test_insert_reaction_ignores_non_payoff_reasons():
    spine = _spine({0: [(float(i), (0.0, 0.0, 10.0, 10.0)) for i in range(10)]})
    shots = [sp.Shot(0.0, 10.0, sp.SHOT_SINGLE, [0], None)]
    directives = [{"start": 4.0, "end": 6.0, "x_position": 0.5, "reason": "speaking"}]
    result = sp.insert_reaction_shots(shots, directives, spine, frame_width=100.0)
    assert result == shots


def test_insert_reaction_window_bounded_by_max_reaction_seconds():
    spine = _spine({
        0: [(float(i), (0.0, 0.0, 10.0, 10.0)) for i in range(20)],
        1: [(float(i), (900.0 + i, 0.0, 10.0, 10.0)) for i in range(20)],  # slight drift = real motion
    })
    shots = [sp.Shot(0.0, 20.0, sp.SHOT_SINGLE, [0], None)]
    # A long 10s directive window -- the inserted reaction shot must still
    # be capped at max_reaction_seconds, not span the whole directive.
    directives = [{"start": 5.0, "end": 15.0, "x_position": 0.9, "reason": "referenced"}]
    result = sp.insert_reaction_shots(shots, directives, spine, frame_width=1000.0,
                                      max_reaction_seconds=2.0, min_shot_seconds=1.0)
    reaction = [s for s in result if s.shot_type == sp.SHOT_REACTION][0]
    assert reaction.duration <= 2.0 + 1e-9


# ---------------------------------------------------------------------------
# is_directive_corroborated / the corroboration gate — the actual fix for
# a hallucinated directive reaching the render (HANDOFF_FRAMING.md's
# documented causing_reaction beat with nobody actually reacting)
# ---------------------------------------------------------------------------

def test_corroborated_when_track_absent_entirely():
    spine = _spine({0: [(0.0, (0, 0, 10, 10))]})  # nothing near t=5
    assert sp.is_directive_corroborated(spine, 0, 5.0, 6.0) is False


def test_corroborated_when_track_id_unknown():
    spine = _spine({0: [(5.0, (0, 0, 10, 10))]})
    assert sp.is_directive_corroborated(spine, 99, 5.0, 6.0) is False


def test_not_corroborated_when_perfectly_static():
    # Two detections in the window, box never moves at all -- weak evidence
    # anything reaction-worthy is happening (this is exactly the scenario
    # the earlier splice-mechanics tests above bypass with require_corroboration=False).
    spine = _spine({0: [(5.0, (0.0, 0.0, 100.0, 100.0)), (5.5, (0.0, 0.0, 100.0, 100.0))]})
    assert sp.is_directive_corroborated(spine, 0, 5.0, 6.0) is False


def test_corroborated_when_visible_motion():
    spine = _spine({0: [(5.0, (0.0, 0.0, 100.0, 100.0)), (5.5, (20.0, 0.0, 100.0, 100.0))]})
    assert sp.is_directive_corroborated(spine, 0, 5.0, 6.0) is True


def test_corroboration_falls_back_to_presence_only_for_single_detection():
    # Only one sample in the window -- cannot prove or disprove movement,
    # so a single real detection should NOT be penalized for it.
    spine = _spine({0: [(5.2, (0.0, 0.0, 100.0, 100.0))]})
    assert sp.is_directive_corroborated(spine, 0, 5.0, 6.0) is True


def test_insert_reaction_shots_rejects_uncorroborated_directive_by_default():
    # Same scenario as the splice-mechanics test above, but WITHOUT
    # disabling corroboration -- the perfectly static track must now be
    # rejected, and the base shot list must come back unchanged.
    spine = _spine({
        0: [(float(i), (0.0, 0.0, 100.0, 100.0)) for i in range(10)],
        1: [(float(i), (900.0, 0.0, 100.0, 100.0)) for i in range(10)],  # zero motion
    })
    shots = [sp.Shot(0.0, 10.0, sp.SHOT_SINGLE, [0], (0, 0, 100, 100))]
    directives = [{"start": 4.0, "end": 6.0, "x_position": 0.95, "reason": "causing_reaction"}]
    result = sp.insert_reaction_shots(shots, directives, spine, frame_width=1000.0,
                                      min_shot_seconds=1.0)
    assert result == shots


def test_insert_reaction_shots_accepts_corroborated_directive_by_default():
    spine = _spine({
        0: [(float(i), (0.0, 0.0, 100.0, 100.0)) for i in range(10)],
        1: [(float(i), (900.0 + i * 5, 0.0, 100.0, 100.0)) for i in range(10)],  # visible motion
    })
    shots = [sp.Shot(0.0, 10.0, sp.SHOT_SINGLE, [0], (0, 0, 100, 100))]
    directives = [{"start": 4.0, "end": 6.0, "x_position": 0.95, "reason": "causing_reaction"}]
    result = sp.insert_reaction_shots(shots, directives, spine, frame_width=1000.0,
                                      min_shot_seconds=1.0)
    assert len(result) == 3
    assert result[1].shot_type == sp.SHOT_REACTION


# ---------------------------------------------------------------------------
# two_shot_crop_rect + apply_two_shot
# ---------------------------------------------------------------------------

def test_two_shot_crop_rect_covers_both():
    spine = _spine({
        0: [(0.0, (0.0, 0.0, 10.0, 10.0))],
        1: [(0.0, (90.0, 0.0, 10.0, 10.0))],
    })
    rect = sp.two_shot_crop_rect(spine, 0, 1, 0.0, 1.0)
    assert rect == (0.0, 0.0, 100.0, 10.0)  # spans from track0's left edge to track1's right edge


def test_apply_two_shot_widens_when_addressee_known():
    spine = _spine({
        0: [(float(i), (0.0, 0.0, 10.0, 10.0)) for i in range(10)],
        1: [(float(i), (500.0, 0.0, 10.0, 10.0)) for i in range(10)],
    })
    shots = [sp.Shot(0.0, 10.0, sp.SHOT_SINGLE, [0], (0, 0, 10, 10))]
    addressee_per_second = [1] * 10  # track 0 addresses track 1 the whole shot
    result = sp.apply_two_shot(shots, addressee_per_second, spine, min_shot_seconds=1.0)
    assert len(result) == 1
    assert result[0].shot_type == sp.SHOT_TWO_SHOT
    assert set(result[0].track_ids) == {0, 1}


def test_apply_two_shot_skips_when_addressee_changes_mid_shot():
    spine = _spine({
        0: [(float(i), (0.0, 0.0, 10.0, 10.0)) for i in range(10)],
        1: [(float(i), (500.0, 0.0, 10.0, 10.0)) for i in range(10)],
        2: [(float(i), (900.0, 0.0, 10.0, 10.0)) for i in range(10)],
    })
    shots = [sp.Shot(0.0, 10.0, sp.SHOT_SINGLE, [0], None)]
    addressee_per_second = [1] * 5 + [2] * 5  # not a single consistent addressee
    result = sp.apply_two_shot(shots, addressee_per_second, spine, min_shot_seconds=1.0)
    assert result == shots


def test_apply_two_shot_skips_when_no_addressee():
    shots = [sp.Shot(0.0, 10.0, sp.SHOT_SINGLE, [0], None)]
    result = sp.apply_two_shot(shots, [None] * 10, {}, min_shot_seconds=1.0)
    assert result == shots


# ---------------------------------------------------------------------------
# snap_shots_to_speech — the cut follows the sentence (I9)
# ---------------------------------------------------------------------------

def _words(*spans):
    return [{"word": "w", "start": s, "end": e} for s, e in spans]


def test_word_gaps_ignores_coarticulation():
    # 0.05s between words is run-on speech, not a place you can cut.
    gaps = sp.word_gaps(_words((0.0, 1.0), (1.05, 2.0), (2.5, 3.0)))
    assert len(gaps) == 1
    assert gaps[0] == pytest.approx(2.25)   # midpoint of the 2.0-2.5 silence


def test_word_gaps_tolerates_no_words():
    assert sp.word_gaps(None) == []
    assert sp.word_gaps([]) == []


def test_boundary_snaps_to_the_nearest_silence():
    shots = [sp.Shot(0.0, 5.15, sp.SHOT_SINGLE, [1]),
             sp.Shot(5.15, 11.0, sp.SHOT_SINGLE, [2])]
    out = sp.snap_shots_to_speech(shots, [5.0])
    assert out[0].end == pytest.approx(5.0)
    assert out[1].start == pytest.approx(5.0)


def test_a_boundary_too_far_from_any_gap_stays_put():
    shots = [sp.Shot(0.0, 5.0, sp.SHOT_SINGLE, [1]),
             sp.Shot(5.0, 11.0, sp.SHOT_SINGLE, [2])]
    out = sp.snap_shots_to_speech(shots, [9.0])       # 4s away
    assert out[0].end == pytest.approx(5.0)


def test_forced_boundaries_never_move():
    """The video is physically spliced there — the cut has to be at the
    splice, not near it."""
    shots = [sp.Shot(0.0, 5.0, sp.SHOT_SINGLE, [1]),
             sp.Shot(5.0, 11.0, sp.SHOT_SINGLE, [2])]
    out = sp.snap_shots_to_speech(shots, [4.9], forced_boundaries=[5.0])
    assert out[0].end == pytest.approx(5.0)


def test_the_clip_in_and_out_are_never_moved():
    """They are already sentence-anchored upstream."""
    shots = [sp.Shot(0.0, 5.0, sp.SHOT_SINGLE, [1]),
             sp.Shot(5.0, 11.0, sp.SHOT_SINGLE, [2])]
    out = sp.snap_shots_to_speech(shots, [0.2, 5.0, 10.8])
    assert out[0].start == pytest.approx(0.0)
    assert out[-1].end == pytest.approx(11.0)


def test_a_snap_that_would_starve_a_shot_is_skipped():
    shots = [sp.Shot(0.0, 2.0, sp.SHOT_SINGLE, [1]),
             sp.Shot(2.0, 11.0, sp.SHOT_SINGLE, [2])]
    out = sp.snap_shots_to_speech(shots, [0.5], min_shot_seconds=1.8)
    assert out[0].end == pytest.approx(2.0)


def test_snapping_preserves_total_coverage():
    shots = [sp.Shot(0.0, 5.15, sp.SHOT_SINGLE, [1]),
             sp.Shot(5.15, 11.0, sp.SHOT_SINGLE, [2])]
    out = sp.snap_shots_to_speech(shots, [5.0])
    assert sum(s.duration for s in out) == pytest.approx(11.0)
    assert all(a.end == b.start for a, b in zip(out, out[1:]))
