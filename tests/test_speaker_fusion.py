"""Phase 3's actual deliverable: one speaker->track binding per clip,
decided from all the evidence at once, that cannot flip mid-turn. Pure
Python — no video, no GPU, no ASD model — so this is fully testable here,
and it's the piece that replaced the v2 engine's per-frame speaker-match
(reframe_v2._apply_asd_speaker_boost, now removed).
"""
import speaker_fusion as sf


def _spine(tracks):
    """tracks: {track_id: [(t, box), ...]} -> face_spine-shaped dict."""
    return {
        tid: {"frames": [t for t, _ in samples],
              "boxes": [b for _, b in samples]}
        for tid, samples in tracks.items()
    }


# ---------------------------------------------------------------------------
# resolve_speaker_bindings
# ---------------------------------------------------------------------------

def test_binds_speaker_to_consistently_evidenced_track():
    # Speaker A talks seconds 0-4, ASD points at track 0 every second.
    per_second_speaker = ["A"] * 5
    predicted_track = [0, 0, 0, 0, 0]
    bindings = sf.resolve_speaker_bindings(per_second_speaker, predicted_track)
    assert bindings == {"A": 0}


def test_does_not_bind_below_min_agreement():
    # ASD flickers evenly between two tracks for the same speaker — exactly
    # the "reacting listener" ambiguity that caused the original bug. A dead
    # 50/50 tie is the clearest case of "no confident majority," and the
    # default min_agreement (0.6) is deliberately set above 0.5 so a tie
    # cannot bind.
    per_second_speaker = ["A"] * 4
    predicted_track = [0, 1, 0, 1]
    bindings = sf.resolve_speaker_bindings(per_second_speaker, predicted_track)
    assert "A" not in bindings


def test_binds_on_majority_even_with_some_noise():
    # 4 out of 5 seconds point at track 0 -> 0.8 agreement, above the
    # default 0.5 bar, so the momentary noise does not block a real binding.
    per_second_speaker = ["A"] * 5
    predicted_track = [0, 0, 0, 1, 0]
    bindings = sf.resolve_speaker_bindings(per_second_speaker, predicted_track)
    assert bindings == {"A": 0}


def test_does_not_bind_below_min_seconds():
    # Only half a second of evidence (rounds to under min_seconds=1.0 by
    # having zero qualifying seconds isn't quite it -- use a single second
    # against a stricter min_seconds to prove the floor is enforced).
    per_second_speaker = ["A"]
    predicted_track = [0]
    bindings = sf.resolve_speaker_bindings(per_second_speaker, predicted_track,
                                           min_seconds=2.0)
    assert "A" not in bindings


class TestActiveFromIdentityMap:
    def test_confirmed_labels_map_to_their_tracks(self):
        speaker = ["A", "A", "B", None, "C"]
        mapping = {"A": 3, "B": 7}
        assert sf.active_from_identity_map(speaker, mapping) == [3, 3, 7, None, None]

    def test_unconfirmed_labels_are_none_not_guessed(self):
        # Director v2 rule: an unmapped speaker must never default to a
        # track (no host-default) — None sends the caller to wide/hold.
        speaker = ["A", "C", "C"]
        mapping = {"A": 1}
        assert sf.active_from_identity_map(speaker, mapping) == [1, None, None]

    def test_empty_map_yields_all_none(self):
        assert sf.active_from_identity_map(["A", "B"], {}) == [None, None]


# ---------------------------------------------------------------------------
# decisive_seconds — throwing out the coin-flip seconds before they vote
# ---------------------------------------------------------------------------

def test_decisive_seconds_from_margins():
    # LR-ASD's per-second lead of the winner over the next best face.
    margins = [0.4, 0.02, None, 0.10]
    assert sf.decisive_seconds(margins, 0.10) == [True, False, False, True]


def test_no_margin_data_means_no_gate():
    # LR-ASD unavailable (or an older caller): every second keeps counting,
    # exactly as before.
    assert sf.decisive_seconds(None) is None


def test_only_decisive_seconds_vote_for_a_binding():
    # Three coin-flip seconds point at track 1 (a reacting listener scoring
    # almost as high as the talker); two confident ones point at track 0.
    # Counting every second binds A to the WRONG face on a bare majority.
    per_second_speaker = ["A"] * 5
    predicted_track = [1, 1, 1, 0, 0]
    assert sf.resolve_speaker_bindings(
        per_second_speaker, predicted_track) == {"A": 1}
    decisive = [False, False, False, True, True]
    assert sf.resolve_speaker_bindings(
        per_second_speaker, predicted_track, decisive_ps=decisive) == {"A": 0}


# ---------------------------------------------------------------------------
# match_box_to_track — the head sits at the TOP of a speaker box
# ---------------------------------------------------------------------------

def test_asd_box_covering_two_faces_picks_the_head_at_its_top():
    # One ASD speaker box swallows the talker (small face, near the box top)
    # and a listener leaning in below with a much larger face box. Best-IoU
    # rewards the larger face; the head region picks the person the box was
    # actually drawn around.
    spine = _spine({
        0: [(0.0, (110.0, 95.0, 55.0, 55.0))],
        1: [(0.0, (85.0, 130.0, 110.0, 155.0))],
    })
    box = (80.0, 90.0, 120.0, 200.0)
    assert sf._box_iou(box, (85.0, 130.0, 110.0, 155.0)) > sf._box_iou(
        box, (110.0, 95.0, 55.0, 55.0))
    assert sf.match_box_to_track(spine, 0.0, box) == 0


def test_falls_back_to_iou_when_no_face_centre_is_inside_the_box():
    # A speaker box offset off the face: no face centre lands inside it, so
    # the original IoU match still resolves the track.
    spine = _spine({0: [(0.0, (0.0, 0.0, 100.0, 100.0))]})
    box = (51.0, 0.0, 90.0, 100.0)
    assert sf.match_box_to_track(spine, 0.0, box) == 0


# ---------------------------------------------------------------------------
# smooth_track_sequence — crowd-scene stabilization
# ---------------------------------------------------------------------------

def test_stable_sequence_is_unchanged():
    assert sf.smooth_track_sequence([1, 1, 1, 1]) == [1, 1, 1, 1]


def test_single_second_flicker_is_removed():
    # One bad ASD second must not create a fake speaker change.
    assert sf.smooth_track_sequence([1, 1, 2, 1, 1]) == [1, 1, 1, 1, 1]


def test_real_change_survives():
    # A sustained run of the new track outvotes the old one.
    assert sf.smooth_track_sequence([1, 1, 1, 2, 2, 2]) == [1, 1, 1, 2, 2, 2]


def test_none_votes_nothing_but_gap_fills_from_neighbors():
    assert sf.smooth_track_sequence([1, None, 1]) == [1, 1, 1]
    assert sf.smooth_track_sequence([None, None]) == [None, None]


def test_rapid_alternation_collapses():
    # Sub-window back-and-forth is flicker, not an exchange worth splitting.
    assert sf.smooth_track_sequence([1, 2, 1, 2]) == [1, 1, 2, 2]


def test_fuse_speaker_tracks_smooths_before_votes_and_fallback():
    # A flickering ASD signal must neither bind speaker A to track 1 nor
    # hop the fallback; smoothing concentrates it on the real track 0.
    per_second_speaker = ["A"] * 6
    spine = _spine({0: [(t, (0.0, 0.0, 10.0, 10.0)) for t in range(6)],
                    1: [(t, (100.0, 100.0, 10.0, 10.0)) for t in range(6)]})
    # ASD points at track 1 (the reacting listener) for one bad second.
    boxes = [(0.0, 0.0, 10.0, 10.0)] * 6
    boxes[2] = (100.0, 100.0, 10.0, 10.0)
    bindings, active = sf.fuse_speaker_tracks(
        boxes, spine, [{"start": 0, "end": 6, "text": "", "speaker": "A"}],
        0.0, 6.0)
    assert bindings == {"A": 0}
    assert active == [0, 0, 0, 0, 0, 0]


def test_none_speaker_or_none_track_contributes_no_vote():
    per_second_speaker = ["A", None, "A", "A"]
    predicted_track = [0, 0, None, 0]
    # Only seconds 0 and 3 are valid evidence (index 1 has no speaker, index
    # 2 has no ASD call) -- both point at track 0, so it should still bind.
    bindings = sf.resolve_speaker_bindings(per_second_speaker, predicted_track,
                                           min_seconds=1.0)
    assert bindings == {"A": 0}


def test_multiple_speakers_bind_independently():
    per_second_speaker = ["A", "A", "B", "B", "B"]
    predicted_track = [0, 0, 1, 1, 1]
    bindings = sf.resolve_speaker_bindings(per_second_speaker, predicted_track)
    assert bindings == {"A": 0, "B": 1}


def test_empty_input_binds_nothing():
    assert sf.resolve_speaker_bindings([], []) == {}


# ---------------------------------------------------------------------------
# per_second_active_track
# ---------------------------------------------------------------------------

def test_per_second_active_track_expands_binding():
    per_second_speaker = ["A", "A", "B", None]
    bindings = {"A": 0, "B": 1}
    result = sf.per_second_active_track(per_second_speaker, bindings)
    assert result == [0, 0, 1, None]


def test_per_second_active_track_unbound_speaker_is_none():
    # "C" appears in the transcript but never got a confident binding --
    # must surface as None, never a guessed track.
    per_second_speaker = ["A", "C"]
    bindings = {"A": 0}
    result = sf.per_second_active_track(per_second_speaker, bindings)
    assert result == [0, None]


# ---------------------------------------------------------------------------
# The core claim: a binding, once decided, cannot flip mid-turn
# ---------------------------------------------------------------------------

def test_binding_does_not_flip_on_a_single_bad_frame_mid_turn():
    # Speaker A's whole turn is 10 seconds. ASD is correct on every second
    # except one ambiguous blip at second 5 (points at the wrong track for
    # just that instant) -- today's per-frame matching could flip the camera
    # right there. The fused binding must not: it's decided once, from all
    # 10 seconds of evidence, so a single-second wobble cannot move it.
    per_second_speaker = ["A"] * 10
    predicted_track = [0, 0, 0, 0, 0, 7, 0, 0, 0, 0]  # one bad second
    bindings = sf.resolve_speaker_bindings(per_second_speaker, predicted_track)
    active = sf.per_second_active_track(per_second_speaker, bindings)
    assert bindings == {"A": 0}
    assert active == [0] * 10  # every second, including the bad one, resolves to 0


# ---------------------------------------------------------------------------
# apply_gated_rebinding — the escape hatch from a binding that came out wrong
# ---------------------------------------------------------------------------

def test_sustained_contradiction_rebinds_from_that_second_on():
    # The clip bound speaker A to track 0 (the host). From second 4 on,
    # decisive ASD says track 1 every single second — "held on the host while
    # she wasn't talking". The 4th contradicting second flips the binding,
    # and it stays flipped through the end of the clip.
    speaker = ["A"] * 10
    predicted = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    bindings, active = sf.apply_gated_rebinding(
        speaker, predicted, {"A": 0}, [0] * 10)
    assert bindings == {"A": 1}
    assert active == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]


def test_a_single_blip_never_rebinds():
    # The one property the one-decision-per-clip design bought us, kept.
    speaker = ["A"] * 10
    predicted = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    bindings, active = sf.apply_gated_rebinding(
        speaker, predicted, {"A": 0}, [0] * 10)
    assert bindings == {"A": 0}
    assert active == [0] * 10


def test_agreement_decays_the_case_against_the_binding():
    # Alternating disagreement is ambiguity, not a wrong binding: each
    # agreeing second knocks one contradiction off, so it never reaches four.
    speaker = ["A"] * 16
    predicted = [1, 0] * 8
    bindings, _ = sf.apply_gated_rebinding(
        speaker, predicted, {"A": 0}, [0] * 16)
    assert bindings == {"A": 0}


def test_contradictions_spread_past_the_window_never_accumulate():
    # Five disagreeing seconds, five seconds apart — never four inside one
    # 8-second window, so a slow drip of noise cannot re-bind anything.
    speaker = ["A"] * 21
    predicted = [0] * 21
    for i in (0, 5, 10, 15, 20):
        predicted[i] = 1
    bindings, _ = sf.apply_gated_rebinding(
        speaker, predicted, {"A": 0}, [0] * 21, rebind_window=8)
    assert bindings == {"A": 0}


def test_no_flip_flop_after_a_rebind():
    # Once A re-binds to track 1, a couple of seconds pointing back at track
    # 0 must not undo it — reverting needs its own sustained case.
    speaker = ["A"] * 12
    predicted = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
    bindings, active = sf.apply_gated_rebinding(
        speaker, predicted, {"A": 0}, [0] * 12)
    assert bindings == {"A": 1}
    assert active[7:] == [1] * 5


def test_only_decisive_contradictions_count_toward_a_rebind():
    speaker = ["A"] * 10
    predicted = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    decisive = [True] * 4 + [False] * 6   # the disagreement is a coin flip
    bindings, active = sf.apply_gated_rebinding(
        speaker, predicted, {"A": 0}, [0] * 10, decisive_ps=decisive)
    assert bindings == {"A": 0}
    assert active == [0] * 10


def test_unbound_labels_keep_their_per_second_fallback():
    # "C" never bound; per_second_active_track already fell back to ASD for
    # those seconds and re-binding must not touch them.
    speaker = ["A", "C", "C", "C"]
    predicted = [0, 5, 5, 5]
    bindings, active = sf.apply_gated_rebinding(
        speaker, predicted, {"A": 0}, [0, 5, 5, 5])
    assert bindings == {"A": 0}
    assert active == [0, 5, 5, 5]


def test_rebinding_can_be_switched_off():
    speaker = ["A"] * 10
    predicted = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    bindings, active = sf.apply_gated_rebinding(
        speaker, predicted, {"A": 0}, [0] * 10, rebind_seconds=0)
    assert bindings == {"A": 0}
    assert active == [0] * 10


# ---------------------------------------------------------------------------
# per_second_speaker_label (moved from eval/ground_truth, retested here as
# the production entry point)
# ---------------------------------------------------------------------------

def test_per_second_speaker_label_basic():
    segments = [
        {"start": 0.0, "end": 2.5, "speaker": "A"},
        {"start": 2.5, "end": 5.0, "speaker": "B"},
    ]
    result = sf.per_second_speaker_label(segments, 0.0, 5.0)
    assert result[0] == "A"
    assert result[4] == "B"


def test_per_second_speaker_label_resolves_names():
    segments = [{"start": 0.0, "end": 2.0, "speaker": "A"}]
    result = sf.per_second_speaker_label(segments, 0.0, 2.0, speaker_names={"A": "host"})
    assert result[0] == "host"


# ---------------------------------------------------------------------------
# fuse_speaker_tracks — full pipeline, end to end
# ---------------------------------------------------------------------------

def test_fuse_speaker_tracks_end_to_end():
    spine = _spine({
        0: [(t, (0.0, 0.0, 10.0, 10.0)) for t in range(5)],
        1: [(t, (100.0, 100.0, 10.0, 10.0)) for t in range(5)],
    })
    # ASD points at track 0's box every second.
    asd_boxes = [(0.0, 0.0, 10.0, 10.0)] * 5
    segments = [{"start": 0.0, "end": 5.0, "speaker": "A"}]

    bindings, active = sf.fuse_speaker_tracks(
        asd_boxes, spine, segments, clip_start=0.0, clip_end=5.0)

    assert bindings == {"A": 0}
    assert active == [0, 0, 0, 0, 0]


def test_fuse_speaker_tracks_resolves_speaker_names():
    spine = _spine({0: [(t, (0.0, 0.0, 10.0, 10.0)) for t in range(3)]})
    asd_boxes = [(0.0, 0.0, 10.0, 10.0)] * 3
    segments = [{"start": 0.0, "end": 3.0, "speaker": "A"}]

    bindings, active = sf.fuse_speaker_tracks(
        asd_boxes, spine, segments, clip_start=0.0, clip_end=3.0,
        speaker_names={"A": "host"})

    assert bindings == {"host": 0}
    assert active == [0, 0, 0]


def test_fuse_speaker_tracks_rebinds_on_sustained_decisive_contradiction():
    # The whole-clip vote binds A to track 0 (12 of 18 seconds), which is
    # right for the first two thirds and wrong for the last third. Without
    # re-binding the camera holds track 0 to the end; with it, the tail
    # corrects itself once the contradiction is sustained and decisive.
    spine = _spine({
        0: [(t, (0.0, 0.0, 10.0, 10.0)) for t in range(18)],
        1: [(t, (100.0, 100.0, 10.0, 10.0)) for t in range(18)],
    })
    boxes = [(0.0, 0.0, 10.0, 10.0)] * 12 + [(100.0, 100.0, 10.0, 10.0)] * 6
    segments = [{"start": 0.0, "end": 18.0, "speaker": "A"}]

    bindings, active = sf.fuse_speaker_tracks(
        boxes, spine, segments, clip_start=0.0, clip_end=18.0,
        asd_per_second_margin=[0.5] * 18)

    assert bindings == {"A": 1}
    assert active[:15] == [0] * 15
    assert active[15:] == [1] * 3


def test_fuse_speaker_tracks_holds_when_the_contradiction_is_a_coin_flip():
    # Same footage, but LR-ASD was never sure about the second half (the two
    # faces scored within the decisive margin). Nothing re-binds.
    spine = _spine({
        0: [(t, (0.0, 0.0, 10.0, 10.0)) for t in range(18)],
        1: [(t, (100.0, 100.0, 10.0, 10.0)) for t in range(18)],
    })
    boxes = [(0.0, 0.0, 10.0, 10.0)] * 12 + [(100.0, 100.0, 10.0, 10.0)] * 6
    segments = [{"start": 0.0, "end": 18.0, "speaker": "A"}]

    bindings, active = sf.fuse_speaker_tracks(
        boxes, spine, segments, clip_start=0.0, clip_end=18.0,
        asd_per_second_margin=[0.5] * 12 + [0.01] * 6)

    assert bindings == {"A": 0}
    assert active == [0] * 18


# ---------------------------------------------------------------------------
# The no-guess rule and the speaker-lock score (plan §5.6)
# ---------------------------------------------------------------------------

class TestNoGuessRule:
    def test_a_mapped_speaker_is_always_framed(self):
        assert sf.per_second_active_track(["A", "A"], {"A": 7}, [3, 3]) == [7, 7]

    def test_an_unmapped_speaker_goes_wide_not_to_the_nearest_face(self):
        """Clip 2's defect: the host is talking, the picture is the group."""
        assert sf.per_second_active_track(["HOST"], {}, [9]) == [None]

    def test_an_unlabelled_second_still_uses_asd(self):
        """Nobody said who is talking, so the model that watches faces wins."""
        assert sf.per_second_active_track([None], {}, [9]) == [9]

    def test_policy_asd_restores_the_old_behaviour(self):
        assert sf.per_second_active_track(
            ["HOST"], {}, [9], unmapped_policy="asd") == [9]

    def test_no_signals_at_all_is_still_none(self):
        assert sf.per_second_active_track([None, None], {}, None) == [None, None]


class TestSpeakerLockScore:
    def test_perfect_lock(self):
        assert sf.speaker_lock_score(["A", "A"], {"A": 1}, [1, 1]) == 1.0

    def test_a_wrong_frame_lowers_the_score(self):
        assert sf.speaker_lock_score(["A", "A"], {"A": 1}, [1, 2]) == 0.5

    def test_wide_seconds_are_excluded_not_counted_as_misses(self):
        """The score measures AIM, not coverage. A second that honestly went
        wide because the speaker was unmappable is not a framing error."""
        assert sf.speaker_lock_score(["A", "B"], {"A": 1}, [1, None]) == 1.0

    def test_nothing_to_score_returns_none(self):
        assert sf.speaker_lock_score([None, None], {}, [None, None]) is None
        assert sf.speaker_lock_score(["A"], {}, [None]) is None


class TestCrowdPluralityBinding:
    """A 60% majority is a high bar with 6-11 faces on screen: the vote for
    one speaker scatters across neighbours, nothing binds, and the whole clip
    goes wide. A clear plurality is not the same ambiguity as a coin flip."""

    def test_a_clear_plurality_binds_in_a_crowd(self):
        # 5 votes for track 0, spread of 1 each across four neighbours.
        speaker = ["A"] * 9
        predicted = [0, 0, 0, 0, 0, 1, 2, 3, 4]
        assert sf.resolve_speaker_bindings(speaker, predicted) == {"A": 0}

    def test_a_dead_tie_still_binds_nothing(self):
        """The case the majority bar was actually written for."""
        assert "A" not in sf.resolve_speaker_bindings(["A"] * 4, [0, 1, 0, 1])

    def test_a_narrow_lead_inside_the_plurality_band_still_binds_nothing(self):
        # 5 vs 4 vs 1: agreement 0.5 sits in the plurality band, but the lead
        # over the runner-up is only 1.25x — that is a contest, not a winner.
        speaker = ["A"] * 10
        predicted = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2]
        assert "A" not in sf.resolve_speaker_bindings(speaker, predicted)


class TestBindingDiagnostics:
    def test_it_separates_no_diarization_from_no_match(self):
        no_asd = sf.binding_diagnostics(["A", "A"], [None, None], {})
        assert no_asd["labelled"] == 2 and no_asd["matched"] == 0
        assert no_asd["voting"] == 0        # nothing could vote

        no_labels = sf.binding_diagnostics([None, None], [1, 1], {})
        assert no_labels["labelled"] == 0 and no_labels["matched"] == 2

    def test_it_reports_the_top_share_for_an_unbound_label(self):
        d = sf.binding_diagnostics(["A"] * 4, [0, 1, 0, 1], {})
        track, share, total, spread = d["labels"]["A"]
        assert (track, share, total, spread) == (0, 0.5, 4, 2)
        assert d["bound"] == 0
