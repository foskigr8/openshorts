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
