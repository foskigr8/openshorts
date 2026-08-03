"""The evidence-tiered subject policy.

These encode the measured findings that motivated replacing accumulating
size-score selection (see subject_policy's module docstring): strong
speaker evidence must reach the screen immediately, weak evidence must be
damped, and the camera must hold rather than drift onto whoever is biggest.
"""
import pytest

from subject_policy import (
    Evidence,
    SubjectPolicy,
    TIER_ASD,
    TIER_DIARIZED,
    TIER_DIRECTIVE,
    TIER_HOLD,
    TIER_MOUTH,
    TIER_SIZE,
)

FPS = 25.0


def cands(*specs):
    """specs: (id, x, w) -> candidate dicts with plausible boxes."""
    return [{"id": i, "box": [x, 100, w, w]} for i, x, w in specs]


# A big incumbent and a small challenger: the exact shape that defeated the
# old accumulator (a 2x-larger host outweighed any boost on the speaker).
HOST = (1, 100, 400)
GUEST = (2, 900, 200)


def test_lip_sync_beats_a_much_bigger_face_immediately():
    p = SubjectPolicy(FPS)
    c = cands(HOST, GUEST)
    box, cid, tier = p.decide(c, Evidence(asd_id=2), 0)
    assert cid == 2
    assert tier == TIER_ASD


def test_lip_sync_switch_is_not_gated_by_the_weak_evidence_floor():
    p = SubjectPolicy(FPS, min_shot_hold_seconds=5.0,
                      absolute_min_shot_seconds=0.0)
    c = cands(HOST, GUEST)
    p.decide(c, Evidence(asd_id=1), 0)
    # One frame later, well inside the 5s floor, the other person speaks.
    _, cid, tier = p.decide(c, Evidence(asd_id=2), 1)
    assert (cid, tier) == (2, TIER_ASD)


def test_diarization_is_used_when_lip_sync_is_absent():
    p = SubjectPolicy(FPS)
    _, cid, tier = p.decide(cands(HOST, GUEST), Evidence(diarized_id=2), 0)
    assert (cid, tier) == (2, TIER_DIARIZED)


def test_lip_sync_outranks_diarization_when_they_disagree():
    p = SubjectPolicy(FPS)
    _, cid, tier = p.decide(cands(HOST, GUEST),
                            Evidence(asd_id=1, diarized_id=2), 0)
    assert (cid, tier) == (1, TIER_ASD)


def test_transcript_evidence_outranks_a_directive():
    """Gemini's shot direction was measured pointing at the wrong person; the
    diarized speaker wins when the two disagree."""
    p = SubjectPolicy(FPS)
    _, cid, tier = p.decide(cands(HOST, GUEST),
                            Evidence(diarized_id=2, directive_id=1), 0)
    assert (cid, tier) == (2, TIER_DIARIZED)


def test_directive_is_used_when_no_speaker_evidence_exists():
    p = SubjectPolicy(FPS)
    _, cid, tier = p.decide(cands(HOST, GUEST), Evidence(directive_id=2), 0)
    assert (cid, tier) == (2, TIER_DIRECTIVE)


def test_a_directive_waits_out_the_shot_hold_floor():
    p = SubjectPolicy(FPS, min_shot_hold_seconds=1.0)  # 25 frames
    c = cands(HOST, GUEST)
    p.decide(c, Evidence(asd_id=1), 0)
    _, cid, _ = p.decide(c, Evidence(directive_id=2), 5)
    assert cid == 1, "a directive must not chop a 0.2s-old shot in half"
    _, cid, tier = p.decide(c, Evidence(directive_id=2), 30)
    assert (cid, tier) == (2, TIER_DIRECTIVE)


def test_mouth_motion_needs_repeated_agreement():
    p = SubjectPolicy(FPS, min_shot_hold_seconds=0.0, mouth_confirm=3,
                      absolute_min_shot_seconds=0.0)
    c = cands(HOST, GUEST)
    p.decide(c, Evidence(asd_id=1), 0)
    for f in (1, 2):
        _, cid, _ = p.decide(c, Evidence(mouth_id=2), f)
        assert cid == 1, "one or two mouth samples is noise, not evidence"
    _, cid, tier = p.decide(c, Evidence(mouth_id=2), 3)
    assert (cid, tier) == (2, TIER_MOUTH)


def test_non_consecutive_mouth_samples_never_accumulate():
    p = SubjectPolicy(FPS, min_shot_hold_seconds=0.0, mouth_confirm=3,
                      absolute_min_shot_seconds=0.0)
    c = cands(HOST, GUEST)
    p.decide(c, Evidence(asd_id=1), 0)
    for f in range(1, 12, 2):
        p.decide(c, Evidence(mouth_id=2), f)      # challenger
        _, cid, _ = p.decide(c, Evidence(mouth_id=1), f + 1)  # incumbent
        assert cid == 1


def test_camera_holds_the_subject_when_all_evidence_disappears():
    """The failure this replaces: with no signal, selection fell through to
    box area and drifted onto the biggest face (the host with the mic)."""
    p = SubjectPolicy(FPS)
    c = cands(HOST, GUEST)
    p.decide(c, Evidence(asd_id=2), 0)
    for f in range(1, 40):
        _, cid, tier = p.decide(c, Evidence(), f)
        assert (cid, tier) == (2, TIER_HOLD)


def test_size_only_decides_when_nobody_is_framed():
    p = SubjectPolicy(FPS)
    _, cid, tier = p.decide(cands(HOST, GUEST), Evidence(), 0)
    assert (cid, tier) == (1, TIER_SIZE)


def test_size_cannot_jump_the_camera_on_a_single_detector_blink():
    p = SubjectPolicy(FPS, size_confirm=4)
    c = cands(HOST, GUEST)
    p.decide(c, Evidence(asd_id=2), 0)
    # The framed subject vanishes from detection for one sample.
    box, cid, tier = p.decide(cands(HOST), Evidence(), 1)
    assert box is None, "hold the last framing instead of jumping to the host"


def test_size_takes_over_once_the_subject_is_really_gone():
    p = SubjectPolicy(FPS, size_confirm=3)
    p.decide(cands(HOST, GUEST), Evidence(asd_id=2), 0)
    for f in (1, 2):
        p.decide(cands(HOST), Evidence(), f)
    _, cid, tier = p.decide(cands(HOST), Evidence(), 3)
    assert (cid, tier) == (1, TIER_SIZE)


def test_a_source_cut_lets_the_camera_re_decide_freely():
    p = SubjectPolicy(FPS, min_shot_hold_seconds=10.0,
                      absolute_min_shot_seconds=10.0)
    c = cands(HOST, GUEST)
    p.decide(c, Evidence(asd_id=1), 0)
    _, cid, tier = p.decide(c, Evidence(directive_id=2, scene_changed=True), 2)
    assert (cid, tier) == (2, TIER_DIRECTIVE)


def test_evidence_for_an_undetected_person_is_ignored():
    """A stale id from a previous shot must not blank the frame."""
    p = SubjectPolicy(FPS)
    _, cid, tier = p.decide(cands(HOST), Evidence(asd_id=99), 0)
    assert (cid, tier) == (1, TIER_SIZE)


def test_no_candidates_yields_no_decision():
    p = SubjectPolicy(FPS)
    assert p.decide([], Evidence(asd_id=1), 0) == (None, None, None)


def test_returned_box_is_the_callers_own_list_object():
    """Downstream bookkeeping (boosted sets, split cells) matches on box
    identity, so the policy must not copy."""
    p = SubjectPolicy(FPS)
    c = cands(HOST, GUEST)
    box, _, _ = p.decide(c, Evidence(asd_id=2), 0)
    assert box is c[1]["box"]


def test_a_shot_confirmed_by_lip_sync_becomes_protected_as_strong():
    p = SubjectPolicy(FPS, min_shot_hold_seconds=1.0,
                      absolute_min_shot_seconds=0.0)
    c = cands(HOST, GUEST)
    p.decide(c, Evidence(), 0)                 # tier 6, host
    _, _, tier = p.decide(c, Evidence(asd_id=1), 1)
    assert tier == TIER_ASD


def test_summary_reports_the_tier_mix():
    p = SubjectPolicy(FPS)
    c = cands(HOST, GUEST)
    p.decide(c, Evidence(asd_id=2), 0)
    p.decide(c, Evidence(), 1)
    assert "lip-sync" in p.summary() and "held" in p.summary()


# --- subject identity is the BOX, not the id -------------------------------
# The tracker relabels the same face constantly (confirmed id one frame, a
# fresh negative provisional the next). Before this, the camera read every
# relabel as a cut: 25 subject switches in a 20s clip whose source has 9.

def test_a_relabelled_but_unmoved_subject_is_not_a_cut():
    p = SubjectPolicy(FPS)
    p.decide(cands(HOST, GUEST), Evidence(asd_id=2), 0)
    started = p.shot_started
    # Same person, same place, brand-new provisional id.
    relabelled = [{"id": 1, "box": [100, 100, 400, 400]},
                  {"id": -7, "box": [905, 100, 200, 200]}]
    _, cid, _ = p.decide(relabelled, Evidence(asd_id=-7), 5)
    assert cid == -7
    assert p.shot_started == started, "no cut: the subject never changed"


def test_the_held_subject_is_found_by_overlap_when_its_id_vanishes():
    """Otherwise the camera falls through to size and lands on the host."""
    p = SubjectPolicy(FPS)
    p.decide(cands(HOST, GUEST), Evidence(asd_id=2), 0)
    relabelled = [{"id": 1, "box": [100, 100, 400, 400]},
                  {"id": -9, "box": [902, 102, 200, 200]}]
    _, cid, tier = p.decide(relabelled, Evidence(), 5)
    assert (cid, tier) == (-9, TIER_HOLD)


def test_a_genuinely_different_person_at_a_new_position_is_a_cut():
    p = SubjectPolicy(FPS, absolute_min_shot_seconds=0.0)
    p.decide(cands(HOST, GUEST), Evidence(asd_id=2), 0)
    started = p.shot_started
    _, cid, _ = p.decide(cands(HOST, GUEST), Evidence(asd_id=1), 5)
    assert cid == 1
    assert p.shot_started != started, "different person and place: a real cut"


# --- one person, two box shapes -------------------------------------------
# Detection merges MediaPipe faces with YOLO bodies, so the same person
# arrives as a 135x135 face on one sample and a 624x384 body on the next.
# Their IoU is 0.08; treating that as a new subject cut the camera on every
# alternation.

def test_a_face_and_the_body_it_belongs_to_are_one_subject():
    from subject_policy import same_subject
    face = (855, 156, 135, 135)
    body = (669, 99, 624, 384)
    assert same_subject(face, body, frame_width=1920)


def test_a_face_inside_a_different_persons_body_is_not_the_same_subject():
    from subject_policy import same_subject
    body = (669, 99, 624, 384)
    other_face = (1230, 300, 120, 120)   # inside the body box, but far right
    assert not same_subject(other_face, body, frame_width=1920)


def test_switching_between_face_and_body_boxes_is_not_a_cut():
    p = SubjectPolicy(FPS)
    face = [{"id": 5, "box": [855, 156, 135, 135]}]
    p.decide(face, Evidence(asd_id=5), 0, frame_width=1920)
    started = p.shot_started
    body = [{"id": -2, "box": [669, 99, 624, 384]}]
    _, cid, _ = p.decide(body, Evidence(asd_id=-2), 4, frame_width=1920)
    assert cid == -2
    assert p.shot_started == started


def test_no_shot_may_be_shorter_than_the_absolute_floor():
    """Even lip-sync, the strongest evidence in the system, cannot produce a
    0.17s shot — that is the 7-person-lineup flicker."""
    p = SubjectPolicy(FPS, absolute_min_shot_seconds=0.5)   # 12 frames
    c = cands(HOST, GUEST)
    p.decide(c, Evidence(asd_id=1), 0)
    _, cid, _ = p.decide(c, Evidence(asd_id=2), 4)
    assert cid == 1, "0.16s is not a shot"
    _, cid, tier = p.decide(c, Evidence(asd_id=2), 13)
    assert (cid, tier) == (2, TIER_ASD)


def test_the_absolute_floor_never_blocks_a_source_cut():
    p = SubjectPolicy(FPS, absolute_min_shot_seconds=5.0)
    c = cands(HOST, GUEST)
    p.decide(c, Evidence(asd_id=1), 0)
    _, cid, _ = p.decide(c, Evidence(asd_id=2, scene_changed=True), 2)
    assert cid == 2, "the director already cut; re-deciding is free"


def test_a_new_shot_lands_exactly_on_the_detection():
    from subject_policy import stabilize_box
    assert stabilize_box((10, 20, 30, 40), None) == (10, 20, 30, 40)


def test_the_same_subject_eases_between_face_and_body_boxes():
    """MediaPipe face vs YOLO head-and-chest for one person. Snapping
    between the two moved the crop on every detection."""
    from subject_policy import stabilize_box
    face = (855, 156, 135, 135)
    body = (669, 99, 624, 384)
    eased = stabilize_box(body, face, blend=0.35)
    for i in range(4):
        assert min(face[i], body[i]) <= eased[i] <= max(face[i], body[i])
    assert eased[2] < (face[2] + body[2]) / 2, "eased, not jumped"


def test_repeated_easing_converges_on_the_new_detection():
    from subject_policy import stabilize_box
    box, target = (0, 0, 100, 100), (0, 0, 500, 500)
    for _ in range(30):
        box = stabilize_box(target, box, blend=0.35)
    assert abs(box[2] - 500) < 1
