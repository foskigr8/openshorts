import pytest

import reframe_v2

from reframe_v2 import (
    COLOR_GRADE_FILTER,
    DELIVERY_MIN_WIDTH,
    MOUTH_ACTIVITY_THRESHOLD,
    PRIMARY_ASSERT_BOOST,
    PRIMARY_RETURN_BOOST,
    REACTION_MOTION_THRESHOLD,
    REACTION_SCORE_BOOST,
    SPEAKER_ANCHOR_TOLERANCE,
    SPEECH_ACTIVITY_SCORE_BOOST,
    SPLIT_HEAD_FRACTION,
    SPLIT_RECT_MAX_HEIGHT_FRACTION,
    SPLIT_RECT_MAX_WIDTH_FRACTION,
    UNIFIED_CROP_RATIO,
    ZOOM_EMPHASIS,
    _sendcmd_timestamp,
    _apply_primary_assert_boost,
    _apply_primary_return_bias,
    _apply_reaction_boost,
    _apply_speech_activity_boost,
    _directive_split_corroborated,
    _discount_stale_yolo,
    _banter_two_shot_eligible,
    _has_second_subject,
    _is_reaction_beat,
    _merge_person_candidates,
    _reaction_two_shot_boxes,
    _pick_primary_anchor,
    _primary_speaker_label,
    _resolve_speaker_binding,
    _subject_cell_rect,
    _update_mouth_activity,
    _zoom_confirm,
    _zoom_for_target,
    _zoom_hold_decision,
    _hook_secondary,
    _hook_union,
    cell_initial_xy,
    cell_scale_target,
    cell_xy_sendcmd_lines,
    cell_zoom_sendcmd_lines,
    dedupe_sendcmd_lines,
    delivery_size,
    split_cell_size,
    unified_filtergraph,
    unified_split_filtergraph,
)

VERTICAL = 9 / 16


def test_sendcmd_lines_dedupe_to_change_points():
    xs = [100, 100, 100, 104, 104, 110]
    lines = dedupe_sendcmd_lines(xs, fps=30.0)
    assert lines == [
        "0.000000 crop@c x 100;",
        "0.099999 crop@c x 104;",
        "0.166666 crop@c x 110;",
    ]


def test_sendcmd_timestamp_never_lands_after_its_frame():
    # Regression for the wrong-frame flash at hard cuts (plan item 3): a
    # 4-decimal stamp like "0.0667" is LARGER than frame 2's exact pts
    # (2/30 = 0.0666667), so ffmpeg's sendcmd misses that frame and fires
    # one frame late. The emitted stamp must stay strictly below i/fps while
    # remaining above the previous frame's time.
    fps = 30.0
    for i in range(1, 60):
        stamp = float(_sendcmd_timestamp(i, fps))
        exact = i / fps
        assert stamp < exact, f"frame {i}: {stamp} >= exact {exact}"
        assert stamp >= (i - 1) / fps, f"frame {i}: {stamp} < previous frame"


def test_sendcmd_static_camera_is_single_command():
    assert len(dedupe_sendcmd_lines([250] * 300, fps=30.0)) == 1


class TestDeliverySize:
    """A 720p source must not produce a 406x720 clip (prod audit, 25-jul-2026)."""

    def test_720p_source_is_upscaled_to_the_delivery_floor(self):
        assert delivery_size(1280, 720, VERTICAL) == (1080, 1920)

    def test_1080p_source_lands_on_the_floor_exactly(self):
        assert delivery_size(1920, 1080, VERTICAL) == (1080, 1920)

    def test_tiny_source_still_reaches_the_floor(self):
        # 360x640 sources were shipping as-is; platforms treat that as junk.
        assert delivery_size(640, 360, VERTICAL) == (1080, 1920)

    def test_high_res_source_is_never_downscaled(self):
        # 4K source: the native crop is already well past the floor, so keep it.
        assert delivery_size(3840, 2160, VERTICAL) == (1216, 2160)

    def test_already_vertical_source_keeps_its_full_height(self):
        assert delivery_size(1080, 1920, VERTICAL) == (1080, 1920)

    def test_square_output_uses_the_same_floor(self):
        assert delivery_size(1280, 720, 1.0) == (1080, 1080)

    def test_dimensions_are_always_even(self):
        for w, h in [(1280, 720), (1920, 1080), (854, 480), (3840, 2160), (641, 361)]:
            out_w, out_h = delivery_size(w, h, VERTICAL)
            assert out_w % 2 == 0 and out_h % 2 == 0, (w, h)

    def test_never_returns_below_the_floor(self):
        for w, h in [(320, 180), (640, 360), (1280, 720), (1920, 1080)]:
            assert delivery_size(w, h, VERTICAL)[0] >= DELIVERY_MIN_WIDTH

    def test_aspect_ratio_is_preserved(self):
        for w, h in [(1280, 720), (640, 360), (3840, 2160)]:
            out_w, out_h = delivery_size(w, h, VERTICAL)
            assert abs(out_w / out_h - VERTICAL) < 0.01, (w, h)


class TestUnifiedFiltergraph:
    """One consistent 3:4-shaped crop, letterboxed into the output canvas —
    used for every frame regardless of single-subject or group content,
    instead of switching between a narrow TRACK crop and a separate wide
    GENERAL layout (explicit user direction, 31-jul-2026)."""

    def test_ratio_is_wider_than_a_single_subject_crop_but_narrower_than_output(self):
        # 3:4 sits between the old 9:16 single-subject crop (too narrow for
        # a group) and the full output canvas (which would need no letterbox
        # at all) — that's the whole point of picking it.
        assert VERTICAL < UNIFIED_CROP_RATIO < 1.0

    def test_contains_blur_and_dynamic_crop_sendcmd(self):
        graph = unified_filtergraph(1080, 1920, 810, 1080, "/tmp/cmd.txt", 100)
        assert "gblur" in graph
        assert "sendcmd=f='/tmp/cmd.txt'" in graph
        assert "crop@c=w=810:h=1080:x=100:y=0" in graph

    def test_foreground_fills_full_output_width(self):
        graph = unified_filtergraph(1080, 1920, 810, 1080, "/tmp/cmd.txt", 0)
        fg = graph.split("[fga]")[-1]
        assert "scale=1080:" in fg

    def test_foreground_height_is_even_and_letterboxed(self):
        # crop_w/crop_h = 810/1080 = 0.75; scaled to out_w=1080 gives
        # fg_h = 1080 * (1080/810) = 1440 — well under out_h=1920, so it
        # DOES letterbox (this is the point: consistent shape, not full-fill).
        graph = unified_filtergraph(1080, 1920, 810, 1080, "/tmp/cmd.txt", 0)
        fg = graph.split("[fga]")[-1]
        fg_h = int(fg.split("scale=1080:")[1].split(",")[0])
        assert fg_h % 2 == 0
        assert fg_h < 1920

    def test_overlay_is_horizontally_flush_vertically_centered(self):
        graph = unified_filtergraph(1080, 1920, 810, 1080, "/tmp/cmd.txt", 0)
        assert "overlay=x=0:y=(H-h)/2" in graph

    def test_color_grade_applied_after_overlay_and_output_still_labeled_v(self):
        # The grade must run on the fully-composited frame (after the
        # overlay), and the graph must still terminate in [v] — that's the
        # label render() maps to the output stream.
        graph = unified_filtergraph(1080, 1920, 810, 1080, "/tmp/cmd.txt", 0)
        assert graph.endswith(f"{COLOR_GRADE_FILTER}[v]")
        overlay_pos = graph.index("overlay=x=0:y=(H-h)/2")
        grade_pos = graph.index(COLOR_GRADE_FILTER)
        assert grade_pos > overlay_pos


class TestReactionBoost:
    """A non-speaker who moves/gestures sharply between detection samples
    gets a score boost so they can outscore a passively-framed speaker in
    the tracker's existing size-based selection — the missing signal for
    real editors' reaction cuts (~30% of cuts in studied references go to
    a non-speaker doing something visually notable, not the speaker)."""

    def _cand(self, x, y, w, h):
        return {'box': [x, y, w, h], 'score': w * h}

    def test_no_previous_candidates_is_a_no_op(self):
        candidates = [self._cand(100, 100, 50, 50)]
        original_score = candidates[0]['score']
        _apply_reaction_boost(candidates, None)
        assert candidates[0]['score'] == original_score

    def test_stationary_face_gets_no_boost(self):
        prev = [self._cand(100, 100, 50, 50)]
        candidates = [self._cand(101, 100, 50, 50)]  # 1px drift, not motion
        original_score = candidates[0]['score']
        _apply_reaction_boost(candidates, prev)
        assert candidates[0]['score'] == original_score

    def test_sharp_lateral_movement_gets_boosted(self):
        prev = [self._cand(100, 100, 50, 50)]
        # center moves from 125 to 165 = 40px shift on a 50px-wide box -> 0.8 ratio, over threshold
        candidates = [self._cand(140, 100, 50, 50)]
        original_score = candidates[0]['score']
        _apply_reaction_boost(candidates, prev)
        assert candidates[0]['score'] == pytest.approx(original_score * REACTION_SCORE_BOOST)

    def test_sharp_size_change_gets_boosted(self):
        # A lean-in/gesture that changes box height sharply, even with a
        # centered x position, should also count as motion.
        prev = [self._cand(100, 100, 50, 50)]
        candidates = [self._cand(100, 100, 50, 90)]  # height jumps 50 -> 90
        original_score = candidates[0]['score']
        _apply_reaction_boost(candidates, prev)
        assert candidates[0]['score'] == pytest.approx(original_score * REACTION_SCORE_BOOST)

    def test_unmatched_new_candidate_gets_no_boost(self):
        # Someone just entered frame — one sample isn't motion evidence.
        prev = [self._cand(100, 100, 50, 50)]
        candidates = [self._cand(900, 100, 50, 50)]  # far away -> no match within threshold
        original_score = candidates[0]['score']
        _apply_reaction_boost(candidates, prev)
        assert candidates[0]['score'] == original_score

    def test_matches_each_candidate_to_its_nearest_previous_by_position(self):
        prev = [self._cand(100, 100, 50, 50), self._cand(500, 100, 50, 50)]
        candidates = [
            self._cand(103, 100, 50, 50),   # near prev[0], stationary -> no boost
            self._cand(560, 100, 50, 50),   # near prev[1], moved 60px -> boosted
        ]
        _apply_reaction_boost(candidates, prev)
        assert candidates[0]['score'] == 50 * 50
        assert candidates[1]['score'] == pytest.approx(50 * 50 * REACTION_SCORE_BOOST)

    def test_threshold_constant_is_a_fraction_not_a_pixel_count(self):
        # Sanity check on the tuned constant's shape, not an exact value.
        assert 0 < REACTION_MOTION_THRESHOLD < 3
        assert REACTION_SCORE_BOOST > 1


class TestMergePersonCandidates:
    """YOLO body candidates fill in for MediaPipe's face detector when it
    finds nothing at all (a face mask, a head turned away) — ground-truthed
    31-jul-2026 on a clip where the actual speaker vanished from tracking
    entirely at points. Must not double-count a person who already has a
    good face detection."""

    def _cand(self, x, w):
        return {'box': [x, 100, w, 50], 'score': w * 50}

    def test_yolo_candidate_added_when_no_overlapping_face(self):
        faces = [self._cand(100, 50)]
        yolo = [self._cand(500, 60)]
        merged = _merge_person_candidates(faces, yolo)
        assert len(merged) == 2

    def test_yolo_candidate_dropped_when_it_matches_an_existing_face(self):
        faces = [self._cand(100, 50)]
        yolo = [self._cand(105, 55)]  # same person, slightly different box
        merged = _merge_person_candidates(faces, yolo)
        assert len(merged) == 1

    def test_no_face_candidates_keeps_all_yolo_candidates(self):
        # The masked/turned-away case: face detection found nobody, YOLO
        # candidates are the only way this person stays trackable.
        yolo = [self._cand(100, 50), self._cand(500, 60)]
        merged = _merge_person_candidates([], yolo)
        assert len(merged) == 2

    def test_no_yolo_candidates_keeps_face_candidates_unchanged(self):
        faces = [self._cand(100, 50)]
        merged = _merge_person_candidates(faces, [])
        assert merged == faces


class TestMouthActivityAndSpeechBoost:
    """The only direct audio-to-visual speaker-IDENTITY link in the system:
    matches mouth-position movement across frames to tell WHICH visible face
    is producing the current audio, instead of picking whoever's biggest."""

    def _cand(self, x, mouth_frac):
        return {'box': [x, 100, 50, 50], 'score': 50 * 50, 'mouth_frac': mouth_frac}

    def test_first_sample_has_no_activity_yet(self):
        candidates = [self._cand(100, 0.5)]
        _update_mouth_activity(candidates, None)
        assert candidates[0]['mouth_activity'] == 0.0

    def test_moving_mouth_across_samples_builds_activity(self):
        c1 = [self._cand(100, 0.5)]
        _update_mouth_activity(c1, None)
        c2 = [self._cand(101, 0.6)]
        _update_mouth_activity(c2, c1)
        c3 = [self._cand(100, 0.48)]
        _update_mouth_activity(c3, c2)
        assert c3[0]['mouth_activity'] == pytest.approx(0.6 - 0.48)

    def test_static_mouth_stays_at_zero_activity(self):
        c1 = [self._cand(100, 0.5)]
        _update_mouth_activity(c1, None)
        c2 = [self._cand(100, 0.5)]
        _update_mouth_activity(c2, c1)
        c3 = [self._cand(100, 0.5)]
        _update_mouth_activity(c3, c2)
        assert c3[0]['mouth_activity'] == 0.0

    def test_no_active_speaker_is_a_no_op(self):
        candidates = [self._cand(100, 0.5)]
        candidates[0]['mouth_activity'] = 0.5
        original = candidates[0]['score']
        _apply_speech_activity_boost(candidates, None)
        assert candidates[0]['score'] == original

    def test_single_candidate_is_a_no_op(self):
        # Nothing to compare against — reaction/size scoring is enough.
        candidates = [self._cand(100, 0.5)]
        candidates[0]['mouth_activity'] = 1.0
        original = candidates[0]['score']
        _apply_speech_activity_boost(candidates, 'A')
        assert candidates[0]['score'] == original

    def test_clear_mouth_movement_winner_gets_boosted(self):
        talker = self._cand(100, 0.5)
        talker['mouth_activity'] = MOUTH_ACTIVITY_THRESHOLD * 2
        silent = self._cand(500, 0.5)
        silent['mouth_activity'] = 0.0
        candidates = [talker, silent]
        _apply_speech_activity_boost(candidates, 'A')
        assert talker['score'] == pytest.approx(50 * 50 * SPEECH_ACTIVITY_SCORE_BOOST)
        assert silent['score'] == 50 * 50

    def test_marginal_difference_does_not_force_a_pick(self):
        a = self._cand(100, 0.5)
        a['mouth_activity'] = MOUTH_ACTIVITY_THRESHOLD * 1.05
        b = self._cand(500, 0.5)
        b['mouth_activity'] = MOUTH_ACTIVITY_THRESHOLD * 1.0
        candidates = [a, b]
        _apply_speech_activity_boost(candidates, 'A')
        assert a['score'] == 50 * 50
        assert b['score'] == 50 * 50

    def test_below_threshold_activity_never_boosts(self):
        a = self._cand(100, 0.5)
        a['mouth_activity'] = MOUTH_ACTIVITY_THRESHOLD * 0.5
        b = self._cand(500, 0.5)
        b['mouth_activity'] = 0.0
        candidates = [a, b]
        _apply_speech_activity_boost(candidates, 'A')
        assert a['score'] == 50 * 50


class TestSendcmdRectDedupe:
    """The unified renderer drives crop@c's full rect (w/h/x/y) so a zoom
    changes crop size and position at the same timestamp, while a pure pan
    still collapses to a single change-point line."""

    def test_tuple_rects_emit_all_four_params(self):
        rects = [(100, 0, 810, 1080)] * 5 + [(120, 10, 810, 1080)] * 5
        lines = dedupe_sendcmd_lines(rects, fps=30.0)
        assert lines == [
            "0.000000 crop@c w 810; 0.000000 crop@c h 1080; "
            "0.000000 crop@c x 100; 0.000000 crop@c y 0;",
            "0.166666 crop@c w 810; 0.166666 crop@c h 1080; "
            "0.166666 crop@c x 120; 0.166666 crop@c y 10;",
        ]

    def test_zoom_change_is_a_change_point_even_with_same_x(self):
        # Zoom in without panning: x unchanged but w/h shrink — must still
        # emit a line (whole-tuple dedupe).
        rects = [(100, 0, 810, 1080)] * 3 + [(100, 108, 688, 918)] * 3
        lines = dedupe_sendcmd_lines(rects, fps=30.0)
        assert len(lines) == 2
        assert "crop@c w 688;" in lines[1]
        assert "crop@c h 918;" in lines[1]

    def test_bare_x_values_keep_legacy_format(self):
        assert dedupe_sendcmd_lines([100, 100, 104], fps=30.0) == [
            "0.000000 crop@c x 100;",
            "0.066666 crop@c x 104;",
        ]


class TestUnifiedFiltergraphVertical:
    def test_initial_y_is_wired_into_crop(self):
        graph = unified_filtergraph(1080, 1920, 810, 1080, "/tmp/cmd.txt", 100,
                                    initial_y=40)
        assert "crop@c=w=810:h=1080:x=100:y=40" in graph

    def test_default_initial_y_is_zero(self):
        graph = unified_filtergraph(1080, 1920, 810, 1080, "/tmp/cmd.txt", 100)
        assert "crop@c=w=810:h=1080:x=100:y=0" in graph

    def test_scale_stays_at_fixed_output_size(self):
        # The zoomed-in look comes from a SMALLER crop scaled up to the same
        # fixed output size — the final scale step must never change.
        graph = unified_filtergraph(1080, 1920, 810, 1080, "/tmp/cmd.txt", 0)
        assert "scale=1080:1440" in graph


class TestMergePersonCandidatesVertical:
    """Dedup must require BOTH x and y proximity — an x-only check merged two
    different people standing at the same x (one behind the other)."""

    def _cand(self, x, y, w=50, h=50):
        return {'box': [x, y, w, h], 'score': w * h}

    def test_same_x_but_different_y_is_not_merged(self):
        faces = [self._cand(100, 100)]
        yolo = [self._cand(105, 400)]  # another person at the same x, far below
        merged = _merge_person_candidates(faces, yolo)
        assert len(merged) == 2

    def test_same_x_and_similar_y_is_merged(self):
        faces = [self._cand(100, 100)]
        yolo = [self._cand(105, 108)]  # same person, slightly different box
        merged = _merge_person_candidates(faces, yolo)
        assert len(merged) == 1


class TestBoostFloor:
    """Reaction/speech boosts must not lift background-small candidates: a
    tiny background person's motion can otherwise out-boost a large
    foreground subject's baseline (user: "focusing on irrelevant things")."""

    def _cand(self, x, y, w, h, raw=None):
        cand = {'box': [x, y, w, h], 'score': w * h}
        if raw is not None:
            cand['raw_score'] = raw
        return cand

    def test_small_candidate_is_not_reaction_boosted(self):
        big = self._cand(100, 100, 200, 200, raw=40000)
        small = self._cand(400, 100, 30, 30, raw=900)
        prev = [self._cand(100, 100, 200, 200), self._cand(400, 100, 30, 30)]
        candidates = [big, small]
        original_big = big['score']
        original_small = small['score']
        _apply_reaction_boost(candidates, prev)
        assert small['score'] == original_small, "background-small reactor must not be boosted"
        assert big['score'] == original_big, "big candidate unchanged (stationary)"

    def test_large_reacting_candidate_still_gets_boosted(self):
        cand = self._cand(100, 100, 100, 100, raw=10000)
        prev = [self._cand(100, 100, 100, 100)]
        candidates = [cand]
        original = cand['score']
        _apply_reaction_boost(candidates, prev)  # no size change -> no motion
        candidates[0]['box'] = [200, 100, 100, 100]  # moved sharply
        candidates[0]['raw_score'] = 10000
        _apply_reaction_boost(candidates, prev)
        assert candidates[0]['score'] == pytest.approx(original * REACTION_SCORE_BOOST)

    def test_small_mouth_mover_is_not_speech_boosted(self):
        big = self._cand(100, 100, 200, 200, raw=40000)
        big['mouth_activity'] = MOUTH_ACTIVITY_THRESHOLD * 2
        small = self._cand(500, 100, 20, 20, raw=400)
        small['mouth_activity'] = MOUTH_ACTIVITY_THRESHOLD * 3
        candidates = [big, small]
        original = small['score']
        _apply_speech_activity_boost(candidates, 'A')
        assert small['score'] == original
        assert big['score'] == 200 * 200


class TestCoveragePolicy:
    """Problem 3's universal rule, as pure helpers: the clip has a PRIMARY
    subject (most talking time) and a cutaway to anyone else is brief — the
    return bias nudges selection back to the primary once a non-boosted
    cutaway overstays."""

    def test_primary_label_is_the_most_talkative(self):
        turns = [(0, 10, 'A'), (10, 20, 'B'), (20, 30, 'B')]
        assert _primary_speaker_label(turns) == 'B'

    def test_no_diarization_has_no_primary(self):
        assert _primary_speaker_label(None) is None
        assert _primary_speaker_label([(0, 10, None), (10, 20, None)]) is None

    def test_return_bias_boosts_the_primary_candidate(self):
        primary = {'id': 1, 'box': [100, 100, 50, 50], 'score': 100}
        secondary = {'id': 2, 'box': [500, 100, 80, 80], 'score': 300}
        assert _apply_primary_return_bias([primary, secondary], 1) is True
        assert primary['score'] == pytest.approx(100 * PRIMARY_RETURN_BOOST)
        assert secondary['score'] == 300

    def test_return_bias_noop_when_primary_is_off_screen(self):
        secondary = {'id': 2, 'box': [500, 100, 80, 80], 'score': 300}
        assert _apply_primary_return_bias([secondary], 1) is False
        assert secondary['score'] == 300


class TestStaleYoloDiscount:
    """Cached YOLO boxes describe an OLD position during camera/subject
    motion — merge them at reduced weight, drop them past max_stale."""

    def _cand(self, x, score):
        return {'box': [x, 100, 50, 50], 'score': score}

    def test_fresh_cache_is_unchanged(self):
        cached = [self._cand(100, 100)]
        merged = _discount_stale_yolo(cached, 0, max_stale=8)
        assert merged[0]['score'] == 100

    def test_stale_cache_is_discounted(self):
        cached = [self._cand(100, 100)]
        merged = _discount_stale_yolo(cached, 4, max_stale=8)
        assert merged[0]['score'] == pytest.approx(100 * 0.5)

    def test_too_stale_cache_is_dropped(self):
        cached = [self._cand(100, 100)]
        assert _discount_stale_yolo(cached, 9, max_stale=8) == []
        assert _discount_stale_yolo([], 9, max_stale=8) == []


class TestZoomPolicy:
    """Push-in only for a dominant, currently-boosted target — emphasis, not
    constant zooming."""

    def _cand(self, x, w=100, h=100, raw=None):
        cand = {'box': [x, 100, w, h], 'score': w * h}
        if raw is not None:
            cand['raw_score'] = raw
        return cand

    def test_unboosted_target_never_zooms(self):
        cand = self._cand(100, raw=10000)
        assert _zoom_for_target([cand], cand['box'], set()) == 1.0

    def test_dominant_boosted_target_zooms_in(self):
        cand = self._cand(100, raw=10000)
        boosted = {id(cand['box'])}
        assert _zoom_for_target([cand], cand['box'], boosted) == ZOOM_EMPHASIS

    def test_background_boosted_target_does_not_zoom(self):
        small = self._cand(100, w=20, h=20, raw=400)
        big = self._cand(500, w=200, h=200, raw=40000)
        boosted = {id(small['box'])}
        assert _zoom_for_target([small, big], small['box'], boosted) == 1.0

    def test_crowd_shot_never_zooms(self):
        # Two comparable subjects (a group/lineup): even a boosted one must
        # not push in — zoom is for a single tightly-locked subject.
        a = self._cand(100, w=100, h=100, raw=10000)
        b = self._cand(400, w=110, h=110, raw=12100)
        boosted = {id(a['box'])}
        assert _zoom_for_target([a, b], a['box'], boosted) == 1.0


class TestZoomConfirmation:
    """A push-in must be SUSTAINED before the zoom target moves — a
    one-sample reaction blip would otherwise cause a push-in/push-out wobble."""

    def test_single_boost_cycle_does_not_zoom(self):
        target, frames = _zoom_confirm(ZOOM_EMPHASIS, 0, 3)
        assert target is None and frames == 1

    def test_sustained_boost_confirms_after_threshold(self):
        target, frames = ZOOM_EMPHASIS, 0
        for _ in range(3):
            target, frames = _zoom_confirm(ZOOM_EMPHASIS, frames, 3)
        assert target == ZOOM_EMPHASIS

    def test_relax_is_immediate_on_first_unboosted_cycle(self):
        target, frames = _zoom_confirm(ZOOM_EMPHASIS, 1, 3)
        assert target is None and frames == 2  # still confirming
        target, frames = _zoom_confirm(1.0, frames, 3)
        assert target == 1.0 and frames == 0


class TestZoomHold:
    """A confirmed push-in is held briefly before relaxing — a pause in the
    boost must not bounce the zoom back out immediately."""

    def test_confirm_arms_the_hold(self):
        target, hold = _zoom_hold_decision(ZOOM_EMPHASIS, 0, 3)
        assert target == ZOOM_EMPHASIS and hold == 3

    def test_relax_is_deferred_while_holding(self):
        target, hold = _zoom_hold_decision(1.0, 3, 3)
        assert target is None and hold == 2  # keep pushing in

    def test_relax_lands_after_the_hold_expires(self):
        target, hold = _zoom_hold_decision(1.0, 1, 3)
        assert target is None and hold == 0
        target, hold = _zoom_hold_decision(1.0, hold, 3)
        assert target == 1.0 and hold == 0


class TestSpeechBoostIdentityGate:
    """A mouth-mover may only be speech-boosted when it is the active
    speaker's VERIFIED identity (bound_id) — or, while unbound, only when it
    is the candidate the camera is already framing. A laughing reactor in a
    group shot must not drag the camera off the actual speaker (ground-
    truthed 31-jul-2026: the host got boosted 3 samples running during the
    primary's line and the crop parked on him for 2+ seconds)."""

    def _cand(self, x, w=60, mid=1):
        return {'box': [x, 100, w, 50], 'score': w * 50, 'raw_score': w * 50,
                'id': mid, 'mouth_activity': 0.05}

    def _talker(self, x, mid=1):
        cand = self._cand(x, mid=mid)
        cand['mouth_activity'] = 0.5  # clearly the mouth-mover
        return cand

    def test_bound_id_matching_winner_is_boosted(self):
        talker = self._talker(100, mid=1)
        silent = self._cand(500, mid=2)
        candidates = [talker, silent]
        original = talker['score']
        _apply_speech_activity_boost(candidates, 'A', bound_id=1)
        assert talker['score'] == pytest.approx(original * SPEECH_ACTIVITY_SCORE_BOOST)

    def test_bound_id_other_than_winner_suppresses_boost(self):
        # The mouth-mover is a different person than the verified speaker
        # (e.g. the host laughing through the primary's line) -> no boost.
        talker = self._talker(100, mid=1)
        silent = self._cand(500, mid=2)
        candidates = [talker, silent]
        original = talker['score']
        _apply_speech_activity_boost(candidates, 'A', bound_id=2)
        assert talker['score'] == original

    def test_unbound_winner_must_be_the_current_target(self):
        talker = self._talker(100, mid=1)
        silent = self._cand(500, mid=2)
        candidates = [talker, silent]
        original = talker['score']
        # Camera is framing id=2 -> a different mouth-mover can't hijack it.
        _apply_speech_activity_boost(candidates, 'A', current_target_id=2)
        assert talker['score'] == original
        # Camera already frames the mouth-mover -> reinforcing boost is fine.
        _apply_speech_activity_boost(candidates, 'A', current_target_id=1)
        assert talker['score'] == pytest.approx(original * SPEECH_ACTIVITY_SCORE_BOOST)


class TestPrimaryAssertBoost:
    """Audio-assert fallback: diarization says the PRIMARY is talking, so
    boost their candidate even with zero mouth activity (hidden face) —
    the fix for the park-on-the-reactor freeze."""

    def _cand(self, cid, w, h=50):
        return {'box': [100, 100, w, h], 'score': w * h, 'raw_score': w * h,
                'id': cid}

    def test_assert_boosts_primary_without_mouth_activity(self):
        primary = self._cand(1, 80)
        secondary = self._cand(2, 60)
        candidates = [primary, secondary]
        assert _apply_primary_assert_boost(candidates, 1) is True
        assert primary['score'] == pytest.approx(80 * 50 * PRIMARY_ASSERT_BOOST)
        assert secondary['score'] == 60 * 50

    def test_assert_rejects_background_sliver(self):
        sliver = self._cand(1, 10)
        big = self._cand(2, 200)
        candidates = [sliver, big]
        assert _apply_primary_assert_boost(candidates, 1) is False
        assert sliver['score'] == 10 * 50

    def test_assert_noop_when_primary_is_off_screen(self):
        candidates = [self._cand(2, 60)]
        assert _apply_primary_assert_boost(candidates, 1) is False

    def test_assert_does_not_double_an_existing_boost(self):
        primary = self._cand(1, 80)
        candidates = [primary, self._cand(2, 60)]
        boosted = {id(primary['box'])}
        _apply_primary_assert_boost(candidates, 1, boosted=boosted)
        assert primary['score'] == 80 * 50  # not multiplied again


class TestPrimaryAnchorPick:
    """The anchor is the candidate the natural selection holds during the
    primary's turns: stable position, dominant presence."""

    def _stats(self, n, total, cx):
        return {'G': {1: {'n': n, 'total': total, 'cx': cx}}}

    def test_stable_dominant_target_is_the_anchor(self):
        stats = self._stats(7, 7, [620, 630, 640, 635, 628, 640, 636])
        assert _pick_primary_anchor(stats, 'G', 1920) == 1

    def test_jumping_target_is_rejected(self):
        stats = self._stats(7, 7, [200, 1200, 200, 1200, 200, 1200, 200])
        assert _pick_primary_anchor(stats, 'G', 1920) is None

    def test_low_presence_is_rejected(self):
        stats = self._stats(2, 10, [620, 630])
        assert _pick_primary_anchor(stats, 'G', 1920) is None

    def test_no_primary_stats_returns_none(self):
        stats = {'A': {2: {'n': 5, 'total': 5, 'cx': [100, 101]}}}
        assert _pick_primary_anchor(stats, 'G', 1920) is None


# Imported here rather than in the module header: this file's header block is
# edited concurrently, and a bottom-of-file import can't collide with it.
from reframe_v2 import (  # noqa: E402
    _apply_directive_boost,
    _directive_at_time,
    _directive_zoom,
)


class TestSceneContextDirection:
    """Third verification layer (main.analyze_scene_context): Gemini watches
    the finished cut and says WHO the camera should be on and why. The
    heuristics can only see size/motion/mouths, so a group laughing at
    someone clowning framed the laughers instead of the clown — these
    directives are the only signal that carries intent."""

    def _cand(self, x, w=100):
        return {'box': [x, 100, w, 100], 'score': w * 100, 'raw_score': w * 100}

    def _d(self, start=0.0, end=5.0, x=0.5, reason="speaking", intensity=0.5):
        return {"start": start, "end": end, "x_position": x,
                "reason": reason, "intensity": intensity, "subject": "someone"}

    def test_directive_lookup_by_time(self):
        ds = [self._d(0, 2), self._d(2, 4, x=0.9)]
        assert _directive_at_time(ds, 1.0)["x_position"] == 0.5
        assert _directive_at_time(ds, 3.0)["x_position"] == 0.9

    def test_no_directive_outside_any_span(self):
        assert _directive_at_time([self._d(0, 2)], 5.0) is None
        assert _directive_at_time(None, 1.0) is None

    def test_boosts_the_candidate_nearest_the_directed_position(self):
        left = self._cand(100)      # centre 150 of 1000 -> 0.15
        right = self._cand(800)     # centre 850 of 1000 -> 0.85
        box = _apply_directive_boost([left, right], self._d(x=0.85), 1000)
        assert box is right['box']
        assert right['score'] > left['score']

    def test_ignores_a_directive_pointing_at_empty_frame(self):
        # A confident instruction aimed where nobody is must fall back to the
        # heuristics rather than dragging the camera onto nothing.
        only = self._cand(100)
        before = only['score']
        assert _apply_directive_boost([only], self._d(x=0.95), 1000) is None
        assert only['score'] == before

    def test_no_directive_is_a_no_op(self):
        c = self._cand(100)
        assert _apply_directive_boost([c], None, 1000) is None
        assert c['score'] == 100 * 100

    def test_payoff_beat_pushes_in_tighter_than_a_talking_head(self):
        speaking = _directive_zoom(self._d(reason="speaking", intensity=0.6))
        cause = _directive_zoom(self._d(reason="causing_reaction", intensity=0.6))
        assert cause < speaking, "the cause of a reaction is the payoff shot"

    def test_low_intensity_stays_wide(self):
        assert _directive_zoom(self._d(intensity=0.1)) == 1.0

    def test_zoom_never_exceeds_the_configured_emphasis(self):
        for reason in ("speaking", "causing_reaction", "referenced", "reacting"):
            z = _directive_zoom(self._d(reason=reason, intensity=1.0))
            assert z >= ZOOM_EMPHASIS - 1e-9

    def test_directive_outranks_the_heuristic_boosts(self):
        # A big, strongly-reacting candidate vs. a smaller directed one: the
        # directive must still win, because it is intent and the other is a
        # proxy for it.
        big = self._cand(100, w=300)
        _apply_reaction_boost([big], None)
        big['score'] *= REACTION_SCORE_BOOST
        small = self._cand(800, w=80)
        _apply_directive_boost([big, small], self._d(x=0.86), 1000)
        assert small['score'] > big['score']



class TestSplitCellSize:
    """Split cells use a FIXED-size crop (ffmpeg trac #10984: crop stops
    providing frames when its height changes via sendcmd), so the window is
    as large as the source allows up to the 720x640 (9:8) cap, rounded even.
    (990x880 was wide/tall enough to cram a whole group into one cell —
    ground-truthed 31-jul-2026 — so the cap shrank to keep a cell to one
    person.)"""

    def test_full_cap_for_large_sources(self):
        assert split_cell_size(1920, 1080) == (720, 640)

    def test_720p_source_uses_source_bounds(self):
        assert split_cell_size(1280, 720) == (720, 640)

    def test_small_source_is_bounded_by_the_source_and_the_aspect(self):
        # Under the caps the window is bounded by the source — but it must
        # STILL match the target aspect. Returning the raw source bounds
        # (640x480 = 1.333 against a 1.125 target) was the bug: the extra
        # width is discarded by the downstream crop, so the cell silently
        # loses subject content rather than filling its half of the canvas.
        w, h = split_cell_size(640, 480)
        assert w <= 640 and h <= 480
        assert w / h == pytest.approx(9 / 8, abs=0.02)

    def test_even_rounding_for_odd_sources(self):
        w, h = split_cell_size(641, 481)
        assert w % 2 == 0 and h % 2 == 0
        assert w <= 641 and h <= 481
        assert w / h == pytest.approx(9 / 8, abs=0.02)


class TestCellCmdFiles:
    """The fixed-size cell crop gets TWO sendcmd files: x/y reposition the
    crop (safe), w/h drive the downstream zoom scale (safe) — never w/h on
    the crop itself, which is the trac #10984 freeze."""

    RECTS = [(318, 0, 726, 646)] * 3 + [(300, 20, 600, 500)] * 3

    def test_xy_lines_emit_only_position_changes(self):
        lines = cell_xy_sendcmd_lines(self.RECTS, fps=30.0, target="crop@ct",
                                      cell_w=990, cell_h=880,
                                      orig_w=1280, orig_h=720)
        assert len(lines) == 2
        assert "crop@ct w" not in "".join(lines)
        assert "crop@ct h" not in "".join(lines)
        assert lines[0] == "0.000000 crop@ct x 186; 0.000000 crop@ct y 0;"
        # second rect center (600, 270) minus half-window (495, 440):
        # x = 105, y clamped to 0
        assert "crop@ct x 105;" in lines[1]

    def test_xy_positions_are_clamped_to_source(self):
        rect = (0, 0, 200, 200)
        assert cell_initial_xy(rect, 990, 880, 1280, 720) == (0, 0)
        rect = (1080, 520, 200, 200)
        assert cell_initial_xy(rect, 990, 880, 1280, 720) == (290, 0)

    def test_zoom_lines_drive_scale_not_crop(self):
        lines = cell_zoom_sendcmd_lines(self.RECTS, fps=30.0, target="scale@st",
                                        cell_w=990, cell_h=880,
                                        out_w=1080, half_h=960)
        assert len(lines) == 2
        assert "crop@st" not in "".join(lines)
        # rect 726x646 -> 1080*990/726 = 1472.7 -> 1474; 960*880/646 = 1307.1 -> 1308
        assert lines[0] == "0.000000 scale@st w 1474; 0.000000 scale@st h 1308;"

    def test_zoom_floor_and_evenness(self):
        # a rect as wide as the cell maps to exactly the output size
        assert cell_scale_target((0, 0, 990, 880), 990, 880, 1080, 960) == (1080, 960)
        # smaller rect zooms IN (larger target), never below the output size
        sw, sh = cell_scale_target((0, 0, 500, 450), 990, 880, 1080, 960)
        assert sw > 1080 and sh > 960 and sw % 2 == 0 and sh % 2 == 0


class TestUnifiedSplitFiltergraph:
    """The split graph must never resize crop@ct/crop@cb — zoom lives in the
    downstream scale (trac #10984)."""

    def _graph(self):
        return unified_split_filtergraph(
            1080, 1920, 660, 880, "/tmp/cmd.txt", 100, 0,
            "/tmp/top_xy.txt", "/tmp/top_zoom.txt",
            "/tmp/bot_xy.txt", "/tmp/bot_zoom.txt",
            990, 880, (186, 0, 1474, 1308), (500, 400, 1080, 960),
            "between(t,10,13)",
            "/tmp/top_frame.txt", "/tmp/bot_frame.txt", (12, 34), (56, 78))

    def test_cells_use_fixed_size_crop_with_xy_sendcmd(self):
        g = self._graph()
        assert "crop@ct=w=990:h=880:x=186:y=0" in g
        assert "crop@cb=w=990:h=880:x=500:y=400" in g
        assert "sendcmd=f='/tmp/top_xy.txt'" in g
        assert "sendcmd=f='/tmp/bot_xy.txt'" in g

    def test_zoom_goes_to_scale_never_crop(self):
        g = self._graph()
        assert "scale@st=1474:1308" in g
        assert "scale@sb=1080:960" in g
        # no w/h sendcmd targeting the crops
        assert "crop@ct w" not in g and "crop@cb w" not in g
        assert "crop@ct h" not in g and "crop@cb h" not in g

    def test_halves_and_overlay_enable(self):
        g = self._graph()
        assert "crop@ft=w=1080:h=960:x=12:y=34,setsar=1[top]" in g
        assert "overlay=x=0:y=0:enable='between(t,10,13)'" in g
        assert "overlay=x=0:y=960:enable='between(t,10,13)'" in g

    def test_color_grade_applied_and_output_still_labeled_v(self):
        g = self._graph()
        assert g.endswith(f"{COLOR_GRADE_FILTER}[v]")



class TestHookTwoShot:
    """The hook two-shot must pair the star with the person who can actually
    share the fixed 3:4 crop — the naive biggest-other rule picks a distant
    third person, the union overflows the crop, and the crop falls on the
    gap (ground-truthed: opening landed on the host, girl out of frame)."""

    def _cand(self, cid, x, y, w, h):
        return {"id": cid, "box": [x, y, w, h], "score": w * h}

    def test_picks_nearby_substantial_candidate_over_distant_biggest(self):
        # girl at x~531; host face small but near; distant third person
        # larger but far away — the hook must pair with the HOST.
        girl = self._cand(1, 531, 78, 303, 315)
        host = self._cand(0, 1149, 63, 111, 111)
        distant = self._cand(2, 1743, 99, 174, 267)
        sec = _hook_secondary(girl, [girl, host, distant], 660, 1920, 880,
                              exclude_ids={1})
        assert sec is host  # distant is bigger (46458 > 12321) but too far

    def test_falls_back_to_biggest_when_nobody_is_near(self):
        girl = self._cand(1, 100, 78, 303, 315)
        far = self._cand(2, 1700, 99, 174, 267)
        sec = _hook_secondary(girl, [girl, far], 660, 1920, 880,
                              exclude_ids={1})
        assert sec is far

    def test_union_pad_capped_to_crop_width(self):
        # raw span 1149+111-531 = 729 > 660 crop -> no padding at all
        u = _hook_union([531, 78, 303, 315], [1149, 63, 111, 111],
                        660, 1920, 880)
        assert u[2] == 729
        # span 600 fits raw but 12% pad (672) would overflow -> cap to 660
        u3 = _hook_union([0, 0, 300, 300], [300, 0, 300, 300],
                         660, 1920, 880)
        assert u3[2] == 660
        # normal case (room to spare) still pads
        u2 = _hook_union([100, 100, 100, 100], [200, 100, 100, 100],
                         660, 1920, 880)
        assert u2[2] > 200  # padded beyond the raw 200px span

    def test_overflow_span_gets_no_pad(self):
        u = _hook_union([0, 0, 300, 300], [1000, 0, 300, 300],
                        660, 1920, 880)
        assert u[2] == 1300  # raw span, no padding


class TestSubjectCellRect:
    """A split cell must stay ONE-person-sized even for a big subject box —
    ground-truthed 31-jul-2026: a 990x880 cell (>half the 1920-wide source)
    crammed a whole standing group into one cell. The width/height caps are
    a hard backstop on top of SPLIT_WIDTH_MULT/SPLIT_HEIGHT_FLOOR."""

    def test_normal_box_is_not_capped(self):
        # A modest face box well under the cap keeps the width-mult sizing.
        rect = _subject_cell_rect([900, 400, 100, 100], 1920, 1080)
        assert rect[2] <= 1920 * SPLIT_RECT_MAX_WIDTH_FRACTION
        assert rect[3] <= 1080 * SPLIT_RECT_MAX_HEIGHT_FRACTION

    def test_large_body_box_is_capped_to_one_person_size(self):
        # A big YOLO body box (someone standing close to camera): the naive
        # width-mult/height-floor sizing alone would blow well past a
        # one-person crop; the caps must clamp it.
        rect = _subject_cell_rect([200, 100, 900, 900], 1920, 1080)
        assert rect[2] <= int(1920 * SPLIT_RECT_MAX_WIDTH_FRACTION) + 2
        assert rect[3] <= int(1080 * SPLIT_RECT_MAX_HEIGHT_FRACTION) + 2

    def test_rect_stays_within_frame(self):
        rect = _subject_cell_rect([0, 0, 900, 900], 1920, 1080)
        x, y, w, h = rect
        assert x >= 0 and y >= 0
        assert x + w <= 1920 and y + h <= 1080

    def test_rect_is_head_anchored(self):
        # A face box at [900, 400, 100, 100]: the cell center must sit in
        # the HEAD region (box top + SPLIT_HEAD_FRACTION of box height),
        # not a torso/upper-body crop, and the width must stay one-person
        # (≤ SPLIT_RECT_MAX_WIDTH_FRACTION of the source).
        rect = _subject_cell_rect([900, 400, 100, 100], 1920, 1080)
        x, y, w, h = rect
        cx = x + w / 2.0
        cy = y + h / 2.0
        assert abs(cx - 950) <= 3  # horizontal center on the face
        expect_cy = 400 + 100 * SPLIT_HEAD_FRACTION
        assert abs(cy - expect_cy) <= 3
        assert w <= 1920 * SPLIT_RECT_MAX_WIDTH_FRACTION


class TestResolveSpeakerBinding:
    """Identity via face-position anchors: id fast-path when the anchor's
    id is still among today's candidates, else nearest-by-position within
    SPEAKER_ANCHOR_TOLERANCE, else the live speaker_to_id fallback."""

    def _cand(self, cid, x, w=100):
        return {"id": cid, "box": [x, 100, w, 100], "score": w * 100}

    def test_id_fast_path_when_anchor_id_present(self):
        candidates = [self._cand(1, 100), self._cand(2, 800)]
        anchors = {0: {"H": {"id": 2, "cx": 850}}}
        assert _resolve_speaker_binding(candidates, "H", 0, anchors, {}, 1920) == 2

    def test_stale_id_falls_back_to_nearest_within_tolerance(self):
        # Anchor id 99 no longer exists this frame (reassigned in a later
        # scene) — position (cx=850) still matches candidate 2 at x=800
        # (center 850) within 0.15*1920=288.
        candidates = [self._cand(1, 100), self._cand(2, 800)]
        anchors = {4: {"H": {"id": 99, "cx": 850}}}
        assert _resolve_speaker_binding(candidates, "H", 4, anchors, {}, 1920) == 2

    def test_position_too_far_falls_back_to_live_binding(self):
        candidates = [self._cand(1, 100), self._cand(2, 800)]
        anchors = {4: {"H": {"id": 99, "cx": 1850}}}  # nobody is anywhere near this
        assert _resolve_speaker_binding(
            candidates, "H", 4, anchors, {"H": 1}, 1920) == 1

    def test_no_anchor_and_no_live_binding_returns_none(self):
        candidates = [self._cand(1, 100)]
        assert _resolve_speaker_binding(candidates, "H", 4, {}, {}, 1920) is None

    def test_no_label_returns_none(self):
        candidates = [self._cand(1, 100)]
        assert _resolve_speaker_binding(candidates, None, 0, {}, {}, 1920) is None


class TestDirectiveSplitCorroboration:
    """A directive-driven split (causing_reaction/reacting) must be backed
    by real motion evidence and spatially distinct cells — a live Gemini
    run hallucinated a causing_reaction beat at 3.5-6.6s with nobody
    actually reacting, which this gate exists to kill."""

    def test_no_split_without_reactor_motion(self):
        assert not _directive_split_corroborated(
            100, [], 24, (0, 0, 200, 200), (800, 0, 200, 200), 1920)

    def test_split_with_nearby_motion_and_distinct_cells(self):
        assert _directive_split_corroborated(
            100, [95], 24, (0, 0, 200, 200), (800, 0, 200, 200), 1920)

    def test_motion_outside_corroboration_window_does_not_count(self):
        assert not _directive_split_corroborated(
            100, [50], 24, (0, 0, 200, 200), (800, 0, 200, 200), 1920)

    def test_same_subject_in_both_cells_is_rejected(self):
        # centers 100 and 220 are only 120px apart on a 1920-wide frame —
        # well under the 0.15*1920=288 distinctness floor.
        assert not _directive_split_corroborated(
            100, [95], 24, (0, 0, 200, 200), (120, 0, 200, 200), 1920)

    def test_missing_cell_rect_is_rejected(self):
        assert not _directive_split_corroborated(
            100, [95], 24, None, (800, 0, 200, 200), 1920)


# --- Detection cache-and-replay (plan item 9) ------------------------------
# _learn_speaker_anchors caches raw pre-boost candidates; the camera pass
# replays them instead of decoding + detecting the clip a second time. main
# is stubbed out wholesale (the host test env doesn't carry torch/cv2) — the
# detectors and tracker are the only pieces the two passes touch.
class TestDetectionCacheReplay:
    import types as _types

    def _make_fake_main(self):
        import types

        class FakeTracker:
            def __init__(self, cooldown_frames=0, unlock_frames=set(),
                         identity=None):
                self.active_speaker_id = None
                self.last_switch_frame = -10 ** 9
                self.identity = identity
                self.identity_frames = 0

            def set_identity_frame(self, frame):
                self.identity_frames += 1

            def assign_ids(self, candidates, frame_number, orig_w):
                # Stable per-sample ids by left-to-right position, matching
                # the anchor pass and the camera pass identically.
                for i, c in enumerate(sorted(candidates, key=lambda c: c["box"][0])):
                    c["id"] = i + 1

            def get_target_id(self, candidates, frame_number, orig_w, continuity_bias=True):
                if not candidates:
                    return None, None
                best = max(candidates, key=lambda c: c["score"])
                self.active_speaker_id = best["id"]
                return best["box"], best["id"]

        calls = {"face": 0, "yolo": 0}

        def detect_face_candidates(frame):
            calls["face"] += 1
            return [{"box": [100, 100, 80, 80]}]  # left person (small coords)

        def detect_person_candidates_yolo(frame):
            calls["yolo"] += 1
            return [{"box": [300, 100, 90, 200]}]  # right person

        fake = types.ModuleType("main")
        fake.DETECT_STRIDE = 2
        fake.YOLO_FALLBACK_STRIDE = 4
        fake.SpeakerTracker = FakeTracker
        fake.detect_face_candidates = detect_face_candidates
        fake.detect_person_candidates_yolo = detect_person_candidates_yolo
        fake.calls = calls
        return fake

    def _frame_bytes_for(self, orig_w, orig_h):
        import reframe_v2 as rv
        small_w = min(rv.ANALYSIS_MAX_WIDTH, orig_w)
        if small_w % 2:
            small_w -= 1
        small_h = max(int(orig_h * small_w / orig_w), 2)
        if small_h % 2:
            small_h += 1
        return small_w * small_h * 3, small_w, small_h

    def _fake_proc(self, total_frames, frame_bytes):
        import types

        class FakePipe:
            def __init__(self, payload):
                self._payload = payload
                self._off = 0

            def read(self, n):
                if self._off >= len(self._payload):
                    return b""
                chunk = self._payload[self._off:self._off + n]
                self._off += len(chunk)
                return chunk

            def close(self):
                pass

        class FakeProc:
            def __init__(self):
                self.stdout = FakePipe(b"\x00" * (frame_bytes * total_frames))
                self.returncode = 0

            def wait(self):
                return 0

        return FakeProc()

    def test_anchor_pass_caches_every_sample_frame_pre_boost(self, monkeypatch, tmp_path):
        import sys
        import reframe_v2 as rv
        fake_main = self._make_fake_main()
        monkeypatch.setitem(sys.modules, "main", fake_main)
        monkeypatch.setattr(rv, "SPLIT_SCREEN", False)

        frame_bytes, small_w, small_h = self._frame_bytes_for(1920, 1080)
        total = 12
        monkeypatch.setattr(
            rv.subprocess, "Popen",
            lambda *a, **kw: self._fake_proc(total, frame_bytes))

        anchors, detections = rv._learn_speaker_anchors(
            "fake.mp4", [(0, total)], 30, 1920, 1080,
            [(0, total, "host")])

        # Anchors learned for the active speaker (the bigger/right person is
        # the natural-selection target every sample).
        assert 0 in anchors
        host = anchors[0].get("host")
        assert host is not None and host["id"] == 2
        # Boxes are scaled to source pixels (scale=1920/640=3): cx = 3*(300+45).
        assert host["cx"] == 1035.0

        # Every DETECT_STRIDE frame is cached, plus the total-frame key.
        assert set(range(0, total, fake_main.DETECT_STRIDE)) <= set(
            k for k in detections if isinstance(k, int))
        assert detections["_total_frames"] == total
        # Raw pre-boost contract: cached scores equal the scaled box area
        # (nothing boosted, no tracker ids stamped).
        scale = 1920 / small_w
        for fr in (0, 4):
            for cand in detections[fr]:
                x, y, w, h = cand["box"]
                assert w == round(80 * scale) or w == round(90 * scale)
                assert cand["score"] == w * h
                assert "id" not in cand

    def test_camera_pass_makes_zero_additional_detector_calls(self, monkeypatch, tmp_path):
        import sys
        import reframe_v2 as rv
        fake_main = self._make_fake_main()
        monkeypatch.setitem(sys.modules, "main", fake_main)
        monkeypatch.setattr(rv, "SPLIT_SCREEN", False)

        frame_bytes, _, _ = self._frame_bytes_for(1920, 1080)
        total = 12
        monkeypatch.setattr(
            rv.subprocess, "Popen",
            lambda *a, **kw: self._fake_proc(total, frame_bytes))

        class FakeCameraman:
            crop_width = 800
            crop_height = 1400

            def __init__(self):
                self.force_next_update = False
                self._box = (10, 20, 100, 120)

            def update_target(self, box, zoom_target=None):
                pass

            def get_crop_box(self, force_snap=False):
                return self._box

        cameraman = FakeCameraman()
        tracker = fake_main.SpeakerTracker()

        rects, split_info = rv._analyze_trajectory(
            "fake.mp4", [(0, total)], 30, 1920, 1080,
            cameraman, tracker, speaker_turns=[(0, total, "host")])

        # The camera pass replayed the cache: the ONLY detector calls are the
        # anchor pass's own (6 face samples at stride 2, 3 YOLO at stride 4).
        assert fake_main.calls == {"face": 6, "yolo": 3}
        # One rect per frame, all well-formed.
        assert len(rects) == total
        for x1, y1, w, h in rects:
            assert w > 0 and h > 0
        assert split_info is None

    def test_no_speaker_turns_falls_back_to_own_detection(self, monkeypatch, tmp_path):
        import sys
        import reframe_v2 as rv
        fake_main = self._make_fake_main()
        monkeypatch.setitem(sys.modules, "main", fake_main)
        monkeypatch.setattr(rv, "SPLIT_SCREEN", False)

        frame_bytes, _, _ = self._frame_bytes_for(1920, 1080)
        total = 12
        monkeypatch.setattr(
            rv.subprocess, "Popen",
            lambda *a, **kw: self._fake_proc(total, frame_bytes))

        class FakeCameraman:
            crop_width = 800
            crop_height = 1400

            def __init__(self):
                self.force_next_update = False
                self._box = (10, 20, 100, 120)

            def update_target(self, box, zoom_target=None):
                pass

            def get_crop_box(self, force_snap=False):
                return self._box

        calls_before = dict(fake_main.calls)
        rects, _ = rv._analyze_trajectory(
            "fake.mp4", [(0, total)], 30, 1920, 1080,
            FakeCameraman(), fake_main.SpeakerTracker(), speaker_turns=None)
        assert len(rects) == total
        # No anchor pass ran, so the camera pass did its own detection.
        assert fake_main.calls["face"] == calls_before["face"] + total // fake_main.DETECT_STRIDE


# --- Split-screen reaction beat (round-5 spec 1.1) -------------------------
# The old inline check compared id(candidate DICT) against sets of
# id(candidate BOX) — never equal, so reaction beats (and both split paths)
# were permanently dead. These tests pin the fixed box-identity comparison.
class TestReactionBeat:
    def _cand(self, box, pid=1):
        return {"box": box, "id": pid, "score": 100.0, "raw_score": 100.0}

    def test_box_id_in_boosted_triggers_beat(self):
        cand = self._cand([10, 20, 100, 100])
        boosted = {id(cand["box"])}
        assert _is_reaction_beat([cand], boosted, set(), None, None, None) is True

    def test_dict_id_collision_does_not_trigger(self):
        # The old bug: id(c) (the dict) could equal an unrelated box id in the
        # set and STILL not match (it never did) — with the fix only the box's
        # own id counts. Build a set containing the DICT's id but not the
        # box's id: no beat.
        cand = self._cand([10, 20, 100, 100])
        boosted = {id(cand)}  # dict id, not box id
        assert _is_reaction_beat([cand], boosted, set(), None, None, None) is False

    def test_speech_boosted_candidate_is_excluded(self):
        cand = self._cand([10, 20, 100, 100])
        boosted = {id(cand["box"])}
        speech = {id(cand["box"])}
        assert _is_reaction_beat([cand], boosted, speech, None, None, None) is False

    def test_current_target_is_excluded(self):
        cand = self._cand([10, 20, 100, 100])
        boosted = {id(cand["box"])}
        assert _is_reaction_beat([cand], boosted, set(), cand["box"], None, None) is False

    def test_effective_primary_is_excluded(self):
        cand = self._cand([10, 20, 100, 100], pid=7)
        boosted = {id(cand["box"])}
        assert _is_reaction_beat([cand], boosted, set(), None, 7, None) is False

    def test_directed_box_suppresses_reaction_beat(self):
        cand = self._cand([10, 20, 100, 100])
        boosted = {id(cand["box"])}
        assert _is_reaction_beat([cand], boosted, set(), None, None, cand["box"]) is False

    def test_unboosted_candidate_never_triggers(self):
        cand = self._cand([10, 20, 100, 100])
        assert _is_reaction_beat([cand], set(), set(), None, None, None) is False


# --- Split-vs-override decision + reaction two-shot (round-5 spec 2.1/2.2) ---
class TestDirectiveSplitFallback:
    def _cand(self, box, pid):
        return {"box": box, "id": pid, "score": 50.0, "raw_score": 50.0}

    def test_second_subject_available_means_split(self):
        directed = self._cand([100, 100, 120, 160], 1)
        other = self._cand([400, 100, 110, 150], 2)
        assert _has_second_subject([directed, other], directed["box"]) is True

    def test_single_subject_falls_back_to_override(self):
        directed = self._cand([100, 100, 120, 160], 1)
        assert _has_second_subject([directed], directed["box"]) is False

    def test_directed_box_not_in_candidates_is_false(self):
        other = self._cand([400, 100, 110, 150], 2)
        assert _has_second_subject([other], [999, 999, 10, 10]) is False

    def test_duplicate_id_does_not_count_as_second_subject(self):
        directed = self._cand([100, 100, 120, 160], 1)
        ghost = self._cand([400, 100, 110, 150], 1)  # same tracker id
        assert _has_second_subject([directed, ghost], directed["box"]) is False


class TestReactionTwoShot:
    def test_close_pair_returns_union(self):
        union = _reaction_two_shot_boxes(
            [100, 100, 120, 160], [260, 100, 110, 150], 400, 1080, 1920)
        assert union is not None
        assert union[0] <= 100 and (union[0] + union[2]) >= 370

    def test_far_pair_returns_none(self):
        union = _reaction_two_shot_boxes(
            [100, 100, 120, 160], [900, 100, 110, 150], 400, 1080, 1920)
        assert union is None

    def test_missing_boxes_returns_none(self):
        assert _reaction_two_shot_boxes(None, [0, 0, 10, 10], 400, 1080, 1920) is None
        assert _reaction_two_shot_boxes([0, 0, 10, 10], None, 400, 1080, 1920) is None


class TestSplitCellHeadAnchor:
    """The final crop of a split cell must be anchored on the subject.

    Regression guard for heads being cut off in split screen. The cell window
    is clamped to stay inside the source frame, and in a 16:9 frame heads sit
    high, so the clamp pins the window at y=0 and the subject ends up ABOVE the
    window's centre. The old code then cropped back to the cell size from the
    CENTRE, removing exactly the part containing the head.
    """

    ORIG_W, ORIG_H = 1920, 1080
    OUT_W, HALF_H = 1080, 720

    def _head_fraction(self, head_top, anchored):
        """Where the head lands vertically in the rendered cell (0=top, 1=bottom)."""
        cell_w, cell_h = reframe_v2.split_cell_size(self.ORIG_W, self.ORIG_H)
        box = (900, head_top, 150, 190)
        rect = reframe_v2._subject_cell_rect(box, self.ORIG_W, self.ORIG_H)
        wx, wy = reframe_v2.cell_initial_xy(rect, cell_w, cell_h,
                                            self.ORIG_W, self.ORIG_H)
        sw, sh = reframe_v2.cell_scale_target(rect, cell_w, cell_h,
                                              self.OUT_W, self.HALF_H)
        hy = rect[1] + rect[3] / 2.0
        sy = (hy - wy) * sh / cell_h
        if anchored:
            _x, y = reframe_v2.cell_frame_xy(rect, (wx, wy), cell_w, cell_h,
                                             (sw, sh), self.OUT_W, self.HALF_H)
        else:
            y = (sh - self.HALF_H) / 2.0   # the old centred crop
        return (sy - y) / self.HALF_H

    def test_centred_crop_would_cut_the_head_off(self):
        # Documents the bug: a subject high in frame lands ABOVE the cell.
        assert self._head_fraction(60, anchored=False) < 0

    def test_anchored_crop_keeps_the_head_in_frame(self):
        for head_top in (60, 180, 420):
            frac = self._head_fraction(head_top, anchored=True)
            assert 0.0 < frac < 1.0, f"head at {frac} for head_top={head_top}"

    def test_head_sits_at_the_configured_height(self):
        for head_top in (60, 180, 420):
            frac = self._head_fraction(head_top, anchored=True)
            assert abs(frac - reframe_v2.SPLIT_CELL_HEAD_Y) < 0.02

    def test_final_crop_never_leaves_the_magnified_image(self):
        cell_w, cell_h = reframe_v2.split_cell_size(self.ORIG_W, self.ORIG_H)
        for head_top in (0, 60, 300, 800, 1000):
            box = (900, head_top, 150, 190)
            rect = reframe_v2._subject_cell_rect(box, self.ORIG_W, self.ORIG_H)
            wxy = reframe_v2.cell_initial_xy(rect, cell_w, cell_h,
                                             self.ORIG_W, self.ORIG_H)
            sw, sh = reframe_v2.cell_scale_target(rect, cell_w, cell_h,
                                                  self.OUT_W, self.HALF_H)
            x, y = reframe_v2.cell_frame_xy(rect, wxy, cell_w, cell_h,
                                            (sw, sh), self.OUT_W, self.HALF_H)
            assert 0 <= x <= max(0, sw - self.OUT_W)
            assert 0 <= y <= max(0, sh - self.HALF_H)


class TestSplitCellAspectFollowsOutput:
    """Split cells must be shaped from the ACTUAL output, not a constant.

    SPLIT_CELL_ASPECT was hardcoded 9/8, correct only for a 9:16 canvas. On the
    3:4 output in real use, half the canvas is 1080x720 (1.5), so a 9:8 cell
    scaled to 1080x960 and was then centre-cropped to 1080x720 — discarding 25%
    of each subject's height. Split screen showed a middle band of each person,
    defeating its own purpose.
    """

    @pytest.mark.parametrize("out_w,out_h", [(1080, 1920), (1080, 1440), (1080, 1080)])
    def test_cell_fills_half_the_canvas_with_no_crop(self, out_w, out_h):
        aspect = reframe_v2.split_cell_aspect(out_w, out_h)
        cw, ch = reframe_v2.split_cell_size(1920, 1080, aspect=aspect)
        half_h = out_h // 2
        # Scaling the cell to the output width must land within a pixel of the
        # half-canvas height — any shortfall is content the final crop removes.
        scaled_h = out_w * ch / cw
        assert abs(scaled_h - half_h) <= 2, (
            f"{out_w}x{out_h}: cell {cw}x{ch} scales to {scaled_h:.0f}px, "
            f"needs {half_h}px")

    def test_three_four_is_not_nine_eight(self):
        """Guards the specific regression: 3:4 must not reuse the 9:16 shape."""
        assert reframe_v2.split_cell_aspect(1080, 1440) == pytest.approx(1.5)
        assert reframe_v2.split_cell_aspect(1080, 1920) == pytest.approx(1.125)

    def test_subject_rect_uses_the_passed_aspect(self):
        box = (900, 200, 160, 200)
        r34 = reframe_v2._subject_cell_rect(box, 1920, 1080, aspect=1.5)
        r916 = reframe_v2._subject_cell_rect(box, 1920, 1080, aspect=1.125)
        assert r34[2] / r34[3] == pytest.approx(1.5, abs=0.05)
        assert r916[2] / r916[3] == pytest.approx(1.125, abs=0.05)

    def test_cell_never_exceeds_the_source(self):
        for aspect in (1.125, 1.5, 2.0):
            cw, ch = reframe_v2.split_cell_size(640, 360, aspect=aspect)
            assert cw <= 640 and ch <= 360
            assert cw % 2 == 0 and ch % 2 == 0


def test_asd_match_requires_a_decisive_nearest_face():
    """In a lineup the ASD box sits between several faces and detection
    jitter flips the nearest one from sample to sample. An ambiguous match
    must report nothing rather than become a cut."""
    asd_box = (1000, 200, 200, 200)
    ambiguous = [{"id": 1, "box": [900, 200, 200, 200], "score": 1.0},
                 {"id": 2, "box": [1100, 200, 200, 200], "score": 1.0}]
    assert reframe_v2._apply_asd_speaker_boost(ambiguous, asd_box, 1920) is None
    assert ambiguous[0]["score"] == 1.0, "no boost on an ambiguous match"


def test_asd_match_is_accepted_when_one_face_is_clearly_nearest():
    asd_box = (1000, 200, 200, 200)
    clear = [{"id": 1, "box": [1005, 205, 200, 200], "score": 1.0},
             {"id": 2, "box": [1700, 600, 200, 200], "score": 1.0}]
    assert reframe_v2._apply_asd_speaker_boost(clear, asd_box, 1920) == 1


def test_jcut_pre_roll_proposes_the_next_long_speaker_early():
    """Owner spec: hard cut to Speaker B 0.5s before their audio begins."""
    turns = [(0, 100, "A"), (100, 260, "B"), (260, 400, "A")]
    # 12 frames before B's turn (0.5s @ 24fps) -> B, jcut.
    label, jcut = reframe_v2._effective_speaker_label(turns, 88, 24)
    assert (label, jcut) == ("B", True)
    # 2s before B's turn -> still A, no jcut.
    label, jcut = reframe_v2._effective_speaker_label(turns, 52, 24)
    assert (label, jcut) == ("A", False)


def test_short_utterance_is_never_proposed_early():
    """A 1s 'yeah/right' agreement must not drag the camera away early."""
    turns = [(0, 120, "A"), (120, 144, "B"), (144, 300, "A")]
    label, jcut = reframe_v2._effective_speaker_label(turns, 110, 24)
    assert (label, jcut) == ("A", False), "short turn must not j-cut"
    # Inside the short turn itself the label is still the speaker.
    label, jcut = reframe_v2._effective_speaker_label(turns, 130, 24)
    assert label == "B"


def test_banter_two_shot_fires_when_the_floor_blocks_a_strong_switch():
    """Owner tip 1.3: a strong speaker switch the 1.5s floor blocks means both
    speakers are visible — frame both instead of lagging on the old one."""
    from subject_policy import Evidence, TIER_HOLD
    assert _banter_two_shot_eligible(TIER_HOLD, Evidence(asd_id=2), 1)
    assert _banter_two_shot_eligible(TIER_HOLD, Evidence(diarized_id=2), 1)
    assert _banter_two_shot_eligible(TIER_HOLD, Evidence(jcut_id=2), 1)


def test_banter_two_shot_does_not_fire_when_the_switch_landed():
    from subject_policy import (
        Evidence, TIER_ASD, TIER_HOLD, TIER_REACTION, TIER_DIRECTIVE)
    assert not _banter_two_shot_eligible(TIER_ASD, Evidence(asd_id=2), 1)
    assert not _banter_two_shot_eligible(TIER_REACTION, Evidence(asd_id=2), 1)
    # A weak (directive) proposal is not a speaker identification — the
    # camera must not widen on it.
    assert not _banter_two_shot_eligible(TIER_HOLD, Evidence(directive_id=2), 1)
    assert not _banter_two_shot_eligible(TIER_HOLD, Evidence(asd_id=1), 1)
