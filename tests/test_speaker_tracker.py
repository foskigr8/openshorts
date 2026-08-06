"""Speaker switching hysteresis.

Regression cover for the camera swinging between subjects: the switch cooldown
used to fall through and switch anyway whenever the active speaker happened to
be missing from the current frame's candidates — a blink or one motion-blurred
frame was enough, which is exactly when the cooldown is needed. Measured on a
12s clip (25-jul-2026), 3 of 7 target switches jumped the cooldown this way.
"""
import pytest

main = pytest.importorskip("main")  # needs cv2/mediapipe, absent in minimal CI

WIDTH = 1280
COOLDOWN = 30


def _face(center_x, size=120, score=None):
    """A candidate box centred on ``center_x``."""
    return {"box": [center_x - size / 2, 100, size, size],
            "score": score if score is not None else size * size}


def _tracker():
    return main.SpeakerTracker(cooldown_frames=COOLDOWN)


def _lock_onto(tracker, x, frame=0):
    """Drive the tracker until it is locked on the subject at ``x``."""
    for f in range(frame, frame + 5):
        tracker.get_target([_face(x)], f, WIDTH)
    return tracker.active_speaker_id


class TestSwitchCooldown:
    def test_locks_onto_a_lone_speaker(self):
        t = _tracker()
        assert _lock_onto(t, 300) is not None

    def test_holds_when_the_active_speaker_blinks_out(self):
        # THE BUG: speaker A vanishes for one frame while B is on screen.
        # Inside the cooldown the tracker must hold, not hand the camera to B.
        t = _tracker()
        a = _lock_onto(t, 300)
        box = t.get_target([_face(1000)], 6, WIDTH)   # only B visible
        assert box is None, "must hold instead of switching mid-cooldown"
        assert t.active_speaker_id == a, "the active speaker must not change"

    def test_holds_on_the_active_speaker_when_both_are_visible(self):
        t = _tracker()
        a = _lock_onto(t, 300)
        box = t.get_target([_face(300), _face(1000)], 6, WIDTH)
        assert box is not None
        assert t.active_speaker_id == a

    def test_switches_once_the_cooldown_expires(self):
        # The hold is bounded: someone who really left the shot is released.
        t = _tracker()
        a = _lock_onto(t, 300)
        t.get_target([_face(1000)], 6, WIDTH)          # held
        assert t.active_speaker_id == a
        for f in range(COOLDOWN + 10, COOLDOWN + 20):  # past the window
            t.get_target([_face(1000)], f, WIDTH)
        assert t.active_speaker_id != a, "cooldown must not hold forever"

    def test_no_candidates_holds_rather_than_recentres(self):
        t = _tracker()
        _lock_onto(t, 300)
        assert t.get_target([], 6, WIDTH) is None

    def test_repeated_dropouts_do_not_accumulate_switches(self):
        # A flickering detection used to switch on every dropout. The lock is
        # taken at frame 0, so the cooldown window runs up to frame COOLDOWN.
        t = _tracker()
        a = _lock_onto(t, 300)
        for f in range(6, COOLDOWN):
            t.get_target([_face(1000)] if f % 2 else [_face(300), _face(1000)],
                         f, WIDTH)
        assert t.active_speaker_id == a


class TestJumpConfirmation:
    """A lone huge target jump is a detector error, not a person moving."""

    def _cam(self, crop=600, video_w=1920):
        cam = main.SmoothedCameraman(crop, 1080, video_w, 1080,
                                     aspect_ratio=9 / 16)
        cam.target_center_x = 500.0
        return cam

    def test_mid_band_moves_are_ema_damped(self):
        # 6-aug-2026 (TARGET_EMA): detector box-centre wobble in the band
        # between the dead zone and the safe zone used to commit instantly and
        # the chase followed the noise. A light EMA now damps it — the target
        # still moves (50% of the way), but not to the raw noisy centre.
        cam = self._cam()
        cam.update_target([540, 0, 20, 20])          # centre 550, well inside
        assert cam.target_center_x == 525.0

    def test_a_single_big_jump_is_ignored(self):
        cam = self._cam()
        before = cam.target_center_x
        cam.update_target([1400, 0, 40, 40])         # centre 1420, far away
        assert cam.target_center_x == before, "one outlier must not move the frame"

    def test_a_sustained_big_jump_is_followed(self):
        cam = self._cam()
        for _ in range(cam.jump_confirm_frames):
            cam.update_target([1400, 0, 40, 40])
        assert cam.target_center_x == 1420

    def test_contradictory_outliers_do_not_confirm_each_other(self):
        # Detector flapping between two wrong places must not add up to a move.
        cam = self._cam()
        before = cam.target_center_x
        for _ in range(cam.jump_confirm_frames * 2):
            cam.update_target([1400, 0, 40, 40])     # far right
            cam.update_target([20, 0, 40, 40])       # far left
        assert cam.target_center_x == before

    def test_a_confirmed_move_resets_the_counter(self):
        cam = self._cam()
        for _ in range(cam.jump_confirm_frames):
            cam.update_target([1400, 0, 40, 40])
        assert cam.target_center_x == 1420
        cam.update_target([1410, 0, 20, 20])         # small follow-up
        assert cam.target_center_x == 1420

    def test_no_detection_leaves_the_target_alone(self):
        cam = self._cam()
        cam.update_target(None)
        assert cam.target_center_x == 500


class TestSpeakerChangeFrames:
    """speaker_change_frames() — AssemblyAI diarization -> clip-relative
    unlock-frame windows for SpeakerTracker."""

    def test_empty_without_speaker_labels(self):
        transcript = {"segments": [{"start": 0, "end": 5, "text": "hi"}]}  # no 'speaker' key
        assert main.speaker_change_frames(transcript, 0.0, 10.0, fps=30.0) == set()

    def test_empty_transcript(self):
        assert main.speaker_change_frames(None, 0.0, 10.0, fps=30.0) == set()
        assert main.speaker_change_frames({}, 0.0, 10.0, fps=30.0) == set()

    def test_no_changes_when_single_speaker_throughout(self):
        transcript = {"segments": [
            {"start": 0, "end": 3, "speaker": "A"},
            {"start": 3, "end": 6, "speaker": "A"},
        ]}
        assert main.speaker_change_frames(transcript, 0.0, 10.0, fps=30.0) == set()

    def test_detects_a_change_and_windows_around_it(self):
        transcript = {"segments": [
            {"start": 0, "end": 3, "speaker": "A"},
            {"start": 3, "end": 6, "speaker": "B"},  # change at clip-relative t=3s
        ]}
        frames = main.speaker_change_frames(transcript, 0.0, 10.0, fps=10.0, unlock_window_s=0.5)
        # change at 3s * 10fps = frame 30, +/- 5 frames (0.5s * 10fps) window
        assert frames == set(range(25, 36))

    def test_offsets_by_clip_start(self):
        transcript = {"segments": [
            {"start": 100.0, "end": 103.0, "speaker": "A"},
            {"start": 103.0, "end": 106.0, "speaker": "B"},
        ]}
        # clip starts at absolute 100s, change at absolute 103s -> clip-relative 3s
        frames = main.speaker_change_frames(transcript, 100.0, 110.0, fps=10.0, unlock_window_s=0.5)
        assert frames == set(range(25, 36))

    def test_filters_segments_outside_clip_range(self):
        transcript = {"segments": [
            {"start": 0, "end": 3, "speaker": "A"},       # before clip
            {"start": 100, "end": 103, "speaker": "A"},   # in clip
            {"start": 103, "end": 106, "speaker": "B"},   # in clip -> change
            {"start": 500, "end": 503, "speaker": "C"},   # after clip
        ]}
        frames = main.speaker_change_frames(transcript, 100.0, 110.0, fps=10.0, unlock_window_s=0.5)
        assert frames == set(range(25, 36))

    def test_multiple_changes_produce_multiple_windows(self):
        transcript = {"segments": [
            {"start": 0, "end": 1, "speaker": "A"},
            {"start": 1, "end": 2, "speaker": "B"},  # change at t=1s -> frame 10
            {"start": 2, "end": 3, "speaker": "A"},  # change at t=2s -> frame 20
        ]}
        frames = main.speaker_change_frames(transcript, 0.0, 5.0, fps=10.0, unlock_window_s=0.2)
        assert 10 in frames and 20 in frames
        assert 5 not in frames  # well outside either window


class TestAudioInformedUnlock:
    """SpeakerTracker.unlock_frames: suppresses the sticky bonus only at
    known audio speaker-turn changes, without touching switch_cooldown."""

    def test_stays_on_active_speaker_without_unlock(self):
        t = _tracker()
        a = _lock_onto(t, 300, frame=0)
        # B is slightly bigger (higher raw score) but not 3x bigger — normal
        # sticky hysteresis should keep A active throughout.
        for f in range(5, COOLDOWN + 40):
            t.get_target([_face(300, size=100), _face(1000, size=110)], f, WIDTH)
        assert t.active_speaker_id == a

    def test_switches_at_an_unlock_frame_when_visual_evidence_favors_the_other(self):
        unlock_at = COOLDOWN + 39
        t = main.SpeakerTracker(cooldown_frames=COOLDOWN, unlock_frames={unlock_at})
        a = _lock_onto(t, 300, frame=0)
        for f in range(5, unlock_at):
            t.get_target([_face(300, size=100), _face(1000, size=110)], f, WIDTH)
        assert t.active_speaker_id == a, "must still be sticky before the unlock frame"
        t.get_target([_face(300, size=100), _face(1000, size=110)], unlock_at, WIDTH)
        assert t.active_speaker_id != a, "unlock frame must let real visual evidence win"

    def test_unlock_does_not_bypass_the_switch_cooldown(self):
        # An unlock frame still inside the cooldown window must NOT force a
        # switch — only the artificial sticky bonus is suppressed, the
        # separate anti-jitter cooldown guard stays fully intact.
        t = main.SpeakerTracker(cooldown_frames=COOLDOWN, unlock_frames={6})
        a = _lock_onto(t, 300, frame=0)
        box = t.get_target([_face(1000, size=200)], 6, WIDTH)  # only B visible, well within cooldown
        assert box is None, "cooldown hold must still apply even at an unlock frame"
        assert t.active_speaker_id == a


class TestSpeakerTurnFrameRanges:
    """speaker_turn_frame_ranges() — the audio half of dynamic reaction-
    camera switching: which single speaker (if any) is active at each
    frame, covering the whole clip with no gaps."""

    def test_single_speaker_covers_whole_range(self):
        transcript = {"segments": [{"start": 0, "end": 10, "speaker": "A"}]}
        ranges = main.speaker_turn_frame_ranges(transcript, 0.0, 10.0, fps=10.0, total_frames=100)
        assert ranges == [(0, 100, "A")]

    def test_two_speakers_produce_two_ranges(self):
        transcript = {"segments": [
            {"start": 0, "end": 5, "speaker": "A"},
            {"start": 5, "end": 10, "speaker": "B"},
        ]}
        ranges = main.speaker_turn_frame_ranges(transcript, 0.0, 10.0, fps=10.0, total_frames=100)
        assert ranges == [(0, 50, "A"), (50, 100, "B")]

    def test_gap_between_segments_is_marked_none(self):
        transcript = {"segments": [
            {"start": 0, "end": 3, "speaker": "A"},
            {"start": 7, "end": 10, "speaker": "B"},
        ]}
        ranges = main.speaker_turn_frame_ranges(transcript, 0.0, 10.0, fps=10.0, total_frames=100)
        assert ranges == [(0, 30, "A"), (30, 70, None), (70, 100, "B")]

    def test_leading_and_trailing_silence_marked_none(self):
        transcript = {"segments": [{"start": 3, "end": 7, "speaker": "A"}]}
        ranges = main.speaker_turn_frame_ranges(transcript, 0.0, 10.0, fps=10.0, total_frames=100)
        assert ranges == [(0, 30, None), (30, 70, "A"), (70, 100, None)]

    def test_no_transcript_yields_one_none_range_for_whole_clip(self):
        ranges = main.speaker_turn_frame_ranges(None, 0.0, 10.0, fps=10.0, total_frames=100)
        assert ranges == [(0, 100, None)]

    def test_whisper_transcript_without_speaker_labels_is_all_none(self):
        transcript = {"segments": [{"start": 0, "end": 10, "text": "no speaker key"}]}
        ranges = main.speaker_turn_frame_ranges(transcript, 0.0, 10.0, fps=10.0, total_frames=100)
        assert ranges == [(0, 100, None)]

    def test_offsets_by_clip_start(self):
        transcript = {"segments": [{"start": 100.0, "end": 105.0, "speaker": "A"}]}
        ranges = main.speaker_turn_frame_ranges(transcript, 100.0, 105.0, fps=10.0, total_frames=50)
        assert ranges == [(0, 50, "A")]

    def test_segments_outside_clip_range_are_ignored(self):
        transcript = {"segments": [
            {"start": -100, "end": -50, "speaker": "A"},  # before clip
            {"start": 0, "end": 10, "speaker": "B"},
            {"start": 500, "end": 600, "speaker": "C"},   # after clip
        ]}
        ranges = main.speaker_turn_frame_ranges(transcript, 0.0, 10.0, fps=10.0, total_frames=100)
        assert ranges == [(0, 100, "B")]


class TestSmoothedCameramanEasing:
    """Problem 1: motion must be eased (ramp-up/cruise/ease-in) with NO
    overshoot and monotonic convergence — the old constant-speed + hard
    overshoot-snap read as mechanical."""

    def _cam(self, video_w=1920, video_h=1080, aspect=9 / 16):
        cam = main.SmoothedCameraman(1080, 1920, video_w, video_h,
                                     aspect_ratio=aspect)
        return cam

    def test_converges_to_target_without_overshoot(self):
        cam = self._cam()
        cam.force_next_update = True
        cam.update_target([1400, 400, 40, 40])  # target centre 1420 (in-range)
        target = cam.target_center_x
        prev = cam.current_center_x
        crossed = False
        for _ in range(600):
            x1, _, x2, _ = cam.get_crop_box()
            # Convergence is a property of the eased CAMERA, not of the crop
            # window: thirds placement offsets the window from the subject on
            # purpose, so reading the window centre would measure composition
            # rather than easing.
            center = cam.current_center_x
            assert center <= target + 0.6, "must never overshoot the target"
            assert center >= prev - 0.6, "must move monotonically toward target"
            prev = center
            if abs(center - target) <= 0.6:
                crossed = True
                break
        assert crossed, "camera must actually reach the target"

    def test_small_moves_apply_without_a_frozen_dead_zone(self):
        # The old dead-zone froze the camera inside 25% of the crop width;
        # an eased chase should start compensating immediately (but gently).
        cam = self._cam()
        cam.force_next_update = True
        cam.update_target([cam.video_width / 2 + 10, 400, 40, 40])
        x1, _, x2, _ = cam.get_crop_box()
        assert x1 != cam.video_width / 2 - cam.crop_width / 2

    def test_velocity_ramps_up_not_instant(self, monkeypatch):
        # Eased travel is no longer the default (see CAMERA_STYLE — the genre
        # cuts, it does not pan), but the pan path is still supported and must
        # still ramp rather than starting at full speed.
        monkeypatch.setattr(main, "CAMERA_STYLE", "pan")
        cam = self._cam()
        cam.force_next_update = True
        cam.update_target([1400, 400, 40, 40])
        steps = []
        for _ in range(8):
            before = cam.current_center_x
            cam.get_crop_box()
            steps.append(cam.current_center_x - before)
        # First frame's step is small (acceleration-limited ramp-up), and
        # steps grow for a while before the exponential tail takes over.
        assert steps[0] < 4.0, "must not start at full cruise speed"
        assert any(b > a + 0.01 for a, b in zip(steps, steps[1:])), (
            "velocity should ramp up, not stay constant")

    def test_scene_snap_still_snaps_immediately(self):
        cam = self._cam()
        cam.force_next_update = True
        cam.update_target([1400, 400, 40, 40])
        x1, _, x2, _ = cam.get_crop_box(force_snap=True)
        # The crop centre is no longer the subject: an off-centre subject is
        # placed on a third with looking room (COMPOSE_THIRDS). What must hold
        # is that the snap put the SUBJECT where composition asks, inside the
        # crop with margin.
        crop_w = x2 - x1
        pos = (cam.target_center_x - x1) / crop_w
        assert 0.13 <= pos <= 0.87, f"subject stranded at {pos:.2f} of the crop"

    def test_head_anchor_keeps_body_box_heads_in_frame(self):
        """A YOLO head-and-chest box centred at 0.5 puts the head at the top
        of the crop (measured: 'top of head cropped' on Pop The Balloon span
        2, 4-aug-2026). The vertical anchor must sit at CAMERA_HEAD_ANCHOR
        down the box, not at the box centre."""
        cam = self._cam()
        cam.force_next_update = True
        # YOLO body box: head sits in the top ~16% of this rect.
        cam.update_target([669, 99, 624, 384])
        assert cam.target_center_y == pytest.approx(
            99 + 384 * main.CAMERA_HEAD_ANCHOR, abs=0.01)
        assert cam.target_center_y < 99 + 384 / 2, \
            "body-box centering is exactly what cut heads off"
        # A face box is head-only: the anchor stays near the box's top half.
        cam.force_next_update = True
        cam.update_target([855, 156, 135, 135])
        assert cam.target_center_y == pytest.approx(
            156 + 135 * main.CAMERA_HEAD_ANCHOR, abs=0.01)

    def test_crop_places_the_head_on_the_top_third_line(self):
        """The head anchor must sit CAMERA_HEAD_Y down the FINAL crop, not
        dead centre — eyes on the top-third grid line (owner framing spec,
        4-aug-2026), matching the split-cell path's SPLIT_CELL_HEAD_Y."""
        cam = self._cam()
        cam.force_next_update = True
        # Mid-frame subject: only then does the source frame contain enough
        # room for the crop to place the head at 36% (a subject near the top
        # clamps to y=0, one near the bottom to y=video_height-crop_h — the
        # source simply has no headroom there).
        cam.update_target([960, 440, 40, 40], zoom_target=0.85)
        x1, y1, x2, y2 = cam.get_crop_box(force_snap=True)
        h = y2 - y1
        head = 440 + 40 * main.CAMERA_HEAD_ANCHOR
        assert h > 0
        assert (head - y1) / h == pytest.approx(main.CAMERA_HEAD_Y, abs=0.03)


class TestSmoothedCameramanZoom:
    """Problem 2: an eased zoom state (crop-size scale) + dynamic y so a
    push-in stays framed on the subject's face instead of cropping heads."""

    def _cam(self, video_w=1920, video_h=1080, aspect=3 / 4):
        return main.SmoothedCameraman(1080, 1920, video_w, video_h,
                                      aspect_ratio=aspect)

    def test_zoom_target_is_clamped_and_accepted(self):
        cam = self._cam()
        cam.force_next_update = True
        cam.update_target([100, 100, 40, 40], zoom_target=0.85)
        assert cam.target_zoom == pytest.approx(0.85)
        cam.update_target([100, 100, 40, 40], zoom_target=0.1)
        assert cam.target_zoom >= cam.min_zoom

    def test_zoom_contracts_the_crop_window(self):
        cam = self._cam()
        base_w = cam.crop_width
        cam.force_next_update = True
        cam.update_target([960, 400, 40, 40], zoom_target=0.85)
        # Focal length changes ride along with a cut (see CAMERA_STYLE); the
        # reference edits never zoom continuously inside a held shot.
        x1, y1, x2, y2 = cam.get_crop_box(force_snap=True)
        assert x2 - x1 < base_w, "zoomed-in crop must be narrower than base"

    def test_zoomed_crop_is_centered_on_the_face_y(self):
        cam = self._cam()
        cam.force_next_update = True
        # Face mid-frame; a zoomed crop must follow the HEAD anchor, not the
        # box centre (y=500 box, h=40 -> 500 + 40*0.16 = 506.4). Box-centre
        # framing is what cut heads off on body boxes (4-aug-2026).
        cam.update_target([960, 500, 40, 40], zoom_target=0.85)
        x1, y1, x2, y2 = cam.get_crop_box(force_snap=True)
        assert y1 > 0, "zoomed-in crop must leave the top of the frame"
        assert cam.current_center_y == pytest.approx(
            500 + 40 * main.CAMERA_HEAD_ANCHOR, abs=2)

    def test_full_height_crop_keeps_y_at_zero(self):
        cam = self._cam()
        cam.force_next_update = True
        cam.update_target([960, 100, 40, 40])  # zoom stays 1.0
        x1, y1, x2, y2 = cam.get_crop_box()
        assert y1 == 0 and y2 == cam.video_height

    def test_zoom_never_exceeds_source_bounds(self):
        cam = self._cam()
        cam.force_next_update = True
        cam.update_target([960, 100, 40, 40], zoom_target=0.85)
        for _ in range(400):
            x1, y1, x2, y2 = cam.get_crop_box()
            assert x1 >= 0 and y1 >= 0
            assert x2 <= cam.video_width and y2 <= cam.video_height


class TestKenBurnsAntiStaticDrift:
    """The anti-static sinusoidal drift is OFF by default. At a subtle
    amplitude it advances far less than the whole pixel the crop is cut at,
    so it renders not as gentle motion but as a locked frame that twitches a
    pixel every couple of seconds — it caused the "freezing" complaint it was
    added to cure. A held shot should be genuinely held; the tunable stays so
    it can be dialled back in deliberately."""

    def _cam(self, amplitude=None):
        cam = main.SmoothedCameraman(1080, 1920, 1920, 1080,
                                     aspect_ratio=3 / 4, fps=30.0)
        if amplitude is not None:
            cam._test_amplitude = amplitude
        return cam

    def test_held_crop_is_perfectly_locked_by_default(self):
        cam = self._cam()
        xs = {cam.get_crop_box()[0] for _ in range(400)}
        assert len(xs) == 1, "a held crop must not twitch sub-pixel"

    def test_drift_is_disabled_by_default(self):
        assert main.STATIC_DRIFT_AMPLITUDE == 0.0

    def test_drift_still_applies_when_deliberately_enabled(self, monkeypatch):
        monkeypatch.setattr(main, "STATIC_DRIFT_AMPLITUDE", 0.05)
        cam = self._cam()
        xs = set()
        for _ in range(main.STATIC_DRIFT_FRAMES):
            xs.add(cam.get_crop_box()[0])
        assert len(xs) == 1, "no drift before the static threshold"
        for _ in range(400):
            xs.add(cam.get_crop_box()[0])
        assert len(xs) > 1, "enabled drift must actually move the crop"
        amplitude = int(cam.crop_width * 0.05)
        assert max(xs) - min(xs) <= 2 * amplitude + 2, "drift must stay bounded"

    def test_moving_camera_resets_static_counter(self):
        cam = self._cam()
        for _ in range(60):
            cam.get_crop_box()
        assert cam._static_frames > main.STATIC_DRIFT_FRAMES
        cam.force_next_update = True
        cam.update_target([1400, 100, 40, 40])  # far jump, accepted (forced)
        cam.get_crop_box()
        assert cam._static_frames == 0

    def test_drift_never_escapes_the_source_bounds(self):
        # Park the camera at the extreme right edge; drift must stay in-frame.
        cam = self._cam()
        cam.force_next_update = True
        cam.update_target([1800, 100, 40, 40])
        for _ in range(600):
            cam.get_crop_box()
        for _ in range(400):
            x1, _, x2, _ = cam.get_crop_box()
            assert 0 <= x1 < x2 <= cam.video_width
