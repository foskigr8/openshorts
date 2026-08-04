"""Long-shot follow: a held shot stays locked; a LONG-held shot tracks."""
import main as m


def cam():
    c = m.SmoothedCameraman(1080, 1440, 1920, 1080, aspect_ratio=0.75, fps=25)
    c.force_next_update = True
    c.update_target([900, 400, 200, 200])
    c.get_crop_box(force_snap=True)
    return c


def test_a_fresh_shot_stays_pixel_locked():
    """Between cuts nothing moves — that is what makes a held shot unable
    to stutter."""
    c = cam()
    before = c.current_center_x
    for _ in range(10):
        c.update_target([905, 400, 200, 200])   # tiny drift
        c.get_crop_box()
    assert c.current_center_x == before


def feed(c, box, n=6):
    """A big move only commits after the jump-confirm gate sees it repeatedly."""
    for _ in range(n):
        c.update_target(box)


def test_a_fresh_shot_still_re_frames_instantly_on_a_big_move():
    c = cam()
    start = c.current_center_x
    feed(c, [1700, 400, 200, 200])              # way outside the safe zone
    c.get_crop_box()
    assert c.current_center_x != start, "the camera must re-frame"
    # A subject near the frame edge cannot be centred without the crop running
    # off the source, so the camera clamps — the owner's "if the person cannot
    # be centered, at least keep them at the edge" rule. What must hold is that
    # they are IN the crop.
    x1, _, x2, _ = c.get_crop_box()
    assert x1 <= 1800 <= x2, "the subject must be inside the crop window"


def test_a_long_held_shot_follows_smoothly_instead_of_teleporting():
    """Owner spec: 'instead of constant jittering that distracts the eyes, we
    can use tracking if the camera doesn't change for long'."""
    c = cam()
    for _ in range(c.long_shot_follow_frames + 5):   # hold the shot
        c.get_crop_box()
    start = c.current_center_x
    feed(c, [1700, 400, 200, 200])
    c.get_crop_box()
    moved = abs(c.current_center_x - start)
    assert moved > 0, "a long-held shot must follow a departing subject"
    assert moved < abs(c.target_center_x - start), "it must EASE, not teleport"


def test_the_follow_eventually_arrives():
    c = cam()
    for _ in range(c.long_shot_follow_frames + 5):
        c.get_crop_box()
    feed(c, [1700, 400, 200, 200])
    for _ in range(400):
        c.get_crop_box()
    x1, _, x2, _ = c.get_crop_box()
    assert x1 <= 1800 <= x2, "the follow must end with the subject framed"


def test_a_long_held_shot_stays_pixel_locked_while_the_subject_is_inside():
    """Owner spec, 4-aug-2026 (§2h(2a)): a held shot is visually still. The
    follow may only correct when the subject approaches the crop edge — a
    subject who is merely off-centre but comfortably inside the crop must not
    pull the camera at all (the old continuous correction after 3s is the
    drift the review called 'robotic')."""
    c = cam()
    for _ in range(c.long_shot_follow_frames + 5):   # hold the shot
        c.get_crop_box()
    start = c.current_center_x
    # Box centre 250px right of the crop centre (drifted past the 25% safe
    # zone) but still well inside the 810px-wide crop with margin: crop is
    # [595, 1405] at centre 1000, edge margin is 81px, so a centre at 1250
    # is neither near the edge nor outside it.
    feed(c, [start - 100 + 250, 400, 200, 200])
    c.get_crop_box()
    assert c.current_center_x == start, \
        "an off-centre-but-visible subject must NOT pull a long-held shot"


def test_a_long_held_shot_follows_when_the_subject_reaches_the_edge():
    """The flip side of the edge gate: once the subject actually approaches
    the crop edge, the long-held shot eases them back inside instead of
    snapping."""
    c = cam()
    for _ in range(c.long_shot_follow_frames + 5):
        c.get_crop_box()
    start = c.current_center_x
    # Near the crop edge (crop is [595, 1405] at centre 1000; centre 1390 is
    # within the 81px edge margin).
    feed(c, [start - 100 + 390, 400, 200, 200])
    c.get_crop_box()
    moved = abs(c.current_center_x - start)
    assert moved > 0, "an edge-approaching subject must be followed"
    assert moved < abs(c.target_center_x - start), "it must EASE, not teleport"


def test_y_head_anchor_does_not_re_frame_a_full_height_crop():
    """Regression, 4-aug-2026 (§2h(2a)): at zoom 1.0 the crop is full-height
    and y is clamped to the middle, so the head-anchor y-offset (~350px above
    the clamped centre) is permanent and uncorrectable. Counting it made
    `drifted` true on EVERY frame, which re-snapped every fresh shot on the
    first small target change — the mid-shot jolts measured on Pop The
    Balloon. y must not count as drift when it cannot move the crop."""
    c = cam()
    start = c.current_center_x
    # A target move beyond the x dead zone but INSIDE the x safe zone must
    # not re-frame a fresh shot: the subject is still fully framed.
    c.force_next_update = True
    c.update_target([1000, 100, 200, 200])  # centre 1100, head anchor y=132
    c.get_crop_box()
    assert c.current_center_x == start, \
        "a full-height crop must not re-frame on the uncorrectable y offset"
