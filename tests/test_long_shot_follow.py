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
