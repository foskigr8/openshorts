"""Horizontal composition: centre when you can, thirds when you can't.

Owner spec, 4-aug-2026: "The composition of the person that is talking is
center. If the person cannot be centered, that's when you can use rule of
thirds and make sure the person is at least at the edge."
"""
import main as m


def cam_at(x, w=1920, h=1080):
    c = m.SmoothedCameraman(1080, 1440, w, h, aspect_ratio=0.75, fps=25)
    c.force_next_update = True
    c.update_target([x - 100, 400, 200, 200])      # face centred on x
    c.get_crop_box(force_snap=True)
    x1, _, x2, _ = c.get_crop_box()
    return x1, x2, (x - x1) / float(x2 - x1)       # subject's position in crop


def test_a_centred_subject_is_centred():
    _, _, pos = cam_at(960)
    assert abs(pos - 0.5) < 0.05, f"subject at {pos:.2f}, expected centre"


def test_a_left_subject_sits_on_the_left_third_with_looking_room():
    """Someone on the left of the room is facing right — the space belongs
    in front of them, not behind."""
    _, _, pos = cam_at(300)
    assert pos < 0.45, f"subject at {pos:.2f}, expected left of centre"


def test_a_right_subject_sits_on_the_right_third():
    _, _, pos = cam_at(1620)
    assert pos > 0.55, f"subject at {pos:.2f}, expected right of centre"


def test_a_subject_at_the_extreme_edge_is_still_fully_inside_the_crop():
    """Containment beats composition — the 'at least at the edge' rule."""
    # A subject 60px from the source edge cannot be given inner margin
    # without the crop leaving the frame, so containment is the only
    # guarantee that holds at the extremes.
    for x in (60, 120, 1800, 1870):
        x1, x2, _ = cam_at(x)
        assert x1 <= x <= x2, f"subject at {x} fell outside the crop"
    # Where the frame allows it, the subject keeps real margin.
    for x in (500, 1400):
        _, _, pos = cam_at(x)
        assert 0.10 <= pos <= 0.90, f"subject stranded at {pos:.2f}"


def test_thirds_can_be_switched_off(monkeypatch):
    # x=600 is far enough from the edge that centring is achievable, so the
    # flag makes a visible difference (nearer the edge the clamp decides).
    _, _, thirds = cam_at(600)
    monkeypatch.setattr(m, "COMPOSE_THIRDS", False)
    _, _, centred = cam_at(600)
    assert abs(centred - 0.5) < 0.05
    assert thirds < centred - 0.1, "thirds must actually shift the framing"
