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


def test_a_subject_is_centred_wherever_the_frame_allows_it():
    """Owner spec: "if the person can be centered, then center the person."
    Centring is possible for x in [crop_w/2, W - crop_w/2] = [405, 1515] on a
    1920-wide source. Two people facing each other on this format sit at
    roughly 450 and 1500 — both inside that band, and both were being pushed
    17% off centre by the old thirds default."""
    for x in (450, 600, 960, 1200, 1500):
        _, _, pos = cam_at(x)
        assert abs(pos - 0.5) < 0.05, f"subject at {x} landed at {pos:.2f}"


def test_a_subject_outside_the_centrable_band_is_clamped_not_lost():
    """Beyond the band the crop would leave the source, so the subject rides
    toward the crop edge — the "at least at the edge" rule. No choice is
    involved; geometry decides."""
    for x in (200, 1750):
        x1, x2, pos = cam_at(x)
        assert x1 <= x <= x2
        assert abs(pos - 0.5) > 0.1, "should be off centre, but only because clamped"


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


def test_thirds_can_be_switched_on_for_looking_room(monkeypatch):
    """The offset still exists for deliberate looking-room work; it is just
    not the default any more."""
    _, _, centred = cam_at(600)
    monkeypatch.setattr(m, "COMPOSE_THIRDS", True)
    _, _, thirds = cam_at(600)
    assert abs(centred - 0.5) < 0.05
    assert thirds < centred - 0.1, "thirds must actually shift the framing"
