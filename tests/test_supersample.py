"""CROP_SUPERSAMPLE / TARGET_EMA / head-containment (6-aug-2026).

The camera emits crop positions on a 1/supersample grid and the renderer
upscales the source by the same factor, so an eased follow can step half a
source pixel instead of stalling below 1px and then jumping (the owner's
"why is it not stabilized" complaint).
"""
import main as m
import reframe_v2


def test_dedupe_sendcmd_lines_scales_into_2x_space():
    # A 0.5-source-px step (the 2x grid) must become whole 2x-pixels in the
    # sendcmd file, and the emitted w/h must match the scaled crop.
    rects = [(100.0, 50.0, 400.0, 600.0), (100.5, 50.5, 400.5, 600.5)]
    lines = reframe_v2.dedupe_sendcmd_lines(rects, fps=30.0, scale=2)
    assert len(lines) == 2
    assert "x 200" in lines[0] and "y 100" in lines[0]
    assert "w 800" in lines[0] and "h 1200" in lines[0]
    # Second rect lands on even 2x pixels (201, 101, 801, 1201).
    assert "x 201" in lines[1] and "y 101" in lines[1]
    assert "w 801" in lines[1] and "h 1201" in lines[1]


def test_unified_filtergraph_supersamples_the_input():
    graph = reframe_v2.unified_filtergraph(
        1080, 1920, 1620, 2160, "/tmp/cmd.txt", 200, initial_y=100,
        supersample=2)
    assert "scale=iw*2:ih*2:flags=bicubic" in graph
    assert "crop@c=w=1620:h=2160:x=200:y=100" in graph


def test_unified_filtergraph_default_is_unchanged():
    graph = reframe_v2.unified_filtergraph(
        1080, 1920, 810, 1080, "/tmp/cmd.txt", 100, initial_y=50)
    assert "scale=iw*2" not in graph
    assert "crop@c=w=810:h=1080:x=100:y=50" in graph


def test_unified_split_filtergraph_supersamples_the_input():
    graph = reframe_v2.unified_split_filtergraph(
        1080, 1920, 1620, 2160, "/tmp/main.txt", 200, 100,
        "/tmp/top_xy.txt", "/tmp/top_z.txt",
        "/tmp/bot_xy.txt", "/tmp/bot_z.txt",
        1440, 1280, (300, 200, 1080, 960), (900, 300, 1080, 960),
        "between(t,0,2)", "/tmp/top_f.txt", "/tmp/bot_f.txt",
        (0, 0), (0, 0), supersample=2)
    assert "scale=iw*2:ih*2:flags=bicubic" in graph
    assert "crop@c=w=1620:h=2160" in graph
    assert "crop@ct=w=1440:h=1280:x=300:y=200" in graph


def test_camera_emits_on_the_sub_pixel_grid():
    c = m.SmoothedCameraman(1080, 1440, 1920, 1080, aspect_ratio=0.75,
                            fps=25, supersample=2)
    c.force_next_update = True
    c.update_target([900, 400, 200, 200])  # centre 1000, head anchor y=432
    c.get_crop_box(force_snap=True)
    # Drive a small committed move (past the dead zone, inside the safe zone)
    # so the emitted rect has to change on the 0.5px grid.
    c.force_next_update = True
    c.update_target([903, 400, 200, 200])  # centre 1003
    x1, y1, x2, y2 = c.get_crop_box()
    assert x1 == round(x1 * 2) / 2, "x positions must sit on the 0.5 grid"
    assert y1 == round(y1 * 2) / 2, "y positions must sit on the 0.5 grid"


def test_target_ema_damps_committed_centers():
    m.TARGET_EMA = 0.5
    try:
        c = m.SmoothedCameraman(1080, 1440, 1920, 1080, aspect_ratio=0.75,
                                fps=25)
        c.force_next_update = True
        c.update_target([900, 400, 200, 200])  # centre 1000, commits exactly
        c.get_crop_box(force_snap=True)
        # Mid-band move (1000 -> 1040): commits, but EMA halves the jump.
        c.force_next_update = False
        c.update_target([940, 400, 200, 200])
        assert c.target_center_x == 1020.0, \
            "EMA must blend, not snap, mid-band target moves"
    finally:
        m.TARGET_EMA = 0.5


def test_force_snap_bypasses_the_ema():
    m.TARGET_EMA = 0.5
    try:
        c = m.SmoothedCameraman(1080, 1440, 1920, 1080, aspect_ratio=0.75,
                                fps=25)
        c.force_next_update = True
        c.update_target([900, 400, 200, 200])  # centre 1000
        assert c.target_center_x == 1000.0, \
            "a cut-time snap must land exactly, never blended"
    finally:
        m.TARGET_EMA = 0.5


def test_head_anchor_is_contained_when_target_drifts():
    c = m.SmoothedCameraman(1080, 1440, 1920, 1080, aspect_ratio=0.75,
                            fps=25)
    c.force_next_update = True
    # Push the head anchor well above the crop centre: at zoom=1.0 the crop
    # is full-height so containment is trivial; force a zoomed crop by
    # updating with a tight zoom target.
    c.update_target([1000, 100, 200, 200], zoom_target=0.7)
    c.get_crop_box(force_snap=True)
    # Pretend the committed target drifts out of the emitted crop.
    c.target_center_y = 50.0
    x1, y1, x2, y2 = c.get_crop_box()
    assert y1 <= c.target_center_y <= y2, \
        "the head anchor must never fall outside the emitted crop"
