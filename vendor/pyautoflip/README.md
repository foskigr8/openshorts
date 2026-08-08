# Vendored: pyautoflip 0.2.1

Upstream: https://github.com/AhmedHisham1/pyautoflip
License: MIT (see `LICENSE`) — Copyright (c) 2024 Ahmed Hisham
Vendored: 2026-08-08, from the PyPI sdist `pyautoflip-0.2.1.tar.gz`

## Why vendored instead of a pip dependency

`pyautoflip`'s dependency set is incompatible with this stack:

```
torch>=2.11.0        opencv-python>=4.11.0.86
numpy>=1.24.0        mediapipe>=0.10.21
insightface>=0.7.3   onnxruntime>=1.21.0
```

`numpy>=1.24.0` is unbounded, so it resolves to numpy 2.x — the exact breakage
documented in `CLAUDE.md` (the `numpy<2` guard line in `requirements.txt` exists
to stop insightface's chain from doing this). It would also force a torch bump.

The files taken here import only `cv2`, `numpy` and `onnxruntime`, all of which
are already in the stack, so vendoring them adds **no new dependency**.

## What is vendored

| File | Purpose |
|---|---|
| `saliency_detector.py` | UNISAL saliency via ONNX Runtime, verbatim. Standalone — no pyautoflip-internal imports. |
| `unisal.onnx` + `unisal.onnx.data` | UNISAL weights (~13 MB), ~17 ms/frame on CPU |
| `split_screen.py` | `find_split_faces` + `render_split_screen_from_centers`, verbatim from `cropping/saliency_cropper.py` |

Both `.py` files are byte-identical to upstream so they can be diffed against a
future release. OpenShorts logic that *uses* them lives in `reframe_v3.py`.

## What is deliberately NOT vendored, and why

Reviewed at 0.2.1 and rejected — these are the reasons, recorded so the decision
is not silently revisited:

- **`cropping/camera_motion.py` (`CameraMotionHandler`).** Its `STATIONARY` mode
  sets the crop to the *average* of key-frame positions
  (`avg_x = int(sum(key_xs) / len(key_xs))`). On a two-speaker scene that lands
  the crop between both people — the "half a person" framing this rebuild
  exists to fix. `shot_planner.py` already does better (per-shot static crop
  with a containment check). Its `is_talking_head` heuristic is also
  dimensionally wrong: it divides a frame position by the *crop* width
  (`key_crop_windows[0][0][2]`) rather than the frame width, so the test is
  near-meaningless.

- **`compute_crop_window`.** Returns `(crop_x, 0, crop_w, frame_h)` — Y is
  always 0 and height is always the full frame. It has no vertical composition
  at all, so head-anchoring (`CAMERA_HEAD_ANCHOR`, `CAMERA_HEAD_Y` in
  `main.py`) cannot be expressed through it.

- **`apply_padding_to_crop`.** Despite the name it does not letterbox; it
  `cv2.resize`s the crop to fill the full output and then darkens bands over
  the *stretched* result. When the crop aspect differs from the target (i.e.
  the `WIDE_CROP_FACTOR = 1.30` path, which is precisely the two-person case)
  faces are geometrically distorted. Our renderer letterboxes instead.

- **`detection/face_detector.py`, `detection/shot_boundary.py`.** Superseded by
  `face_spine.py` (tracked identities) and `shot_planner.py` (knows about
  jump-cut splice points, which a pixel-based shot detector cannot see).

## Upgrading

Re-download the sdist, diff `saliency_detector.py` and the two split-screen
functions against these copies, and re-check the rejected list above — several
entries are bugs that may be fixed upstream, at which point taking more of the
library becomes worth reconsidering.
