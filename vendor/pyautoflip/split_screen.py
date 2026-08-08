"""
Split-screen geometry, vendored VERBATIM from pyautoflip 0.2.1.

Source: pyautoflip/cropping/saliency_cropper.py (MIT, Copyright (c) 2024 Ahmed Hisham)
See vendor/pyautoflip/README.md for why these two functions are vendored and the
rest of that module is not.

DO NOT EDIT. This file is kept byte-identical to upstream so it can be diffed
against a future release. OpenShorts' own behaviour (speaker-aware split
decisions) lives in reframe_v3.py and calls into these.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


def find_split_faces(face_rects, frame_w, frame_h, target_aspect):
    """
    Check if 2+ faces are too far apart to fit in one crop.
    Returns [(cx, cy), (cx, cy)] normalized 0-1, or None.
    """
    if not face_rects or len(face_rects) < 2:
        return None

    crop_w = int(frame_h * target_aspect[0] / target_aspect[1])
    crop_w_norm = crop_w / frame_w

    face_centers = sorted(
        [((fx + fw / 2) / frame_w, (fy + fh / 2) / frame_h)
         for (fx, fy, fw, fh) in face_rects],
        key=lambda c: c[0]
    )
    left, right = face_centers[0], face_centers[-1]

    if right[0] - left[0] > crop_w_norm:
        return [left, right]
    return None


def render_split_screen_from_centers(frame, face_centers, target_aspect):
    """
    Render a 2-panel split-screen frame, each panel centered on a face.
    face_centers: [(cx_norm, cy_norm), (cx_norm, cy_norm)]
    """
    h, w = frame.shape[:2]
    aspect_w, aspect_h = target_aspect
    target_ratio = aspect_w / aspect_h

    out_w = int(h * target_ratio)
    out_h = h
    divider_h = 4
    panel_h = (out_h - divider_h) // 2
    panel_ratio = out_w / panel_h

    src_crop_w = min(int(h * target_ratio), w)
    src_crop_h = min(int(src_crop_w / panel_ratio), h)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y_cursor = 0

    for i, (cx_norm, cy_norm) in enumerate(face_centers):
        cx_px = int(cx_norm * w)
        cy_px = int(cy_norm * h)
        crop_x = max(0, min(cx_px - src_crop_w // 2, w - src_crop_w))
        crop_y = max(0, min(cy_px - src_crop_h // 2, h - src_crop_h))

        strip = frame[crop_y:crop_y + src_crop_h, crop_x:crop_x + src_crop_w]
        panel = cv2.resize(strip, (out_w, panel_h), interpolation=cv2.INTER_LANCZOS4)
        canvas[y_cursor:y_cursor + panel_h, :] = panel
        y_cursor += panel_h

        if i == 0:
            canvas[y_cursor:y_cursor + divider_h, :] = (40, 40, 40)
            y_cursor += divider_h

    return canvas
