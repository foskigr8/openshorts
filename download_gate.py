"""Pre-download client capability gate (stdlib-only).

The download ladder probes each client with ``extract_info(download=False)``
before pulling the file. YouTube serves DIFFERENT format lists per client,
and a spoofed client can be offered only ~360p progressive — the ladder
used to download that whole low-res file before the post-download HD gate
rejected it. This module answers "could this client's format list plausibly
produce a floor-passing source?" so the ladder can skip straight to the next
strategy.

Conservative by contract: any uncertainty returns True (let the download
decide), so a wrong judgement can never skip a client that actually had HD —
it can only ever skip a download the HD gate would have thrown away anyway.
"""

from __future__ import annotations

import os


class ClientCannotServeFloor(Exception):
    """Raised by the ladder when a client's probe list cannot serve the HD
    floor — the download is skipped in favour of the next strategy."""

    def __init__(self, best_height):
        self.best_height = best_height
        super().__init__(f"client can only serve {best_height}p")


def _floor_and_cap():
    """(min_height, max_height_or_None) from the same env the HD gate uses:
    MIN_SOURCE_HEIGHT (1080 floor) and SOURCE_MAX_HEIGHT (1440 cap; 0 =
    no cap)."""
    floor = int(os.environ.get("MIN_SOURCE_HEIGHT", "1080"))
    raw_max = (os.environ.get("SOURCE_MAX_HEIGHT") or "1440").strip() or "1440"
    if raw_max in ("0", "unlimited", "none"):
        return floor, None
    return floor, int(raw_max)


def can_serve_hd_floor(info) -> bool:
    """Could this client's probe format list produce a floor-passing source?

    A format qualifies when its height is within [MIN_SOURCE_HEIGHT,
    SOURCE_MAX_HEIGHT] and it is not audio-only, not AV1, and not 10-bit VP9
    (vp09.00.40) — the same codecs the HD format chain selects and Turing's
    NVDEC can decode. Mirrors the chain loosely on purpose: if ANY plausible
    HD format exists, this returns True so the download's own format
    selector + the post-download probe stay the final authority.

    Returns True when the probe has no format list at all (cannot judge —
    never skip on uncertainty).
    """
    fmts = info.get("formats") or []
    if not fmts:
        return True
    floor, cap = _floor_and_cap()
    for f in fmts:
        h = f.get("height") or 0
        if h < floor or (cap is not None and h > cap):
            continue
        vc = str(f.get("vcodec") or "none").lower()
        if vc in ("none", "") or vc.startswith(("audio", "mp4a")):
            continue  # audio-only stream
        if vc.startswith("av01"):
            continue  # AV1 — Turing's NVDEC has no AV1 decoder
        if vc.startswith("vp09") and not vc.startswith("vp09.00.10"):
            continue  # 10-bit / other VP9 profiles the chain never selects
        return True
    return False
