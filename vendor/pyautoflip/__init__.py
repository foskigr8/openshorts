"""
Vendored subset of pyautoflip 0.2.1 (MIT). See README.md for provenance and for
the list of upstream modules deliberately excluded.
"""

from vendor.pyautoflip.split_screen import (
    find_split_faces,
    render_split_screen_from_centers,
)

__all__ = ["find_split_faces", "render_split_screen_from_centers"]

# NOTE: SaliencyDetector is intentionally NOT re-exported here. Importing it
# pulls in onnxruntime and loads a ~13 MB model on first construction; callers
# import it directly from vendor.pyautoflip.saliency_detector so that modules
# which only need the split-screen geometry stay cheap to import.
