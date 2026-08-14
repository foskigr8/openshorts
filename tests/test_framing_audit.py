"""The gate's verdict logic, locked against the calibration baseline.

`eval/framing_audit.py` is calibrated so both reference shorts pass and all
three known-bad renders fail. That calibration lives in real video files which
are not in the repo, so these tests freeze the MEASURED numbers instead: if
somebody loosens a threshold far enough to let clip 1 through, this fails.

A gate that passes the known-bad clips is not a gate.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "eval"))

framing_audit = pytest.importorskip("framing_audit")


def metrics(**over):
    """A clip that passes everything, overridden per test."""
    base = dict(
        clip="synthetic.mp4", duration_s=30.0, frames_sampled=60, cuts=10,
        median_shot_s=2.9, short_shot_frac=0.05,
        eyeline_median=0.217, face_h_median=0.155,
        empty_frac=0.02, head_top_frac=0.0, seam_frac=0.0, gap_frac=0.02,
        split_share=0.0)
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# The measured baselines (see the module docstring in eval/framing_audit.py)
# ---------------------------------------------------------------------------

REFERENCE_A = metrics(eyeline_median=0.218, face_h_median=0.149,
                      empty_frac=0.10, gap_frac=0.03, split_share=0.02,
                      seam_frac=0.0, median_shot_s=2.90, short_shot_frac=0.05)
REFERENCE_B = metrics(eyeline_median=0.220, face_h_median=0.163,
                      empty_frac=0.0, gap_frac=0.0, split_share=0.0,
                      seam_frac=0.0, median_shot_s=3.27, short_shot_frac=0.11)
FLOP_1 = metrics(eyeline_median=0.562, face_h_median=0.219, empty_frac=0.0,
                 gap_frac=0.03, split_share=1.0, seam_frac=0.68,
                 median_shot_s=1.96, short_shot_frac=0.0)
FLOP_2 = metrics(eyeline_median=0.412, face_h_median=0.092, empty_frac=0.21,
                 gap_frac=0.05, split_share=0.04, seam_frac=0.0,
                 median_shot_s=1.58, short_shot_frac=0.29)
FLOP_4 = metrics(eyeline_median=0.385, face_h_median=0.100, empty_frac=0.27,
                 gap_frac=0.03, split_share=0.08, seam_frac=0.0,
                 median_shot_s=1.23, short_shot_frac=0.40)


@pytest.mark.parametrize("name,m", [("reference A", REFERENCE_A),
                                    ("reference B", REFERENCE_B)])
def test_the_reference_clips_pass(name, m):
    fail, _ = framing_audit.verdict(m)
    assert fail == [], f"{name} must pass the gate, got {fail}"


@pytest.mark.parametrize("name,m", [("clip 1", FLOP_1), ("clip 2", FLOP_2),
                                    ("clip 4", FLOP_4)])
def test_every_known_bad_clip_fails(name, m):
    fail, _ = framing_audit.verdict(m)
    assert fail, f"{name} must NOT pass the gate"


def test_clip_1_fails_specifically_on_the_seam():
    """The defect the whole rebuild started from: 68% of split frames with the
    seam through a face."""
    fail, warn = framing_audit.verdict(FLOP_1)
    assert any("seam" in f for f in fail)
    assert any("I3" in f for f in fail)
    assert any("split screen on 100%" in w for w in warn)


def test_clips_2_and_4_fail_on_subject_size_or_emptiness():
    fail_2, _ = framing_audit.verdict(FLOP_2)
    assert any("I4" in f for f in fail_2)
    fail_4, _ = framing_audit.verdict(FLOP_4)
    assert any("show nobody" in f for f in fail_4)


# ---------------------------------------------------------------------------
# Each invariant fires on its own
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,value,tag", [
    ("head_top_frac", 0.40, "I2"),
    ("eyeline_median", 0.55, "I3"),
    ("eyeline_median", 0.10, "I3"),
    ("face_h_median", 0.06, "I4"),
    ("face_h_median", 0.35, "I4"),
    ("empty_frac", 0.40, "I5"),
    ("seam_frac", 0.30, "I5"),
    ("gap_frac", 0.60, "I6"),
    ("median_shot_s", 1.0, "I9"),
    ("short_shot_frac", 0.50, "I9"),
])
def test_each_invariant_fails_on_its_own(field, value, tag):
    fail, _ = framing_audit.verdict(metrics(**{field: value}))
    assert any(tag in f for f in fail), f"{field}={value} should trip {tag}"


def test_a_high_split_share_warns_but_does_not_fail():
    """Split screen is a device, not a defect. Only its geometry can fail a
    clip; the share is a diagnostic that the two-shot test is misfiring."""
    fail, warn = framing_audit.verdict(metrics(split_share=0.95))
    assert fail == []
    assert warn and "two-shot" in warn[0]


def test_a_clean_clip_passes_with_no_warnings():
    fail, warn = framing_audit.verdict(metrics())
    assert fail == [] and warn == []
