"""Progress snapshots and .ready markers (plan items 6 + 11)."""
import json
import os

import pipeline_progress as pp


def test_write_progress_writes_atomic_snapshot(tmp_path):
    pp.write_progress(str(tmp_path), "transcribe")
    payload = json.loads((tmp_path / "progress.json").read_text())
    assert payload["stage"] == "transcribe"
    assert payload["overall_pct"] == 25
    assert payload["clips_done"] == 0
    assert payload["clips_total"] == 0
    # No temp files left behind (the atomic replace cleaned up).
    assert not list(tmp_path.glob(".progress.json.tmp"))


def test_render_stage_interpolates_between_analyze_and_render(tmp_path):
    pp.write_progress(str(tmp_path), "render", clips_done=2, clips_total=4)
    payload = json.loads((tmp_path / "progress.json").read_text())
    # 45 + (90-45) * 2/4 = 45 + 22 = 67
    assert payload["overall_pct"] == 67
    assert payload["stage_pct"] == 50  # 2/4 of the render stage itself
    assert payload["clips_done"] == 2
    assert payload["clips_total"] == 4


def test_render_progress_clamps_and_zero_total_is_safe(tmp_path):
    pp.write_progress(str(tmp_path), "render", clips_done=3, clips_total=3)
    assert json.loads((tmp_path / "progress.json").read_text())["overall_pct"] == 90
    pp.write_progress(str(tmp_path), "render", clips_done=1, clips_total=0)
    # No division by zero: zero clips means nothing left to interpolate, so
    # the render-stage weight stands.
    assert json.loads((tmp_path / "progress.json").read_text())["overall_pct"] == 90


def test_finalize_is_100(tmp_path):
    pp.write_progress(str(tmp_path), "finalize", clips_done=3, clips_total=3)
    payload = json.loads((tmp_path / "progress.json").read_text())
    assert payload["overall_pct"] == 100
    assert payload["stage_pct"] == 100


def test_non_granular_stage_has_no_stage_pct(tmp_path):
    pp.write_progress(str(tmp_path), "analyze")
    assert json.loads((tmp_path / "progress.json").read_text())["stage_pct"] is None


def test_step_fields_ride_along_for_the_live_panel(tmp_path):
    pp.write_progress(str(tmp_path), "render", clips_done=2, clips_total=4,
                      step="rendering clip 2/4", step_pct=50)
    payload = json.loads((tmp_path / "progress.json").read_text())
    assert payload["step"] == "rendering clip 2/4"
    assert payload["step_pct"] == 50
    pp.write_progress(str(tmp_path), "analyze", note="detecting scenes")
    payload = json.loads((tmp_path / "progress.json").read_text())
    assert payload["note"] == "detecting scenes"
    assert "step" not in payload  # step is optional


def test_mark_clip_ready_writes_marker(tmp_path):
    pp.mark_clip_ready(str(tmp_path), "mytitle_clip_1.mp4")
    assert (tmp_path / "mytitle_clip_1.mp4.ready").exists()
    assert not (tmp_path / "mytitle_clip_1.mp4.ready.tmp").exists()


def test_mark_clip_ready_never_raises(tmp_path):
    # Nonexistent dir + garbage filename must be a no-op, not a crash.
    pp.mark_clip_ready(str(tmp_path / "nope"), "x" * 400)
    pp.write_progress(str(tmp_path / "nope"), "analyze")


class TestEtaEstimate:
    def test_record_stage_durations_rolling_mean(self, tmp_path):
        pp.record_stage_durations(str(tmp_path), {"render": 100.0, "analyze": 50.0})
        pp.record_stage_durations(str(tmp_path), {"render": 120.0})
        data = json.loads((tmp_path / "stage_durations.json").read_text())
        assert data["render"] == {"n": 2, "mean": 110.0}
        assert data["analyze"] == {"n": 1, "mean": 50.0}

    def test_eta_uses_remaining_current_stage_plus_later_stages(self, tmp_path):
        pp.record_stage_durations(str(tmp_path), {
            "download": 20.0, "transcribe": 30.0,
            "analyze": 40.0, "render": 100.0, "finalize": 5.0,
        })
        # In render at 50%: half of render (50) + finalize (5) = 55.
        eta = pp.estimate_eta_seconds(
            str(tmp_path), {"stage": "render", "stage_pct": 50})
        assert eta == 55

    def test_eta_no_data_returns_none(self, tmp_path):
        assert pp.estimate_eta_seconds(
            str(tmp_path), {"stage": "render", "stage_pct": 50}) is None

    def test_eta_unknown_stage_returns_none(self, tmp_path):
        pp.record_stage_durations(str(tmp_path), {"render": 100.0})
        assert pp.estimate_eta_seconds(
            str(tmp_path), {"stage": "nonsense", "stage_pct": 50}) is None
