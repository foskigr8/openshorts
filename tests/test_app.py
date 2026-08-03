"""Ready-marker gating for in-progress clip surfacing (plan item 6)."""
import os

import app
from app import _collect_ready_clips, _canonical_clip_file


def _clip(**overrides):
    d = {"start": 1.0, "end": 20.0, "video_title_for_youtube_short": "t"}
    d.update(overrides)
    return d


def _touch(path):
    with open(path, "w") as f:
        f.write("x")


def test_no_clips_surfaced_before_ready_marker(tmp_path):
    base = "mytitle"
    _touch(tmp_path / f"{base}_clip_1.mp4")
    clips = [_clip()]
    ready = _collect_ready_clips(str(tmp_path), base, clips, "job-1")
    assert ready == []


def test_ready_marker_gates_clean_file(tmp_path):
    base = "mytitle"
    _touch(tmp_path / f"{base}_clip_1.mp4")
    _touch(tmp_path / f"{base}_clip_1.mp4.ready")
    clips = [_clip()]
    ready = _collect_ready_clips(str(tmp_path), base, clips, "job-1")
    assert len(ready) == 1
    assert ready[0]["video_url"] == "/videos/job-1/mytitle_clip_1.mp4"


def test_subtitled_derivative_wins_with_marker(tmp_path):
    base = "mytitle"
    clean = tmp_path / f"{base}_clip_1.mp4"
    _touch(clean)
    subtitled = tmp_path / f"subtitled_1700000000_{base}_clip_1.mp4"
    _touch(subtitled)
    _touch(tmp_path / f"{base}_clip_1.mp4.ready")
    # The canonical resolver prefers the newest derived file.
    assert _canonical_clip_file(str(tmp_path), base, 0) == subtitled.name
    ready = _collect_ready_clips(str(tmp_path), base, [_clip()], "job-1")
    assert ready[0]["video_url"] == f"/videos/job-1/{subtitled.name}"


def test_empty_file_not_surfaced_even_with_marker(tmp_path):
    base = "mytitle"
    _touch(tmp_path / f"{base}_clip_1.mp4.ready")
    (tmp_path / f"{base}_clip_1.mp4").write_text("")  # zero bytes
    ready = _collect_ready_clips(str(tmp_path), base, [_clip()], "job-1")
    assert ready == []


def test_marker_absent_for_second_clip_leaves_it_pending(tmp_path):
    base = "mytitle"
    _touch(tmp_path / f"{base}_clip_1.mp4")
    _touch(tmp_path / f"{base}_clip_1.mp4.ready")
    _touch(tmp_path / f"{base}_clip_2.mp4")  # rendered but captioning not done
    clips = [_clip(start=1.0), _clip(start=30.0)]
    ready = _collect_ready_clips(str(tmp_path), base, clips, "job-1")
    assert len(ready) == 1
    assert ready[0]["video_url"].endswith("_clip_1.mp4")


def test_get_status_serves_progress_snapshot(monkeypatch, tmp_path):
    """/api/status must include the live progress.json written by main.py."""
    import json
    from fastapi.testclient import TestClient
    import app as app_mod

    job_id = "progress-test-job"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "progress.json").write_text(json.dumps({
        "stage": "render", "overall_pct": 67, "clips_done": 2, "clips_total": 4,
    }))
    app_mod.jobs[job_id] = {
        "status": "processing",
        "logs": ["queued"],
        "output_dir": str(job_dir),
    }
    try:
        res = TestClient(app_mod.app).get(f"/api/status/{job_id}")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "processing"
        assert body["progress"]["stage"] == "render"
        assert body["progress"]["overall_pct"] == 67
        assert body["progress"]["clips_done"] == 2
        assert body["result"] is None
    finally:
        app_mod.jobs.pop(job_id, None)


def test_get_status_reports_missing_progress_as_null(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_mod

    job_id = "no-progress-test-job"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    app_mod.jobs[job_id] = {
        "status": "queued",
        "logs": ["queued"],
        "output_dir": str(job_dir),
    }
    try:
        res = TestClient(app_mod.app).get(f"/api/status/{job_id}")
        assert res.status_code == 200
        assert res.json()["progress"] is None
    finally:
        app_mod.jobs.pop(job_id, None)


def test_system_status_reports_storage_usage(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_mod

    (tmp_path / "clip.mp4").write_bytes(b"x" * (100 * 1024 * 1024))  # 100 MiB
    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app_mod, "OUTPUT_MAX_GB", 100)

    body = TestClient(app_mod.app).get("/api/system").json()
    assert body["storage"]["cap_gb"] == 100
    assert body["storage"]["used_gb"] > 0
    assert body["storage"]["pct"] == 0  # 2 MiB of 100 GB rounds to 0%


def test_history_status_derivation(monkeypatch, tmp_path):
    """/api/history must stamp per-job status from progress.json, and keep
    failed jobs visible with empty media fields (rail's Failed filter)."""
    import json
    from fastapi.testclient import TestClient
    import app as app_mod

    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))

    # Completed job: progress.json at finalize + one playable clip.
    done = tmp_path / "job-done"
    done.mkdir()
    (done / "progress.json").write_text(json.dumps({"stage": "finalize"}))
    (done / "title_metadata.json").write_text(json.dumps({"shorts": [{"title": "A"}]}))
    (done / "title_clip_1.mp4").write_bytes(b"x")
    # Failed job: metadata but no playable clips.
    failed = tmp_path / "job-failed"
    failed.mkdir()
    (failed / "title_metadata.json").write_text(json.dumps({"shorts": [{"title": "B"}]}))
    # Processing job: progress.json mid-render + one playable clip.
    running = tmp_path / "job-running"
    running.mkdir()
    (running / "progress.json").write_text(json.dumps({"stage": "render"}))
    (running / "title_metadata.json").write_text(json.dumps({"shorts": [{"title": "C"}]}))
    (running / "title_clip_1.mp4").write_bytes(b"x")

    videos = TestClient(app_mod.app).get("/api/history").json()["videos"]
    by_job = {}
    for v in videos:
        by_job.setdefault(v["job_id"], []).append(v)
    assert by_job["job-done"][0]["status"] == "completed"
    assert by_job["job-running"][0]["status"] == "processing"
    assert by_job["job-failed"][0]["status"] == "failed"
    assert by_job["job-failed"][0]["view_url"] == ""


def test_self_host_projects_list_and_restore(monkeypatch, tmp_path):
    """Self-host projects must be listable and re-openable from disk (round 4:
    'projects are not even saving' — the cloud-only gate made reopen 404)."""
    import json
    from fastapi.testclient import TestClient
    import app as app_mod

    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app_mod, "BILLING_ENABLED", False)

    job = tmp_path / "job-abc"
    job.mkdir()
    (job / "title_metadata.json").write_text(json.dumps({
        "shorts": [{"start": 1.0, "end": 20.0,
                    "video_title_for_youtube_short": "The real hook"}],
    }))
    (job / "title_clip_1.mp4").write_bytes(b"x")

    client = TestClient(app_mod.app)
    projects = client.get("/api/projects").json()["projects"]
    assert projects[0]["job_id"] == "job-abc"
    assert projects[0]["title"] == "The real hook"

    restored = client.post("/api/projects/job-abc/restore").json()
    assert restored["job_id"] == "job-abc"
    assert restored["status"] == "completed"
    assert restored["result"]["clips"][0]["video_url"].endswith("title_clip_1.mp4")
    assert restored["project_state"]["clips"][0]["server_file"] == "title_clip_1.mp4"
    assert restored["project_state"]["clips"][0]["active_layers"] == []
    # The in-memory record is registered so edit endpoints can operate.
    assert app_mod.jobs["job-abc"]["output_dir"] == str(job)


def test_self_host_restore_unknown_job_404(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_mod
    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app_mod, "BILLING_ENABLED", False)
    assert TestClient(app_mod.app).post("/api/projects/nope/restore").status_code == 404


def test_newest_metadata_file_and_sorted_glob(tmp_path):
    """Round-5 spec 4.1: metadata selection must be newest-by-mtime, never
    arbitrary glob order."""
    import time
    from app import _newest_metadata_file, _sorted_metadata
    older = tmp_path / "a_metadata.json"
    newer = tmp_path / "b_metadata.json"
    older.write_text("{}")
    time.sleep(0.01)
    newer.write_text("{}")
    assert _newest_metadata_file(str(tmp_path)) == str(newer)
    matches = _sorted_metadata([str(older), str(newer)])
    assert matches[0] == str(newer)
    assert _newest_metadata_file(str(tmp_path / "empty")) is None


def test_find_completed_job_for_source(monkeypatch, tmp_path):
    """Round-5 same-source reuse: a completed on-disk job for the same URL is
    found (and only when it has playable, ready clips)."""
    import json
    import app as app_mod
    from app import _find_completed_job_for_source
    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))

    done = tmp_path / "job-done"
    done.mkdir()
    (done / "title_metadata.json").write_text(json.dumps({
        "source_url": "https://youtube.com/watch?v=abc",
        "shorts": [{"start": 1.0, "end": 20.0}],
    }))
    (done / "title_clip_1.mp4").write_bytes(b"x")
    (done / "title_clip_1.mp4.ready").write_text("1")

    other = tmp_path / "job-other"
    other.mkdir()
    (other / "title_metadata.json").write_text(json.dumps({
        "source_url": "https://youtube.com/watch?v=other",
        "shorts": [{"start": 1.0, "end": 20.0}],
    }))
    (other / "title_clip_1.mp4").write_bytes(b"x")
    (other / "title_clip_1.mp4.ready").write_text("1")

    noclip = tmp_path / "job-noclip"
    noclip.mkdir()
    (noclip / "title_metadata.json").write_text(json.dumps({
        "source_url": "https://youtube.com/watch?v=abc",
        "shorts": [{"start": 1.0, "end": 20.0}],
    }))

    assert _find_completed_job_for_source("https://youtube.com/watch?v=abc") == "job-done"
    assert _find_completed_job_for_source("https://youtube.com/watch?v=other") == "job-other"
    # Same URL but no playable/ready clips -> not reusable.
    assert _find_completed_job_for_source("https://youtube.com/watch?v=nope") is None


def test_source_images_endpoint_generates_stills(monkeypatch, tmp_path):
    """The Source section's stills endpoint lazily extracts frames and returns
    URLs served from the static mount."""
    import json
    from fastapi.testclient import TestClient
    import app as app_mod

    job = tmp_path / "job-img"
    job.mkdir()
    (job / "title_metadata.json").write_text(json.dumps({
        "source_file": "source.mp4",
        "shorts": [{"start": 1.0, "end": 20.0}],
    }))
    (job / "source.mp4").write_bytes(b"fakevideo")

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd[0])
        if cmd[0] == "ffprobe":
            return type("R", (), {"stdout": "120.0"})()
        # ffmpeg: create two fake frames so the endpoint sees cached output.
        frames = job / "source_frames"
        frames.mkdir(exist_ok=True)
        (frames / "source_01.jpg").write_bytes(b"j")
        (frames / "source_02.jpg").write_bytes(b"j")
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(app_mod, "SOURCE_FRAME_COUNT", 2)

    body = TestClient(app_mod.app).get("/api/source/job-img/images").json()
    assert len(body["frames"]) == 2
    assert body["frames"][0]["url"].startswith("/videos/job-img/source_frames/")
    assert "ffprobe" in calls


def test_derived_source_file_classification():
    """Round-5: the source fallback must never mistake a clip/derivative for
    the source video."""
    from app import _is_derived_source_file
    assert _is_derived_source_file("title_clip_1.mp4") is True
    assert _is_derived_source_file("subtitled_1700000000_title_clip_1.mp4") is True
    assert _is_derived_source_file("temp_title_clip_2.mp4") is True
    assert _is_derived_source_file("source_preview.mp4") is True
    assert _is_derived_source_file("autosubs_1234.ass") is True
    assert _is_derived_source_file("My_Original_Video.mp4") is False


def test_resolve_source_path_fallback_without_metadata(monkeypatch, tmp_path):
    """A job that failed before metadata was written must still resolve its
    source from the job dir (largest non-derived video)."""
    import app as app_mod
    from app import _resolve_source_path
    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app_mod, "UPLOAD_DIR", str(tmp_path / "uploads"))

    job = tmp_path / "job-failed"
    job.mkdir()
    (job / "video_title.mp4").write_bytes(b"x" * 5000)  # the source
    (job / "video_title_clip_1.mp4").write_bytes(b"x" * 1000)  # a clip
    assert _resolve_source_path("job-failed") == str(job / "video_title.mp4")


class TestJobCreatedAt:
    """History dates must reflect when a job was GENERATED.

    The list used the job directory's mtime, which a directory updates on every
    file add/remove — so subtitling, re-rendering or deleting a clip re-dated
    the project. Measured across a real library, every job showed the wrong
    DAY, drifting up to 43.5 hours.
    """

    def test_prefers_the_explicit_marker(self, tmp_path):
        job = tmp_path / "job"
        job.mkdir()
        (job / ".created").write_text("1700000000.0")
        (job / "clip.mp4").write_bytes(b"x")
        assert app._job_created_at(str(job)) == 1700000000.0

    def test_falls_back_to_the_oldest_file_not_the_directory(self, tmp_path):
        import os
        job = tmp_path / "job"
        job.mkdir()
        first = job / "first.mp4"
        first.write_bytes(b"x")
        os.utime(first, (1_700_000_000, 1_700_000_000))
        later = job / "later.mp4"
        later.write_bytes(b"x")
        os.utime(later, (1_700_090_000, 1_700_090_000))
        # Directory mtime is "now" (a file was just added); the answer must be
        # the oldest FILE, i.e. when generation began.
        assert app._job_created_at(str(job)) == 1_700_000_000

    def test_marker_is_written_once_and_not_overwritten(self, tmp_path):
        job = tmp_path / "job"
        job.mkdir()
        app._mark_job_created(str(job))
        first = (job / ".created").read_text()
        app._mark_job_created(str(job))
        assert (job / ".created").read_text() == first

    def test_missing_directory_does_not_raise(self, tmp_path):
        assert app._job_created_at(str(tmp_path / "nope")) > 0
