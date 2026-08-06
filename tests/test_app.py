"""Ready-marker gating for in-progress clip surfacing (plan item 6)."""
import json
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


# --- Server-held keys + persistent storage ---------------------------------


def _system(monkeypatch, **env):
    """Call /api/system with a controlled environment."""
    from fastapi.testclient import TestClient
    import app as app_mod

    for name in ("GEMINI_API_KEY", "ASSEMBLYAI_API_KEY", "ELEVENLABS_API_KEY",
                 "UPLOAD_POST_API_KEY", "HF_TOKEN", "HF_STORAGE_REPO"):
        monkeypatch.delenv(name, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return TestClient(app_mod.app).get("/api/system").json()


def test_system_reports_which_keys_the_server_holds(monkeypatch):
    """The dashboard gates its job form on a browser-stored key; on Kaggle the
    server already has one, so it needs to be able to ask."""
    body = _system(monkeypatch, GEMINI_API_KEY="k", ASSEMBLYAI_API_KEY="a")
    assert body["server_keys"]["gemini"] is True
    assert body["server_keys"]["assemblyai"] is True
    assert body["server_keys"]["elevenlabs"] is False


def test_system_never_leaks_key_material(monkeypatch):
    """Presence only. /api/system is unauthenticated."""
    secret = "AIzaSy-super-secret-value"
    body = _system(monkeypatch, GEMINI_API_KEY=secret)
    assert secret not in json.dumps(body)
    assert body["server_keys"]["gemini"] is True


def test_system_reports_no_server_keys_on_a_bare_selfhost(monkeypatch):
    body = _system(monkeypatch)
    assert body["server_keys"] == {"gemini": False, "assemblyai": False,
                                   "elevenlabs": False, "upload_post": False}


def test_system_reports_storage_backend_state(monkeypatch):
    body = _system(monkeypatch)
    assert body["storage_backend"]["configured"] is False
    body = _system(monkeypatch, HF_TOKEN="t", HF_STORAGE_REPO="u/r")
    assert body["storage_backend"] == {"provider": "huggingface",
                                       "configured": True, "repo": "u/r"}


def test_history_serves_a_wiped_clip_from_storage(monkeypatch, tmp_path):
    """The Kaggle case: /kaggle/working was wiped, metadata survived. The clip
    must still be listed, pointing at the restore endpoint."""
    from fastapi.testclient import TestClient
    import app as app_mod

    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setenv("HF_STORAGE_REPO", "u/r")

    job_dir = tmp_path / "job-wiped"
    job_dir.mkdir()
    (job_dir / "vid_metadata.json").write_text(json.dumps({
        "shorts": [{"start": 0, "end": 20, "storage_key": "jobs/job-wiped/vid_clip_1.mp4",
                    "storage_filename": "vid_clip_1.mp4",
                    "video_title_for_youtube_short": "Kept"}],
    }))
    # NOTE: no clip file on disk — that is the point.

    body = TestClient(app_mod.app).get("/api/history").json()
    entry = [v for v in body["videos"] if v["job_id"] == "job-wiped"]
    assert len(entry) == 1
    assert entry[0]["title"] == "Kept"
    assert entry[0]["view_url"] == "/api/storage/job-wiped/vid_clip_1.mp4"
    assert entry[0]["storage"] == "huggingface"


def test_history_still_hides_a_clip_with_no_local_file_and_no_backup(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_mod

    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_STORAGE_REPO", raising=False)

    job_dir = tmp_path / "job-gone"
    job_dir.mkdir()
    (job_dir / "vid_metadata.json").write_text(json.dumps(
        {"shorts": [{"start": 0, "end": 20}]}))

    body = TestClient(app_mod.app).get("/api/history").json()
    entries = [v for v in body["videos"] if v["job_id"] == "job-gone"]
    # Surfaced as a failed job (no media), never as a playable clip.
    assert all(v["view_url"] == "" for v in entries)


def test_restore_endpoint_serves_the_local_file_when_present(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_mod

    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    job_dir = tmp_path / "job-local"
    job_dir.mkdir()
    (job_dir / "clip_1.mp4").write_bytes(b"video-bytes")

    res = TestClient(app_mod.app).get("/api/storage/job-local/clip_1.mp4")
    assert res.status_code == 200
    assert res.content == b"video-bytes"


def test_restore_endpoint_pulls_from_storage_when_local_is_gone(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_mod
    import hf_storage

    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setenv("HF_STORAGE_REPO", "u/r")
    (tmp_path / "job-restore").mkdir()

    pulled = []

    def fake_download(key, local_path):
        pulled.append(key)
        with open(local_path, "wb") as f:
            f.write(b"restored")
        return True

    monkeypatch.setattr(hf_storage, "download_file", fake_download)
    res = TestClient(app_mod.app).get("/api/storage/job-restore/clip_1.mp4")
    assert res.status_code == 200
    assert res.content == b"restored"
    assert pulled == ["jobs/job-restore/clip_1.mp4"]


def test_restore_endpoint_404s_when_the_clip_is_truly_gone(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_mod

    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "job-none").mkdir()
    res = TestClient(app_mod.app).get("/api/storage/job-none/clip_1.mp4")
    assert res.status_code == 404


def test_restore_endpoint_never_escapes_the_job_directory(monkeypatch, tmp_path):
    """A traversing filename must not reach a file outside the job dir.

    Two layers stop it: a path with a slash does not match the two-segment
    route at all (the SPA catch-all answers instead), and anything that does
    match is basename-d before it touches the filesystem.
    """
    from fastapi.testclient import TestClient
    import app as app_mod

    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / "job-safe").mkdir()
    (tmp_path / "secret.txt").write_text("do not serve me")

    client = TestClient(app_mod.app)
    for url in ("/api/storage/job-safe/..%2Fsecret.txt",
                "/api/storage/job-safe/../secret.txt"):
        assert b"do not serve me" not in client.get(url).content

    # The sibling file exists one level up; asking for it by bare name must
    # 404 rather than resolve against OUTPUT_DIR.
    assert client.get("/api/storage/job-safe/secret.txt").status_code == 404


def test_restore_endpoint_404s_for_an_unknown_job(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_mod

    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    res = TestClient(app_mod.app).get("/api/storage/no-such-job/clip_1.mp4")
    assert res.status_code == 404


def test_backup_is_skipped_entirely_when_storage_is_unconfigured(monkeypatch, tmp_path):
    """A self-host with no HF credentials must behave exactly as before."""
    import app as app_mod
    import hf_storage

    monkeypatch.delenv("HF_TOKEN", raising=False)
    calls = []
    monkeypatch.setattr(hf_storage, "upload_file",
                        lambda *a, **k: calls.append(a))
    app_mod.jobs["job-x"] = {"status": "processing", "logs": [],
                             "output_dir": str(tmp_path)}
    try:
        app_mod._hf_backup_ready_clips("job-x", str(tmp_path))
    finally:
        app_mod.jobs.pop("job-x", None)
    assert calls == []


def test_storage_key_is_recorded_atomically_in_metadata(monkeypatch, tmp_path):
    import app as app_mod

    meta = tmp_path / "vid_metadata.json"
    meta.write_text(json.dumps({"shorts": [{"start": 0, "end": 10},
                                           {"start": 20, "end": 30}]}))
    app_mod._hf_record_storage_key(str(tmp_path), 1, "vid_clip_2.mp4",
                                   "jobs/j/vid_clip_2.mp4")
    data = json.loads(meta.read_text())
    assert data["shorts"][1]["storage_key"] == "jobs/j/vid_clip_2.mp4"
    assert data["shorts"][1]["storage_filename"] == "vid_clip_2.mp4"
    assert "storage_key" not in data["shorts"][0], "must not touch other clips"
    assert not list(tmp_path.glob("*.tmp")), "tmp file must be replaced, not left"


def test_recording_a_key_for_a_missing_clip_index_is_ignored(tmp_path):
    import app as app_mod

    meta = tmp_path / "vid_metadata.json"
    meta.write_text(json.dumps({"shorts": [{"start": 0, "end": 10}]}))
    app_mod._hf_record_storage_key(str(tmp_path), 9, "x.mp4", "jobs/j/x.mp4")
    assert json.loads(meta.read_text())["shorts"] == [{"start": 0, "end": 10}]


# --- Per-job caption placement ---------------------------------------------


def test_caption_margin_parsing_clamps_and_rejects_junk():
    """Mirrors generate_ass's own 0..200 clamp, so nothing out of range can
    reach the ASS style line."""
    from app import _parse_caption_margin

    assert _parse_caption_margin("120") == 120
    assert _parse_caption_margin(43) == 43
    assert _parse_caption_margin("9999") == 200
    assert _parse_caption_margin("-5") == 0
    assert _parse_caption_margin("") is None       # -> server default
    assert _parse_caption_margin(None) is None
    assert _parse_caption_margin("bottom") is None


def test_caption_positions_are_the_three_generate_ass_supports():
    import app as app_mod
    import subtitles

    assert set(app_mod.CAPTION_POSITIONS) == {"top", "middle", "bottom"}
    # generate_ass's align_map is the source of truth for what is renderable.
    for position in app_mod.CAPTION_POSITIONS:
        assert position in ("top", "middle", "bottom")
    assert subtitles.AUTO_CAPTION_STYLE["alignment"] in app_mod.CAPTION_POSITIONS


def test_job_logs_download_serves_local_persisted_log(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import app as app_mod

    job_id = "log-local-test"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "logs.jsonl").write_text(
        '{"ts": 1700000000, "text": "local line"}\n')
    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))
    res = TestClient(app_mod.app).get(f"/api/jobs/{job_id}/logs")
    assert res.status_code == 200
    assert "local line" in res.text
    assert "logs.txt" in res.headers.get("content-disposition", "")


def test_job_logs_download_falls_back_to_hf_storage(monkeypatch, tmp_path):
    """6-aug-2026: a wiped Kaggle session must still download the log."""
    import json
    from fastapi.testclient import TestClient
    import app as app_mod

    job_id = "log-restore-test"
    monkeypatch.setattr(app_mod, "OUTPUT_DIR", str(tmp_path))

    class _FakeHF:
        @staticmethod
        def configured():
            return True

        @staticmethod
        def job_key(job_id, filename):
            return f"jobs/{job_id}/{filename}"

        @staticmethod
        def download_file(key, local_path):
            with open(local_path, "w") as f:
                f.write(json.dumps({"ts": 1700000000,
                                    "text": "hello from hf"}) + "\n")
            return True

    monkeypatch.setattr(app_mod, "hf_storage", _FakeHF)
    res = TestClient(app_mod.app).get(f"/api/jobs/{job_id}/logs")
    assert res.status_code == 200
    assert "hello from hf" in res.text
