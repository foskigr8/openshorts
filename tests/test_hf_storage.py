"""Tests for HuggingFace persistent clip storage.

The load-bearing property is that this NEVER breaks a job: unconfigured it is
a no-op, and a failing upload is logged and swallowed. A self-host with no HF
credentials must behave exactly as it did before the module existed.
"""

import sys
import types

import pytest

import hf_storage


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Each test gets a clean client cache and repo-creation memo."""
    monkeypatch.setattr(hf_storage, "_api", None)
    monkeypatch.setattr(hf_storage, "_ensured_repos", set())
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_STORAGE_REPO", raising=False)
    yield


class FakeApi:
    """Stands in for huggingface_hub.HfApi."""

    def __init__(self, token=None, fail_upload=False, fail_repo=False):
        self.token = token
        self.uploaded = []
        self.created = []
        self.deleted = []
        self.files = []
        self.fail_upload = fail_upload
        self.fail_repo = fail_repo

    def create_repo(self, repo_id=None, repo_type=None, private=None, exist_ok=None):
        if self.fail_repo:
            raise RuntimeError("401 unauthorized")
        self.created.append((repo_id, repo_type, private))

    def upload_file(self, path_or_fileobj=None, path_in_repo=None,
                    repo_id=None, repo_type=None):
        if self.fail_upload:
            raise RuntimeError("connection reset")
        self.uploaded.append((path_or_fileobj, path_in_repo, repo_id, repo_type))

    def list_repo_files(self, repo_id=None, repo_type=None):
        return list(self.files)

    def delete_file(self, path_in_repo=None, repo_id=None, repo_type=None):
        self.deleted.append(path_in_repo)


def _install_fake_hub(monkeypatch, api):
    """Inject a fake huggingface_hub so the tests never touch the network."""
    module = types.ModuleType("huggingface_hub")
    module.HfApi = lambda token=None: api
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return module


def _configure(monkeypatch, repo="user/openshorts-clips"):
    monkeypatch.setenv("HF_TOKEN", "hf_write_token")
    monkeypatch.setenv("HF_STORAGE_REPO", repo)


def _clip(tmp_path, name="clip_1.mp4", size=2048):
    p = tmp_path / name
    p.write_bytes(b"\0" * size)
    return str(p)


# --- Configuration ----------------------------------------------------------


def test_unconfigured_is_a_no_op(tmp_path, monkeypatch):
    api = FakeApi()
    _install_fake_hub(monkeypatch, api)
    assert hf_storage.configured() is False
    assert hf_storage.upload_file(_clip(tmp_path), "jobs/j/clip_1.mp4") is None
    assert hf_storage.download_file("jobs/j/clip_1.mp4", str(tmp_path / "out.mp4")) is False
    assert api.uploaded == []


def test_token_without_repo_is_not_configured(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    assert hf_storage.configured() is False


def test_repo_without_token_is_not_configured(monkeypatch):
    monkeypatch.setenv("HF_STORAGE_REPO", "user/repo")
    assert hf_storage.configured() is False


def test_status_reports_presence_but_never_the_token(monkeypatch):
    _configure(monkeypatch)
    status = hf_storage.status()
    assert status == {"provider": "huggingface", "configured": True,
                      "repo": "user/openshorts-clips"}
    assert "hf_write_token" not in repr(status)


def test_status_hides_the_repo_when_unconfigured():
    assert hf_storage.status()["repo"] is None


# --- Key layout -------------------------------------------------------------


def test_job_key_groups_by_job():
    assert hf_storage.job_key("abc123", "video_clip_1.mp4") == \
        "jobs/abc123/video_clip_1.mp4"


def test_job_key_strips_directories():
    """A path must never escape the job prefix."""
    assert hf_storage.job_key("abc", "/tmp/evil/../clip.mp4") == "jobs/abc/clip.mp4"


def test_public_url_points_at_the_dataset_repo(monkeypatch):
    _configure(monkeypatch)
    assert hf_storage.public_url("jobs/j/c.mp4") == (
        "https://huggingface.co/datasets/user/openshorts-clips/resolve/main/jobs/j/c.mp4")


# --- Upload -----------------------------------------------------------------


def test_upload_creates_the_repo_once_then_uploads(tmp_path, monkeypatch):
    _configure(monkeypatch)
    api = FakeApi()
    _install_fake_hub(monkeypatch, api)

    key = hf_storage.job_key("job1", "clip_1.mp4")
    assert hf_storage.upload_file(_clip(tmp_path), key) == key
    assert hf_storage.upload_file(_clip(tmp_path, "clip_2.mp4"),
                                  hf_storage.job_key("job1", "clip_2.mp4"))
    assert len(api.created) == 1, "repo creation must be memoized"
    assert [u[1] for u in api.uploaded] == ["jobs/job1/clip_1.mp4",
                                            "jobs/job1/clip_2.mp4"]
    assert api.uploaded[0][3] == "dataset"


def test_repo_is_private_by_default(tmp_path, monkeypatch):
    _configure(monkeypatch)
    api = FakeApi()
    _install_fake_hub(monkeypatch, api)
    hf_storage.upload_file(_clip(tmp_path), "jobs/j/c.mp4")
    assert api.created[0][2] is True


def test_repo_can_be_made_public_explicitly(tmp_path, monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setenv("HF_STORAGE_PRIVATE", "0")
    api = FakeApi()
    _install_fake_hub(monkeypatch, api)
    hf_storage.upload_file(_clip(tmp_path), "jobs/j/c.mp4")
    assert api.created[0][2] is False


def test_a_failed_upload_returns_none_and_does_not_raise(tmp_path, monkeypatch, capsys):
    """The job must survive a broken backup — the clip is still on local disk."""
    _configure(monkeypatch)
    _install_fake_hub(monkeypatch, FakeApi(fail_upload=True))
    assert hf_storage.upload_file(_clip(tmp_path), "jobs/j/c.mp4") is None
    # And it must be LOUD: the S3 path this replaces logged neither successes
    # nor failures, so a broken backup looked identical to a working one.
    assert "FAILED" in capsys.readouterr().out


def test_an_unreachable_repo_does_not_raise(tmp_path, monkeypatch):
    _configure(monkeypatch)
    _install_fake_hub(monkeypatch, FakeApi(fail_repo=True))
    assert hf_storage.upload_file(_clip(tmp_path), "jobs/j/c.mp4") is None


def test_upload_of_a_missing_file_is_reported_not_raised(monkeypatch):
    _configure(monkeypatch)
    api = FakeApi()
    _install_fake_hub(monkeypatch, api)
    assert hf_storage.upload_file("/does/not/exist.mp4", "jobs/j/c.mp4") is None
    assert api.uploaded == []


def test_missing_library_degrades_to_no_op(tmp_path, monkeypatch, capsys):
    _configure(monkeypatch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import raises
    assert hf_storage.upload_file(_clip(tmp_path), "jobs/j/c.mp4") is None
    assert "not installed" in capsys.readouterr().out


# --- Delete -----------------------------------------------------------------


def test_delete_prefix_removes_only_that_jobs_files(monkeypatch):
    _configure(monkeypatch)
    api = FakeApi()
    api.files = ["jobs/j1/a.mp4", "jobs/j1/b.mp4", "jobs/j2/c.mp4"]
    _install_fake_hub(monkeypatch, api)
    assert hf_storage.delete_prefix("jobs/j1/") == 2
    assert api.deleted == ["jobs/j1/a.mp4", "jobs/j1/b.mp4"]


def test_delete_prefix_is_a_no_op_when_unconfigured(monkeypatch):
    _install_fake_hub(monkeypatch, FakeApi())
    assert hf_storage.delete_prefix("jobs/j1/") == 0
