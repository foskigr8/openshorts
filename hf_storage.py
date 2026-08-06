"""HuggingFace Hub storage for finished clips — persistence past the session.

Kaggle wipes ``/kaggle/working`` when the session ends (12h cap, and it dies
sooner than that often enough), so clips rendered there evaporate. This uploads
each clip to a private HF dataset repo as it finishes, and records the key in
the job metadata so History can still serve it after the local file is gone.

**Why HuggingFace and not R2.** An earlier plan chose Cloudflare R2 because
``cloud/storage.py`` already speaks S3 and this would have been a config
change. That reasoning ignored the blocker: Cloudflare requires a credit card,
which the owner does not want to provide. Backblaze B2 wants payment details
too. HuggingFace needs neither, and ``HF_TOKEN`` already exists as a Kaggle
secret. The honest trade-off is that HF is not S3-compatible, so this is a new
module rather than a reused one.

Shape deliberately mirrors ``cloud/storage.py`` so the two read as siblings.

Fails soft everywhere: unconfigured, every function is a no-op returning
``None``/``False``, and a self-host behaves exactly as it did before. An upload
failure is logged and never fails the job — the clip is still on local disk;
losing the backup is not worth losing the render.

Environment
-----------
HF_TOKEN         a **write** access token (read-only tokens cannot upload)
HF_STORAGE_REPO  dataset repo id, e.g. "username/openshorts-clips"
HF_STORAGE_PRIVATE  "0" to create the repo public (default: private)
"""

import os
import threading

_api = None
_api_lock = threading.Lock()
_ensured_repos = set()


def configured() -> bool:
    """True when both the token and the target repo are set."""
    return bool(os.environ.get("HF_TOKEN", "").strip()
                and os.environ.get("HF_STORAGE_REPO", "").strip())


def repo_id() -> str:
    return os.environ.get("HF_STORAGE_REPO", "").strip()


def status() -> dict:
    """Presence-only summary for /api/system. Never returns the token."""
    return {
        "provider": "huggingface",
        "configured": configured(),
        "repo": repo_id() if configured() else None,
    }


def _client():
    """Lazily build the HfApi client. None when unconfigured or missing deps."""
    global _api
    if not configured():
        return None
    with _api_lock:
        if _api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError:
                print("⚠️ HF storage: huggingface_hub is not installed — "
                      "`pip install huggingface_hub` to enable clip backup.")
                return None
            _api = HfApi(token=os.environ["HF_TOKEN"].strip())
    return _api


def _ensure_repo(api) -> bool:
    """Create the dataset repo on first use. Idempotent, once per process."""
    rid = repo_id()
    if rid in _ensured_repos:
        return True
    private = os.environ.get("HF_STORAGE_PRIVATE", "1").strip().lower() not in (
        "0", "false", "no")
    try:
        api.create_repo(repo_id=rid, repo_type="dataset", private=private,
                        exist_ok=True)
        _ensured_repos.add(rid)
        return True
    except Exception as e:
        print(f"⚠️ HF storage: cannot reach dataset repo {rid} "
              f"({type(e).__name__}: {e})")
        return False


def job_key(job_id, filename) -> str:
    """Key layout: jobs/<job_id>/<filename>.

    Keeps one job's clips together, which makes a later delete-by-prefix
    straightforward and matches how cloud/storage.py groups by job.
    """
    return f"jobs/{job_id}/{os.path.basename(filename)}"


def public_url(key) -> str:
    """The resolve URL for a stored key.

    For a PRIVATE repo this URL still requires an Authorization header, so it
    is a durable identifier rather than a link that works in any browser. The
    dashboard reaches content through the API, which holds the token.
    """
    return f"https://huggingface.co/datasets/{repo_id()}/resolve/main/{key}"


def upload_file(local_path, key) -> "str | None":
    """Upload one file. Returns the key on success, None on any failure.

    Never raises: callers run this beside a finished render and a failed
    backup must not fail the job.
    """
    api = _client()
    if api is None:
        return None
    if not local_path or not os.path.exists(local_path):
        print(f"⚠️ HF storage: nothing to upload at {local_path}")
        return None
    if not _ensure_repo(api):
        return None
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=key,
            repo_id=repo_id(),
            repo_type="dataset",
        )
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"☁️  HF storage: uploaded {key} ({size_mb:.1f} MB)")
        return key
    except Exception as e:
        # Logged loudly on purpose. The S3 path this replaces pinned boto's
        # loggers to CRITICAL and reported neither successes nor failures, so
        # a broken backup looked exactly like a working one.
        print(f"❌ HF storage: upload of {key} FAILED "
              f"({type(e).__name__}: {e}) — the clip is still on local disk")
        return None


def download_file(key, local_path) -> bool:
    """Fetch a stored key back to local disk. False on any failure."""
    api = _client()
    if api is None:
        return False
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    try:
        cached = hf_hub_download(
            repo_id=repo_id(), filename=key, repo_type="dataset",
            token=os.environ["HF_TOKEN"].strip())
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        if os.path.abspath(cached) != os.path.abspath(local_path):
            import shutil
            shutil.copyfile(cached, local_path)
        return True
    except Exception as e:
        print(f"⚠️ HF storage: download of {key} failed "
              f"({type(e).__name__}: {e})")
        return False


def delete_prefix(prefix) -> int:
    """Delete every stored file under a prefix. Returns how many were removed."""
    api = _client()
    if api is None:
        return 0
    try:
        files = api.list_repo_files(repo_id=repo_id(), repo_type="dataset")
    except Exception as e:
        print(f"⚠️ HF storage: list failed ({type(e).__name__}: {e})")
        return 0
    removed = 0
    for path in [f for f in files if f.startswith(prefix)]:
        try:
            api.delete_file(path_in_repo=path, repo_id=repo_id(),
                            repo_type="dataset")
            removed += 1
        except Exception as e:
            print(f"⚠️ HF storage: delete of {path} failed "
                  f"({type(e).__name__}: {e})")
    return removed
