#!/usr/bin/env python3
"""Is the Kaggle host ACTUALLY working, or did it merely start?

Lives in the repo rather than in the notebook on purpose: the notebook has to
be re-imported into Kaggle by hand every time it changes, so anything that
might be improved belongs on this side of `git pull`. The notebook is reduced
to secrets + clone + two shell calls (see openshorts_kaggle.ipynb).

Each check is independent and says which thing failed rather than "error".
Exit code is 0 when every check passes, 1 otherwise, so a notebook cell fails
visibly.

    python3 kaggle_smoke_test.py            # all checks
    python3 kaggle_smoke_test.py --skip-youtube
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request

BASE = os.environ.get("SMOKE_BASE_URL", "http://localhost:8000")
REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def api(path, timeout=15):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as r:
        return r.status, r.read()


def check(label, fn, results):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}", flush=True)
    results.append((label, ok))
    return ok


# --- individual checks ------------------------------------------------------


def _api_alive():
    return api("/api/system")[0] == 200, "/api/system 200"


def _dashboard():
    """Served by the SAME process as the API (single-origin mode)."""
    status, body = api("/")
    return (status == 200 and b'<div id="root"' in body,
            f"/ returned {len(body)} bytes of HTML")


def _cuda():
    """GPU reachable from the app's own imports, not just nvidia-smi."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import torch;print(torch.cuda.is_available(), torch.cuda.device_count())"],
        capture_output=True, text=True, cwd=REPO_DIR).stdout.strip()
    return out.startswith("True"), out


def _gpu_sharding():
    """Both T4s usable by clip workers. Informational on a 1-GPU host."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import gpu_affinity;print(gpu_affinity.available_devices())"],
        capture_output=True, text=True, cwd=REPO_DIR).stdout.strip()
    devices = json.loads(out or "[]")
    if not devices:
        return False, "no CUDA devices visible to clip workers"
    if len(devices) == 1:
        return True, f"1 GPU ({devices}) — clip workers will share it"
    return True, f"{len(devices)} GPUs {devices} — clip workers spread across them"


def _nvenc():
    """Without NVENC every render silently falls back to x264 (slow)."""
    out = subprocess.run(
        "ffmpeg -hide_banner -encoders 2>/dev/null | grep -c nvenc",
        shell=True, capture_output=True, text=True).stdout.strip()
    return out.isdigit() and int(out) > 0, f"{out} nvenc encoders"


def _transcription():
    """AssemblyAI (API + diarization) vs local whisper.

    Local whisper cost 254s of a ~420s job in the measured run AND produced no
    diarization, which is subject_policy's TIER_DIARIZED framing evidence.
    """
    # LAST line only: _select_backend() prints its own explanation to stdout
    # when it auto-selects, so the naive read captured the log line plus the
    # answer and matched neither.
    stdout = subprocess.run(
        [sys.executable, "-c",
         "import transcribe_backends as t;print(t._select_backend())"],
        capture_output=True, text=True, cwd=REPO_DIR).stdout.strip()
    out = stdout.splitlines()[-1].strip() if stdout else ""
    if out == "assemblyai":
        return True, "assemblyai (API, with diarization)"
    if os.environ.get("ASSEMBLYAI_API_KEY"):
        return False, f"key is set but backend resolved to '{out}'"
    return False, "local whisper — slow, and NO diarization (set ASSEMBLYAI_API_KEY)"


def _stage3():
    """The clip SELECTOR. A missing skill package or key does not crash
    anything — the job quietly produces weaker clips instead."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import os, viral_clip_finder as v;"
         "print(v.skill_available(), len(v._provider_candidates()),"
         " os.environ.get('VIRAL_ENGINE','auto'))"],
        capture_output=True, text=True, cwd=REPO_DIR)
    parts = (out.stdout or "").strip().split()
    if len(parts) != 3:
        return False, (out.stderr or out.stdout).strip()[-200:]
    have_skill, providers, engine = parts[0] == "True", int(parts[1]), parts[2]
    if not have_skill:
        return False, "viral_clip_finder_skill/ is missing from the clone"
    if providers == 0:
        return False, "no provider key (GEMINI_API_KEY / NARRATIVE_GEMINI_API_KEY)"
    return True, f"engine={engine}, skill package present, {providers} provider(s)"


def _storage():
    """Clips survive the session ending only if HF storage is configured."""
    status, body = api("/api/system")
    backend = json.loads(body).get("storage_backend") or {}
    if backend.get("configured"):
        return True, f"huggingface -> {backend.get('repo')}"
    return False, ("not configured — clips are wiped with /kaggle/working when "
                   "the session ends (set HF_TOKEN + HF_STORAGE_REPO)")


def _server_keys():
    """The dashboard should not ask for a key the server already has."""
    status, body = api("/api/system")
    keys = json.loads(body).get("server_keys") or {}
    if keys.get("gemini"):
        return True, "gemini key present server-side — the UI will not prompt"
    return False, "no server-side Gemini key; the dashboard will demand one"


def _youtube():
    """FUNCTIONAL cookie check.

    cookie_health.py only checks that cookie NAMES exist and reports OK for a
    jar YouTube rejects, so it is deliberately not used here.
    """
    # `python -m yt_dlp`, not the console script: the script is not always on
    # PATH (it was missing on the Kaggle host even with the package installed).
    cmd = [sys.executable, "-m", "yt_dlp", "--skip-download",
           "--print", "%(title)s", "https://youtu.be/ua9Z0Lq3QVA"]
    have_cookies = os.path.exists(os.path.join(REPO_DIR, "cookies.txt"))
    if have_cookies:
        cmd[3:3] = ["--cookies", "cookies.txt"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                       cwd=REPO_DIR)
    title = (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else ""
    if title:
        how = "with cookies" if have_cookies else "anonymously (no cookie jar needed)"
        return True, f"{title} — fetched {how}"
    # Measured 5-aug-2026: Kaggle's IP is not flagged and downloads fine with no
    # cookies at all, so a missing jar is only worth reporting if the fetch
    # ALSO failed — which is the case here.
    detail = (r.stderr or "").strip()[-200:]
    if not have_cookies:
        detail += " (no cookies.txt; paste YOUTUBE_COOKIES if the IP is flagged)"
    return False, detail


def _face_id():
    """Named-speaker enrichment. Dormant by design unless FACE_ID_DB is set —
    reported either way so "I configured it and nothing happened" is visible
    here rather than inferred from a job that silently used anonymous labels."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import face_id, os;"
         "print(face_id.available(), face_id.default_ctx_id(),"
         " bool(os.environ.get('FACE_ID_DB')))"],
        capture_output=True, text=True, cwd=REPO_DIR)
    parts = (out.stdout or "").strip().split()
    if len(parts) != 3:
        return False, (out.stderr or out.stdout).strip()[-200:]
    ready, ctx, configured = parts[0] == "True", parts[1], parts[2] == "True"
    if not configured:
        return True, "off (no FACE_ID_DB) — speakers stay anonymous, which is fine"
    if not ready:
        return False, ("FACE_ID_DB is set but insightface is not usable — "
                       "speakers will stay anonymous")
    return True, f"active, running on GPU {ctx}"


def _po_token():
    """The PO token provider — what stops YouTube's bot wall on a cloud IP.

    Measured 6-aug-2026: without it, every download strategy failed with "Sign
    in to confirm you're not a bot" from a clean Kaggle IP and no cookies.
    """
    base = os.environ.get("BGUTIL_BASE_URL", "").strip()
    if not base:
        try:
            req = urllib.request.Request("http://127.0.0.1:4416/ping", method="GET")
            with urllib.request.urlopen(req, timeout=2) as r:
                if r.status == 200:
                    base = "http://127.0.0.1:4416"
                    os.environ["BGUTIL_BASE_URL"] = base
        except Exception:
            pass
    if not base:
        return False, ("no BGUTIL_BASE_URL — downloads run with no PO token and "
                       "will likely hit the bot wall")
    try:
        with urllib.request.urlopen(f"{base}/ping", timeout=5) as r:
            if r.status == 200:
                return True, f"provider responding at {base}"
            return False, f"{base}/ping returned {r.status}"
    except Exception as e:
        return False, f"{base} unreachable ({type(e).__name__})"


CHECKS = [
    ("API", _api_alive),
    ("Dashboard", _dashboard),
    ("CUDA in app env", _cuda),
    ("GPU sharding", _gpu_sharding),
    ("NVENC", _nvenc),
    ("Transcription backend", _transcription),
    ("Stage 3 clip selection", _stage3),
    ("Server-side keys", _server_keys),
    ("Face ID", _face_id),
    ("Persistent storage", _storage),
    ("PO token provider", _po_token),
    ("YouTube download", _youtube),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-youtube", action="store_true",
                        help="skip the cookie check (it is the slowest)")
    args = parser.parse_args()

    results = []
    for label, fn in CHECKS:
        if args.skip_youtube and label == "YouTube download":
            continue
        check(label, fn, results)

    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} passed")
    failed = [label for label, ok in results if not ok]
    if failed:
        print("\nFailed: " + ", ".join(failed))
        if "YouTube download" in failed:
            if not os.path.exists(os.path.join(REPO_DIR, "cookies.txt")):
                print("  YouTube: no cookies.txt. Kaggle usually downloads fine "
                      "without one — if this failed for another reason, read the "
                      "detail above before adding YOUTUBE_COOKIES.")
            else:
                print("  YouTube: the jar expired (~3h lifetime). Re-paste "
                      "YOUTUBE_COOKIES from a fresh local cookies.txt.")
        if "PO token provider" in failed:
            print("  PO token: this is the usual cause of a failed YouTube "
                  "download from a cloud IP. Re-run kaggle_bootstrap.sh and read "
                  "its 'YouTube PO token provider' section, or check "
                  "/tmp/openshorts-logs/bgutil.log.")
        if "Persistent storage" in failed:
            print("  Storage: see PLAN_CAPTIONS_AND_STORAGE.md for the "
                  "HF_TOKEN / HF_STORAGE_REPO setup (no credit card needed).")
        if "Transcription backend" in failed:
            print("  Transcription: add ASSEMBLYAI_API_KEY as a Kaggle secret "
                  "to get diarization and drop ~200s from every job.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
