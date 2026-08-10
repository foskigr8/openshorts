"""Per-source persistent store: reuse the source video, transcript and
context blob for a URL that has already been processed.

Keyed by the YouTube video ID when one is in the URL (``ua9Z0Lq3QVA``), else
a SHA-1 of the URL. Layout under ``SOURCE_CACHE_DIR`` (default: ``sources/``
next to the repo — point it at a mounted Kaggle Dataset for cross-session
persistence):

    sources/<key>/meta.json       {"url", "title", "filename"}
    sources/<key>/<filename>      the downloaded source video (hardlinked)
    sources/<key>/transcript.json the transcript (never changes per source)
    sources/<key>/context.json    the Gemini context blob

main.py checks the store BEFORE downloading / transcribing / context-calling:
a re-run of a seen project skips the download (saves time + bandwidth) and
the transcription (saves AssemblyAI API calls). The transcript for a source
never changes, so re-spending API calls on it is pure waste.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess

_YT_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?[^#\s]*v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{6,})")


def video_id_from_url(url):
    """YouTube video ID from a watch/shorts/embed/youtu.be URL, else None."""
    if not url:
        return None
    m = _YT_ID_RE.search(url)
    return m.group(1) if m else None


def source_key(url):
    """Stable cache key: the YouTube video ID, else a URL hash."""
    vid = video_id_from_url(url)
    if vid:
        return vid
    return hashlib.sha1((url or "").encode("utf-8")).hexdigest()[:16]


def cache_dir():
    return os.environ.get("SOURCE_CACHE_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sources")


def _key_dir(url):
    return os.path.join(cache_dir(), source_key(url))


def _read_json(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _link_file(src, dst):
    """Hardlink with cross-device fallback — a 3.6GiB source must never be
    copied per job."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def lookup(url):
    """Return cached artifacts for ``url``, or None if the source was never
    stored. Keys: video_path, title, filename, transcript, context_blob —
    transcript/context may be None if only the source was saved."""
    if not url:
        return None
    d = _key_dir(url)
    meta = _read_json(os.path.join(d, "meta.json"))
    if not meta:
        return None
    filename = str(meta.get("filename") or "")
    video_path = os.path.join(d, filename) if filename else ""
    if not video_path or not os.path.exists(video_path):
        return None
    # AV1 sources are never reusable: Turing's NVDEC has no AV1 hardware
    # decoder, so an AV1 file would silently force CPU decode everywhere.
    # Treat them as a cache miss so the fixed download policy re-fetches a
    # VP9/H.264 source. ffprobe reports codec_name "av1"; yt-dlp's format tag
    # is "av01" — match both.
    if str(meta.get("codec") or "").lower().startswith(("av1", "av01")):
        return None
    return {
        "video_path": video_path,
        "title": str(meta.get("title") or ""),
        "filename": filename,
        "transcript": _read_json(os.path.join(d, "transcript.json")),
        "context_blob": _read_json(os.path.join(d, "context.json")),
    }


def save_source(url, video_path, title):
    """Store the source video (hardlinked) + title metadata. Safe to call
    right after download — later saves add transcript/context on top."""
    if not url or not video_path or not os.path.exists(video_path):
        return
    d = _key_dir(url)
    os.makedirs(d, exist_ok=True)
    filename = os.path.basename(video_path)
    _link_file(video_path, os.path.join(d, filename))
    codec = _probe_codec(video_path)
    meta_path = os.path.join(d, "meta.json")
    tmp = meta_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"url": url, "title": title or "", "filename": filename,
                   "codec": codec},
                  f, ensure_ascii=False)
    os.replace(tmp, meta_path)


def _probe_codec(video_path):
    """Best-effort video codec name (e.g. vp09/avc1/av01) via ffprobe."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def save_transcript(url, transcript):
    if not url or not transcript:
        return
    d = _key_dir(url)
    if not os.path.isdir(d):
        return
    path = os.path.join(d, "transcript.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False)
    os.replace(tmp, path)


def save_context(url, context_blob):
    if not url or not context_blob:
        return
    d = _key_dir(url)
    if not os.path.isdir(d):
        return
    path = os.path.join(d, "context.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(context_blob, f, ensure_ascii=False)
    os.replace(tmp, path)


def video_path_for_key(key):
    """Absolute path of the cached source video for a store key, or None."""
    if not key:
        return None
    d = os.path.join(cache_dir(), key)
    meta = _read_json(os.path.join(d, "meta.json"))
    if not meta:
        return None
    filename = str(meta.get("filename") or "")
    path = os.path.join(d, filename)
    return path if filename and os.path.exists(path) else None


def list_sources():
    """Every saved source, newest first — feeds the dashboard's Sources tab.

    Shape: {"key", "title", "filename", "url", "size_bytes", "updated_at",
    "has_transcript", "has_context"}. Entries without a stored video file are
    skipped (a partial write must never surface a broken card).
    """
    base = cache_dir()
    if not os.path.isdir(base):
        return []
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return []
    out = []
    for key in entries:
        d = os.path.join(base, key)
        if not os.path.isdir(d):
            continue
        meta = _read_json(os.path.join(d, "meta.json"))
        if not meta:
            continue
        filename = str(meta.get("filename") or "")
        video_path = os.path.join(d, filename)
        if not filename or not os.path.exists(video_path):
            continue
        try:
            size = os.path.getsize(video_path)
            mtime = os.path.getmtime(video_path)
        except OSError:
            continue
        out.append({
            "key": key,
            "title": str(meta.get("title") or key),
            "filename": filename,
            "url": str(meta.get("url") or ""),
            "size_bytes": size,
            "updated_at": mtime,
            "has_transcript": os.path.exists(os.path.join(d, "transcript.json")),
            "has_context": os.path.exists(os.path.join(d, "context.json")),
        })
    out.sort(key=lambda s: s["updated_at"], reverse=True)
    return out
