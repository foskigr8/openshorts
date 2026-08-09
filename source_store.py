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
    meta_path = os.path.join(d, "meta.json")
    tmp = meta_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"url": url, "title": title or "", "filename": filename},
                  f, ensure_ascii=False)
    os.replace(tmp, meta_path)


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
