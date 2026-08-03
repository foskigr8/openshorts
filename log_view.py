"""Log curation for the job log view.

Self-hosted instances see the raw pipeline output (useful for debugging).
Cloud/paying users get a curated, whitelist-based view: friendly progress
lines only — no file paths, model names, encoder details or pipeline
internals. Anything not matched by a rule is hidden.
"""
import re
import time

# Strips any slashy path token (output/…/clip.mp4, /app/uploads/x.mp4, …).
_PATH_RE = re.compile(r'(?:/?[\w.\-]+/)+[\w.\-]+')


def _strip_paths(line):
    return _PATH_RE.sub('', line).rstrip(' :')


# Ordered rules; first match wins. Replacement is a template using the
# match's groups, or None to keep the (path-stripped) line verbatim.
_RULES = [
    # Worker/job lifecycle + errors: keep, minus any paths.
    (re.compile(r'^(Job started|Process finished|Process failed|'
                r'Execution error|No metadata|❌)'), None),
    # Live transcription progress emitted by transcribe_backends.
    (re.compile(r'^🎙️ Transcribing… \d+%'), None),
    (re.compile(r'Transcribing (video|audio)'), '🎙️ Transcribing audio…'),
    (re.compile(r'Found (\d+) viral clips'), '🔥 Found {0} viral clips!'),
    (re.compile(r'Processing Clip (\d+)'), '🎬 Creating clip {0}…'),
    (re.compile(r'Clip (\d+) ready'), '✅ Clip {0} ready'),
]


def friendly_log_line(line):
    """Map one raw log line to its cloud-visible form, or None to hide it."""
    text = line.get("text") if isinstance(line, dict) else line
    stripped = str(text or "").strip()
    if not stripped:
        return None
    for pattern, template in _RULES:
        match = pattern.search(stripped)
        if match:
            if template is None:
                return _strip_paths(stripped)
            return template.format(*match.groups())
    return None


def friendly_logs(logs):
    """Curated log list for cloud users, consecutive duplicates collapsed."""
    out = []
    for line in logs:
        friendly = friendly_log_line(line)
        if friendly and (not out or out[-1] != friendly):
            out.append(friendly)
    return out


# --- Self-host plain view (round 3, item 3) --------------------------------
# Self-host previously dumped the full unfiltered subprocess stream by design,
# which is how a raw signed yt-dlp CDN URL or a base64 blob ended up as one
# giant unwrapped line in the UI. These rules drop only obviously-noisy lines;
# everything else stays visible for debugging. An explicit raw toggle opts
# back into the fully unfiltered stream.
_NOISE_URL_RE = re.compile(r'^https?://\S{120,}$')
_NOISE_B64_RE = re.compile(r'^[A-Za-z0-9+/=\-_]{140,}$')
_NOISE_YTDLP_PREFIXES = (
    '[debug]', '[youtube] Extracting URL', '[youtube] Identified',
    '[info] ', '[Merger] ', '[FixupMpegTS] ', '[Metadata] ', '[VideoID] ',
    '[ExtractAudio] ', '[VideoConvertor] ', '[downloader] ',
    '[download] Destination:', '[download] Downloading item',
)
_KEEP_YTDLP_PROGRESS = re.compile(r'^\[download\]\s+\d+(\.\d+)?%')

# Round-4 teardown: the self-host default view must narrate the pipeline for
# the person LOOKING at the screen, not dump a machine's diary. These patterns
# map internal lines to a human sentence (or hide them outright when they have
# no user-facing meaning — e.g. dependency version banners). The raw toggle
# still shows the unfiltered original.
_FRIENDLY_TRANSLATIONS = [
    # The single highest-liability string on the screen ("spoof" reads as
    # deception to a non-engineer) — translate, never show verbatim.
    (re.compile(r'^Download attempt: ios-spoof$'), 'Fetching your video…'),
    (re.compile(r'^Download attempt: (.*)$'), 'Fetching your video ({0})…'),
    (re.compile(r'^Found YOUTUBE_COOKIES env var.*$'), None),
    (re.compile(r'^Debug: Cookies file created.*$'), None),
    (re.compile(r'^Debug: yt-dlp version.*$'), None),
    (re.compile(r'^INFO: Created TensorFlow Lite.*$'), None),
    (re.compile(r'^WARNING:?.*$'), None),  # dependency warnings, not job state
]


def plain_log_line(entry):
    """One log entry -> the user-facing line, or None when it's noise."""
    text = entry.get("text") if isinstance(entry, dict) else str(entry)
    line = str(text or "").strip()
    if not line:
        return None
    for pattern, template in _FRIENDLY_TRANSLATIONS:
        match = pattern.match(line)
        if match:
            if template is None:
                return None
            return template.format(*match.groups())
    if _NOISE_URL_RE.match(line) or _NOISE_B64_RE.match(line):
        return None
    if line.startswith(_NOISE_YTDLP_PREFIXES) and not _KEEP_YTDLP_PROGRESS.match(line):
        return None
    return line


def plain_logs(logs):
    """Self-host default view: real timestamps + noise-filtered lines."""
    out = []
    for entry in logs:
        line = plain_log_line(entry)
        if line is None:
            continue
        ts = entry.get("ts") if isinstance(entry, dict) else None
        out.append({"ts": float(ts) if ts is not None else time.time(), "text": line})
    return out
