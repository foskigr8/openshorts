#!/usr/bin/env python3
"""Which yt-dlp strategy actually downloads from THIS host?

The answer differs by network. This studio's IP is flagged by YouTube and needs
a cookie jar; Kaggle's may not be, and a datacenter IP can fail in a different
way (SABR-only formats, bot wall on the media request but not on metadata).
Guessing wastes a session, so this tries each strategy end to end — a real 5
second download, not a metadata fetch — and reports which ones work.

    python3 kaggle_download_probe.py [URL]

Run it on the host that will do the downloading.
"""
import os
import subprocess
import sys
import tempfile
import time

URL = sys.argv[1] if len(sys.argv) > 1 else "https://youtu.be/ua9Z0Lq3QVA"
POT = os.environ.get("BGUTIL_BASE_URL", "").strip()
COOKIES = next((p for p in ("cookies.txt", "/app/cookies.txt", "/kaggle/working/openshorts/cookies.txt")
                if os.path.isfile(p) and os.path.getsize(p) > 0), None)

FMT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]/b"


def strategy(name, extra):
    """Returns (ok, detail). A strategy only counts if BYTES land on disk."""
    out = os.path.join(tempfile.mkdtemp(), "probe.%(ext)s")
    cmd = ["yt-dlp", "--no-warnings", "--download-sections", "*0-5",
           "--force-keyframes-at-cuts", "-f", FMT, "-o", out]
    if POT:
        cmd += ["--extractor-args", f"youtubepot-bgutilhttp:base_url={POT}"]
    cmd += extra + [URL]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return False, "timed out after 180s"
    got = [f for f in os.listdir(os.path.dirname(out)) if f.startswith("probe.")]
    if got:
        size = os.path.getsize(os.path.join(os.path.dirname(out), got[0]))
        if size > 10_000:
            return True, f"{size/1e6:.1f} MB in {time.time()-t0:.0f}s"
        return False, f"file was only {size} bytes"
    err = (r.stderr or "").strip().splitlines()
    tail = err[-1][:150] if err else f"exit {r.returncode}, no output"
    return False, tail


print(f"URL     : {URL}")
print(f"POT     : {POT or 'NOT SET (no PO token — expect more bot walls)'}")
print(f"cookies : {COOKIES or 'none on disk'}\n")

# Ordered cheapest/most-likely-first. Anonymous is listed first deliberately:
# if the host's IP is clean, no cookie management is needed at all, which is by
# far the most robust setup — nothing to expire.
tests = [
    ("anonymous (default clients)", []),
    ("anonymous + ios client",
     ["--extractor-args", "youtube:player_client=ios,android,web"]),
    ("anonymous + tv_embed",
     ["--extractor-args", "youtube:player_client=tv_embed,android,mweb,web"]),
    ("anonymous + web_safari + fetch_pot",
     ["--extractor-args", "youtube:player_client=web_safari,tv;fetch_pot=always"]),
]
if COOKIES:
    tests += [
        (f"cookies ({COOKIES})", ["--cookies", COOKIES]),
        ("cookies + web_safari + fetch_pot",
         ["--cookies", COOKIES,
          "--extractor-args", "youtube:player_client=web_safari,tv;fetch_pot=always"]),
    ]

winners = []
for name, extra in tests:
    ok, detail = strategy(name, extra)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if ok:
        winners.append(name)

print()
if winners:
    print(f"WORKS HERE: {winners[0]}")
    if winners[0].startswith("anonymous"):
        print("  This host does not need cookies at all — the most robust outcome,")
        print("  since there is nothing to expire. Prefer this over a cookie jar.")
else:
    print("NOTHING WORKED. This host cannot fetch YouTube media directly.")
    print("  Fall back to uploading the source instead of downloading it:")
    print("   * attach it as a Kaggle Dataset and point the job at /kaggle/input/...")
    print("   * or use the dashboard's file-upload path, which never touches yt-dlp")
