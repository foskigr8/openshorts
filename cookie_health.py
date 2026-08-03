#!/usr/bin/env python3
"""Report whether .env currently holds a USABLE YouTube cookie jar.

Exists because the failure that kept biting was silent: the refresher would
die (or export a half-session), .env would keep whatever stale cookies it had,
and nothing surfaced the problem until a job failed minutes later with
"Sign in to confirm you're not a bot".

Exit codes:
  0  jar is complete (all core SSO cookies present)
  1  jar is missing / incomplete  -> a browser re-login is needed
"""
import os
import re
import sys

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
COOKIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")

# The Google SSO set yt-dlp needs to present a real authenticated session.
# A jar with only YouTube's visitor/consent cookies (or only the third-party
# __Secure-3PSID slot) reads to YouTube as a broken login and is treated MORE
# harshly than an anonymous request.
CORE = {"SID", "__Secure-1PSID"}
IDENT = {"SAPISID", "__Secure-1PAPISID", "LOGIN_INFO"}


def jar_names(env_path=ENV_PATH):
    try:
        with open(env_path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return set()
    m = re.search(r'YOUTUBE_COOKIES="([^"]*)"', content, re.DOTALL)
    if not m:
        return set()
    names = set()
    for line in m.group(1).split("\n"):
        parts = line.split("\t")
        if len(parts) >= 7:
            names.add(parts[5])
    return names


def main():
    # The LIVE jar is cookies.txt (main.py reads the file, not the env); check
    # it first and fall back to .env for legacy setups.
    names = set()
    try:
        with open(COOKIES_FILE, encoding="utf-8") as f:
            for line in f:
                parts = line.split("\t")
                if len(parts) >= 7:
                    names.add(parts[5])
    except OSError:
        names = jar_names()
    if not names:
        print("COOKIES: MISSING — no usable cookie jar found")
        return 1
    core = names & CORE
    if core and "LOGIN_INFO" in names:
        print(f"COOKIES: OK — {len(names)} cookies, "
              f"core={sorted(core)} LOGIN_INFO=present")
        return 0
    print(f"COOKIES: INCOMPLETE — {len(names)} cookies but missing "
          f"{'core login (SID/__Secure-1PSID)' if not core else 'LOGIN_INFO'}. "
          "YouTube will treat this as a bot.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
