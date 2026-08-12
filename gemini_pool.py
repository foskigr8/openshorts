"""Round-robin pool of Gemini API keys for parallel pipeline calls.

Spreads the picker's planning calls and the render-time scene-direction
calls (main.py's analyze_scene_context) across multiple keys so a single
key's rate limit doesn't serialize the whole job. A key that errors during
a call is assumed rate-limited/broken and gets skipped for the rest of the
run rather than retried immediately — the caller decides what to do about
the failed call itself.
"""
import itertools
import os
import threading


class GeminiKeyPool:
    def __init__(self, keys):
        self._keys = list(dict.fromkeys(k for k in (keys or []) if k))  # de-dup, preserve order
        self._bad = set()
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(range(len(self._keys))) if self._keys else None

    def __bool__(self):
        return bool(self._keys)

    def __len__(self):
        return len(self._keys)

    def acquire(self):
        """Returns the next usable key, round-robin, skipping any marked bad.
        None if the pool is empty or every key has been marked bad."""
        if not self._keys:
            return None
        with self._lock:
            for _ in range(len(self._keys)):
                idx = next(self._cycle)
                key = self._keys[idx]
                if key not in self._bad:
                    return key
            return None

    def mark_bad(self, key):
        if key is None:
            return
        with self._lock:
            self._bad.add(key)


def pool_from_env():
    """Builds a pool from GEMINI_API_KEY (primary) + GEMINI_API_KEYS (extra,
    comma-separated — the Settings "extra keys" list, joined by app.py before
    it reaches this subprocess)."""
    primary = os.environ.get("GEMINI_API_KEY")
    extra = [k.strip() for k in (os.environ.get("GEMINI_API_KEYS") or "").split(",") if k.strip()]
    keys = ([primary] if primary else []) + extra
    return GeminiKeyPool(keys)


def fallback_model_names():
    """Fallback Gemini models used when the primary hits a transient failure
    (quota/overload/high-demand 503), tried in order. GEMINI_FALLBACK_MODEL
    wins (comma-separated = a chain); the default is gemini-3.5-flash —
    when gemini-3.1-flash-lite is high-demand, retrying the SAME model just
    keeps hitting the wall. gemini-2.5-flash was the old default until
    Google retired it (404 'no longer available to new users', 12-aug-2026),
    which killed every fallback attempt — the chain now skips models that
    report NOT_FOUND instead of dying on them."""
    raw = os.environ.get("GEMINI_FALLBACK_MODEL", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return ["gemini-3.5-flash"]


def fallback_model_name():
    """First fallback model (backward-compatible single-name accessor)."""
    names = fallback_model_names()
    return names[0] if names else None


TRANSIENT_TOKENS = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "quota",
    "rate limit", "rate_limit", "500", "INTERNAL", "overloaded", "Deadline",
)


def is_transient_error(exc) -> bool:
    """True when a Gemini call failed for a retryable/rate-limit reason the
    fallback model could plausibly dodge (429 quota, 503 overload, 5xx)."""
    msg = str(exc).lower()
    return any(tok.lower() in msg for tok in TRANSIENT_TOKENS)


def _model_unavailable(exc) -> bool:
    """True when the MODEL itself is gone — 404 NOT_FOUND / "no longer
    available" / "model not found". A retired fallback will never succeed,
    so it must be SKIPPED (try the next in the chain), never treated as a
    fatal error or retried into the sand."""
    msg = str(exc).lower()
    return ("not_found" in msg or "404" in msg
            or "no longer available" in msg or "model not found" in msg)


def generate_with_fallback(client, model_name, contents, config=None,
                           max_attempts=1, log=print):
    """One generate_content call that switches models on transient failure.

    Tries `model_name` up to max_attempts; if every attempt failed with a
    transient error, moves down the fallback chain (GEMINI_FALLBACK_MODEL,
    comma-separated) with max_attempts each. A model that reports itself
    unavailable (404/retired) is skipped immediately. Non-transient errors
    (policy blocks, schema errors) propagate immediately — retrying those
    only burns quota. Returns the first successful response, else raises
    the last exception.
    """
    import time
    models = [model_name]
    for fb in fallback_model_names():
        if fb and fb != model_name and fb not in models:
            models.append(fb)
    last_exc = None
    for mi, model in enumerate(models):
        for attempt in range(1, max_attempts + 1):
            try:
                return client.models.generate_content(
                    model=model, contents=contents, config=config)
            except Exception as exc:
                last_exc = exc
                if _model_unavailable(exc):
                    if log:
                        log(f"⚠️ Model {model} unavailable "
                            f"({str(exc)[:120]}) — skipping to the next model")
                    break  # this model is gone; move down the chain
                if not is_transient_error(exc):
                    raise
                more_models = mi < len(models) - 1
                if not more_models and attempt == max_attempts:
                    continue  # exhausted everything — raise last_exc below
                wait = 5 * (2 ** (attempt - 1))
                if more_models and attempt == max_attempts:
                    log(f"⚠️ Gemini {model} transient failure after {attempt} attempt(s) "
                        f"({str(exc)[:120]}); switching to {models[mi + 1]}")
                else:
                    log(f"⚠️ Gemini transient error on {model} "
                        f"(attempt {attempt}/{max_attempts}); retrying in {wait}s: {str(exc)[:120]}")
                time.sleep(wait)
    raise last_exc
