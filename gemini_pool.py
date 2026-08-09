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


def fallback_model_name():
    """Secondary Gemini model used when the primary hits a transient failure
    (quota/overload). GEMINI_FALLBACK_MODEL, else the primary GEMINI_MODEL
    (no-op fallback), else the pipeline default."""
    return (os.environ.get("GEMINI_FALLBACK_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or "gemini-3.1-flash-lite")


TRANSIENT_TOKENS = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "quota",
    "rate limit", "rate_limit", "500", "INTERNAL", "overloaded", "Deadline",
)


def is_transient_error(exc) -> bool:
    """True when a Gemini call failed for a retryable/rate-limit reason the
    fallback model could plausibly dodge (429 quota, 503 overload, 5xx)."""
    msg = str(exc).lower()
    return any(tok.lower() in msg for tok in TRANSIENT_TOKENS)


def generate_with_fallback(client, model_name, contents, config=None,
                           max_attempts=1, log=print):
    """One generate_content call that switches models on transient failure.

    Tries `model_name` up to max_attempts; if every attempt failed with a
    transient error and a different fallback model is configured, retries up
    to max_attempts more on the fallback. Non-transient errors (policy
    blocks, schema errors, 404s) propagate immediately — retrying those only
    burns quota. Returns the first successful response, else raises the last
    exception.
    """
    import time
    models = [model_name]
    fb = fallback_model_name()
    if fb and fb != model_name:
        models.append(fb)
    last_exc = None
    for mi, model in enumerate(models):
        for attempt in range(1, max_attempts + 1):
            try:
                return client.models.generate_content(
                    model=model, contents=contents, config=config)
            except Exception as exc:
                last_exc = exc
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
