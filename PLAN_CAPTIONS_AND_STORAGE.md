# Plan: per-job caption placement + HuggingFace persistent storage

> **STATUS (6-aug-2026): both parts implemented.** Part 1 follows the
> `captions`/`zoom_mode` pattern exactly (`MediaInput.jsx` → `/api/process` →
> `CAPTION_POSITION`/`CAPTION_MARGIN_V` → `subtitles.AUTO_CAPTION_STYLE`).
> Part 2 is `hf_storage.py` + per-clip upload from `app.py`'s poll loop +
> `/api/storage/{job_id}/{filename}` for restore. Notes on where the
> implementation differs from the text below:
>
> - The history fallback emits `/api/storage/...`, not a raw HF URL: the repo is
>   **private**, so its resolve URL needs an Authorization header and cannot be
>   put in a browser `src`. The server holds the token and streams the file,
>   restoring it to disk so the next request is served locally.
> - `_clamp_number` had to move ABOVE `AUTO_CAPTION_STYLE` in `subtitles.py`
>   (it is now used at import time). Still one clamp, as instructed.
>
> Not yet done, and it needs a real run: verification steps 3 and 4 (clips
> appearing in the HF repo mid-job, and History surviving a session restart).
> Tests cover the logic with a stubbed `HfApi`; they cannot prove the round trip.

**Audience:** implementing engineer (DeepSeek).
**Repo:** `github.com/foskigr8/openshorts`, branch `session/framing-work`.
If you lack repo access, ask the owner to add you — every change is in this repo.
Read `KAGGLE.md` and `KAGGLE_OPTIMIZATION_PLAN.md` first; `HANDOFF_FRAMING.md`
§2e lists measurement traps that have already cost this project real time.

Baseline: **711 tests passing**
(`docker exec openshorts-backend sh -c 'cd /app && python3 -m pytest tests/ -q'`).
A drop is a regression you caused.

---

# PART 1 — Caption placement chosen in the UI, before processing

## Context

Captions currently burn at a fixed position. `CAPTION_POSITION` was added as an
env var, but that is deployment-wide and the owner needs it **per job**: for
talking-head clips captions belong near the bottom *but lifted slightly off the
edge*; for other footage middle reads better. Choosing after the fact via the
Subtitle modal means re-encoding every clip.

Two separate controls are needed, and they are different things:

- **position** — `top` / `middle` / `bottom`. Already supported end to end:
  `align_map = {'top': 8, 'middle': 5, 'bottom': 2}` in `subtitles.generate_ass`
  (~line 478), and the Subtitle modal already offers all three
  (`dashboard/src/components/SubtitleModal.jsx:41` `POSITION_OPTIONS`).
- **how far up from the edge** — this is ASS `MarginV`, *not* alignment.
  `SAFE_MARGIN_V = 43` (`subtitles.py:272`) is the current default and
  `generate_ass` already takes a `margin_v` argument, clamped `0..200`
  (see the style line built around `subtitles.py:579`). "Bottom but slightly
  up" = bottom alignment with a larger `margin_v`.

## The pattern to copy

`captions`, `zoom_mode` and `style_variant` already flow UI → API → job. Follow
them exactly; do not invent a new mechanism.

| layer | file:line | what to add |
|---|---|---|
| UI state + submit | `dashboard/src/App.jsx:845-847` (JSON) and `861-863` (FormData) | send `caption_position` and `caption_margin` beside `captions` |
| API parse | `app.py:1583-1589` (where `captions_on`, `zoom_mode`, `style_variant` are normalised) | parse + validate both; also read from `body.get(...)` near `app.py:1625` for the JSON path |
| API → job | `app.py:1774-1776` (`env["AUTO_CAPTIONS"]`, `env["SCENE_STRATEGY_OVERRIDE"]`) | `env["CAPTION_POSITION"]`, `env["CAPTION_MARGIN_V"]` |
| consumer | `subtitles.py:300-310` `AUTO_CAPTION_STYLE` | `alignment` already reads `CAPTION_POSITION`; add `margin_v` reading `CAPTION_MARGIN_V`, defaulting to `SAFE_MARGIN_V` |

Validation rules: position must be one of `top|middle|bottom`, anything else
falls back to `bottom`. Margin clamps to `0..200` — `generate_ass` already
clamps via `_clamp_number`, so reuse that rather than adding a second clamp.

## UI

Put the control in the **job submission form**, next to the existing captions
toggle, and only enable it when captions are on. Reuse `SegmentedControl` and
`POSITION_OPTIONS` from `SubtitleModal.jsx:41` — export them rather than
duplicating the list.

For the margin, a 3-step choice is friendlier than a pixel slider and matches
how the owner described the problem:

| label | alignment | margin_v |
|---|---|---|
| Bottom | `bottom` | `43` (`SAFE_MARGIN_V`) |
| Bottom, raised | `bottom` | `~120` |
| Middle | `middle` | n/a (MarginV is ignored for centre alignment) |

**Important:** per-line `MarginV` override is only meaningful for bottom
alignment — `subtitles.py:617-619` computes `line_margin_v` only when
`ass_alignment == 2`. Do not expose a margin control for middle/top; it will
appear to do nothing.

## Verification

1. Full suite green (711+).
2. New tests in `tests/`: `generate_ass` emits `Alignment,...,MarginV` matching
   each combination; an invalid position falls back to bottom; margin clamps.
3. End to end: submit three jobs (bottom / bottom-raised / middle) and confirm
   the burned captions differ. **Watch the output** — do not judge from the
   `.ass` file alone.

---

# PART 2 — Persistent storage on HuggingFace Hub

## Context

Kaggle wipes `/kaggle/working` at session end (12h cap), so clips evaporate.

**Decision: HuggingFace Hub, not Cloudflare R2.** An earlier draft of this plan
recommended R2 because `cloud/storage.py` already speaks S3 — that reasoning
ignored the blocker: **Cloudflare requires a credit card.** HuggingFace does
not, and the owner **already has `HF_TOKEN` configured as a Kaggle secret**, so
the credentials exist today. Card-free alternatives considered and rejected:
Backblaze B2 (also wants payment details) and Kaggle Datasets (works, no new
account, but awkward to browse and tied to Kaggle).

Trade-off, stated honestly: this is *more* work than R2 would have been, because
`cloud/storage.py` cannot be reused — HF is not S3-compatible. It needs a new
uploader. That cost is worth paying to avoid a blocked signup.

## Owner setup steps (no code)

1. **huggingface.co** → sign up / log in (free, no card).
2. **Settings → Access Tokens → Create new token.**
   - Type: **Write** (read-only cannot upload)
   - Name it e.g. `openshorts-storage`
   - Copy it — shown once.
3. **Create a private Dataset repo:** huggingface.co → *New* → **Dataset**.
   - Name e.g. `openshorts-clips`
   - Visibility: **Private**
   - Full id is `<your-username>/openshorts-clips`
4. **Kaggle secrets** (Add-ons → Secrets) — create and **tick the checkbox** so
   they attach to the notebook:
   - `HF_TOKEN` — the write token (may already exist; confirm it is a *write*
     token, not read)
   - `HF_STORAGE_REPO` — `<your-username>/openshorts-clips`
5. Optionally add both to the local `.env` to use the same store from the studio.

Free tier is generous for this (well beyond R2's 10 GB) and egress is free.

## Implementation

### New module `hf_storage.py`

Mirror the shape of `cloud/storage.py` so the two are recognisable as siblings:

```python
upload_file(local_path, key) -> str | None   # returns the repo path
download_file(key, local_path) -> bool
public_url(key) -> str                        # resolve/<repo>/<key>
configured() -> bool                          # bool(HF_TOKEN and HF_STORAGE_REPO)
```

Use `huggingface_hub.HfApi.upload_file(path_or_fileobj=..., path_in_repo=...,
repo_id=..., repo_type="dataset")`. Add `huggingface_hub` to
`requirements.txt` (it is likely already present transitively via
faster-whisper — check before adding).

Key layout: `jobs/<job_id>/<filename>` — keeps a job's clips together and makes
`delete` by prefix straightforward later.

**Fail soft.** If `configured()` is false, every function is a no-op returning
`None`/`False`. A self-host without HF must behave exactly as today.

### Upload per clip, not per job — the load-bearing part

`app.py:1391` currently calls `upload_job_artifacts(output_dir, job_id)` **after
the whole `main.py` subprocess exits**, on a `run_in_executor` that is never
awaited. On Kaggle that is wrong twice: a session dying mid-job uploads nothing,
and a failed upload is invisible because the job is already marked `completed`
at `app.py:1385`.

Upload **as each clip finishes**. `main.py` already writes a `.ready` marker per
clip and logs `✅ Clip N ready` — hook there, or watch for the marker in
`app.py`'s poll loop. **Log every success and failure**; the current path logs
neither, and `s3_uploader.py` pins boto's loggers to `CRITICAL` (lines 9-11) so
even library warnings vanish. Do not repeat that.

### Make clips retrievable after the session dies

Nothing records a storage key today: `upload_job_artifacts` returns `None`, and
`main.py:3227-3245` writes the metadata JSON *before* any upload with no storage
field. `/api/history` (`app.py:1948+`) reads local disk only.

- Write the HF key into the job metadata as each clip uploads. That write is
  already atomic (`_meta_tmp` + `os.replace`) — preserve that.
- In `/api/history`, when the local file is **missing** but an HF key exists,
  emit the HF URL instead of the local `/videos/...` path.

This is what turns "backup" into persistence: History keeps working after
`/kaggle/working` is wiped.

### Surface state

- `GET /api/system`: add `storage: {"provider": "huggingface", "configured": true}`
  — **presence only, never the token**.
- `kaggle_bootstrap.sh`: add a line to the Configuration block, mirroring the
  existing `GEMINI_API_KEYS` / `ASSEMBLYAI_API_KEY` reporting.

## Files to touch

| file | change |
|---|---|
| `hf_storage.py` | **new** — the uploader |
| `app.py` | per-clip upload; HF URL fallback in `/api/history`; status in `/api/system` |
| `main.py` | record the HF key per clip in the metadata JSON |
| `kaggle_bootstrap.sh` | report storage configured/unset |
| `requirements.txt` | `huggingface_hub` if not already present |
| `tests/` | first tests for this path — there are currently **none** for storage |

## Verification

1. Full suite green (711+).
2. Unit tests with a stubbed `HfApi`: key layout, no-op when unconfigured,
   presigned/public URL fallback when the local file is absent, and that a
   failed upload does **not** fail the job.
3. On Kaggle: run a job and confirm clips appear in the HF dataset repo **while
   the job is still running**, not only at the end.
4. **The real test:** stop the Kaggle session, start a fresh one, open the
   dashboard, and confirm History still lists the clips and they play.
5. Confirm a run with `HF_TOKEN`/`HF_STORAGE_REPO` unset behaves exactly as
   today.

---

# Order of work

1. **Part 1** (caption placement) — self-contained, follows an existing pattern,
   immediately useful.
2. **Part 2** (HF storage) — bigger, and the per-clip upload restructure is the
   risky half. Do the `hf_storage.py` module and its tests first, wire the
   per-clip hook second, and the `/api/history` fallback last.

# Rules

- **Verify by running, not by reading.** Every "measured" claim in these docs
  came from a real run; match that standard.
- **Watch the video** before declaring caption placement good.
- Do not commit secrets. `cookies.txt` and `.env` are gitignored; keep it so.
  Never print a token or a clone URL containing one.
