# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenShorts is an AI-powered vertical video generator that transforms long YouTube videos or local uploads into viral-ready short clips (9:16 format) for TikTok, Instagram Reels, and YouTube Shorts. Uses Google Gemini 2.0 Flash for viral moment detection and title generation.

## Development Commands

### Local Development (Docker)
```bash
docker compose up --build   # Build and run full stack
```
- Backend: http://localhost:8000 (FastAPI/Uvicorn)
- Frontend: http://localhost:5175 (Vite proxies API calls to backend)

### Frontend Only (Dashboard)
```bash
cd dashboard
npm install
npm run dev       # Dev server with HMR (port 5173)
npm run build     # Production build
npm run lint      # ESLint (strict, --max-warnings 0)
```

### Backend Only
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Architecture

### Core Processing Pipeline
1. **Ingest** - YouTube download (yt-dlp) or local upload
2. **Transcription** - faster-whisper with word-level timestamps
3. **Scene Detection** - PySceneDetect for segment boundaries
4. **AI Analysis** - Gemini identifies 3-15 viral moments (15-60 sec each)
5. **FFmpeg Extraction** - Precise clip cutting
6. **AI Cropping** - Vertical reframing with subject tracking
7. **Effects/Subtitles** - Optional AI-generated FFmpeg filters
8. **Hook Overlay** - Text overlays with styled fonts
9. **Voice Dubbing** - Optional ElevenLabs AI translation (30+ languages)
10. **S3 Backup** - Silent background upload
11. **Social Distribution** - Upload-Post API (async upload)

### Key Files
| File | Purpose |
|------|---------|
| `main.py` | Core video processing: transcription, scene detection, clip extraction, vertical reframing |
| `app.py` | FastAPI server with async job queue and REST endpoints |
| `editor.py` | Gemini AI integration for dynamic video effects (FFmpeg filter generation) |
| `hooks.py` | Hook text overlay generation with font rendering |
| `s3_uploader.py` | AWS S3 upload with caching |
| `subtitles.py` | SRT generation, FFmpeg subtitle burning, and dubbed video transcription |
| `translate.py` | ElevenLabs dubbing API for AI voice translation |
| `picker.py` | Stage 3 planner: ONE Gemini call over the whole transcript + the context brain returns the EXACT requested clip count (keep-looking loop), sentence-anchored, niche-agnostic |
| `context_layer.py` | Pre-download Gemini "brain": sends the YouTube LINK to Gemini in parallel with the download (own key) for a 3-part audiovisual summary consumed by the picker |
| `source_store.py` | Per-source persistent store: a seen URL reuses the downloaded video (hardlink), transcript and context blob — no re-download, no re-transcription API spend |
| `face_id.py` | Optional named-identity enrichment (InsightFace known-faces DB) that upgrades the transcript to named-speaker input for the picker; fails open |
| `hf_storage.py` | HuggingFace Hub clip storage: clips upload as each one finishes so they survive a Kaggle session ending; fails soft when unconfigured |
| `gpu_affinity.py` | Per-clip-worker GPU assignment (thread-local torch device) so a second GPU is not left idle |
| `kaggle_smoke_test.py` | Post-boot health check for the Kaggle host — lives in the repo so it improves via `git pull`, not a notebook re-import |
| `face_spine.py` | Framing rebuild Phase 1: ONE shared face-track spine (SCRFD + ByteTrack + ArcFace re-id across cuts). Boxes are **pixels**, `(x, y, w, h)` |
| `speaker_fusion.py` | Phase 3: one speaker↔track binding per clip, replacing per-frame pixel-position matching |
| `shot_planner.py` | Phase 4: plans the whole shot list up front — a few `(start, end, shot_type, track_ids, crop_rect)` entries, each with ONE static crop for its duration, so intra-shot jitter is structurally impossible |
| `reframe_v3.py` | Phase 4b: composition. Attention map (UNISAL saliency + role-weighted faces), containment-first crop geometry with vertical head placement, split-screen decision, and a fail-loud validation stage |
| `vendor/pyautoflip/` | Vendored subset of pyautoflip 0.2.1 (MIT): UNISAL saliency + split-screen geometry. **Read its README before taking more of that library** — its cropper, camera-motion handler and padding were reviewed and rejected, with reasons |
| `eval/` | Phase 0: framing measurement harness. Every framing fix ships with a metric that moved |
| `dashboard/src/App.jsx` | Main React component with state management |
| `dashboard/src/components/TranslateModal.jsx` | Voice dubbing UI with language selection |
| `dashboard/vite-plugin-seo.js` | Build-time SEO surface: injects crawler-visible homepage content, emits static pages, sitemap.xml and llms.txt |
| `dashboard/seo/data.js` | Single source of truth for pricing, pipeline and competitor facts used by every generated page |

### Framing (what actually runs)

`reframe_v3` is the only reframe engine. The v1/v2 engines (MediaPipe/YOLO
detectors, `SmoothedCameraman`, `subject_policy`) were removed — there is no
`REFRAME_ENGINE` switch and no fallback. The chain is:

`face_spine.py` (SCRFD + ByteTrack + ArcFace) → `speaker_fusion.py` (one
speaker↔track binding per clip) → `shot_planner.py` (one static crop per
shot) → `reframe_v3.py` (saliency-aware composition, fail-loud
`validate_composition`).

See `MASTER_PLAN_FRAMING_REBUILD.md` for why MediaPipe/YOLO were replaced and
how each v3 decision was tested.

### Stage 3 picker (what actually runs)

`picker.py` IS the planner: one Gemini call reads the WHOLE transcript (1M
context — no slicing) plus the pre-loaded context brain and returns the
EXACT requested number of distinct viral moments. No skill engine, no
narrative engine, no vision confirmation — the old dual-engine machinery
(`viral_clip_finder.py`, `viral_clip_finder_skill/`, `deepseek_worker.py`,
`VIRAL_ENGINE`, `VISION_CONFIRM_FALLBACK`) was deleted.

- The count is ALWAYS explicit (`--clip-count`, dashboard slider). Auto
  count was removed — the picker fulfills the requested count at all costs
  via a bounded keep-looking loop (forbidden-span passes), and a physical
  shortfall (video too short) is surfaced in logs + `metadata.json`.
- `context_layer.py` runs in parallel with the download: the YouTube LINK is
  sent to Gemini (`Part.from_uri`), which returns a 3-part audiovisual brain
  (summary / highlights with approximate timestamps / parts people would
  love) written to `gemini_context.json`. Uses its own key
  (`CONTEXT_GEMINI_API_KEY`) so it never shares the picker's rate budget.
- The shared tail in `main.py` is unchanged: ASR `term_corrections` applied
  BEFORE the word list is built → `_extend_start_for_preceding_question` →
  `_snap_candidates` (sentence-anchored: a boundary proposed mid-sentence
  snaps to that sentence's own start/end, so mid-sentence opens are
  impossible) → scene clamp → overlap dedup.
- The clip contract is load-bearing: `keep_spans` is a list of
  `{"start": float, "end": float}` **dicts** consumed by the jump-cutter;
  `narrative_summary` / `hook_type` / titles / descriptions / `viral_hook_text`
  ride along in `metadata.json` for the dashboard.

Env vars:

- `CONTEXT_GEMINI_API_KEY` — dedicated key for the pre-download context
  layer (default: reuses `GEMINI_API_KEY`)
- `SOURCE_CACHE_DIR` — where `source_store.py` keeps per-source videos,
  transcripts and context blobs (default: `sources/` next to the repo;
  point it at a mounted Kaggle Dataset for cross-session persistence)
- `GEMINI_MODEL` / `GEMINI_FALLBACK_MODEL` / `GEMINI_API_KEYS` — picker
  model + pool (unchanged)

Note the two engines resolve keys differently: the narrative engine is
`NARRATIVE_GEMINI_API_KEY`-only and is **inert without it** (so it cannot act
as the `auto` safety net), while the skill engine falls back to
`GEMINI_API_KEY`. On Kaggle only the latter is set by default, so the skill
engine is the sole Stage 3 path there — `kaggle_bootstrap.sh` prints which.
- `FACE_ID_DB` — path to a folder of `{name}.jpg` headshots; activates the
  optional named-identity enrichment layer (`FACE_ID_MODEL`, `FACE_ID_CTX`,
  `FACE_ID_SAMPLE_FPS`, `FACE_ID_THRESHOLD`, `FACE_ID_MIN_COVERAGE`,
  `FACE_ID_MIN_TURNS_S`, `FACE_ID_MAX_SECONDS` tune it). Fails open —
  anonymous speakers are used when InsightFace is missing, the DB is empty, or
  nothing matches. `enrich_if_configured` returns the `{label: name}` mapping
  (not the trajectory) because that is what the skill's transcript formatter
  looks speakers up in. Defaults to the LAST GPU (GPU 1 on a 2xT4) so it does
  not compete with the render pipeline on GPU 0; `FACE_ID_CTX=-1` forces CPU.

`insightface` is in `requirements.txt`, but **installing it unconstrained
breaks the stack** (measured 6-aug-2026): its chain pulls numpy 2.5.1 and
opencv-python-headless 5.x through `albumentations -> albucore`, and the pinned
ultralytics/torch stack does not work on numpy 2. Hence the `numpy<2`
guard line in requirements.txt — it is not imported by anything, it exists to
stop that. `kaggle_bootstrap.sh` re-applies the same protection differently: it
strips numpy/scipy/opencv lines (Kaggle's are CUDA-built), so it generates a
constraints file from the versions already installed on the host and passes it
to pip, then verifies numpy is still 1.x afterwards. The ONNX runtime is
separate again — `onnxruntime-gpu` is ~2GB and lives in the Dockerfile's GPU
block, with the bootstrap installing it on Kaggle.

### Kaggle host

`kaggle_bootstrap.sh` brings the stack up natively (no Docker on Kaggle) and
`kaggle_smoke_test.py` verifies it. The notebook is deliberately thin — secrets,
clone/pull, then those two files — so changes ship with `git pull` instead of a
manual `.ipynb` re-import. Things that were silently degrading there:

- **Transcription.** `TRANSCRIBE_BACKEND` chose the backend and defaulted to
  `whisper`, so setting `ASSEMBLYAI_API_KEY` alone did nothing: 254s of a ~420s
  job, and no diarization at all (that strips the speaker-binding pipeline of
  its diarized-tier evidence). `_select_backend()` now prefers assemblyai when
  the key is set and the backend is unset; an explicit `TRANSCRIBE_BACKEND`
  still wins.
- **Storage.** `/kaggle/working` is wiped at session end. `hf_storage.py`
  uploads each clip **as it finishes** (not at job end, which loses everything
  when a session dies mid-job) and records the key in the job metadata;
  `/api/history` then serves missing files through `/api/storage/{job}/{file}`.
  HuggingFace, not R2/B2 — neither can be signed up for without a credit card.
- **Keys.** The dashboard gated its job form on a browser-stored key even when
  the server had one. `/api/system` now reports `server_keys` (presence only)
  and the gate respects it.
- **The second GPU.** Clips render on a ThreadPoolExecutor in ONE process, so
  `CUDA_VISIBLE_DEVICES` per worker cannot work — it is read once at CUDA init.
  `gpu_affinity.py` uses the thread-local torch device instead. Only LR-ASD
  shards; per-clip face-spine detection runs inside each worker, so expect a
  partial gain on multi-clip jobs, not 2×.

### Captions

Placement is chosen **per job** in the submission form, not afterwards in the
Subtitle modal (which re-encodes every clip). Position and margin are separate
controls: "bottom but lifted off the edge" is bottom alignment with a larger
ASS `MarginV`, not a different alignment. `MarginV` only applies to bottom
alignment — `generate_ass` computes a per-line margin only when
`ass_alignment == 2` — so the UI offers the raised option there alone.
UI → `caption_position`/`caption_margin` → `CAPTION_POSITION`/`CAPTION_MARGIN_V`
→ `subtitles.AUTO_CAPTION_STYLE`.

### SEO / AI-crawler surface

The dashboard is a client-rendered SPA with hash routing, so the HTML served for
`/` used to contain an empty `<div id="root">`. Googlebot renders JavaScript and
saw the real page; GPTBot, ClaudeBot and PerplexityBot do not and measured the
homepage as zero characters of text. `vite-plugin-seo.js` fixes that at build time:

- Injects the content of `seo/landing-fallback.js` into `#root`. React's
  `createRoot().render()` replaces it on mount, so users get the app and
  non-executing clients get the copy. **Keep it in sync with `Landing.jsx`.**
- Emits the standalone pages under `/alternatives`, `/free-ai-clip-generator`,
  `/open-source-video-clipper` and `/how-openshorts-works` as flat `.html` files.
  nginx resolves the clean URL through `try_files $uri $uri.html`; serving them as
  directories instead makes nginx 301 to a trailing slash and every canonical
  would then point at a redirect.
- Generates `sitemap.xml` and `llms.txt` from the same page list, so they cannot
  drift. Do not add a static `public/sitemap.xml` back.

When editing pricing anywhere, edit `seo/data.js` too. Nothing on the site should
say "OpenShorts is free" without naming the Cloud price in the same breath: both
are true of different editions and quoting only the first one is what makes AI
answers describe the paid product as free.

### Video Reframing (v3) — attention, not just the active speaker

The old v1/v2 engines decided the crop **per frame, reactively**, which is why
jitter was visible even when nothing about who should be on screen changed. The
v3 chain plans instead: `face_spine` (who) → `speaker_fusion` (who is talking)
→ `shot_planner` (who holds the frame, for how long) → `reframe_v3` (what the
crop looks like). Each shot carries ONE static crop, so intra-shot re-aiming
cannot happen.

Two things about `reframe_v3` are load-bearing:

- **Active-speaker detection is not enough, by construction.** ASD answers "who
  is talking", which is the wrong question during a reaction — someone pops a
  balloon, someone winces, and the speaker is not the attraction. UNISAL
  saliency answers "where would a human look". `build_attention_map` fuses
  both; neither rules alone.
- **Bystander suppression is multiplicative (`×0.25`), not just absent boost.**
  Upstream pyautoflip uses `np.maximum(region, FACE_WEIGHT)`, which can only
  raise a value — so a bystander standing on a bright background keeps full
  saliency and still drags the centre of mass. That *is* the "biggest/brightest
  face wins" bug. If you touch `build_attention_map`, keep suppression able to
  scale values down.

Coordinates are **pixels** throughout (`face_spine` stores raw InsightFace
bboxes); normalized coords appear only at the boundary with the vendored
split-screen helpers. `validate_composition` raises `CompositionError` rather
than repairing — the framing bugs this rebuild fixes all shipped silently,
because a crop that cut a person in half still produced a playable file.

### API Endpoints
| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/process` | Submit video for processing |
| GET | `/api/status/{job_id}` | Poll job status and logs |
| POST | `/api/edit` | Apply AI video effects |
| POST | `/api/subtitle` | Generate and apply subtitles (auto-transcribes dubbed videos) |
| POST | `/api/hook` | Add text hook overlays |
| POST | `/api/translate` | AI voice dubbing via ElevenLabs |
| GET | `/api/translate/languages` | List supported dubbing languages |
| POST | `/api/social/post` | Post to social media (async upload) |

### Concurrency Model
Async job queue with semaphore-based concurrency control. Configure via `MAX_CONCURRENT_JOBS` env var (default: 5). Jobs auto-cleanup after 1 hour.

## Environment Variables

**Server-side (.env):**
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_S3_BUCKET` - For S3 backup
- `MAX_CONCURRENT_JOBS` - Concurrent processing limit (default: 5)
- `VITE_API_URL` - Production API URL override
- `VITE_OPENPANEL_API_URL`, `VITE_OPENPANEL_CLIENT_ID` - Optional product analytics, read at **build** time. Unset (the default, including every self-hosted build) means no analytics is initialised and no third-party script is loaded. `dashboard/index.html` also gates reporting on an `ANALYTICS_HOSTS` allowlist, so a build carrying credentials stays inert on any other host.
- `OPENPANEL_CLIENT_ID`, `OPENPANEL_CLIENT_SECRET` - Optional **server-side** analytics (`cloud/analytics.py`), same opt-in rule: unset means a silent no-op. Reports job outcomes with the user's job index, which the browser cannot do reliably — a render finishes after the tab is often gone, and ad-blockers eat a share of client events. Needs a *write* client; the read client used for querying is a different credential.

**Client-side (localStorage, encrypted):**
- `GEMINI_API_KEY` - Google Gemini API key (required)
- `ELEVENLABS_API_KEY` - ElevenLabs API key for voice dubbing (optional)
- `UPLOAD_POST_API_KEY` - Upload-Post API key for social posting (optional)

> API keys are stored encrypted in the browser and sent via headers only when needed. Never stored server-side.

## Tech Stack
- **Backend:** Python 3.11, FastAPI, google-genai, faster-whisper, insightface (SCRFD/ArcFace), ultralytics (ByteTrack/BoT-SORT only), LR-ASD, ONNX Runtime (UNISAL saliency), opencv-python, yt-dlp, FFmpeg, httpx
- **Frontend:** React 18, Vite 4, Tailwind CSS 3.4
- **External APIs:** Google Gemini, ElevenLabs Dubbing, Upload-Post
- **Infrastructure:** Docker + Docker Compose, AWS S3
