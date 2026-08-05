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
| `viral_clip_finder.py` | Stage 3 judgment engine: the viral-clip-finder skill (15 frameworks, 8-axis rubric, 18 anti-patterns, niche playbooks) as a schema-enforced LLM call returning scored clips with cut briefs + rejected candidates |
| `face_id.py` | Optional named-identity enrichment (InsightFace known-faces DB) that upgrades the transcript to named-speaker input for the skill; fails open |
| `viral_clip_finder_skill/` | Bundled skill package (SKILL.md + references) consumed by `viral_clip_finder.py` |
| `dashboard/src/App.jsx` | Main React component with state management |
| `dashboard/src/components/TranslateModal.jsx` | Voice dubbing UI with language selection |
| `dashboard/vite-plugin-seo.js` | Build-time SEO surface: injects crawler-visible homepage content, emits static pages, sitemap.xml and llms.txt |
| `dashboard/seo/data.js` | Single source of truth for pricing, pipeline and competitor facts used by every generated page |

### Viral Clip Finder (Stage 3 engine)

Stage 3 ("find viral moments") is engine-selectable via `VIRAL_ENGINE`:

- `auto` (default): the viral-clip-finder skill runs first; the existing
  narrative-arc engine (deepseek_worker → Gemini 2-pass) is the fallback on
  any provider failure, so a hiccup can't zero out a job.
- `skill`: the skill is a hard requirement — failure fails the job loudly.
- `narrative`: the pre-upgrade behavior exactly.

Every engine's candidates pass through the shared Gemini Vision confirmation
and word-snapping tail (`_vision_confirm_candidates` / `_snap_candidates` in
`main.py`), so the cut/reframe/subtitle stages are unchanged. The skill's
extra fields (score, patterns, risk flags, cold-open, caption text, B-roll
cues, cut list) ride along in `metadata.json` for the dashboard and editors.

**Both engines emit the same clip contract**, and that is load-bearing:

- `keep_spans` is a list of `{"start": float, "end": float}` **dicts**
  (`deepseek_worker.KeepSpan`). Every consumer in `main.py` reads them with
  `span.get("start")`. Emitting `[start, end]` pairs instead raises
  `AttributeError` inside `_extend_keep_spans_to_cover_boundaries` — which is
  swallowed by the engine's fallback and shows up as "the skill engine never
  runs", not as an error. `tests/test_viral_clip_finder.py` pins this against
  the real functions compiled out of `main.py`.
- `narrative_summary` is what `confirm_clip_with_vision` shows the vision
  judge; the skill path aliases `why_it_hits` onto it. Without it the judge
  reviews clips with no idea what they are supposed to resolve.
- `term_corrections` (ASR mishearings) must be applied **before** the word
  list is built, or captions render the misheard spelling.

Env vars:

- `VIRAL_ENGINE` — `auto` | `skill` | `narrative` (default `auto`)
- `VCF_ALLOW_DEEPSEEK` — `1` opts DeepSeek's own API back into the provider
  chain (default: Gemini only, matching the existing narrative engine)
- `VCF_LONG_FORM_REFS` — always include the seamless-cutting reference
  (default: only when long-context clips are requested)
- `VCF_REFERENCES` — `full` (default) | `lean` | explicit `a.md,b.md` list.
  The full set is ~124KB (~35k tokens) prepended to every Stage 3 call; that
  breadth is the point, `lean` trades it for cost/latency.

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
  looks speakers up in.

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

### Dual-Mode Video Reframing
- **TRACK Mode** (single subject): MediaPipe face detection + YOLOv8 fallback with "Heavy Tripod" stabilization
- **GENERAL Mode** (groups/landscapes): Blurred background layout preserving full width

### Key Classes
- `SmoothedCameraman` - Stabilized camera movement with safe zone logic (prevents jitter)
- `SpeakerTracker` - Prevents rapid speaker switching, handles temporary occlusions

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
- **Backend:** Python 3.11, FastAPI, google-genai, faster-whisper, ultralytics (YOLOv8), mediapipe, opencv-python, yt-dlp, FFmpeg, httpx
- **Frontend:** React 18, Vite 4, Tailwind CSS 3.4
- **External APIs:** Google Gemini, ElevenLabs Dubbing, Upload-Post
- **Infrastructure:** Docker + Docker Compose, AWS S3
