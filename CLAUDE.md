# CLAUDE.md

OpenShorts — AI vertical-video generator: long YouTube videos / local
uploads → viral-ready 9:16 shorts for TikTok, Reels and YouTube Shorts.
This is the **single source of truth** for what actually runs, in what
order, on what hardware, and how to set the environment up so jobs don't
fail. If a doc contradicts this file, this file wins.

Branch: `claude/gemini-vision-clip-picking-bikvuy` (Kaggle notebook pulls it
in Cell 2).

## The pipeline, stage by stage

1. **Source lookup** — `source_store.py` checks the persistent `sources/`
   cache keyed by YouTube video ID. A seen URL reuses the downloaded video
   (hardlink), the transcript and the context blob: **no re-download, no
   re-transcription, no re-context API spend**.
2. **Context brain** — `context_layer.py` sends the YouTube LINK to Gemini
   (`Part.from_uri`, Google-side processing) in a background thread while
   the download runs. Returns the 3-part audiovisual brain (summary /
   highlights / parts people would love) into `gemini_context.json`. Own
   key (`CONTEXT_GEMINI_API_KEY`), rotates across all keys on 429, falls
   back to `gemini-3.5-flash` on transient errors (a comma-separated
   `GEMINI_FALLBACK_MODEL` is a chain; a retired model that 404s is skipped,
   not fatal). Re-runs of a cached
   source whose first attempt never landed retry the link call (the pipeline
   waits out a fair budget — `CONTEXT_WAIT_TARGET` — so the blob can land
   and get cached; on fresh runs the download+transcribe runway already
   covers it, so the wait stays short).
3. **Download** — yt-dlp. **1080p is the hard floor, nothing below is ever
   accepted** (no 720p/480p/360p): the format chain has no sub-1080 option
   and the HD gate rejects anything lower. Option B caps at 1440p
   (`SOURCE_MAX_HEIGHT`) and only picks GPU-decodable codecs — H.264 first
   (up to 1080p, always 8-bit), then 8-bit VP9, then H.265. 10-bit VP9 and
   AV1 are excluded (Turing's NVDEC can't decode either — they'd silently
   CPU-decode). A video whose GPU-decodable streams cap below 1080p fails
   loudly instead of shipping soft clips. The landed codec + pixel format
   print in the source specs. Each ladder strategy probes the client's full
   format list BEFORE downloading and skips straight to the next strategy
   when that client can't serve the HD floor at all (a spoofed client that
   YouTube only offers ~360p no longer costs a full low-res download); the
   final strategy is exempt so the low-quality restore path keeps working.
4. **Transcribe** — AssemblyAI (API, diarization + sentiment + highlights)
   when `ASSEMBLYAI_API_KEY` is set, else CPU faster-whisper. Silent video →
   `get_visual_clips` (Gemini watches the footage).
5. **Scene detection** — TransNetV2 over the whole source (GPU decode, tiny
   48×27 frames). Only used to clamp a clip's END to a shot boundary.
   `SCENE_GPU_ONLY=1` (default): if the GPU can't decode the file, scene
   detection skips with a clear log line — no CPU decode, no PySceneDetect.
6. **Picker** — `picker.py` IS the planner: ONE Gemini call over the whole
   transcript (1M context, no slicing) + the context brain, returning the
   EXACT requested clip count. Bounded keep-looking loop (max 3 passes,
   already-picked spans become forbidden). No "no viral moment found" path;
   a physical shortfall is surfaced in logs + metadata. Rotates across all
   `GEMINI_API_KEY` + `GEMINI_API_KEYS` on transient failures.
7. **Tail** (instant, CPU): ASR term corrections → question backstop →
   sentence-anchored snap (a boundary proposed mid-sentence snaps to that
   sentence's own start/end — mid-sentence opens are impossible) → scene
   clamp → overlap dedup.
8. **Render** per clip (GPU): cut (NVDEC decode) → face spine (insightface
   on onnxruntime-gpu) → LR-ASD (torch CUDA) → speaker fusion → shot planner
   (one static crop per shot) → UNISAL saliency (CPU, ~17ms × shots) →
   NVENC encode. The render-time scene-context director
   (`analyze_scene_context`, Gemini) supplies focus directives per clip.
   Reframe prints timed sub-steps (`↳ face spine …`, `↳ LR-ASD …`, …) so it
   never looks frozen.
9. **Finalize** — metadata, per-clip `.ready` markers, HF backup
   (`hf_storage.py`, clips upload as they finish).

## Where things run

| Stage | Tool | Where |
|---|---|---|
| Context brain, picker, scene direction | Gemini API | API (no GPU) |
| Transcription | AssemblyAI / faster-whisper | API / CPU |
| Download | yt-dlp + ffmpeg | network/CPU |
| Scene detection | ffmpeg NVDEC + TransNetV2 (torch) | GPU |
| Face spine | insightface on onnxruntime-gpu | GPU |
| LR-ASD | torch (vendor/lrasd) | GPU |
| Saliency | UNISAL ONNX (CPUExecutionProvider) | CPU (tiny) |
| ffmpeg decode/encode | NVDEC / NVENC | GPU |

The GPU-or-nothing contract applies everywhere decode matters:
`REQUIRE_GPU_DECODE=1` (default) fails a job at startup if the ffmpeg on
PATH cannot decode on the GPU; `SCENE_GPU_ONLY=1` skips scene detection
rather than CPU-decoding; the download never picks AV1.

## Environment — the deterministic setup (Kaggle)

`kaggle_bootstrap.sh` is the single setup path; `requirements-gpu.txt` is the
single pinned source for the GPU runtime. The bootstrap **uninstalls
conflicting CPU defaults before installing** so there is exactly one of each
tool on the path.

Notebook: Cell 1 (secrets) → Cell 2 (git pull) → Cell 3 (bootstrap + serve)
→ Cell 4 (smoke test). **Cell 3 must print all of these** — if any is
missing, fix it before running a job:

```
==> nvenc-capable ffmpeg
    ffmpeg already has GPU encode + CUDA decode/filters — nothing to do
    REQUIRE_GPU_DECODE=1 — strict GPU-or-nothing mode confirmed

==> Face ID runtime (onnxruntime-gpu from requirements-gpu.txt)
    onnxruntime-gpu installed: CUDAExecutionProvider available — face spine on GPU

==> Configuration
    encoder=nvenc  whisper_device=cuda  gpus=2
    onnxruntime: CUDAExecutionProvider ACTIVE (face spine on GPU)
```

Every job log must then open with:

```
🖥️  Pipeline diagnostics:
   ffmpeg: /usr/bin/ffmpeg   (or the nvenc build path)
   GPU decode (NVDEC + CUDA filters): YES
   GPU encode (NVENC h264): YES
   torch CUDA: True (2 device(s))
   onnxruntime: 1.28.0 providers=[... 'CUDAExecutionProvider' ...]
```

GPU pieces that were silently wrong and are now deterministic:
- **ffmpeg**: Kaggle's apt ffmpeg is CPU-only. The bootstrap installs a
  decode-capable BtbN build to `FFMPEG_DIR` (default
  `/kaggle/working/ffmpeg-nvenc`); `main.py` prepends it to PATH itself (and
  adds `/usr/lib/x86_64-linux-gnu` + `/usr/local/cuda/lib64` to
  `LD_LIBRARY_PATH`) so the subprocess always finds it. If missing, the job
  self-heals by downloading it; if the download fails, the job fails loudly.
- **onnxruntime**: Kaggle ships the CPU build → insightface (face spine) ran
  on CPU while GPUs idled ("stuck in Reframe engine v3 with GPU at 0"). The
  bootstrap now uninstalls CPU onnxruntime and installs the pinned
  `onnxruntime-gpu==1.28.0` from `requirements-gpu.txt`.
- **Codec**: AV1 is excluded from downloads (T4 can't NVDEC AV1); the cache
  rejects AV1 files too.
- **Keys**: the picker + context rotate across `GEMINI_API_KEY` and every
  `GEMINI_API_KEYS` extra on 429 — no more quota wall with unused keys.

## Environment variables

- `GEMINI_API_KEY` — required (picker + context primary).
- `GEMINI_API_KEYS` — extra keys (comma-separated); picker + context rotate
  across them on transient failures.
- `CONTEXT_GEMINI_API_KEY` — dedicated key for the pre-download context
  layer (default: reuses `GEMINI_API_KEY`).
- `CONTEXT_WAIT_TARGET` — total seconds of runway the context thread should
  have had before the picker runs (default `35`). A cached re-run starts the
  thread ~now, so this is how long the pipeline waits for the blob there;
  a fresh run that already gave the thread minutes waits only the 5s floor.
  `join()` returns the instant the thread finishes, so a stuck 504 never
  blocks the job beyond this budget.
- `ASSEMBLYAI_API_KEY` — transcription backend (diarization + sentiment +
  highlights). Missing → CPU whisper.
- `SOURCE_CACHE_DIR` — source store location (default `sources/` next to the
  repo). Point at a mounted Kaggle Dataset for cross-session persistence.
- `SOURCE_MAX_HEIGHT` — download height cap, default `1440` (Option B);
  `0` = no cap. **Only GPU-decodable codecs are ever picked**: H.264 first
  (up to 1080p, always 8-bit), then 8-bit VP9, then H.265. 10-bit VP9 and
  AV1 are excluded. **1080p is the hard floor** — no 720p/480p/360p is ever
  accepted; a video whose GPU-decodable streams cap below 1080p fails loudly.
- `MIN_SOURCE_HEIGHT` — quality gate floor, default **1080** (a sub-1080 or
  non-GPU-decodable source fails the job loudly; `ALLOW_LOW_QUALITY_SOURCE=1`
  is the explicit escape hatch).
- `REQUIRE_GPU_DECODE` — `1` (default; fail if GPU decode unavailable) |
  `warn` (loud warning, run CPU — emergencies only) | `0`.
- `CONVERSATION_FRAMING` — master switch for the local conversation
  director (default `1`): vertical split-screen during genuine two-person
  back-and-forth (single → split → single), driven by LR-ASD + diarization +
  stable anonymous face identities. `0` reverts to the pre-conversation
  pipeline. `SPLIT=0` keeps the speaker-binding fix but skips the beat
  planner; `SPLIT_MIN_EXCHANGE_S=2.5` / `SPLIT_MIN_SPAN_S=2.0` tune the
  exchange windows. An exchange only cuts to a two-track edit when BOTH
  participants actually have faces on screen during the window (no face →
  the normal single/wide shot holds — the "show what's actually there"
  rule), and then as a VERTICAL SPLIT only when they are too far apart for
  one crop; people working together in one frame become a TWO_SHOT instead.
  `VSPLIT_BAND_FRAC=0` (default) stacks the two panels edge-to-edge with no
  separating bar — each panel fills half the frame and frames its person
  head-and-shoulders (9:8 for 9:16 output, no letterbox, no distortion), and
  captions overlay the seam (middle) whenever a vertical split is on screen
  while keeping the user's chosen position (usually bottom) everywhere else
  in the clip. `VSPLIT_BAND_FRAC>0` re-adds a band. `CAPTION_POSITION` only
  affects non-split clips.
- `SPEAKER_SMOOTH_WINDOW` — majority-filter window (default `3`) for the
  per-second LR-ASD speaker track before it votes for bindings or falls
  through as the per-second fallback: a one-second crowd flicker (reacting
  listener vs talker) no longer flips the frame to the wrong person or
  fabricates a split exchange. `0`/`1` disables the smoothing.
- `ASD_DECISIVE_MARGIN` — how far LR-ASD's winning face must lead the next
  best on-screen face, that second, for the second to vote at all (default
  `0` — every second counts, the pre-experiment behaviour; the framing
  experiment's `0.10` gate degraded the last session). The model already
  computes the lead (`score_clip`'s
  `per_second_margin`); seconds where two faces scored within the margin are
  the model saying "could be either", and they no longer pollute the
  one-binding-per-clip vote. `0` counts every second (the old behaviour).
- `SPEAKER_REBIND_SECONDS` / `SPEAKER_REBIND_WINDOW` — the gate on
  correcting a binding mid-clip (default `0` = disabled, the pre-experiment
  behaviour; `4` and `8` were the experiment's values). One binding per clip
  killed the per-frame flip-flop but meant a binding that came out WRONG held
  the camera on the wrong person for the whole clip. Now, when a bound
  speaker is talking and decisive ASD points at a different track for
  `SPEAKER_REBIND_SECONDS` seconds inside a `SPEAKER_REBIND_WINDOW`-second
  window, the label re-binds from that second to the clip's end. Agreement
  with the current binding decays the case, so a one-second blip — or an
  alternating, ambiguous signal — can never trigger it.
  `SPEAKER_REBIND_SECONDS=0` disables re-binding entirely.
- `IDENTITY_CONFIRM` — director v2's Gemini face-identity confirmation
  (default `0` = off; the rendered-clip review showed the per-clip maps
  can't reliably tell speakers apart in a crowd — the LR-ASD fusion is the
  primary again. Opt in with `1` to experiment): after
  each clip's face spine, Gemini receives labeled face
  crops + the diarized transcript and returns the speaker->face map
  (`identity_confirm.py`). When it lands, framing is **ASR-first** — the
  transcript decides WHO, the map decides WHICH FACE, and unmapped speakers
  go wide/hold instead of defaulting to a wrong face (no host-default).
  Fail-open: any error falls back to the LR-ASD fusion unchanged.
  `IDENTITY_CONFIRM_MODEL` picks the model (default `GEMINI_MODEL` /
  `gemini-3.1-flash-lite`).

## Dashboard ordering & data files

- History and the Generated Shorts side rail sort **newest-to-oldest at all
  times**: each clip's `created_at` is its own file mtime (when it was
  actually generated), jobs are ordered by their newest clip, and the job
  dir's `.created` marker self-heals on first read so restored/touched files
  can't keep reshuffling history.
- Metadata JSON never appears as a video: the dead-job label skips non-media
  files, and the raw `*_metadata.json` files live in a collapsible
  **storage · data files** subsection of the History tab (hidden by default)
  backed by `GET /api/history/meta` and `DELETE /api/history/meta`
  (`?job_id=` for one job, no arg = delete for all).
- `VSPLIT_TRACK` — per-panel smart-crop tracking for vertical splits
  (default `1`): each panel's crop glides (AutoFlip-style) to keep its
  person framed as they move — head-anchored with headroom above the hair,
  dead-zone hysteresis (micro head-bobs don't pan), lerp smoothing (cinematic
  glide), hard-cut snap (pixel-diff scene cut), boundary clamping, a
  containment projection that never lets the face leave the crop, and a
  full-width fallback when the face is lost. `0` keeps the static
  union-box panels (the pre-tracking behavior). A panel is framed the way a
  SINGLE shot would frame that person, then stacked: the tracked panels use
  the same breathing room as the static engine
  (`reframe_v3.DEFAULT_SIDE_MARGIN` / `DEFAULT_VERT_MARGIN`), and
  `VSPLIT_PANEL_MIN_FRAC=0.45` floors a panel crop at ~half the source
  height so a small or distant face can never zoom into a passport close-up.
  Tuning knobs:
  `VSPLIT_HEADROOM=0.18`, `VSPLIT_SIDE_MARGIN=0.55`, `VSPLIT_VERT_MARGIN=0.35`,
  `VSPLIT_PANEL_MIN_FRAC=0.45` (fraction of source height),
  `MIN_CROP_FRAC=0.45` (same floor for regular single/two-shot crops — the
  blurry-zoom fix: a tiny crop upscaled to 1080x1920 is unreadable),
  `VSPLIT_DEADZONE=0.02` (fraction of crop), `VSPLIT_SMOOTH=0.12` (pan lerp),
  `VSPLIT_SMOOTH_ZOOM=0.06` (zoom lerp), `VSPLIT_FULLWIDTH_FRAC=0.60` (box
  area fraction that triggers the wide-shot fallback), `VSPLIT_LOST_HOLD_FRAMES=30`,
  `VSPLIT_CUT_THRESHOLD=25.0` (64x36 gray mean-abs-diff that counts as a cut).
- **4:3 WIDE shots** — the owner-approved "show everyone" framing: moments
  with no confident subject (reaction / unbound speaker / "both people are
  relevant") render as a 4:3 crop of the source, letterboxed into the 9:16
  frame (fit-width, black bars top/bottom). When faces are known the crop
  ZOOMS to them (no tiny-people dead space). Far-apart pairs the camera can
  still capture in one 4:3 crop get the wide instead of a split; only
  people too far apart for even that become the vertical split. It only
  ever appears in those moments — never as the default framing.
- `SCENE_DIRECTION` — the per-clip Gemini camera-director call, now **`0` by
  default** (the local director replaced it; opt-in only if you want Gemini
  naming who to frame).
- `SCENE_GPU_ONLY` — `1` (default; skip scene detection if GPU can't decode
  the file) | `0` (allow the cheap 48×27 CPU retry + PySceneDetect).
- `SCENE_DETECTION` — `1` (default) | `0` (skip scene detection entirely —
  faster runs, clips lose the end-shot-boundary polish).
- `SCENE_DETECT_TIMEOUT` — TransNetV2 decode cap in seconds (default 300).
- `FFMPEG_DIR` — nvenc ffmpeg location (default `/kaggle/working/ffmpeg-nvenc`).
- `GEMINI_MODEL` / `GEMINI_FALLBACK_MODEL` — picker/context model
  (default `gemini-3.1-flash-lite`; fallback default `gemini-3.5-flash`,
  comma-separated = a chain tried in order. `gemini-2.5-flash` was retired
  by Google (404 "no longer available") and is skipped automatically).
- `FACE_ID_DB` — folder of `{name}.jpg` headshots for optional named-speaker
  enrichment (fails open; anonymous speakers otherwise).
- `YOUTUBE_COOKIES` — optional Netscape cookie blob to dodge YouTube's bot
  wall (`cookies.txt: MISSING` is the only known loose end).
- `OUTPUT_DIR` — job output root (default `$PWD/output`); `OUTPUT_MAX_GB`
  caps it.

## Key files

| File | Purpose |
|---|---|
| `main.py` | Pipeline orchestrator: download, transcribe, picker wiring, render loop, CLI |
| `picker.py` | Stage 3 planner — one Gemini pass, exact clip count, keep-looking loop |
| `context_layer.py` | Pre-download Gemini link brain (3-part audiovisual summary) |
| `source_store.py` | Per-source cache (video/transcript/context) keyed by video ID |
| `scene_detection.py` | TransNetV2 shot boundaries (GPU decode; skip-if-not-GPU) |
| `reframe_v3.py` | Composition engine (saliency + role-weighted faces, static shots) |
| `face_spine.py` | SCRFD + ArcFace face tracks (onnxruntime-gpu) |
| `asd_worker.py` | LR-ASD active-speaker detection (torch CUDA) |
| `shot_planner.py` | One static crop per shot |
| `gpu_affinity.py` | Per-worker thread-local GPU assignment |
| `ffmpeg_utils.py` | Encoder/decoder selection + GPU probes |
| `transcribe_backends.py` | AssemblyAI / whisper selection |
| `app.py` | FastAPI server: jobs, history, sources, storage |
| `kaggle_bootstrap.sh` | Deterministic Kaggle environment setup |
| `requirements-gpu.txt` | Pinned GPU runtime (onnxruntime-gpu==1.28.0) |
| `kaggle_smoke_test.py` | Post-boot health check |

## Troubleshooting — the failures we've actually hit, and the fix

- **Job dies at startup, `GPU decode (NVDEC + CUDA filters): NO`** → the
  decode-capable ffmpeg isn't on PATH. Run Cell 3; confirm the nvenc build
  line. The job also self-heals (downloads the build); if the download
  fails, the job fails loudly with the reason.
- **"Stuck in 🚀 Reframe engine v3" with GPU at 0** → CPU onnxruntime
  (face spine on CPU). Re-run Cell 3 and confirm
  `onnxruntime-gpu installed: CUDAExecutionProvider available`.
- **`TransNetV2 scene detection skipped`** → the GPU couldn't decode that
  file (or the download landed AV1). Check the source specs codec line;
  re-download (AV1 caches are rejected) and it should be `vp9`/`avc1`.
- **`429 RESOURCE_EXHAUSTED`** → set `GEMINI_API_KEYS` (extras) as a Kaggle
  secret; the picker + context rotate across them now.
- **`clip_count is required`** → auto clip count was removed; set the
  dashboard slider (1–40) — the picker fulfills that count at all costs.
- **Source re-downloads every run** → `/kaggle/working` is wiped per
  session; mount a Dataset and set `SOURCE_CACHE_DIR` to it.
- **A job fails with no obvious reason** → the failure log now carries the
  real last output; hit the dashboard **copy** button and paste it.

## Development / tests

- Local: `pip install -r requirements.txt` (heavy GPU deps are
  Dockerfile/bootstrap-only; `requirements-gpu.txt` is the GPU add-on).
- Dashboard: `cd dashboard && npm install && npm run dev` (port 5173) or
  `npm run build`.
- Tests: `python3 -m pytest tests/ -q`. The full suite is 500+ passing; the
  handful of collection errors locally are files needing cv2/scenedetect/
  torch (they run on the Kaggle host).
- Docker: `docker compose up --build` (backend :8000, frontend :5175).

## API endpoints (main ones)

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/process` | Submit video for processing (requires `clip_count`) |
| GET | `/api/status/{job_id}` | Poll job status + live logs |
| GET | `/api/sources` | List saved source videos (persistent store) |
| GET | `/api/sources/{key}/video` | Stream a saved source (Range supported) |
| GET | `/api/history` | Saved clip library, newest first |
| POST | `/api/subtitle` | Generate/apply subtitles |
| POST | `/api/hook` | Hook overlay text |
| POST | `/api/translate` | ElevenLabs dubbing |
