# Kaggle optimization plan

**For the engineer picking this up (DeepSeek or otherwise).**
Repo: `github.com/foskigr8/openshorts`, branch `session/framing-work`.
If you do not have repo access, ask the owner to add you — every change below is
in this repo. Read `KAGGLE.md` first for how the Kaggle host is brought up, and
`HANDOFF_FRAMING.md` §2e for measurement traps that have already cost this
project real time.

Everything here comes from **one real Kaggle run** (job `b86b8c5a`, 5-aug-2026,
2×T4). Timings are from that log. Do not re-derive them; do not contradict them
without a new measurement.

---

## What already works — do not "fix" these

| stage | evidence from the run |
|---|---|
| YouTube download | `Download attempt: anonymous` → `✅ Download succeeded` in **5.16s**, no cookies. Kaggle's IP is not flagged. |
| NVENC encode | `video encoder: h264_nvenc` — GPU encode is live. |
| Framing | `lip-sync 97%, directed 3%`, size **0%**. The framing rebuild is working. |
| Clip selection | 3 clips, `$0.002932` total Gemini spend. |
| Single-origin UI | dashboard served by the API process through one tunnel. |

Total job: **~7 minutes** for a 10.5-minute source → 3 clips.

---

## FINDING 1 — transcription is NOT using the API (biggest win, do first)

**254 seconds of the ~7 minute job was transcription**, and the owner's
suspicion was right: it ran **locally**, not through AssemblyAI.

Proof in the log: `Warning: You are sending unauthenticated requests to the HF
Hub` (faster-whisper pulling its model from HuggingFace), then
`Detected language 'en', 189 segments` — that is faster-whisper's output
format, not AssemblyAI's.

**Root cause** (`transcribe_backends.py:591`):

```python
backend = os.environ.get("TRANSCRIBE_BACKEND", "whisper").strip().lower()
```

The default is `whisper`. Setting `ASSEMBLYAI_API_KEY` alone does **nothing** —
the backend is chosen by `TRANSCRIBE_BACKEND`, and nobody sets it.

### Why this matters far beyond speed

No AssemblyAI means **no diarization**. Diarization is `TIER_DIARIZED` in
`subject_policy.py` — the evidence tier that decides framing whenever lip-sync
has no binding. The run confirms it: the tier mix reads `lip-sync 97%,
directed 3%` with **diarized 0%**, because there were no speaker labels at all.

On a single-speaker video (this one) that costs nothing. On Pop The Balloon —
multi-speaker, the actual target content — it removes the signal that stops the
camera sitting on the wrong person. **Framing quality on real content depends on
this fix.**

### The change

1. In `kaggle_bootstrap.sh`, set the backend when the key is present:
   ```bash
   if [ -n "${ASSEMBLYAI_API_KEY:-}" ]; then
       export TRANSCRIBE_BACKEND="${TRANSCRIBE_BACKEND:-assemblyai}"
       echo "    transcription: assemblyai (API, with diarization)"
   else
       echo "    transcription: local whisper — SLOW (254s measured) and NO diarization"
   fi
   ```
2. Consider making `transcribe_media` prefer `assemblyai` automatically when
   `ASSEMBLYAI_API_KEY` is set and `TRANSCRIBE_BACKEND` is unset. Keep the
   existing fallback-to-whisper on API failure.
3. **Verify by running**, not by reading: the log must show the AssemblyAI path
   and the elapsed transcription time must drop from ~254s to tens of seconds.
   Then confirm a multi-speaker clip shows a non-zero `diarized` share in the
   `🎯 Framing evidence` line.

**Expected: ~7 min job → ~3 min, and framing gets its second evidence tier back.**

---

## FINDING 2 — the dashboard demands API keys the server already has

The owner pasted keys into Kaggle Secrets and the UI still asked for them.

`app.py:112` already falls back to `os.environ.get("GEMINI_API_KEY")`, so the
**server side is fine**. The problem is the dashboard: it is built for
self-hosters who keep their key in the browser (localStorage), so it gates the
UI on a locally-stored key regardless of what the server has.

### The change

1. Extend `GET /api/system` with a block reporting **presence only — never the
   values**:
   ```json
   "server_keys": {"gemini": true, "assemblyai": true, "elevenlabs": false}
   ```
   Derive from `bool(os.environ.get(...))`. Do not return prefixes or lengths.
2. In the dashboard, when `server_keys.gemini` is true, skip the "enter your
   key" gate and show something like *"using the server's key"* in Settings,
   with the local field still available as an override.
3. Rebuild `dashboard/dist` (the bootstrap does this) so Kaggle picks it up.

**Test:** on Kaggle with the secret set, a fresh browser profile must reach the
job form without entering anything.

---

## FINDING 3 — persistent storage: use Cloudflare R2 now, HuggingFace later

The owner asked for a decision between HuggingFace Hub and Cloudflare. Here is
the reasoning, not just the answer.

**Recommendation: Cloudflare R2 as the primary output store. Add HF Hub only if
a bulk/archive tier is actually needed.**

| | Cloudflare R2 | HuggingFace Hub |
|---|---|---|
| free tier | 10 GB storage, **zero egress** | much larger, but designed for datasets/models |
| code needed | **almost none** — `s3_uploader.py` already speaks S3 | a new uploader module (`huggingface_hub`) |
| access model | private buckets, signed URLs | repo-based; private repos are limited on free |
| suits | finished clips (10–30 MB each) | large source archives |

10 GB holds roughly **300–1000 finished clips**. The decisive factor is not
capacity, it is that **R2 is S3-compatible and the repo already has a working
S3 uploader** — so this is a config change plus one small patch, versus writing
and testing a new integration.

### The change

1. `s3_uploader.py` currently passes `region_name` but **no `endpoint_url`**
   (checked — it is not there). R2 requires one. Add:
   ```python
   endpoint = os.environ.get("AWS_S3_ENDPOINT")  # R2: https://<acct>.r2.cloudflarestorage.com
   boto3.client("s3", ..., endpoint_url=endpoint or None)
   ```
   `endpoint_url=None` keeps real AWS behaviour identical — verify that.
2. Kaggle secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `AWS_S3_BUCKET`, `AWS_S3_ENDPOINT`, `AWS_REGION=auto` (R2 wants `auto`).
3. Upload **as each clip finishes**, not at job end — `/kaggle/working` is wiped
   when the session dies, and a 12h cap means it will die mid-job eventually.
4. Surface the uploaded URL in the dashboard so clips are retrievable after the
   session ends.

**If >10 GB is genuinely needed:** add HF Hub as a second tier via
`huggingface_hub.upload_file` to a **private dataset repo**, using the
`HF_TOKEN` that is already a secret. Do this only after R2 works — two storage
backends is twice the failure surface.

---

## FINDING 4 — use the second T4

The run showed three clips rendering **concurrently** (three `Reframe engine v2`
starts within 10 seconds) — so per-clip parallelism already exists. But every
one of them ran on **GPU 0**; GPU 1 was idle the whole job.

Per-clip reframe times: **94.5s, 110.5s** — the dominant cost after
transcription. LR-ASD inside them: 20.1s, 25.8s, 34.3s.

### The change

Shard clip workers across devices. Whatever spawns the per-clip work should set,
per worker:

```
CUDA_VISIBLE_DEVICES=0   # worker A   -> torch sees one GPU, index 0
CUDA_VISIBLE_DEVICES=1   # worker B   -> also index 0 *within its mask*
```

Note the subtlety: **`YOLO_DEVICE` stays `0` in both**, because the mask
renumbers devices. Setting `YOLO_DEVICE=1` under a mask of `1` is an error.

Points to check:
- `main.py` reads `YOLO_DEVICE`; `asd_worker.py` and `scene_detection.py` use
  torch CUDA — all inherit the mask, so no code change beyond how workers spawn.
- NVENC: a T4 has its own encoder per GPU, so two concurrent nvenc sessions are
  fine (consumer cards cap concurrent sessions; T4 does not have that limit).
- Measure before and after with the same source. **Expected ~2× on multi-clip
  jobs, 0% on single-clip jobs** — say which you measured.

Do this **after** Finding 1. Transcription is 254s of a ~420s job; halving the
reframe stage while leaving transcription local is optimising the smaller half.

---

## FINDING 5 — HF_TOKEN warning

```
Warning: You are sending unauthenticated requests to the HF Hub.
```

This is faster-whisper downloading its model anonymously — rate-limited and
slower. The owner already has `HF_TOKEN` as a Kaggle secret; it is simply not
exported.

Add it to the bootstrap's secret block and to the notebook's key list. Note that
**once Finding 1 lands this mostly disappears**, because AssemblyAI does not
download a local model — but keep it for the whisper fallback path.

---

## FINDING 6 — make the notebook thin so it never needs re-importing

The owner's requirement: *"each time we update it, we don't have to reimport."*

Right now the notebook carries real logic (key loading, clone, smoke test), so
improving any of it means re-importing the `.ipynb` into Kaggle by hand.

### The change

Reduce the notebook to a **launcher of three cells** and move everything else
into repo files, which update with `git pull`:

```python
# Cell 1 — clone or update, then hand over to the repo
import os, subprocess
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
for k in ("GITHUB_TOKEN","GEMINI_API_KEY","GEMINI_API_KEYS",
          "ASSEMBLYAI_API_KEY","YOUTUBE_COOKIES","HF_TOKEN",
          "AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY",
          "AWS_S3_BUCKET","AWS_S3_ENDPOINT"):
    try: os.environ[k] = s.get_secret(k)
    except Exception: pass

D = "/kaggle/working/openshorts"
url = f"https://{os.environ['GITHUB_TOKEN']}@github.com/foskigr8/openshorts.git"
if os.path.isdir(D):
    subprocess.run(["git","-C",D,"pull","--ff-only"])
else:
    subprocess.run(["git","clone","--depth","1","-b","session/framing-work",url,D])
os.chdir(D)
```

```python
# Cell 2 — everything else lives in the repo
!bash kaggle_bootstrap.sh
```

```python
# Cell 3 — smoke test, also in the repo
!python3 kaggle_smoke_test.py
```

So: **move the smoke-test cell into `kaggle_smoke_test.py`**, keep the secret
list as the only thing the notebook knows, and make cell 1 idempotent (pull if
present, clone if not). After that, new features arrive with `git pull` and the
notebook never changes again.

Keep the secret name list generous — adding a secret then becomes "create it in
Kaggle", with no notebook edit.

---

## Order of work

1. **Finding 1** (transcription backend) — biggest win, smallest change, and it
   restores a framing signal.
2. **Finding 6** (thin notebook) — do it early so everything after ships by
   `git pull`.
3. **Finding 3** (R2 storage) — outputs stop evaporating.
4. **Finding 2** (server keys in UI) — removes the re-paste friction.
5. **Finding 4** (dual GPU) — the fun one, but the smallest real gain until 1 is
   done.
6. **Finding 5** (HF_TOKEN) — one line.

## Rules

- **Verify by running, not by reading.** Every claim above that says "measured"
  came from a real run; match that standard.
- **Watch the video before declaring framing good.** `HANDOFF_FRAMING.md` §2e
  lists five ways this project has fooled itself with metrics, including a
  vision review that was confidently wrong about geometry.
- **One render per output path.** Two concurrent writers produced a truncated
  file and a round of garbage numbers.
- Run the suite: `docker exec openshorts-backend sh -c 'cd /app && python3 -m
  pytest tests/ -q'` — **711 passing** at the time of writing. A drop is a
  regression you caused.
- Do not commit secrets. `cookies.txt` and `.env` are gitignored; keep it that
  way. Never print a clone URL containing a token.
