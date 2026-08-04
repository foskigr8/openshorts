# Running OpenShorts on Kaggle (dual T4)

Kaggle gives you 2×T4 for free, which is where this pipeline wants to run: on
CPU, LR-ASD alone is ~2× realtime (a 32s clip took 170s to render end to end);
batched on CUDA it measured ~48× realtime.

**Kaggle has no Docker**, so `docker compose` cannot be used. `kaggle_bootstrap.sh`
brings the same application up natively: one Python process serves the API *and*
the built dashboard (single-origin block at the end of `app.py`), and one
cloudflared tunnel exposes it.

---

## Notebook setup (do this once per session)

**Settings panel:**
- Accelerator: **GPU T4 ×2**
- Internet: **On** (requires a phone-verified Kaggle account)
- Persistence: leave off — see "Nothing survives" below

**Secrets** (Add-ons → Secrets). Create these, then attach them to the notebook:

| secret | needed for |
|---|---|
| `GEMINI_API_KEY` | clip selection + scene context. Without it, no clips are chosen. |
| `GEMINI_API_KEYS` | *optional but recommended* — extra keys, comma-separated. `gemini_pool.GeminiKeyPool` rotates to the next on 429/quota/503, which is what keeps a long job alive on free-tier keys. |
| `ASSEMBLYAI_API_KEY` | transcription **with diarization**. Without it the pipeline falls back to faster-whisper, which has no diarization — and diarization is a framing input, so quality drops. |
| `YOUTUBE_COOKIES` | paste the full contents of a working `cookies.txt` (Netscape format). |

## Cell 1 — clone, configure, launch

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS",
             "ASSEMBLYAI_API_KEY", "YOUTUBE_COOKIES"):
    try:
        os.environ[name] = secrets.get_secret(name)
    except Exception:
        print(f"secret not set: {name}")

REPO = "https://github.com/foskigr8/openshorts.git"   # private: use a PAT URL
BRANCH = "session/framing-work"

if not os.path.isdir("/kaggle/working/openshorts"):
    subprocess.run(["git", "clone", "-b", BRANCH, REPO,
                    "/kaggle/working/openshorts"], check=True)
os.chdir("/kaggle/working/openshorts")

!bash kaggle_bootstrap.sh
```

The script prints a `https://<random>.trycloudflare.com` URL. Open it — that is
the full dashboard, same UI as local.

## Cell 2 — keep the session alive

Kaggle kills idle notebooks. The bootstrap backgrounds everything, so the cell
finishes immediately and the session can be reaped while a render is running.

```python
import time
while True:
    time.sleep(60)
    print(".", end="", flush=True)
```

---

## Nothing survives the session

`/kaggle/working` is wiped when the session ends, and a session is capped at
**12 hours** (GPU quota ~30h/week). So:

- **Get outputs off the box as they finish.** The repo already has
  `s3_uploader.py`; set `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
  `AWS_REGION` / `AWS_S3_BUCKET` as Kaggle secrets and clips upload in the
  background. Backblaze B2 works through the same S3 API.
- Or download each clip through the dashboard before you stop the notebook.
- Re-cloning each session costs a few minutes of `pip install` and an `npm
  build`. To skip the npm build, commit `dashboard/dist`.

## Using both GPUs

A single clip renders on one GPU — the stages are sequential, so GPU 1 idles.
The parallelism is **per clip**: a job produces several clips, and they are
independent. Shard them across devices with one worker per GPU:

```
CUDA_VISIBLE_DEVICES=0 ... YOLO_DEVICE=0   # worker A
CUDA_VISIBLE_DEVICES=1 ... YOLO_DEVICE=0   # worker B (device 0 *within* its mask)
```

This is **not yet implemented** — `app.py`'s queue is single-process. Worth
doing only after the single-GPU path is confirmed working, because one T4
already removes most of the wall-clock pain.

## Known sharp edges

1. **Cookies expire in about 3 hours.** Measured on 4-aug-2026: a jar that
   downloaded fine at 17:38 was rejected by 20:43 with *"Sign in to confirm
   you're not a bot"*. Locally an auto-refresh loop drives a headless Chrome to
   renew them; **that loop cannot run on Kaggle**. So a long session will lose
   YouTube access partway through, and the fix is to re-paste the secret. This
   is the weakest part of the whole setup — plan around it (upload sources as a
   Kaggle Dataset for long runs).
2. **`cookie_health.py` is a structural check, not a functional one.** It
   verifies cookie *names* are present and will happily report `OK` for a jar
   YouTube rejects. To actually test:
   `yt-dlp --cookies cookies.txt --skip-download --print "%(title)s" <url>`
3. **The tunnel hostname changes every session.** A named Cloudflare tunnel with
   your own domain fixes this; the quick tunnel used here does not.
4. **SABR warnings are normal.** `Some tv client https formats have been
   skipped… SABR-only streaming experiment` is expected; the download path
   falls through `tv_embed → android → mweb → web`, then an iOS spoof.
5. **Kaggle's base image owns torch/opencv/numpy.** The bootstrap deliberately
   holds those back from `requirements.txt` — reinstalling them is slow and can
   break the image's CUDA build.
