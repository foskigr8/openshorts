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

## Downloads: Kaggle does NOT need cookies

Measured on Kaggle with `kaggle_download_probe.py` (5-aug-2026), real 5-second
downloads, not metadata fetches:

    [PASS] anonymous (default clients)      0.2 MB in 6s
    [PASS] anonymous + ios client           0.1 MB in 4s
    [PASS] anonymous + tv_embed             0.1 MB in 5s
    [FAIL] anonymous + web_safari+fetch_pot  "This video is DRM protected"
    [PASS] cookies (cookies.txt)            0.2 MB in 5s
    [FAIL] cookies + web_safari+fetch_pot    "This video is DRM protected"

**Kaggle's IP is not flagged.** This is the opposite of the studio, where an
unauthenticated request hits the bot wall immediately, and it is a much better
position: nothing expires, so there is no 3-hour cookie clock to manage.

Two things follow:

1. **`YOUTUBE_COOKIES` is optional on Kaggle.** Set it only if you hit a
   region- or age-restricted video, which genuinely needs a signed-in session.
2. **The "DRM protected" failures are not DRM.** That is what a `web`-family
   client reports when it cannot produce a PO token. Those two strategies are
   exactly the ones needing the `bgutil-pot` sidecar, which does not exist on
   Kaggle (`BGUTIL_BASE_URL` is unset there). Nothing to fix — the pipeline
   skips its PO-token attempts automatically when that variable is absent.

The pipeline's own first attempt (`main.py`, the `attempts` list) is
`ios-spoof` with **no cookies and no proxy** — the same strategy the probe
measured passing. So the default path is already the right one for Kaggle.

## Known sharp edges

1. **Cookies expire in about 3 hours — but Kaggle does not need them.**
   Measured on the studio 4-aug-2026: a jar that downloaded fine at 17:38 was
   rejected by 20:43. The headless-Chrome refresh loop cannot run on Kaggle. If
   you do set `YOUTUBE_COOKIES` (only needed for restricted videos), expect it
   to go stale mid-session; the anonymous path has no such clock, so prefer it.
2. **`cookie_health.py` is a structural check, not a functional one.** It
   verifies cookie *names* are present and will happily report `OK` for a jar
   YouTube rejects. To actually test:
   `yt-dlp --cookies cookies.txt --skip-download --print "%(title)s" <url>`
3. **The tunnel hostname changes every session.** A named Cloudflare tunnel with
   your own domain fixes this; the quick tunnel used here does not.
4. **SABR warnings are normal.** `Some tv client https formats have been
   skipped… SABR-only streaming experiment` is expected; the download path
   falls through `tv_embed → android → mweb → web`, then an iOS spoof.
5. **protobuf: mediapipe and Kaggle's TensorFlow cannot both be satisfied.**
   Kaggle ships TensorFlow whose generated `*_pb2.py` need `protobuf >= 5.27`
   (when `runtime_version` appeared); mediapipe pins `protobuf < 5`. Installing
   mediapipe downgrades protobuf and breaks TF, and since mediapipe imports
   `tasks.python -> tensorflow`, `import mediapipe` then dies with
   `ImportError: cannot import name 'runtime_version'`. That killed the first
   Kaggle job (5-aug-2026).

   The bootstrap repairs this by **removing TensorFlow**, which is safe and not
   a hack: this pipeline has no TensorFlow import anywhere, it is not in
   `requirements.txt`, and the working Docker image has no TF at all — verified
   there, `mediapipe 0.10.14` imports against `protobuf 4.25.9`. Scene
   detection (TransNetV2) runs on torch, not TF.

   The repair retests the import after each remedy rather than trusting a
   version pin guessed from outside Kaggle — guessing pins is what caused this.

6. **Kaggle's base image owns torch/opencv/numpy.** The bootstrap deliberately
   holds those back from `requirements.txt` — reinstalling them is slow and can
   break the image's CUDA build.
