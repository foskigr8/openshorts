# Pop The Balloon framing validation — 3 Aug 2026

Both engines rendered over the same two spans of
`Ep_115_Pop_The_Balloon_Or_Find_Love__With_Arlette_Amuli`
(https://youtu.be/ua9Z0Lq3QVA), called exactly like `main.py:1603`
(`render(..., transcript=<full 1877-segment diarized transcript>,
clip_start=..., clip_end=...)`; no focus directives — same as the harness
measurements in the framing session). Old engine = commit `2530352`
(`PYTHONPATH=/tmp/oldengine:/app`), new engine = current
`session/framing-work`.

## Source

Re-downloaded via yt-dlp with `/app/cookies.txt` + bgutil PO-token
(`--extractor-args "youtubepot-bgutilhttp:base_url=http://bgutil-pot:4416"`).
The jar authenticates (both spans downloaded; `pot:bgutil:http` lines in the
verbose log are informational, the old "Script path doesn't exist" warnings
are gone). `/app/cookies.txt` is a live bind-mount copy of `cookies.txt`.

- span 1: 481.28–608.26 (127.0s, 1920x1080@29.97)
- span 2: 2222.807–2250.88 (28.1s, 1920x1080@29.97)

## Measurements

### Span 1 (127s, 11 source shots, 2–6 faces in every detection frame)

| metric | new engine | old engine |
|---|---|---|
| policy decisions | 952 | — |
| ASD named a speaker | 664 frames | — (old ran with USE_ASD=0) |
| framed the ASD speaker | 653/664 = **98%** | — |
| tier mix | lip-sync 69%, diarized 28%, held 3%, **size 0%** | — |
| policy shot commits | 39 (10 pairs <0.7s, mostly micro re-centres in the 5-person lineup) | — |
| pixel-delta cuts / rapid | 48 / 16 | 49 / 8 |
| TransNetV2 scenes | 42 | 42 |
| perceived cuts (Gemini) | 6–7 | 14–25+ |

### Span 2 (28.1s, 3 source shots, 9 source cuts incl. 7 rapid pairs)

| metric | new engine | old engine |
|---|---|---|
| policy decisions | 211 | — |
| ASD named a speaker | 182 frames | — |
| framed the ASD speaker | 176/182 = **97%** | — |
| tier mix | lip-sync 86%, diarized 7%, held 7%, **size 0%** | — |
| policy shot commits | 21 (3 pairs <0.7s; 8 switches in the first 9s) | — |
| pixel-delta cuts / rapid | 21 / 5 | 14 / 3 |
| TransNetV2 scenes | 21 | 13 |
| perceived cuts (Gemini SBS) | 4 purposeful | 7 snapping |

## Video review (Gemini 3.1-flash-lite, fps 4)

### Span 1 — side-by-side (old left / new right)
- Old: misses the speaker (host/mic-holder framed while the guest talks,
  00:03–00:13), constant hunting jitter, frames the gap between people
  (00:06, 00:36), ~14 cuts — **unwatchable**.
- New: consistently on the speaker, stable lock-on, ~6 deliberate cuts —
  **watchable**.

### Span 2 — side-by-side
- Old: lingers on listeners, snaps between subjects 0.5–0.8s apart (00:05,
  00:09), frames dead space, 7 cuts — **unwatchable**.
- New: holds the speaker, smooth transitions, 4 purposeful cuts (00:14–00:16
  pan to the woman as she starts speaking) — **watchable**.

### Individual clips (doc's harsh-editor prompt)
- Span 1 new: speaker tracked throughout, no twitch, no head crop, 7 natural
  cuts — **watchable yes**.
- Span 1 old: interviewer framed while guest speaks (00:04–00:08), constant
  reframing twitch (00:25–00:35, 01:04–01:16), gap framing, 25+ cuts —
  **watchable no**.
- Span 2 new: generally tracks the speaker, but early jitter (00:01–00:05),
  right-side dead space at 00:06, ~16 cuts (most inherited from the source's
  9 cuts in 28s) — **watchable no** (harsh verdict; SBS comparison says yes).
- Span 2 old: off-speaker framing, subjects too low/heads at top edge, gap
  framing, 16 cuts — **watchable no**.

### Pixel cross-checks (studio rule: verify Gemini spatial claims)
- No head cropping at any sampled span-2 frame (12 frames, both engines) —
  the "heads cropped at top edge" claims were **not** reproduced in pixels.
- Frames 18s/20s of span 2 show no person in BOTH engines because the source
  itself is a feet/transition shot there — faithful, not an engine failure.
- Dead space on one side exists in BOTH engines at several span-2 frames; it
  is the fixed 3:4 crop of a 5-person wide lineup, not new-engine-specific.

## Verdict

The new engine is better on the real content. It frames the actual speaker
98%/97% of the time (the old engine never ran LR-ASD — `USE_ASD` shipped at
0), `size` decided 0% of frames on both spans, and reviewers judged it
watchable where they judged the old engine unwatchable. Residual, honestly
reported: in dense banter (span 2, first 9s) the new policy still re-aims
every ~1s on lip-sync evidence, and the fixed 3:4 crop of the wide lineup
leaves side dead space in both engines. No tuning was done against those —
the pixel-delta/scene-count metrics contradict the reviews, so cut frequency
remains a smoke alarm, not a target (per the handoff doc).

## Watch it

UI history (both jobs, status completed):

- `ptb-span1-framing-3aug` — clips: 1 side-by-side, 2 new, 3 old
- `ptb-span2-framing-3aug` — clips: 1 side-by-side, 2 new, 3 old

`curl -s http://localhost:8000/api/history | grep ptb-span`

Source spans in the container: `/tmp/ptb_span1.mp4`, `/tmp/ptb_span2.mp4`;
renders in `/tmp/ptb_span{1,2}_{new,old}.mp4`. Harnesses:
`scratch_ptb/ptb_{render,policycheck,scenecheck,cutsource}.py`.

Tests: **650 passed** (unchanged — no code was modified by this validation).
