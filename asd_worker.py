"""Active Speaker Detection (LR-ASD) — who is actually speaking, from lip-sync.

WHY THIS EXISTS
---------------
Speaker attribution currently comes from AssemblyAI diarization, which clusters
VOICE EMBEDDINGS. That works well on clean English conversation (measured: 9
speakers correctly separated on a dating-show transcript) and fails on
code-switched multilingual audio (measured: the same pipeline collapsed a
multi-person, multi-language video to a single speaker across all 463
segments). Diarization models are trained predominantly on English, so heavy
code-switching is a known weak spot.

Lip movement is language-independent. A mouth moving in sync with the audio
reads the same in Arabic, Tagalog or Portuguese, so ASD degrades gracefully
exactly where diarization degrades badly.

It also answers a question diarization structurally cannot: not just "how many
distinct voices" but "WHICH ON-SCREEN FACE is producing this audio" — which is
what the reframing camera actually needs to decide who to frame.

MODEL
-----
LR-ASD (Liao et al., IJCV 2025), vendored under vendor/lrasd/ (MIT licence).
0.84M parameters, 3.4MB of weights — negligible next to Whisper on a T4.

INPUT CONTRACT (fixed by the pretrained weights, not by us)
-----------------------------------------------------------
  video : grayscale 112x112 face crops at 25 fps
  audio : 16 kHz mono, MFCC numcep=13, winlen=0.025, winstep=0.010 (100 fps)
  ratio : exactly 4 audio frames per video frame

The upstream demo hardcodes ``.cuda()``; this runs on whatever device is
available so the same code path works on a CPU box and on the T4.
"""
import os

import numpy as np

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "lrasd")
DEFAULT_WEIGHTS = os.path.join(_VENDOR_DIR, "weight", "finetuning_TalkSet.model")

# Video/audio frame rates the pretrained network expects. Not tunable.
ASD_VIDEO_FPS = 25
ASD_AUDIO_FPS = 100
FACE_SIZE = 112

# The reference implementation averages predictions over 11 differently-sized
# temporal windows for a small accuracy gain. That is 11x the compute for a
# marginal benefit.
#
# Chunk length is almost pure overhead, not compute: the FLOPs are identical
# either way, but a 1s chunk runs 5x as many Python-loop iterations and
# batch=1 forward passes as a 5s one. Measured on 10s of video: (1,3) took
# 3.17s, (1,) 2.05s, (5,) 1.35s — same arithmetic, 2.3x apart. A 3s window is
# long enough to amortise the loop while still resolving turn-taking at
# conversational speed.
DEFAULT_DURATIONS = (3,)


def available(weights=DEFAULT_WEIGHTS):
    """True when the model can actually be built and its weights exist."""
    if not os.path.exists(weights):
        return False
    try:
        import torch  # noqa: F401
        import python_speech_features  # noqa: F401
        from vendor.lrasd.model.Model import ASD_Model  # noqa: F401
        return True
    except Exception:
        return False


def _durations():
    raw = os.environ.get("ASD_DURATION_SET", "").strip()
    if not raw:
        return DEFAULT_DURATIONS
    try:
        vals = tuple(int(v) for v in raw.replace(",", " ").split() if int(v) > 0)
        return vals or DEFAULT_DURATIONS
    except ValueError:
        return DEFAULT_DURATIONS


def audio_features(wav_path):
    """16 kHz mono WAV -> (N, 13) MFCC at 100 fps, matching the training setup."""
    import python_speech_features
    from scipy.io import wavfile

    rate, audio = wavfile.read(wav_path)
    if audio.ndim > 1:                      # stereo -> mono
        audio = audio.mean(axis=1)
    return python_speech_features.mfcc(
        audio, rate, numcep=13, winlen=0.025, winstep=0.010)


def crop_face(frame, box, scale=0.40):
    """One 112x112 grayscale face crop in the layout the weights expect.

    The reference pads the detected box, resizes to 224x224 and takes the
    centre 112x112 — so the network sees the face filling the frame with a
    little context. Reproduced exactly; a differently-framed crop would be
    out of distribution for the pretrained weights.
    """
    import cv2

    x, y, w, h = [int(v) for v in box]
    cx, cy = x + w / 2.0, y + h / 2.0
    half = max(w, h) * (1.0 + scale) / 2.0
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = int(round(cx + half)), int(round(cy + half))

    fh, fw = frame.shape[:2]
    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - fw), max(0, y2 - fh)
    if pad_l or pad_t or pad_r or pad_b:
        frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x1 += pad_l; x2 += pad_l; y1 += pad_t; y2 += pad_t

    patch = frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
    if patch.size == 0:
        return np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.uint8)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    big = cv2.resize(gray, (224, 224))
    return big[56:168, 56:168]


class ASDScorer:
    """Scores face-crop sequences against an audio track."""

    def __init__(self, weights=DEFAULT_WEIGHTS, device=None):
        import torch
        from vendor.lrasd.model.Model import ASD_Model
        from vendor.lrasd.loss import lossAV

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = ASD_Model().to(self.device)
        self.loss_av = lossAV().to(self.device)

        # The checkpoint is a flat state dict over the whole ASD wrapper
        # ("model.*" and "lossAV.*"), so it's split across the two modules.
        state = torch.load(weights, map_location=self.device, weights_only=False)
        model_sd, loss_sd = {}, {}
        for k, v in state.items():
            if k.startswith("model."):
                model_sd[k[len("model."):]] = v
            elif k.startswith("lossAV."):
                loss_sd[k[len("lossAV."):]] = v
        self.model.load_state_dict(model_sd, strict=False)
        self.loss_av.load_state_dict(loss_sd, strict=False)
        self.model.eval()
        self.loss_av.eval()

    def score(self, faces, mfcc, durations=None):
        """Per-video-frame speaking scores for ONE track.

        faces : (T, 112, 112) uint8 grayscale crops at 25 fps
        mfcc  : (>=4T, 13) MFCC at 100 fps for the SAME time span
        Returns a float array of length T — higher means more likely speaking.
        """
        torch = self.torch
        faces = np.asarray(faces, dtype=np.float32)
        if faces.ndim != 3 or len(faces) == 0:
            return np.zeros(0, dtype=np.float32)

        # Trim both streams to a common length on the exact 4:1 grid the
        # network was trained on; a mismatch here silently misaligns lips
        # from sound, which is the one thing this model must get right.
        length = int(min(len(mfcc) // 4, len(faces)))
        if length <= 0:
            return np.zeros(len(faces), dtype=np.float32)
        mfcc = np.asarray(mfcc, dtype=np.float32)[:length * 4]
        faces = faces[:length]

        all_scores = []
        for duration in (durations or _durations()):
            step = max(1, int(duration))
            # Equal-length chunks go through as ONE batch. The previous loop
            # ran a separate batch=1 forward pass per chunk, which on a GPU
            # means the device sits idle between kernel launches — the work is
            # only ~8.3 GFLOP per second of video, so launch overhead, not
            # arithmetic, was the limit. Batching is also what lets a T4 do
            # this at a useful fraction of its throughput instead of ~5%.
            starts = [i for i in range(0, length, step)
                      if i + step <= length and len(mfcc[i * 4:(i + step) * 4]) == step * 4]
            # Batch on GPU, don't on CPU. Measured both ways on identical
            # input: CUDA 19.1x -> 48.5x realtime batched, but CPU 3.9x -> 2.1x
            # (it is already saturated at batch=1, so a big batch only adds
            # memory pressure). Outputs are bit-identical either way, so this
            # is purely a scheduling choice.
            max_batch = len(starts) if self.device.type == "cuda" else 1
            chunk_scores = []
            with torch.no_grad():
                for b in range(0, len(starts), max(1, max_batch)):
                    grp = starts[b:b + max(1, max_batch)]
                    vb = np.stack([faces[i:i + step] for i in grp])
                    ab = np.stack([mfcc[i * 4:(i + step) * 4] for i in grp])
                    in_v = torch.from_numpy(vb).to(self.device)
                    in_a = torch.from_numpy(ab).to(self.device)
                    emb_a = self.model.forward_audio_frontend(in_a)
                    emb_v = self.model.forward_visual_frontend(in_v)
                    out = self.model.forward_audio_visual_backend(emb_a, emb_v)
                    chunk_scores.extend(self.loss_av.forward(out, labels=None))
                # Ragged tail (a final partial chunk) keeps the single-item
                # path — batching needs uniform shapes.
                tail = starts[-1] + step if starts else 0
                if tail < length:
                    v = faces[tail:length]
                    a = mfcc[tail * 4:length * 4]
                    if len(v) and len(a):
                        in_v = torch.from_numpy(v).unsqueeze(0).to(self.device)
                        in_a = torch.from_numpy(a).unsqueeze(0).to(self.device)
                        emb_a = self.model.forward_audio_frontend(in_a)
                        emb_v = self.model.forward_visual_frontend(in_v)
                        out = self.model.forward_audio_visual_backend(emb_a, emb_v)
                        chunk_scores.extend(self.loss_av.forward(out, labels=None))
            if len(chunk_scores) >= length:
                all_scores.append(np.asarray(chunk_scores[:length], dtype=np.float32))

        if not all_scores:
            return np.zeros(length, dtype=np.float32)
        return np.mean(np.stack(all_scores, axis=0), axis=0)


def _extract_wav(video_path, wav_path):
    import subprocess
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", video_path,
         "-ac", "1", "-ar", "16000", "-vn", wav_path],
        check=True)


def score_clip(video_path, detect_faces, identity=None, max_tracks=6,
               device=None, min_track_seconds=0.6):
    """Who is speaking, per second, for one already-cut clip.

    Returns ``{"per_second": [track_id_or_None, ...],
               "tracks": {track_id: np.ndarray of per-frame scores},
               "fps": 25}``

    Deliberately clip-scoped, never whole-source. The arithmetic is trivial
    (8.3 GFLOP per second of video per track) but it is per TRACK per SECOND,
    so a 47-minute source with two faces on screen is ~140k track-frames —
    tens of minutes of work for footage that will be thrown away. Scoring only
    the handful of clips the selector already chose keeps it bounded.

    ``detect_faces(frame) -> [{'box': (x,y,w,h), 'score': float}, ...]`` is
    injected rather than imported so this module stays independent of main.py
    (which pulls in the whole CV stack) and is testable with a fake detector.
    """
    import cv2
    import tempfile

    workdir = tempfile.mkdtemp(prefix="asd_")
    try:
        wav = os.path.join(workdir, "a.wav")
        _extract_wav(video_path, wav)
        mfcc = audio_features(wav)

        # The network's audio/video stride ratio is baked into its weights, so
        # frames MUST arrive at 25 fps. The obvious way to get there is an
        # ffmpeg `-r 25` re-encode — but that was measured at 14.3s for a 10s
        # clip, three quarters of this whole pass and 25x the cost of the model
        # itself. Re-encoding is also pure waste here: nothing downstream keeps
        # the video, only the crops. So the source is decoded once at its own
        # rate and the nearest real frame to each 25 fps timestamp is selected.
        # Same sampling grid, no encode.
        cap_probe = cv2.VideoCapture(video_path)
        src_fps = cap_probe.get(cv2.CAP_PROP_FPS) or ASD_VIDEO_FPS
        cap_probe.release()
        if src_fps <= 0:
            src_fps = ASD_VIDEO_FPS

        if identity is None:
            try:
                import identity_tracker
                if identity_tracker.available():
                    identity = identity_tracker.IdentityTracker(
                        detection_fps=ASD_VIDEO_FPS, tracker_type="bytetrack")
            except Exception:
                identity = None

        crops, boxes, n_frames = {}, {}, 0
        cap = cv2.VideoCapture(video_path)
        src_index = 0          # index in the SOURCE stream
        next_wanted = 0.0      # source index of the next 25 fps sample
        step = src_fps / float(ASD_VIDEO_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_index < int(round(next_wanted)):
                src_index += 1
                continue       # source frame between 25 fps samples — skip
            src_index += 1
            next_wanted += step
            cands = detect_faces(frame) or []
            if identity is not None:
                identity.update(cands, frame, frame_number=n_frames)
            else:
                for i, c in enumerate(cands):
                    c.setdefault("id", i)
            for c in cands:
                crops.setdefault(c["id"], {})[n_frames] = crop_face(frame, c["box"])
                boxes.setdefault(c["id"], {})[n_frames] = tuple(c["box"])
            n_frames += 1
        cap.release()

        if not n_frames or not crops:
            return {"per_second": [], "per_second_box": [],
                    "per_second_margin": [], "tracks": {},
                    "fps": ASD_VIDEO_FPS}

        # Only score tracks with real screen presence, biggest first — a
        # one-frame false positive costs a whole scoring pass otherwise.
        min_frames = max(1, int(min_track_seconds * ASD_VIDEO_FPS))
        ranked = sorted(((tid, f) for tid, f in crops.items() if len(f) >= min_frames),
                        key=lambda kv: -len(kv[1]))[:max_tracks]
        if not ranked:
            return {"per_second": [], "per_second_box": [],
                    "per_second_margin": [], "tracks": {},
                    "fps": ASD_VIDEO_FPS}

        # No explicit device: use whatever GPU this clip worker was assigned,
        # so concurrent clips do not all pile onto GPU 0 (gpu_affinity). Falls
        # through to ASDScorer's own cuda-if-available choice when nothing was
        # assigned, which is the single-GPU / CPU behaviour.
        if device is None:
            try:
                import gpu_affinity
                device = gpu_affinity.current_device()
            except Exception:
                device = None
        scorer = ASDScorer(device=device)
        blank = np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.uint8)
        tracks = {}
        for tid, byframe in ranked:
            seq = np.stack([byframe.get(i, blank) for i in range(n_frames)])
            scores = scorer.score(seq, mfcc)
            # A frame where this face was absent carries no evidence either
            # way; leaving the model's reading of a black frame in would let
            # "not on screen" masquerade as "confidently silent".
            for i in range(len(scores)):
                if i not in byframe:
                    scores[i] = np.nan
            tracks[tid] = scores

        # Collapse to one speaker per second: the loudest positive mean wins,
        # None when nobody is clearly speaking.
        per_second, per_second_box, per_second_margin = [], [], []
        secs = int(np.ceil(min(len(v) for v in tracks.values()) / ASD_VIDEO_FPS))
        for sec in range(secs):
            lo, hi = sec * ASD_VIDEO_FPS, (sec + 1) * ASD_VIDEO_FPS
            means = []
            for tid, sc in tracks.items():
                window = sc[lo:hi]
                window = window[~np.isnan(window)]
                if window.size == 0:
                    continue
                means.append((float(np.mean(window)), tid))
            best, best_val = None, 0.0
            for val, tid in means:
                if val > best_val:
                    best, best_val = tid, val
            per_second.append(best)
            # How DECISIVE this second was: the winner's lead over the next
            # best on-screen face. A second where two faces score nearly the
            # same is exactly the "reacting listener vs talker" ambiguity that
            # used to bind a speaker to the wrong track — the consumer
            # (speaker_fusion) drops those seconds from the vote instead of
            # letting them pollute the one-per-clip mapping. Nobody else on
            # screen means nothing to confuse the winner with, so the lead is
            # measured against the silence baseline (0.0). None where there
            # was no positive winner at all.
            if best is None:
                per_second_margin.append(None)
            else:
                runner = max((v for v, tid in means if tid != best),
                             default=0.0)
                per_second_margin.append(best_val - max(runner, 0.0))
            # The WHERE matters more than the which. This pass runs its own
            # tracker at 25 fps; the renderer tracks at its own detection
            # stride, so the two id spaces are unrelated and an id would be
            # meaningless across the boundary. A position is comparable in any
            # id space, so callers match the speaker by location instead.
            box = None
            if best is not None:
                seen = boxes.get(best) or {}
                near = [f for f in seen if lo <= f < hi]
                if near:
                    box = seen[min(near, key=lambda f: abs(f - (lo + hi) // 2))]
            per_second_box.append(box)
        return {"per_second": per_second, "per_second_box": per_second_box,
                "per_second_margin": per_second_margin,
                "tracks": tracks, "fps": ASD_VIDEO_FPS}
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
