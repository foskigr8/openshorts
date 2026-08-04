import time
import cv2
import scenedetect
import subprocess
import shutil
import tempfile
import argparse
import re
import sys
import glob
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
import os
import math
import numpy as np
from tqdm import tqdm
import yt_dlp
import mediapipe as mp
# import whisper (replaced by faster_whisper inside function)
from google import genai
from google.genai import types as genai_types

import gemini_worker
import deepseek_worker
import gemini_pool
from clip_selection import build_transcript_windows, snap_clip_to_words
from ffmpeg_utils import (video_encode_args, audio_encode_args, QUALITY,
                          QUALITY_FAST, METADATA_SCRUB)
from pipeline_progress import write_progress as _write_progress
from pipeline_progress import mark_clip_ready as _mark_clip_ready
from pipeline_progress import record_stage_durations
from hardware_defaults import default_clip_workers
from dotenv import load_dotenv
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load environment variables
load_dotenv()

# --- Constants ---
ASPECT_RATIO = 9 / 16

GEMINI_PROMPT_TEMPLATE = """
You are a senior short-form video editor. Read the ENTIRE transcript and word-level timestamps to choose the 3–15 MOST VIRAL moments for TikTok/IG Reels/YouTube Shorts. Each clip must be between 15 and 60 seconds long.

⚠️ FFMPEG TIME CONTRACT — STRICT REQUIREMENTS:
- Return timestamps in ABSOLUTE SECONDS from the start of the video (usable in: ffmpeg -ss <start> -to <end> -i <input> ...).
- Only NUMBERS with decimal point, up to 3 decimals (examples: 0, 1.250, 17.350).
- Ensure 0 ≤ start < end ≤ VIDEO_DURATION_SECONDS.
- Each clip between 15 and 60 s (inclusive).
- Prefer starting 0.2–0.4 s BEFORE the hook and ending 0.2–0.4 s AFTER the payoff.
- Use silence moments for natural cuts; never cut in the middle of a word or phrase.
- STRICTLY FORBIDDEN to use time formats other than absolute seconds.

VIDEO_DURATION_SECONDS: {video_duration}

TRANSCRIPT_TEXT (raw):
{transcript_text}

WORDS_JSON (array of {{w, s, e}} where s/e are seconds):
{words_json}

STRICT EXCLUSIONS:
- No generic intros/outros or purely sponsorship segments unless they contain the hook.
- No clips < 15 s or > 60 s.

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments). Order clips by predicted performance (best to worst). In the descriptions, ALWAYS include a CTA like "Follow me and comment X and I'll send you the workflow" (especially if discussing an n8n workflow):
{{
  "shorts": [
    {{
      "start": <number in seconds, e.g., 12.340>,
      "end": <number in seconds, e.g., 37.900>,
      "video_description_for_tiktok": "<description for TikTok oriented to get views>",
      "video_description_for_instagram": "<description for Instagram oriented to get views>",
      "video_title_for_youtube_short": "<title for YouTube Short oriented to get views 100 chars max>",
      "viral_hook_text": "<SHORT punchy text overlay (max 10 words). MUST BE IN THE SAME LANGUAGE AS THE VIDEO TRANSCRIPT. Examples: 'POV: You realized...', 'Did you know?', 'Stop doing this!'>"
    }}
  ]
}}
"""

# Load the YOLO model once (Keep for backup or scene analysis if needed).
# The device is pinned at INFERENCE time (see detect_person_candidates_yolo)
# — ultralytics 8.4.46 does not accept `device` in the constructor.
model = YOLO('yolov8n.pt')
# YOLO_DEVICE (e.g. "0" in the GPU compose deployment) means this host is
# EXPECTED to have CUDA. Ultralytics 8.4.46 silently falls back to CPU even
# with device="0" when CUDA is broken, which would make the "GPU wired" claim
# untestable — so the env var doubles as a loud startup assertion: a job
# started on a GPU-configured deployment without working CUDA fails here with
# a clear message instead of burning GPU-credit time on CPU.
_YOLO_DEVICE = os.environ.get("YOLO_DEVICE")
if _YOLO_DEVICE and not torch.cuda.is_available():
    raise RuntimeError(
        f"YOLO_DEVICE={_YOLO_DEVICE} is set but CUDA is not available — "
        "refusing to silently fall back to CPU. Fix the GPU wiring (NVIDIA "
        "container runtime / driver) or unset YOLO_DEVICE for CPU operation.")

# --- MediaPipe Setup ---
# Use standard Face Detection (BlazeFace) for speed
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Consecutive detections a large target move must survive before the camera
# follows it (see SmoothedCameraman.update_target). Env-overridable so the
# damping can be dialled back without a deploy; 1 restores the old behaviour.
JUMP_CONFIRM_FRAMES = max(int(os.environ.get("JUMP_CONFIRM_FRAMES", "3")), 1)
# Fraction of crop width inside which a re-target is ignored as detector churn.
# This is what stops the group-shot jitter: faces closer together than this are
# both already framed, so alternating between them is motion with no benefit.
# 0 disables the hysteresis entirely (restores the old always-follow behaviour).
# How far the framed subject may drift from the crop centre before the camera
# corrects. 0.08 -> 0.035 (4-aug-2026): at a 3:4 crop off a 1080p source that
# band was ~65px, and combined with the 0.25 safe zone the speaker could sit a
# quarter of the crop width off centre indefinitely. Owner spec: "the
# composition of the person that is talking is center. If the person cannot be
# centered, that's when you can use rule of thirds."
#
# The band existed to damp A/B churn between two faces. That churn had a real
# cause — unstable ids and a person alternating between a face box and a body
# box — which is now fixed at source (subject_policy.same_subject,
# identity_tracker provisional inheritance), so the compensation can shrink.
RETARGET_DEAD_ZONE_FRAC = float(os.environ.get("RETARGET_DEAD_ZONE_FRAC", "0.035"))

# Long-shot follow (owner spec, 4-aug-2026). A held shot stays locked; only
# after it has run this long may the camera track a drifting subject, and then
# only slowly. Deliberately gentle: panning is what caused every previous
# stutter complaint, so this must never look like a pan.
LONG_SHOT_FOLLOW_SECONDS = float(os.environ.get("LONG_SHOT_FOLLOW_SECONDS", "3.0"))

# Composition. A subject within COMPOSE_SIDE_BAND of a source edge is placed on
# the near third (looking room in front of them); anyone nearer the middle is
# centred. COMPOSE_THIRD=1/3 is the classic line.
COMPOSE_THIRDS = os.environ.get("COMPOSE_THIRDS", "1").strip() not in ("0", "false", "no")
COMPOSE_THIRD = float(os.environ.get("COMPOSE_THIRD", "0.3333"))
COMPOSE_SIDE_BAND = float(os.environ.get("COMPOSE_SIDE_BAND", "0.34"))
# The subject must always sit at least this far inside the crop, whatever the
# thirds placement asks for. Containment beats composition.
COMPOSE_EDGE_MARGIN = float(os.environ.get("COMPOSE_EDGE_MARGIN", "0.14"))
LONG_FOLLOW_RATE = float(os.environ.get("LONG_FOLLOW_RATE", "0.04"))
LONG_FOLLOW_MAX_STEP = float(os.environ.get("LONG_FOLLOW_MAX_STEP", "3.0"))
LONG_FOLLOW_ACCEL = float(os.environ.get("LONG_FOLLOW_ACCEL", "0.35"))

# Where the crop's vertical centre sits inside the detection box, as a
# fraction DOWN the box (0.5 = box centre). Detection hands back either a
# MediaPipe face square or a YOLO head-and-chest rect; centering the 3:4
# crop on a body box's centre puts the head at the top edge and crops it
# off (measured 4-aug-2026 on Pop The Balloon span 2: the shocked-face
# moment rendered with "top of head cropped"). Anchoring on the head keeps
# it in frame with shoulders below — the same fix the split-cell path
# already ground-truthed (SPLIT_HEAD_FRACTION = 0.16, 31-jul-2026).
CAMERA_HEAD_ANCHOR = float(os.environ.get("CAMERA_HEAD_ANCHOR", "0.16"))

# Where the head anchor sits in the FINAL crop, as a fraction down its height.
# 0.36 places the eyes/head on the top-third grid line (user framing spec,
# 4-aug-2026: "place their eyes precisely on the top-third grid line to ensure
# they don't look awkwardly low in the frame") — same value the split-cell
# path uses for its final-crop head placement.
CAMERA_HEAD_Y = float(os.environ.get("CAMERA_HEAD_Y", "0.36"))

# --- Eased camera-motion tuning (SmoothedCameraman) ---
# The old get_crop_box moved at one of two CONSTANT speeds with a hard
# overshoot-snap at the end — no acceleration curve, so pans started and
# stopped abruptly (user: "it's not smooth"). These replace that with an
# acceleration-limited exponential chase: ramps up at EASE_ACCEL px/frame^2,
# cruises at most EASE_MAX_STEP px/frame, then eases in exponentially (the
# step is a fixed fraction of remaining distance, so it never overshoots and
# lands softly). All env-overridable, matching the codebase's tuning style.
# Camera style. "cut" (default) reproduces what this genre actually does;
# "pan" keeps the older eased-travel behaviour.
#
# Measured across four reference Pop-the-Balloon/blind-date shorts
# (31-jul-2026, camera-movement pass): ALL FOUR use 100% hard cuts and zero
# pans between subjects, hold each shot perfectly locked, run 1-4s per shot,
# cut instantly to a reaction and hold it 1-2s, and change focal length only
# AT a cut, never as a continuous move.
#
# That reframes every "stutter" complaint in this project. Sliding a crop
# window across static tripod footage is not a camera move — there is no real
# parallax, so it reads as the picture sliding, and at whole-pixel crop
# granularity it can only ever be judder. A locked shot cannot stutter, and a
# cut is instantaneous. So the fix is not smoother interpolation; it is not
# interpolating at all.
CAMERA_STYLE = os.environ.get("CAMERA_STYLE", "cut").strip().lower()

# Round-5 spec 2.3: every EASE_*/ZOOM_EASE_* constant below is read ONLY
# inside the "pan" branch of get_crop_box — with the default
# CAMERA_STYLE="cut" they are inert. Kept (pan is a supported option) but
# future tuning should not expect them to affect the default path.

EASE_RATE = float(os.environ.get("EASE_RATE", "0.12"))
EASE_ACCEL = float(os.environ.get("EASE_ACCEL", "1.0"))
EASE_MAX_STEP = float(os.environ.get("EASE_MAX_STEP", "14.0"))
EASE_SNAP_EPSILON = 1.0  # px — within this, snap and stop (kills 1px jitter churn)
# Floor on how far an in-progress move may travel per frame. The crop is cut
# at whole pixels, so a move whose eased tail is still "running" at 0.3px/frame
# does not render as slow motion — it renders as FROZEN for three frames and
# then a 1px jump, over and over (user: "the freezing is really annoying").
# Anything below one whole pixel per frame is invisible-then-jumpy, so a move
# either advances a real pixel or it is finished.
EASE_MIN_STEP = float(os.environ.get("EASE_MIN_STEP", "1.0"))
# Zoom is a scale factor on the base crop size (1.0 = full 3:4 crop, <1 = push
# in). Eased the same way, slower than x-pan so a push-in reads as deliberate.
ZOOM_EASE_RATE = float(os.environ.get("ZOOM_EASE_RATE", "0.12"))
ZOOM_EASE_ACCEL = float(os.environ.get("ZOOM_EASE_ACCEL", "0.006"))
ZOOM_EASE_MAX_STEP = float(os.environ.get("ZOOM_EASE_MAX_STEP", "0.045"))
ZOOM_EASE_SNAP_EPSILON = 0.004
# Same floor as EASE_MIN_STEP, in zoom units: the crop size is quantized to
# even pixels, so a zoom creeping slower than ~2px of crop width per frame
# just stalls and jumps like the pan did. On a ~660px crop, 0.004 ≈ 2.6px.
ZOOM_EASE_MIN_STEP = float(os.environ.get("ZOOM_EASE_MIN_STEP", "0.004"))
# Tightest push-in allowed (1.0 = no zoom). 0.7 = ~1.43x magnification.
ZOOM_MIN = float(os.environ.get("ZOOM_MIN", "0.7"))

# --- anti-static Ken Burns drift -------------------------------------------
# A locked-off crop was once thought to read as a frozen frame, so a slow
# sinusoidal pan was added to give a static shot "life".
#
# It is OFF by default now, because it caused the very complaint it was meant
# to cure. At the default 1.5% amplitude over 6s the drift advances ~0.02px
# per frame — far below the whole pixel the crop is actually cut at. So it
# does not render as gentle motion; it renders as a locked frame that twitches
# one pixel every couple of seconds, and a sine's slow turnarounds guarantee
# sub-pixel crawl at both ends no matter how the amplitude is tuned.
#
# A genuinely locked shot is clean, and the honest fix for a shot that feels
# static is a deliberate push-in (see the zoom policy), which now completes
# decisively instead of creeping. Set STATIC_DRIFT_AMPLITUDE to re-enable.
STATIC_DRIFT_FRAMES = max(int(os.environ.get("STATIC_DRIFT_FRAMES", "15")), 1)
STATIC_DRIFT_PERIOD_S = max(float(os.environ.get("STATIC_DRIFT_PERIOD_S", "6.0")), 1.0)
STATIC_DRIFT_AMPLITUDE = float(os.environ.get("STATIC_DRIFT_AMPLITUDE", "0.0"))


class SmoothedCameraman:
    """
    Handles smooth camera movement: eased pans on x, plus an eased zoom
    (crop-size) state and a dynamic y-centre so a push-in stays framed on the
    subject's face. Motion follows an acceleration-limited exponential chase —
    ramps up, cruises at a capped speed, then eases in — instead of the old
    constant-speed-with-hard-overshoot-snap, which read as mechanical
    (user: "much more better but it's not smooth").
    """
    def __init__(self, output_width, output_height, video_width, video_height,
                 aspect_ratio=ASPECT_RATIO, fps=None):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        self.aspect_ratio = aspect_ratio
        self.fps = float(fps) if fps else 30.0
        self._static_frames = 0  # consecutive frames with zero eased motion
        # A shot must have been locked this long before the camera is allowed
        # to FOLLOW a drifting subject instead of teleporting to them.
        self.long_shot_follow_frames = max(
            1, int(LONG_SHOT_FOLLOW_SECONDS * (float(fps) if fps else 30.0)))

        # Base crop dims at zoom=1.0 (the full defined crop). Zoom scales these
        # while keeping the aspect ratio constant — see get_crop_box.
        self.crop_height = video_height
        self.crop_width = int(self.crop_height * aspect_ratio)
        if self.crop_width > video_width:
             self.crop_width = video_width
             self.crop_height = int(self.crop_width / aspect_ratio)

        # Initial State — x pan, y centre (only matters when zoomed in below
        # full height), and zoom level (1.0 = base crop, <1.0 = tighter).
        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2
        self.current_center_y = video_height / 2
        self.target_center_y = video_height / 2
        self.current_zoom = 1.0
        self.target_zoom = 1.0
        self.min_zoom = ZOOM_MIN
        self._vx = 0.0  # x velocity (px/frame), for the eased chase
        self._vy = 0.0  # y velocity (px/frame)
        self._vz = 0.0  # zoom velocity (scale/frame)

        # Safe Zone: 25% of the crop width. Now used only to gate big target
        # jumps (see update_target) — the eased chase below no longer freezes
        # inside it, so small natural drift gets compensating motion instead
        # of abrupt catch-ups (the old "DO NOTHING" dead-zone is gone).
        self.safe_zone_radius = self.crop_width * 0.25

        # Band inside which a new target is treated as detector churn rather
        # than movement (see update_target). Expressed as a fraction of crop
        # width so it scales with the output shape: at 8%, two faces that close
        # together are both comfortably inside the frame already, so switching
        # between them buys nothing and costs stability.
        self.retarget_dead_zone = self.crop_width * RETARGET_DEAD_ZONE_FRAC

        # A target that teleports further than the safe zone in one detection is
        # far more often a detector error — a second face, a false positive, a
        # box snapping to a different body part — than a person who actually
        # moved that far. Committing to it immediately is what made the camera
        # swing: measured on real user footage, 22% of target updates jumped
        # more than the entire safe zone. So a big move has to REPEAT this many
        # times before the camera follows it; a wrong reading disappears on the
        # next detection and never moves the frame.
        #
        # The cost is latency on a genuinely fast move: at DETECT_STRIDE=4 and
        # 30fps, three confirmations is ~0.4s. That reads as an operator being
        # unhurried, which is the look we want, and it is far cheaper than the
        # whip-panning it replaces.
        #
        # Measured over 262s of TRACK footage from two real user videos
        # (26-jul-2026), confirm=1 -> 3: in-scene reversals 0.41/s -> 0.13/s
        # (-69%), camera travel 91px/s -> 60px/s (-34%). Per scene, 54 of 84 get
        # calmer and 23 are unchanged — but 7 get BUSIER, up to 59 -> 108px/s,
        # because committing later can leave the camera further to travel. Net
        # strongly positive, not universally so.
        self.jump_confirm_frames = JUMP_CONFIRM_FRAMES
        self._pending_target = None
        self._pending_count = 0
        # Set by the caller right after a scene/strategy cut (e.g. GENERAL
        # switching to TRACK on a speaker turn) — the jump-confirm gate below
        # exists to filter false-positive detector jumps DURING a continuous
        # shot; it has no such ambiguity to resolve on the first detection
        # of a brand new shot, and gating it there just meant the camera was
        # still crawling towards the real target after the shot requiring it
        # had already ended (ground-truthed 31-jul-2026).
        self.force_next_update = False

    def _eased_step(self, current, target, rate, max_step, max_accel, prev_step,
                    snap_eps, min_step=0.0):
        """One frame of an acceleration-limited exponential chase.

        The step is a fixed fraction (``rate``) of the remaining distance, so
        it decelerates and never overshoots; an acceleration cap adds a
        ramp-up instead of an instant start; a max-step caps cruise speed.
        Returns (new_current, new_step) and snaps exactly onto the target
        inside ``snap_eps`` (which also kills 1px jitter churn in the render).

        ``min_step`` floors the speed of a move that is still in progress —
        an exponential tail asymptotes, so without it the last stretch of
        every move crawls at a fraction of a pixel and the whole-pixel crop
        renders that as stall/jump/stall/jump. With it, a move always makes
        visible progress and then ends, instead of never quite arriving.
        """
        err = target - current
        if abs(err) <= snap_eps:
            return target, 0.0
        step = err * rate
        if abs(step) > abs(prev_step) + max_accel:
            step = math.copysign(abs(prev_step) + max_accel, step)
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
        if min_step and abs(step) < min_step:
            step = math.copysign(min_step, step)
        # Flooring the step can now exceed what's left — land on the target
        # rather than overshooting past it.
        if abs(step) >= abs(err):
            return target, 0.0
        return current + step, step

    def update_target(self, face_box, zoom_target=None):
        """Update the target centre from a detection, ignoring lone big jumps.

        ``zoom_target`` (optional): crop-size scale to ease toward (1.0 = base
        crop, <1.0 = push in). It's only accepted alongside an x-move that
        passed the jump-confirm gate, so a false-positive box can't trigger an
        unwanted zoom either. y/zoom follow the accepted box immediately —
        they're smaller, smoother signals than the x teleports the gate guards.
        """
        if not face_box:
            return
        x, y, w, h = face_box
        new_center = x + w / 2

        if self.force_next_update:
            self.force_next_update = False
            self._pending_target = None
            self._pending_count = 0
            self.target_center_x = new_center
        else:
            # MICRO-MOVE HYSTERESIS — the multi-person jitter fix.
            #
            # The confirm gate below only guards jumps LARGER than the safe
            # zone. In a group shot two faces are usually closer together than
            # that, so the detector alternating between them produced a
            # sub-safe-zone delta that was committed instantly, every single
            # detection. The eased chase then reversed direction continuously:
            # the "subtle jitter" that reads as the camera being unsure.
            #
            # If the new target is this close to the committed one, framing on
            # either subject covers both anyway, so holding is not just calmer,
            # it's visually correct. This is hysteresis, not a dead zone: the
            # target is NOT updated while inside the band, so a genuinely
            # moving subject accumulates error against a fixed reference and
            # crosses the band on their own, then gets followed to their true
            # position. Slow real movement still tracks; A/B churn does not.
            if abs(new_center - self.target_center_x) < self.retarget_dead_zone:
                self._pending_target = None
                self._pending_count = 0
                return
            if abs(new_center - self.target_center_x) > self.safe_zone_radius:
                # Same big move as last time? Count it. Otherwise start counting
                # afresh — two contradictory outliers must not confirm each other.
                if (self._pending_target is not None
                        and abs(new_center - self._pending_target) <= self.safe_zone_radius):
                    self._pending_count += 1
                else:
                    self._pending_target = new_center
                    self._pending_count = 1
                if self._pending_count < self.jump_confirm_frames:
                    return  # not convinced yet — hold the frame
            self._pending_target = None
            self._pending_count = 0
            self.target_center_x = new_center

        # y/zoom follow the accepted target, anchored on the head (see
        # CAMERA_HEAD_ANCHOR) rather than the box centre — a body box centred
        # at 0.5 cuts the head off the top of the crop.
        self.target_center_y = y + h * CAMERA_HEAD_ANCHOR
        if zoom_target is not None:
            self.target_zoom = min(max(float(zoom_target), self.min_zoom), 1.0)

    def _place_frac(self, center_x):
        """Where in the crop the subject should sit, 0..1 (0.5 = centred).

        Returns a third-line placement only when the subject is meaningfully
        off-centre in the SOURCE frame — that is the case where centring
        would clamp and strand them against the crop edge anyway.
        """
        if not COMPOSE_THIRDS:
            return 0.5
        rel = center_x / max(1.0, float(self.video_width))   # 0..1 in source
        if rel <= COMPOSE_SIDE_BAND:
            return COMPOSE_THIRD          # left of frame -> sit on left third
        if rel >= 1.0 - COMPOSE_SIDE_BAND:
            return 1.0 - COMPOSE_THIRD    # right of frame -> right third
        return 0.5

    def get_crop_box(self, force_snap=False):
        """
        Returns the (x1, y1, x2, y2) of the current crop window.

        The aspect ratio is constant at every zoom level (crop_w / crop_h is
        always ``aspect_ratio``); only the absolute size scales with the eased
        ``current_zoom``. y follows the subject's vertical centre so a
        push-in doesn't crop heads off; at zoom=1.0 the crop is full-height
        and y collapses to 0 exactly as before.
        """
        if force_snap:
            self.current_center_x = self.target_center_x
            self.current_center_y = self.target_center_y
            self.current_zoom = self.target_zoom
            self._vx = self._vy = self._vz = 0.0
            self._static_frames = 0
        elif CAMERA_STYLE == "cut":
            # Locked-then-cut (see CAMERA_STYLE). Between cuts NOTHING moves —
            # not position, not zoom — so a held shot is pixel-identical frame
            # to frame and literally cannot stutter. When the subject has
            # genuinely left the safe zone we re-frame instantly rather than
            # travelling, because travelling is the artefact.
            drifted = (abs(self.target_center_x - self.current_center_x) > self.safe_zone_radius
                       or abs(self.target_center_y - self.current_center_y) > self.crop_height * 0.25)
            if drifted and self._static_frames >= self.long_shot_follow_frames:
                # LONG-SHOT FOLLOW (owner spec, 4-aug-2026: "instead of
                # constant jittering that distracts the eyes, we can use
                # tracking if the camera doesn't change for long").
                #
                # A held shot normally moves NOTHING — that is what makes it
                # impossible to stutter, and the reference edits in this genre
                # are 100% hard cuts. But when a shot runs long and the subject
                # walks out of the safe zone, the old behaviour TELEPORTED the
                # crop mid-shot, which reads as a jolt with no cut to justify
                # it. Once a shot has held this long, follow the subject
                # gently instead: slow enough to be invisible, and it only
                # ever engages on shots that have already earned it.
                self.current_center_x, self._vx = self._eased_step(
                    self.current_center_x, self.target_center_x,
                    LONG_FOLLOW_RATE, LONG_FOLLOW_MAX_STEP, LONG_FOLLOW_ACCEL,
                    self._vx, EASE_SNAP_EPSILON, min_step=0.0)
                self.current_center_y, self._vy = self._eased_step(
                    self.current_center_y, self.target_center_y,
                    LONG_FOLLOW_RATE, LONG_FOLLOW_MAX_STEP, LONG_FOLLOW_ACCEL,
                    self._vy, EASE_SNAP_EPSILON, min_step=0.0)
                self.current_zoom = self.target_zoom
                self._vz = 0.0
            elif drifted:
                self.current_center_x = self.target_center_x
                self.current_center_y = self.target_center_y
                # Focal length changes ride along with the cut — the reference
                # edits never zoom continuously inside a held shot.
                self.current_zoom = self.target_zoom
                self._static_frames = 0
                self._vx = self._vy = self._vz = 0.0
            else:
                self._static_frames += 1
                self._vx = self._vy = self._vz = 0.0
        else:
            self.current_center_x, self._vx = self._eased_step(
                self.current_center_x, self.target_center_x,
                EASE_RATE, EASE_MAX_STEP, EASE_ACCEL, self._vx, EASE_SNAP_EPSILON,
                min_step=EASE_MIN_STEP)
            self.current_center_y, self._vy = self._eased_step(
                self.current_center_y, self.target_center_y,
                EASE_RATE, EASE_MAX_STEP, EASE_ACCEL, self._vy, EASE_SNAP_EPSILON,
                min_step=EASE_MIN_STEP)
            self.current_zoom, self._vz = self._eased_step(
                self.current_zoom, self.target_zoom,
                ZOOM_EASE_RATE, ZOOM_EASE_MAX_STEP, ZOOM_EASE_ACCEL, self._vz,
                ZOOM_EASE_SNAP_EPSILON, min_step=ZOOM_EASE_MIN_STEP)
            if (abs(self.current_center_x - self.target_center_x) < EASE_SNAP_EPSILON * 2
                    and abs(self._vx) < 0.5 and abs(self._vy) < 0.5):
                self._static_frames += 1
            else:
                self._static_frames = 0

        # Slow "operator is alive" drift on a truly static crop — see the
        # STATIC_DRIFT_* constants above.
        # Translation ONLY — deliberately not a zoom "breathe". A breathe
        # rescales the crop every single frame, which is the same mechanism
        # that made zoom moves shimmer (see the quantization note below), and
        # a static shot is exactly where that reads worst: nothing else is
        # moving to mask it. A slow sub-pixel-rate pan gives the same "operator
        # is alive" feel and costs no rescaling at all.
        drift_x = 0.0
        if not force_snap and self._static_frames > STATIC_DRIFT_FRAMES:
            phase = (2.0 * math.pi
                     * (self._static_frames - STATIC_DRIFT_FRAMES)
                     / (STATIC_DRIFT_PERIOD_S * self.fps))
            drift_x = self.crop_width * STATIC_DRIFT_AMPLITUDE * math.sin(phase)

        eff_w = self.crop_width * self.current_zoom
        eff_h = self.crop_height * self.current_zoom
        half_w = eff_w / 2
        half_h = eff_h / 2

        # Clamp centres so the crop stays inside the source.
        min_cx, max_cx = half_w, self.video_width - half_w
        if max_cx < min_cx:
            min_cx = max_cx = self.video_width / 2
        # Clamp the EASED state (drift is temporary camera life, NOT part of
        # the eased state — folding it in would make the static detector see
        # the drift as motion and restart its count every frame).
        self.current_center_x = min(max(self.current_center_x, min_cx), max_cx)
        min_cy, max_cy = half_h, self.video_height - half_h
        if max_cy < min_cy:
            min_cy = max_cy = self.video_height / 2
        self.current_center_y = min(max(self.current_center_y, min_cy), max_cy)

        # Quantize the crop SIZE before deriving the rect. Two artefacts,
        # both measured on a real render (31-jul-2026) and both invisible in
        # the trajectory numbers but very visible on playback as "stutter":
        #
        #  1. ASPECT WOBBLE. eff_w/eff_h used to be int()'d independently, so
        #     660x880 (0.75000) became 659x879 (0.74972) on the next frame.
        #     Every crop is rescaled to ONE fixed output size, so a ratio that
        #     moves frame-to-frame stretches/squashes the picture a fraction of
        #     a percent, continuously. Deriving h from w keeps the ratio EXACT.
        #  2. CHROMA SHIMMER. yuv420p subsamples chroma 2x1, so an odd crop
        #     width/height lands the chroma grid on a half-sample offset; the
        #     measured trajectory flipped odd<->even 66 times in 667 frames.
        #     Even SIZES pin that grid without costing anything, because the
        #     size only changes during a zoom.
        #
        # POSITION is deliberately NOT quantized the same way. Rounding x to
        # even was tried and reverted: it turned a smooth 1px/frame pan into
        # "jump 2px, stall, jump 2px" (measured: every step even, 89
        # stall<->move alternations in one clip), which reads as far worse
        # judder than any chroma crawl it prevented. Motion smoothness wins.
        crop_w = max(2, int(round(eff_w / 2.0)) * 2)
        crop_h = max(2, int(round(crop_w / self.aspect_ratio / 2.0)) * 2)
        crop_w = min(crop_w, self.video_width - (self.video_width % 2))
        crop_h = min(crop_h, self.video_height - (self.video_height % 2))
        half_w, half_h = crop_w / 2, crop_h / 2

        # Clamp the EASED STATE, not just the emitted rect — letting the state
        # run past the frame edge pins the output there while the internal
        # value keeps drifting, so the camera then lags on the way back.
        min_cx, max_cx = half_w, self.video_width - half_w
        if max_cx < min_cx:
            min_cx = max_cx = self.video_width / 2
        self.current_center_x = min(max(self.current_center_x, min_cx), max_cx)
        min_cy, max_cy = half_h, self.video_height - half_h
        if max_cy < min_cy:
            min_cy = max_cy = self.video_height / 2
        self.current_center_y = min(max(self.current_center_y, min_cy), max_cy)

        center_x = min(max(self.current_center_x + drift_x, min_cx), max_cx)
        # RULE OF THIRDS / LOOKING ROOM (owner spec, 4-aug-2026: "the frame is
        # not even following the rule of thirds").
        #
        # The crop used to be dead-centred on the subject and then hard-clamped
        # at the frame edge, which is the one composition that always reads as
        # amateur: a person standing off to one side ends up jammed against the
        # crop edge with all the empty room BEHIND them.
        #
        # Someone on the left of the source is almost always facing right (and
        # vice versa) — in this genre the contestants face the host across the
        # room. So place them on the near third and leave the space in front of
        # their gaze. A subject already near frame centre stays centred, which
        # is the owner's first rule: "if the person can be centered, then
        # center the person."
        # Clamp the WINDOW, not the centre: min_cx/max_cx above assume the
        # subject sits dead centre, so applying a thirds offset to an
        # already-clamped centre walks the crop off the subject entirely
        # (caught by test_a_fresh_shot_still_re_frames_instantly_on_a_big_move:
        # subject at x=1800 fell outside a crop ending at 1784).
        raw_cx = self.current_center_x + drift_x
        x1 = int(raw_cx - crop_w * self._place_frac(raw_cx))
        # CONTAINMENT WINS OVER COMPOSITION. Near a source edge the camera
        # centre is already clamped, so a thirds offset can walk the crop off
        # the subject (measured: a subject at x=1800 fell outside a crop
        # ending at 1784). The owner's rule is explicit — "if the person can
        # be centered, center the person; if the person cannot be centered,
        # make sure the person is at least at the edge" — so a thirds
        # placement is only ever applied when the subject still sits inside
        # the frame with margin.
        subj = self.target_center_x
        x1 = max(int(subj - crop_w * (1.0 - COMPOSE_EDGE_MARGIN)),
                 min(x1, int(subj - crop_w * COMPOSE_EDGE_MARGIN)))
        x1 = max(0, min(x1, self.video_width - crop_w))
        # target_center_y is the HEAD anchor (see CAMERA_HEAD_ANCHOR); place
        # it CAMERA_HEAD_Y down the crop instead of dead centre, so the eyes
        # sit on the top-third line and the subject never looks pushed low.
        y1 = max(0, min(int(self.current_center_y - crop_h * CAMERA_HEAD_Y),
                        self.video_height - crop_h))
        return x1, y1, x1 + crop_w, y1 + crop_h

# Active-speaker hysteresis factors in SpeakerTracker: the sticky bonus that
# keeps the camera on whoever it's already framing. STICKY_CONTINUITY_FACTOR
# is used when no reaction/speech boost fired this frame (stay put on weak
# evidence); STICKY_FACTOR otherwise (a real signal may compete).
# Tuned down from 3.0/4.5 (31-jul-2026). Those values held the camera on two
# people for 86% of a 28s clip — 4 shot changes where the reference edits in
# this genre run 1-4s per shot (so ~10-15). Stickiness exists to stop jitter,
# but jitter is now handled by locked-then-cut framing (see CAMERA_STYLE), so
# it no longer has to double as a switch suppressor.
STICKY_FACTOR = float(os.environ.get("STICKY_FACTOR", "1.6"))
STICKY_CONTINUITY_FACTOR = float(os.environ.get("STICKY_CONTINUITY_FACTOR", "2.2"))

class SpeakerTracker:
    """
    Tracks speakers over time to prevent rapid switching and handle temporary obstructions.
    """
    def __init__(self, cooldown_frames=30, unlock_frames=None,
                 identity=None):
        # Optional identity_tracker.IdentityTracker. When present it replaces
        # the legacy x-position matcher in assign_ids with a real multi-object
        # tracker (Kalman motion model + globally optimal assignment + a
        # low-confidence recovery pass). Every id-keyed policy above this —
        # primary-subject bias, speaker binding, split-cell assignment — gets
        # more stable ids for free, without any of them changing.
        self.identity = identity
        # Current BGR frame, set by the renderer each detection. BoT-SORT's
        # camera-motion compensation reads it to separate "the camera panned"
        # from "the person moved" — without it, a handheld/panning source makes
        # every track look like it jumped.
        self._identity_frame = None
        self.active_speaker_id = None
        self.speaker_scores = {}  # {id: score}
        self.last_seen = {}       # {id: frame_number}
        self.locked_counter = 0   # How long we've been locked on current speaker

        # Hyperparameters
        self.switch_cooldown = cooldown_frames              # Minimum frames before switching again
        self.last_switch_frame = -1000

        # Frame numbers near a known AUDIO speaker-turn change (from AssemblyAI
        # diarization, when available — see speaker_change_frames()). At these
        # frames only, the sticky hysteresis bonus below is suppressed so a
        # genuine visual signal can win instead of staying glued to whichever
        # face was previously active. The switch_cooldown debounce is NOT
        # touched — that guard was tuned against real jitter (see the comment
        # in get_target below) and stays fully intact either way; this only
        # removes the ARTIFICIAL bonus, it never forces a switch.
        self.unlock_frames = unlock_frames or frozenset()

        # ID tracking
        self.next_id = 0
        self.known_faces = [] # [{'id': 0, 'center': x, 'last_frame': 123}]

    def get_target(self, face_candidates, frame_number, width):
        """
        Decides which face to focus on.
        face_candidates: list of {'box': [x,y,w,h], 'score': float}
        """
        box, _ = self._get_target(face_candidates, frame_number, width)
        return box

    def get_target_id(self, face_candidates, frame_number, width, continuity_bias=False):
        """Like get_target, but also returns the matched candidate id — the
        current active id when holding, the new id after a switch. The id is
        also stamped onto each input candidate as ``face['id']`` so callers
        can match boosted candidates to tracker decisions by identity.

        ``continuity_bias``: when True (no reaction/speech boost fired this
        frame), the active-speaker sticky factor is raised — a "no clear
        signal -> stay put" bias so a marginal size difference alone can't
        redirect focus (see reframe_v2._analyze_trajectory).
        """
        return self._get_target(face_candidates, frame_number, width,
                                continuity_bias=continuity_bias)

    def set_identity_frame(self, frame):
        """Give the identity tracker the current frame for motion compensation."""
        self._identity_frame = frame

    def assign_ids(self, face_candidates, frame_number, width):
        """Match candidates to known IDs by position, stamping each input
        candidate dict with its ``'id'`` (so callers can apply id-based score
        policies — e.g. the primary-subject return bias — before selection).
        Returns the matched candidate list. Idempotent with _get_target's own
        matching (same positions on the same frame produce the same ids)."""
        if self.identity is not None:
            # Real tracker owns identity. It stamps 'id' in place and is
            # idempotent per frame_number, which matters because the renderer
            # calls this more than once per frame (before and after boosts).
            self.identity.update(face_candidates, frame=self._identity_frame,
                                 frame_number=frame_number)
            for face in face_candidates:
                self.known_faces = [kf for kf in self.known_faces
                                    if kf['id'] != face['id']]
                self.known_faces.append({
                    'id': face['id'],
                    'center': face['box'][0] + face['box'][2] / 2,
                    'last_frame': frame_number,
                })
            return [{'id': f['id'], 'box': f['box'], 'score': f['score']}
                    for f in face_candidates]

        current_candidates = []
        for face in face_candidates:
            x, y, w, h = face['box']
            center_x = x + w / 2

            best_match_id = -1
            min_dist = width * 0.15  # Reduced matching radius to avoid jumping in groups

            # Try to match with known faces seen recently
            for kf in self.known_faces:
                if frame_number - kf['last_frame'] > 30:  # Forgot faces older than 1s (was 2s)
                    continue
                dist = abs(center_x - kf['center'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = kf['id']

            # If no match, assign new ID
            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1
            face['id'] = best_match_id  # let callers match decisions to boxes

            # Update known face
            self.known_faces = [kf for kf in self.known_faces if kf['id'] != best_match_id]
            self.known_faces.append({'id': best_match_id, 'center': center_x, 'last_frame': frame_number})

            current_candidates.append({
                'id': best_match_id,
                'box': face['box'],
                'score': face['score']
            })
        return current_candidates

    def _get_target(self, face_candidates, frame_number, width, continuity_bias=False):
        """
        Shared selection logic behind get_target/get_target_id.
        Returns (box, id) or (None, None).
        """
        # 1. Match faces to known IDs (simple distance tracking)
        current_candidates = self.assign_ids(face_candidates, frame_number, width)

        # 2. Update Scores with decay
        for pid in list(self.speaker_scores.keys()):
             self.speaker_scores[pid] *= 0.85 # Faster decay (was 0.9)
             if self.speaker_scores[pid] < 0.1:
                 del self.speaker_scores[pid]

        # Add new scores
        for cand in current_candidates:
            pid = cand['id']
            # Score is purely based on size (proximity) now that we don't have mouth
            raw_score = cand['score'] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        # 3. Determine Best Speaker
        if not current_candidates:
            # If no one found, maintain last active speaker if cooldown allows
            # to avoid black screen or jump to 0,0
            return None, None
            
        best_candidate = None
        max_score = -1
        
        for cand in current_candidates:
            pid = cand['id']
            total_score = self.speaker_scores.get(pid, 0)
            
            # Hysteresis: HUGE Bonus for current active speaker — suppressed
            # right around a known audio speaker-turn change, so a real
            # switch in who's talking isn't fought by artificial stickiness.
            if pid == self.active_speaker_id and frame_number not in self.unlock_frames:
                # "No clear signal -> stay put": with no reaction/speech boost
                # this frame, stickiness is raised so marginal size noise can't
                # redirect focus off the person we're already framing.
                total_score *= STICKY_CONTINUITY_FACTOR if continuity_bias else STICKY_FACTOR
                
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        # 4. Decide Switch
        if best_candidate:
            target_id = best_candidate['id']
            
            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate['box'], self.active_speaker_id
            
            # New person. The cooldown must hold whether or not the current
            # speaker happens to be detected in THIS frame.
            #
            # It used to fall through and switch when the active speaker was
            # missing from the candidate list — a blink, a head turn or one
            # motion-blurred frame was enough. That is precisely when the
            # cooldown is needed, so it only ever fired when it wasn't: 3 of 7
            # target switches measured on a 12s clip (25-jul-2026) jumped the
            # cooldown this way, and every jump drags the camera across frame.
            #
            # Returning None holds instead: the caller only calls
            # update_target() on a truthy box, so the camera keeps its current
            # target and finishes whatever move it was making. The hold is
            # bounded by the cooldown itself — once it expires, a speaker who
            # really did leave the shot is switched away from normally.
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next((c for c in current_candidates if c['id'] == self.active_speaker_id), None)
                if old_cand:
                    return old_cand['box'], self.active_speaker_id
                return None, None

            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate['box'], target_id

        return None, None


def speaker_change_frames(transcript, clip_start, clip_end, fps, unlock_window_s=0.35):
    """AssemblyAI diarization -> clip-relative frame numbers near a real
    audio speaker-turn change, for SpeakerTracker's unlock_frames.

    Only whisper's own segments have no 'speaker' field (that backend has no
    diarization) — in that case this returns an empty set and SpeakerTracker
    behaves exactly as it always has (pure visual, no audio signal to use).
    fps is expected as frames/second of the CLIP being analyzed (matches
    reframe_v2._analyze_trajectory's frame_number), not the source video.
    """
    segments = [s for s in (transcript or {}).get('segments', [])
               if s.get('speaker') is not None
               and s.get('end', 0) > clip_start and s.get('start', 0) < clip_end]
    segments.sort(key=lambda s: s['start'])

    change_times = []
    last_speaker = None
    for seg in segments:
        speaker = seg.get('speaker')
        if last_speaker is not None and speaker != last_speaker:
            change_times.append(max(0.0, seg['start'] - clip_start))
        last_speaker = speaker

    window_frames = max(1, int(round(unlock_window_s * fps)))
    frames = set()
    for t in change_times:
        center = int(round(t * fps))
        frames.update(range(max(0, center - window_frames), center + window_frames + 1))
    return frames


def speaker_turn_frame_ranges(transcript, clip_start, clip_end, fps, total_frames):
    """AssemblyAI diarization -> clip-relative frame ranges, each tagged with
    the single speaker active throughout (or None where no one clear speaker
    is talking — silence, an untagged whisper transcript, or a genuine gap).

    This is the audio half of dynamic reaction-camera switching: a GENERAL
    (wide/group) scene doesn't have to stay a static wide shot for its whole
    duration just because it opened on a group — when one person is clearly
    talking, that portion can crop in on them like TRACK does, then widen
    back out for a reaction or a new speaker. See reframe_v2._analyze_
    trajectory, which uses this to decide per-frame, and RESEARCH_viral_clip_
    patterns.md §6/§8 for why this matters (real edits cut per speaker turn,
    not per detected scene).

    Returns a list of (start_frame, end_frame, speaker_or_None) covering
    [0, total_frames) with no gaps.
    """
    segments = [s for s in (transcript or {}).get('segments', [])
               if s.get('end', 0) > clip_start and s.get('start', 0) < clip_end]
    segments.sort(key=lambda s: s['start'])

    ranges = []
    cursor = 0.0
    clip_duration = (total_frames / fps) if fps else 0.0
    for seg in segments:
        seg_start = max(0.0, seg.get('start', 0) - clip_start)
        seg_end = min(clip_duration, seg.get('end', 0) - clip_start)
        if seg_end <= seg_start:
            continue
        if seg_start > cursor:
            ranges.append((cursor, seg_start, None))
        ranges.append((seg_start, seg_end, seg.get('speaker')))
        cursor = seg_end
    if cursor < clip_duration:
        ranges.append((cursor, clip_duration, None))

    frame_ranges = []
    for s, e, speaker in ranges:
        sf, ef = int(round(s * fps)), int(round(e * fps))
        sf, ef = max(0, sf), min(total_frames, ef)
        if ef > sf:
            frame_ranges.append((sf, ef, speaker))
    return frame_ranges


# Detectors never need full-resolution frames: MediaPipe returns relative
# coords and YOLO boxes are scaled back up. Running them on a ≤640px copy cuts
# per-frame preprocessing cost hard, which is what dominates CPU-only renders.
DETECT_MAX_WIDTH = 640
# The global MediaPipe graph and YOLO model are NOT thread-safe; clips render
# in parallel, so every inference goes through this lock. Contention is small
# (a few ms per call) — the ffmpeg renders are where the parallel time goes.
DETECT_LOCK = threading.Lock()
# Detect every Nth frame; SmoothedCameraman interpolates between updates.
DETECT_STRIDE = max(int(os.environ.get("DETECT_STRIDE", "4")), 1)
# YOLO fallback (no face found) is far heavier than MediaPipe — extra throttle.
YOLO_FALLBACK_STRIDE = DETECT_STRIDE * 2

# YOLO body-candidate filtering: a confidence floor and a minimum relative
# box area kill obviously-background detections before they ever become
# candidates (ground-truthed 31-jul-2026: a small background person's motion
# could otherwise out-boost a large foreground subject's baseline score).
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.4"))
YOLO_MIN_AREA_FRACTION = float(os.environ.get("YOLO_MIN_AREA_FRACTION", "0.005"))

# Some sources burn a channel-handle watermark strip into the raw footage
# itself (e.g. a row of social handles across the top, a text sticker along
# the bottom) — genuinely part of the source pixels, not something the
# reframe/caption/vision stages can tell apart from real content. Cropping
# it off HERE, before scene detection/face detection/vision confirmation
# ever run, means every downstream stage only ever sees clean frames — not
# just the final render. Per-source (different creators brand differently),
# so off by default; set for a specific source via env.
SOURCE_LOGO_CROP_TOP_PX = max(int(os.environ.get("SOURCE_LOGO_CROP_TOP_PX", "0")), 0)
SOURCE_LOGO_CROP_BOTTOM_PX = max(int(os.environ.get("SOURCE_LOGO_CROP_BOTTOM_PX", "0")), 0)


def source_logo_crop_vf_args():
    """ffmpeg -vf args to strip the configured top/bottom watermark bands,
    or [] when neither is configured (the common case)."""
    if not (SOURCE_LOGO_CROP_TOP_PX or SOURCE_LOGO_CROP_BOTTOM_PX):
        return []
    total = SOURCE_LOGO_CROP_TOP_PX + SOURCE_LOGO_CROP_BOTTOM_PX
    return ["-vf", f"crop=iw:ih-{total}:0:{SOURCE_LOGO_CROP_TOP_PX}"]


def _detection_frame(frame):
    """Downscaled copy for detectors. Returns (small_frame, scale) with
    scale mapping small-frame pixel coords back to the original frame."""
    h, w = frame.shape[:2]
    if w <= DETECT_MAX_WIDTH:
        return frame, 1.0
    scale = w / DETECT_MAX_WIDTH
    small = cv2.resize(frame, (DETECT_MAX_WIDTH, max(int(h / scale), 2)),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def detect_face_candidates(frame):
    """
    Returns list of all detected faces using lightweight FaceDetection.
    Boxes are in ORIGINAL frame coordinates (detection runs downscaled;
    MediaPipe's relative coords make the mapping exact).
    """
    height, width, _ = frame.shape
    small, _scale = _detection_frame(frame)
    rgb_frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    with DETECT_LOCK:
        results = face_detection.process(rgb_frame)
    
    candidates = []
    
    if not results.detections:
        return []
        
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)

        # Mouth keypoint (index 3 of MediaPipe's 6 short-range keypoints),
        # expressed as a fraction of box height — used downstream (reframe_v2)
        # to detect actual mouth movement across frames as a proxy for "is
        # this specific face the one currently speaking," independent of
        # face size. Scale-invariant, so it survives the extra downscaling
        # reframe_v2 applies to boxes.
        mouth_frac = 0.5
        keypoints = detection.location_data.relative_keypoints
        if keypoints and len(keypoints) > 3 and h:
            mouth_y = keypoints[3].y * height
            mouth_frac = (mouth_y - y) / h

        candidates.append({
            'box': [x, y, w, h],
            'score': w * h, # Area as score
            'mouth_frac': mouth_frac,
        })
            
    return candidates

def detect_person_yolo(frame):
    """
    Fallback: Detect largest person using YOLO when face detection fails.
    Returns [x, y, w, h] of the person's 'upper body' approximation, in
    ORIGINAL frame coordinates (inference runs on a downscaled copy).
    """
    small, scale = _detection_frame(frame)
    # Use the globally loaded model
    with DETECT_LOCK:
        results = model(small, verbose=False, classes=[0]) # class 0 is person

    if not results:
        return None

    best_box = None
    max_area = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i * scale) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area > max_area:
                max_area = area
                # Focus on the top 40% of the person (head/chest) for framing
                # This approximates where the face is if we can't detect it directly
                face_h = int(h * 0.4)
                best_box = [x1, y1, w, face_h]

    return best_box


def detect_person_candidates_yolo(frame):
    """Like detect_person_yolo, but returns EVERY detected person (head/chest
    approximation), not just the largest. MediaPipe's face detector is
    trained on visible eyes/nose/mouth and misses a face mask or a head
    turned away entirely — a masked or turned-away subject then has no
    candidate at all, however tracking-worthy their body position is. This
    gives reframe_v2 a body-based candidate to fall back on for exactly
    that case, merged alongside real face candidates rather than only used
    when face detection finds nothing (ground-truthed 31-jul-2026: the
    subject was mid-turn — a face mask would fail identically).
    """
    small, scale = _detection_frame(frame)
    with DETECT_LOCK:
        # YOLO_DEVICE (e.g. "0" in the GPU compose deployment) pins the
        # device explicitly so a CUDA init failure FAILS LOUD here instead
        # of ultralytics silently falling back to CPU (see the startup
        # assertion at model load). Unset on CPU self-hosts -> auto.
        results = model(small, verbose=False, classes=[0], conf=YOLO_CONF,
                        device=_YOLO_DEVICE)

    if not results:
        return []

    frame_area = max(float(frame.shape[0] * frame.shape[1]), 1.0)
    candidates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(i * scale) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            # Background specks are never tracking-worthy subjects: discard
            # boxes that are a negligible fraction of the source frame.
            if (w * h) / frame_area < YOLO_MIN_AREA_FRACTION:
                continue
            # Top 40% of the body (head/chest) approximates where a face
            # would be if we could see it, for framing purposes.
            face_h = int(h * 0.4)
            candidates.append({
                'box': [x1, y1, w, face_h],
                'score': w * face_h,
                'mouth_frac': 0.5,  # no mouth keypoint from a body box
            })
    return candidates

def create_general_frame(frame, output_width, output_height):
    """
    Creates a 'General Shot' frame: 
    - Background: Blurred zoom of original
    - Foreground: Original video scaled to fit width, centered vertically.
    """
    orig_h, orig_w = frame.shape[:2]
    
    # 1. Background (Fill Height)
    # Crop center to aspect ratio
    bg_scale = output_height / orig_h
    bg_w = int(orig_w * bg_scale)
    bg_resized = cv2.resize(frame, (bg_w, output_height), interpolation=cv2.INTER_LINEAR)

    # Crop center of background
    start_x = (bg_w - output_width) // 2
    if start_x < 0: start_x = 0
    background = bg_resized[:, start_x:start_x+output_width]
    if background.shape[1] != output_width:
        background = cv2.resize(background, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    # Blur background: blur at quarter resolution and scale back up — visually
    # identical for a defocused backdrop, an order of magnitude cheaper than a
    # 51px Gaussian at full size.
    small_bg = cv2.resize(background, (max(output_width // 4, 2), max(output_height // 4, 2)),
                          interpolation=cv2.INTER_AREA)
    small_bg = cv2.GaussianBlur(small_bg, (13, 13), 0)
    background = cv2.resize(small_bg, (output_width, output_height),
                            interpolation=cv2.INTER_LINEAR)

    # 2. Foreground (Fit Width)
    scale = output_width / orig_w
    fg_h = int(orig_h * scale)
    foreground = cv2.resize(frame, (output_width, fg_h), interpolation=cv2.INTER_LINEAR)
    
    # 3. Overlay
    y_offset = (output_height - fg_h) // 2
    
    # Clone background to avoid modifying it
    final_frame = background.copy()
    final_frame[y_offset:y_offset+fg_h, :] = foreground
    
    return final_frame

# NOTE: a "route text-heavy scenes to GENERAL" rule was tried here and removed
# on 26-jul-2026. The problem it targets is real — a screencast that happens to
# contain one face gets cropped to the face and its headlines come out cut
# mid-word — but edge density is the wrong signal for it. Measured: a
# constructed talking-head-beside-a-chart scored 0.012 while the SAME shot
# without the panels scored 0.029, because a flat panel of text has far fewer
# edges than ordinary scene detail. Canny measures visual busyness, not text.
# A real fix needs an actual text detector (MSER/EAST) validated against clips
# that contain the failure mode; this corpus has almost none.


def analyze_scenes_strategy(video_path, scenes):
    """
    Analyzes each scene to determine if it should be TRACK (Single person) or GENERAL (Group/Wide).
    Returns list of strategies corresponding to scenes.
    """
    # Manual "Auto Zoom" override (round 3, item 6): auto (default) keeps the
    # per-scene TRACK/GENERAL analysis; track forces every scene to the
    # single-subject close crop; wide forces the group/wide layout.
    override = os.environ.get("SCENE_STRATEGY_OVERRIDE", "auto").strip().lower()
    if override == "track":
        return ['TRACK'] * len(scenes)
    if override in ("general", "wide"):
        return ['GENERAL'] * len(scenes)

    cap = cv2.VideoCapture(video_path)
    strategies = []

    if not cap.isOpened():
        return ['TRACK'] * len(scenes)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for start, end in tqdm(scenes, desc="   Analyzing Scenes"):
        s_f, e_f = start.get_frames(), end.get_frames()
        # Sample 5 frames spread across the scene, clamped inside it (the old
        # start+5/end-5 samples landed outside scenes shorter than ~10 frames).
        margin = min(2, max(0, (e_f - s_f - 1) // 2))
        frames_to_check = sorted(set(
            int(round(f)) for f in np.linspace(s_f + margin, e_f - 1 - margin, 5)
        ))

        face_counts = []
        largest_face_width_ratios = []
        for f_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue

            # Near-black frames (fades, cut-to-black) carry no faces and used
            # to drag single-person scenes into GENERAL. Skip them.
            if frame.mean() < 16:
                continue

            # Detect faces
            candidates = detect_face_candidates(frame)
            face_counts.append(len(candidates))
            if len(candidates) == 1:
                frame_width = frame.shape[1]
                largest_face_width_ratios.append(candidates[0]['box'][2] / frame_width)

        # Decision Logic
        if not face_counts:
            avg_faces = 0
        else:
            avg_faces = sum(face_counts) / len(face_counts)

        # Strategy:
        # 0 faces -> GENERAL (Landscape/B-roll)
        # 1 face, normally framed -> TRACK
        # 1 face, but tiny relative to the frame -> GENERAL. A reaction/
        # commentary edit (small webcam-style face bubble overlaid on a
        # screen recording or the primary footage) also detects as exactly
        # one face, but TRACK-cropping tight around that small corner face
        # drags the crop window away from the actual content the bubble is
        # reacting to (e.g. an on-screen Instagram/UI graphic), squeezing or
        # cutting it off the edge — confirmed on a real reaction-cam clip,
        # 30-jul-2026, where TRACK left the referenced Instagram profile
        # half out of frame. A normally-framed single speaker fills much
        # more of the frame than a corner bubble; 18% of frame width is a
        # conservative cut well below normal talking-head framing.
        # > 1.2 faces -> GENERAL (Group)
        avg_face_width_ratio = (
            sum(largest_face_width_ratios) / len(largest_face_width_ratios)
            if largest_face_width_ratios else None)
        is_small_corner_face = (
            avg_face_width_ratio is not None and avg_face_width_ratio < 0.18)

        if avg_faces > 1.2 or avg_faces < 0.5 or is_small_corner_face:
            strategies.append('GENERAL')
        else:
            strategies.append('TRACK')

    cap.release()

    # Hysteresis: a short scene whose two neighbors agree on the opposite
    # strategy is almost always a sampling miss (profile face, insert shot).
    # Each TRACK<->GENERAL flip is a full on-screen layout change, so flapping
    # is worse than an occasional wrong-but-stable choice.
    max_flip_frames = int(2.0 * fps)
    for i in range(1, len(strategies) - 1):
        dur = scenes[i][1].get_frames() - scenes[i][0].get_frames()
        if (dur < max_flip_frames
                and strategies[i - 1] == strategies[i + 1] != strategies[i]):
            strategies[i] = strategies[i - 1]

    return strategies

def detect_scenes(video_path):
    import scene_detection
    return scene_detection.detect_scenes(video_path)

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


# Byte budget for the sanitized video title used as the stem of every derived
# file. Filesystems cap a name in BYTES (255 on ext4), not characters, and the
# pipeline decorates this stem: "_clip_10.mp4" (12), "subtitled_<ts>_" (21),
# "temp_hook_<hex8>_" (19), "autosubs_<ts>_" + ".ass" (24). Budgeting 120 bytes
# leaves room for all of them stacked and still lands well under the limit.
#
# The old cap was 100 CHARACTERS, which is 300 bytes of Bengali or Arabic — over
# the limit before any decoration. It surfaced as OSError 36 killing the hook
# endpoint in prod on 26-jul-2026.
MAX_TITLE_BYTES = 120


def truncate_bytes(text, max_bytes):
    """Trim ``text`` to a byte budget without splitting a multi-byte character."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", "ignore")


def sanitize_filename(filename):
    """Remove invalid characters from filename and bound it for the filesystem."""
    filename = re.sub(r'[<>:"/\\|?*#]', '', filename)
    filename = filename.replace(' ', '_')
    return truncate_bytes(filename, MAX_TITLE_BYTES)


def download_youtube_video(url, output_dir="."):
    """
    Downloads a YouTube video using yt-dlp.
    Returns the path to the downloaded video and the video title.
    """
    # SSRF guard: block non-http(s) schemes and private/loopback/metadata hosts
    # before handing the URL to yt-dlp.
    from security_utils import assert_public_url
    assert_public_url(url)

    print(f"🔍 Debug: yt-dlp version: {yt_dlp.version.__version__}")
    print("📥 Downloading video from YouTube...")
    step_start_time = time.time()

    cookies_path = '/app/cookies.txt'
    cookies_env = os.environ.get("YOUTUBE_COOKIES")
    # auto_refresh_cookies.sh keeps /app/cookies.txt fresh (a real Netscape
    # jar, rewritten every ~20min from a logged-in Chrome session). The
    # YOUTUBE_COOKIES env value is the LEGACY path and can be stale — a
    # 27-char blob was clobbering the fresh 4.9KB jar on every job, which is
    # exactly the "Sign in to confirm you're not a bot" wall (round-5 live
    # failure, 3-aug-2026). The env is now only a fallback for the first run
    # before the refresh loop has ever written a file: never clobber a newer
    # jar with an older value.
    if os.path.exists(cookies_path) and os.path.getsize(cookies_path) > 0:
        print(f"🍪 Using on-disk cookies jar ({os.path.getsize(cookies_path)} bytes)")
    elif cookies_env:
        print("🍪 No cookies file yet — writing YOUTUBE_COOKIES env value...")
        try:
            with open(cookies_path, 'w') as f:
                f.write(cookies_env)
            if os.path.exists(cookies_path):
                 # Never print file CONTENT here: with a headerless cookies
                 # blob this would leak live YouTube session cookies to logs.
                 print(f"   Debug: Cookies file created. Size: {os.path.getsize(cookies_path)} bytes")
        except Exception as e:
            print(f"⚠️ Failed to write cookies file: {e}")
            cookies_path = None
    else:
        cookies_path = None
        print("⚠️ No cookies file and no YOUTUBE_COOKIES env var — downloads may hit YouTube's bot wall.")
    
    # Optional HTTP proxy. Set PROXY_URL to route downloads through it; unset
    # (self-host) goes direct as before.
    _proxy = os.environ.get("PROXY_URL", "").strip() or None
    if _proxy:
        print("🌐 Using proxy for download.")

    # Two download strategies, tried in order so a break in the HD path degrades
    # gracefully instead of failing the whole job: an HD attempt first, then a
    # conservative fallback (also the only strategy for self-host).
    # PO token provider — YouTube increasingly requires one to trust a client
    # as non-bot, especially from datacenter/cloud IPs. HTTP server mode
    # (BGUTIL_BASE_URL, a persistent sidecar — see docker-compose.yml) is
    # strongly preferred: the provider's own docs say the alternative
    # script-per-request mode (BGUTIL_SCRIPT_PATH) "is NOT recommended" for
    # anything beyond occasional single-shot use. IMPORTANT: this must be
    # merged into EVERY attempt's extractor_args below, not just one — a
    # prior version of this code only wired it into the HD attempt, so every
    # other attempt silently ran with no PO token at all (visible in yt-dlp's
    # debug log as "[pot:...] Script path doesn't exist").
    _bgutil_http = os.environ.get("BGUTIL_BASE_URL", "").strip()
    _bgutil_script = os.environ.get("BGUTIL_SCRIPT_PATH", "").strip()
    if _bgutil_http:
        _pot_args = {'youtubepot-bgutilhttp': {'base_url': [_bgutil_http]}}
    elif _bgutil_script:
        _pot_args = {'youtubepot-bgutilscript': {'script_path': [_bgutil_script]}}
    else:
        _pot_args = {}
    hd_args = dict(_pot_args) if _pot_args else None
    fallback_args = {
        'youtube': {
            'player_client': ['tv_embed', 'android', 'mweb', 'web'],
            'player_skip': ['webpage', 'configs'],
        },
        **_pot_args,
    }
    # Client spoofing, no cookies: impersonate YouTube's iOS app instead of the
    # plain web client. iOS goes first — YouTube's SABR-only streaming rollout
    # has been degrading android's format availability. This alone often
    # dodges the "Sign in to confirm you're not a bot" wall that flags
    # datacenter IPs on the web client, without needing a cookies file at all
    # (and a STALE cookies file can make things worse than none, since an
    # invalid session reads as more suspicious than an anonymous request).
    ios_spoof_args = {
        'youtube': {
            'player_client': ['ios', 'android', 'web'],
        },
        **_pot_args,
    }

    # Cap at 720p ONLY when the bytes actually go through the paid proxy — that
    # cap exists to control bandwidth cost, and the direct attempt has none.
    #
    # This is per-attempt on purpose. Deciding it once from `_proxy` capped the
    # DIRECT attempt too, so with DIRECT_FIRST=1 (which serves most downloads)
    # every YouTube source arrived at 720p and, since the reframe inherits the
    # source height, 80% of delivered clips came out 406x720 (audited 25-jul-2026).
    def _hd_fmt_for(proxy):
        if proxy:
            return ('bestvideo[vcodec^=avc1][height<=720][ext=mp4]+bestaudio[ext=m4a]/'
                    'bestvideo[vcodec^=avc1][height<=720]+bestaudio/'
                    'best[height<=720][ext=mp4]/best[height<=720]/best')
        return ('bestvideo[vcodec^=avc1][height<=1080][ext=mp4]+bestaudio[ext=m4a]/'
                'bestvideo[vcodec^=avc1][height<=1080]+bestaudio/'
                'best[height<=1080][ext=mp4]/best[ext=mp4]/best')
    fallback_fmt = 'best[ext=mp4]/best'

    def _base_opts(extractor_args, proxy, use_cookies=True):
        return {
            'quiet': False, 'verbose': True, 'no_warnings': False,
            'cookiefile': cookies_path if (use_cookies and cookies_path) else None,
            'proxy': proxy, 'socket_timeout': 30, 'retries': 10, 'fragment_retries': 10,
            'nocheckcertificate': True, 'cachedir': False,
            'extractor_args': extractor_args,
            'http_headers': {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ),
            },
        }

    # Wire bytes actually pulled through the (paid) proxy, summed across
    # fragments/streams. Reported to app.py via the PROXY_BYTES= line below.
    _dl_bytes = {"total": 0}

    def _progress_hook(d):
        if d.get('status') == 'finished':
            _dl_bytes["total"] += int(d.get('total_bytes')
                                      or d.get('total_bytes_estimate')
                                      or d.get('downloaded_bytes') or 0)

    def _attempt(extractor_args, fmt, proxy, use_cookies=True):
        _dl_bytes["total"] = 0
        with yt_dlp.YoutubeDL(_base_opts(extractor_args, proxy, use_cookies)) as ydl:
            info = ydl.extract_info(url, download=False)
        sanitized = sanitize_filename(info.get('title', 'youtube_video'))
        expected = os.path.join(output_dir, f'{sanitized}.mp4')
        if os.path.exists(expected):
            os.remove(expected)
        dl_opts = {
            **_base_opts(extractor_args, proxy, use_cookies),
            'format': fmt,
            'outtmpl': os.path.join(output_dir, f'{sanitized}.%(ext)s'),
            'merge_output_format': 'mp4', 'overwrites': True,
            'progress_hooks': [_progress_hook],
        }
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([url])
        return sanitized

    # DIRECT_FIRST=1: try the server's own IP before spending proxy bandwidth.
    # Needs cookies + a PO-token provider — without both, YouTube flags the
    # datacenter IP after the first request (verified in prod, 21-jul-2026).
    _direct_first = (os.environ.get("DIRECT_FIRST", "").strip() == "1"
                     and _proxy and hd_args and cookies_path)

    attempts = (
        [('ios-spoof', ios_spoof_args, fallback_fmt, None, False)]
        + ([('HD-direct', hd_args, _hd_fmt_for(None), None, True)] if _direct_first else [])
        + ([('HD', hd_args, _hd_fmt_for(_proxy), _proxy, True)] if hd_args else [])
        + [('fallback', fallback_args, fallback_fmt, _proxy, True)]
    )

    sanitized_title = None
    last_err = None
    used_proxy = False
    for label, ea, fmt, proxy, use_cookies in attempts:
        # A 403 on the media fetch is usually transient: the googlevideo URL is
        # bound to the IP that extracted it, and the residential proxy rotates
        # its exit IP between requests. Retrying re-extracts and usually lands
        # on a consistent IP (3 of 62 downloads hit this on 22-jul-2026).
        for retry in range(2):
            try:
                print(f"📥 Download attempt: {label}" + (f" (retry {retry})" if retry else ""))
                sanitized_title = _attempt(ea, fmt, proxy, use_cookies)
                used_proxy = proxy is not None
                print(f"✅ Download succeeded ({label}).")
                break
            except Exception as e:
                last_err = e
                print(f"⚠️  Download attempt '{label}' failed: {str(e)[:200]}")
                retryable = '403' in str(e) or 'Forbidden' in str(e)
                if not retryable or retry == 1:
                    break
                time.sleep(3)
        if sanitized_title is not None:
            break

    if sanitized_title is None:
        import sys
        error_msg = f"""
❌ ================================================================= ❌
❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED (all strategies)
❌ ================================================================= ❌
REASON: YouTube blocked the request or the download tooling is out of date.
👇 SOLUTION FOR USER: download the video manually and use the 'Upload Video' tab.
Technical Details: {str(last_err)}
"""
        print(error_msg, file=sys.stdout)
        print(error_msg, file=sys.stderr)
        sys.stdout.flush(); sys.stderr.flush()
        time.sleep(0.5)
        raise last_err

    downloaded_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    if not os.path.exists(downloaded_file):
        for f in os.listdir(output_dir):
            if f.startswith(sanitized_title) and f.endswith('.mp4'):
                downloaded_file = os.path.join(output_dir, f)
                break

    if used_proxy and _dl_bytes["total"]:
        # Machine-parseable marker consumed by app.py's log reader for the
        # monthly proxy-bandwidth counter. Not shown to clients (log filter).
        # Only emitted when the winning attempt actually went through the
        # proxy — direct-first successes are free bandwidth.
        print(f"PROXY_BYTES={_dl_bytes['total']}")
    print(f"✅ Video downloaded in {time.time() - step_start_time:.2f}s: {downloaded_file}")
    return downloaded_file, sanitized_title

def finalize_clip_passthrough(input_video, final_output_video):
    """Keep the clip's native framing (for horizontal/16:9 output).

    The input is the freshly encoded cut, so a stream-copy remux is enough to
    add +faststart — re-encoding here would only cost time and quality.
    """
    if os.path.exists(final_output_video):
        os.remove(final_output_video)
    print(f"🎬 Passthrough (native framing): {input_video}")
    cmd = [
        'ffmpeg', '-y', '-i', input_video,
        '-c', 'copy', *METADATA_SCRUB, '-movflags', '+faststart',
        final_output_video,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
    print(f"✅ Clip saved to {final_output_video}")
    return True, []


def auto_caption_clip(clip_path, transcript, clip_start, clip_end, general_ranges=None):
    """Burn the default caption style onto a finished clip.

    Captions are mandatory for short-form to land, but they were opt-in behind a
    modal and only 9% of delivered clips ever got them (prod audit, 25-jul-2026).
    So every clip now ships captioned by default.

    The captioned file is written ALONGSIDE the clip as
    ``subtitled_<ts>_<clip>.mp4`` — the same convention /api/subtitle uses — so
    the untouched original stays on disk and re-styling from the modal replaces
    the captions instead of burning a second layer over them.

    Returns the captioned path, or None when captions were skipped (silent
    video, no words in range, AUTO_CAPTIONS=0, or any failure — a caption
    problem must never cost the user the clip they already paid for).
    """
    if os.environ.get("AUTO_CAPTIONS", "1").strip() == "0":
        return None
    if not transcript or not transcript.get('segments'):
        return None  # silent video: nothing to caption
    try:
        import subtitles as _subs
        style = _subs.AUTO_CAPTION_STYLE
        output_dir = os.path.dirname(clip_path)
        stem = os.path.basename(clip_path)
        generation_id = int(time.time())
        # The output name MUST stay exactly "subtitled_<ts>_<clip filename>":
        # the modal's walk-back and _canonical_clip_file both reconstruct the
        # clean original from it, so trimming the stem here would orphan the
        # pair. Length is bounded upstream instead, by MAX_TITLE_BYTES at
        # download time. A legacy clip whose name predates that budget can still
        # overflow — that raises OSError 36, which the except below turns into
        # "ship the clip uncaptioned" rather than a broken filename.
        # The .ass path is interpolated INTO an ffmpeg filter string
        # (-vf ass='...'), where a literal apostrophe closes the quote and
        # breaks the filter. Titles carry apostrophes constantly in English
        # ("Earth's", "Don't"), so this name must stay free of the clip stem —
        # which is exactly why /api/subtitle has always used a neutral
        # "subs_<i>_<ts>.ass". Deriving it from the stem silently cost captions
        # on every apostrophe title until 29-jul-2026.
        #
        # The OUTPUT name still carries the stem, and must: the modal's
        # walk-back and _canonical_clip_file reconstruct the clean original
        # from it. That one is only ever passed as an argv element, never
        # inside a filter string, so quoting never applies to it.
        # Unique per clip, not just per second: clips render in parallel
        # (CLIP_WORKERS), so a bare timestamp would collide and let one clip
        # burn another's captions.
        ass_path = os.path.join(
            output_dir, f"autosubs_{generation_id}_{uuid.uuid4().hex[:8]}.ass")
        out_path = os.path.join(output_dir, f"subtitled_{generation_id}_{stem}")

        # Stroke width is derived from font size (see subtitles.
        # auto_stroke_width / CAPTION_STROKE_RATIO), not a fixed px value,
        # so it stays proportional if font_size ever changes.
        border_width = _subs.auto_stroke_width(style["font_size"])

        if not _subs.generate_ass(
                transcript, clip_start, clip_end, ass_path,
                max_chars=style["max_chars"], max_duration=style["max_duration"],
                alignment=style["alignment"], fontsize=style["font_size"],
                font_name=style["font_name"], font_color=style["font_color"],
                border_color=style["border_color"], border_width=border_width,
                highlight_color=style["highlight_color"], effect=style["effect"],
                base_opacity=style["base_opacity"], uppercase=style["uppercase"],
                general_ranges=general_ranges,
                speaker_colors=style.get("speaker_colors", False),
                letter_spacing_ratio=_subs.CAPTION_LETTER_SPACING_RATIO):
            print("   ℹ️ No words in range — clip ships without captions.")
            return None

        _subs.burn_subtitles(
            clip_path, ass_path, out_path,
            alignment=style["alignment"], fontsize=style["font_size"],
            font_name=style["font_name"], font_color=style["font_color"],
            border_color=style["border_color"], border_width=border_width)
        print(f"   💬 Captions burned: {os.path.basename(out_path)}")
        return out_path
    except Exception as e:
        print(f"   ⚠️ Auto-captions failed ({type(e).__name__}: {e}) — "
              f"delivering the clip without them.")
        return None


def render_clip(input_video, final_output_video, output_format="auto",
                transcript=None, clip_start=0.0, clip_end=None,
                focus_directives=None, primary_subject_x=None,
                custom_aspect=None):
    """Route a cut clip through the right renderer for the chosen output format.
    vertical/auto -> 9:16 reframe, square -> 1:1 reframe, horizontal -> keep.

    transcript/clip_start/clip_end are optional — when the caller has them
    (the source video's full transcript plus this clip's absolute time
    range), AssemblyAI-diarized speaker-turn timing informs the active-
    speaker camera tracking (see speaker_change_frames). Without them,
    reframing behaves exactly as before (pure visual tracking).

    focus_directives (see analyze_scene_context): clip-relative shot direction
    from the scene-context layer — who the camera should be on and why. When
    absent, framing falls back to pure heuristic tracking."""
    if output_format == "horizontal":
        return finalize_clip_passthrough(input_video, final_output_video)
    if output_format == "custom" and custom_aspect:
        aspect = custom_aspect
    else:
        aspect = 1.0 if output_format == "square" else ASPECT_RATIO
    return process_video_to_vertical(input_video, final_output_video, aspect_ratio=aspect,
                                     transcript=transcript, clip_start=clip_start, clip_end=clip_end,
                                     focus_directives=focus_directives,
                                     primary_subject_x=primary_subject_x)


# Watermark geometry, as fractions of the clip width/height.
#
# Vertical placement is the whole point: the top and bottom strips of a 9:16
# clip are either black bars or blurred filler (GENERAL layout), so a mark up
# there is cropped away without touching a single pixel of real footage. At 40%
# of the height it sits inside the content band — a 16:9 source letterboxed
# into 9:16 spans roughly 34%-66% — so removing the mark means cutting into the
# picture. Left-aligned, like OpusClip's.
WATERMARK_WIDTH_RATIO = 0.30
WATERMARK_MARGIN_RATIO = 0.05
WATERMARK_Y_RATIO = 0.40
WATERMARK_OPACITY = 0.85


def apply_watermark(video_path):
    """Burn the OpenShorts watermark into a finished clip (free plan).

    One re-encode pass on the final file so every output format (TRACK,
    GENERAL, horizontal passthrough) gets the mark, and later subtitle/hook
    re-encodes keep it — they re-encode the already-marked pixels.
    """
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "assets", "watermark.png")
    if not os.path.exists(logo_path):
        print(f"   ⚠️ Watermark asset missing ({logo_path}); clip kept unmarked.")
        return False

    # Scale the lockup from the clip's real width: overlay can't read the other
    # input's size, and computing it here avoids the deprecated scale2ref.
    try:
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_path],
            stderr=subprocess.STDOUT, timeout=60,
        ).decode().strip().split("x")
        vw, vh = int(probe[0]), int(probe[1])
    except Exception as e:
        print(f"   ⚠️ Could not probe clip for watermark ({e}); clip kept unmarked.")
        return False

    wm_w = max(80, int(vw * WATERMARK_WIDTH_RATIO))
    x = int(vw * WATERMARK_MARGIN_RATIO)
    y = int(vh * WATERMARK_Y_RATIO)
    filt = (
        f"[1:v]scale={wm_w}:-1,format=rgba,"
        f"colorchannelmixer=aa={WATERMARK_OPACITY}[wm];"
        f"[0:v][wm]overlay=x={x}:y={y}"
    )
    tmp_path = video_path + ".wm.mp4"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", logo_path,
           "-filter_complex", filt,
           *video_encode_args(QUALITY), "-c:a", "copy", *METADATA_SCRUB,
           "-movflags", "+faststart", tmp_path]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            timeout=1800)
    if result.returncode == 0 and os.path.exists(tmp_path):
        os.replace(tmp_path, video_path)
        return True
    err = (result.stderr or b"").decode(errors="ignore")[-300:]
    print(f"   ⚠️ Watermark pass failed (clip kept unmarked): {err}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return False


def process_video_to_vertical(input_video, final_output_video, aspect_ratio=ASPECT_RATIO,
                              transcript=None, clip_start=0.0, clip_end=None,
                              focus_directives=None, primary_subject_x=None):
    """
    Core logic to reframe a horizontal video to a target aspect ratio using
    scene detection and Active Speaker Tracking (MediaPipe).
    aspect_ratio: width/height of the output (9/16 vertical, 1.0 square).
    transcript/clip_start/clip_end/focus_directives: optional, see
    render_clip's docstring.
    """
    script_start_time = time.time()

    # v2 engine: analyze downscaled, render natively in ffmpeg. Any failure
    # falls back to the v1 frame loop below so a v2 edge case can't kill jobs.
    if os.environ.get("REFRAME_ENGINE", "v2").strip().lower() != "v1":
        try:
            import reframe_v2
            t0 = time.time()
            result = reframe_v2.render(input_video, final_output_video, aspect_ratio,
                                       transcript=transcript, clip_start=clip_start, clip_end=clip_end,
                                       focus_directives=focus_directives,
                                       primary_subject_x=primary_subject_x)
            print(f"   ⏱️ Reframe v2 total: {time.time() - t0:.1f}s")
            return result
        except Exception as e:
            print(f"   ⚠️ Reframe v2 failed ({type(e).__name__}: {e}) — "
                  f"falling back to v1 frame loop")

    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.aac"

    # Clean up previous temp files if they exist
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(temp_audio_output): os.remove(temp_audio_output)
    if os.path.exists(final_output_video): os.remove(final_output_video)

    print(f"🎬 Processing clip: {input_video}")
    print("   Step 1: Detecting scenes...")
    scenes, fps = detect_scenes(input_video)
    
    if not scenes:
        print("   ❌ No scenes were detected. Using full video as one scene.")
        # If scene detection fails or finds nothing, treat whole video as one scene
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        from scenedetect import FrameTimecode
        scenes = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]

    print(f"   ✅ Found {len(scenes)} scenes.")

    print("\n   🧠 Step 2: Preparing Active Tracking...")
    original_width, original_height = get_video_resolution(input_video)
    
    # Same delivery floor as the v2 engine — a fallback render is still the clip
    # the user posts, so it must not ship sub-HD. The frame loop below already
    # resizes every cropped frame to these dims, so nothing else changes.
    from reframe_v2 import delivery_size
    OUTPUT_WIDTH, OUTPUT_HEIGHT = delivery_size(original_width, original_height,
                                                aspect_ratio)

    # Initialize Cameraman
    cameraman = SmoothedCameraman(OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width,
                                  original_height, aspect_ratio=aspect_ratio,
                                  fps=fps)
    
    # --- New Strategy: Per-Scene Analysis ---
    print("\n   🤖 Step 3: Analyzing Scenes for Strategy (Single vs Group)...")
    scene_strategies = analyze_scenes_strategy(input_video, scenes)
    # scene_strategies is a list of 'TRACK' or 'General' corresponding to scenes
    
    print("\n   ✂️ Step 4: Processing video frames...")
    
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-',
        *video_encode_args(QUALITY_FAST), '-an', temp_video_output
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_number = 0
    current_scene_index = 0
    
    # Pre-calculate scene boundaries
    scene_boundaries = []
    for s_start, s_end in scenes:
        scene_boundaries.append((s_start.get_frames(), s_end.get_frames()))

    # Global tracker for single-person shots. clip_end falls back to the
    # scene boundaries' own total duration when the caller didn't pass one
    # (this v1 path is a rare fallback; v2 above always receives it).
    _clip_end = clip_end
    if _clip_end is None and scene_boundaries:
        _clip_end = clip_start + (scene_boundaries[-1][1] / fps)
    unlock_frames = speaker_change_frames(transcript, clip_start, _clip_end or clip_start, fps) if transcript else set()
    speaker_tracker = SpeakerTracker(cooldown_frames=30, unlock_frames=unlock_frames)

    # Per-stage wall time (server-side diagnostics; hidden from cloud logs).
    stage_seconds = {'detect': 0.0, 'write': 0.0}
    loop_started = time.time()

    with tqdm(total=total_frames, desc="   Processing", file=sys.stdout) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update Scene Index
            if current_scene_index < len(scene_boundaries):
                start_f, end_f = scene_boundaries[current_scene_index]
                if frame_number >= end_f and current_scene_index < len(scene_boundaries) - 1:
                    current_scene_index += 1
            
            # Determine Strategy for current frame based on scene
            current_strategy = scene_strategies[current_scene_index] if current_scene_index < len(scene_strategies) else 'TRACK'
            
            # Apply Strategy
            if current_strategy == 'GENERAL':
                # "Plano General" -> Blur Background + Fit Width
                output_frame = create_general_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                
                # Reset cameraman/tracker so they don't drift while inactive
                cameraman.current_center_x = original_width / 2
                cameraman.target_center_x = original_width / 2
                
            else:
                # "Single Speaker" -> Track & Crop

                # Detect every Nth frame for performance (cameraman smooths in
                # between); the much heavier YOLO fallback gets its own stride.
                if frame_number % DETECT_STRIDE == 0:
                    t_det = time.time()
                    candidates = detect_face_candidates(frame)
                    target_box = speaker_tracker.get_target(candidates, frame_number, original_width)
                    if target_box:
                        cameraman.update_target(target_box)
                    elif frame_number % YOLO_FALLBACK_STRIDE == 0:
                        person_box = detect_person_yolo(frame)
                        if person_box:
                            cameraman.update_target(person_box)
                    stage_seconds['detect'] += time.time() - t_det

                # Snap camera on scene change to avoid panning from previous scene position
                is_scene_start = (frame_number == scene_boundaries[current_scene_index][0])

                x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)

                # Crop
                if y2 > y1 and x2 > x1:
                    cropped = frame[y1:y2, x1:x2]
                    output_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
                else:
                    output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

            t_wr = time.time()
            ffmpeg_process.stdin.write(output_frame.tobytes())
            stage_seconds['write'] += time.time() - t_wr
            frame_number += 1
            pbar.update(1)
    
    loop_total = time.time() - loop_started
    other = loop_total - stage_seconds['detect'] - stage_seconds['write']
    print(f"\n   ⏱️ Frame loop: {loop_total:.1f}s total — "
          f"detect {stage_seconds['detect']:.1f}s, "
          f"encode-wait {stage_seconds['write']:.1f}s, "
          f"decode+render {other:.1f}s ({frame_number} frames)")

    ffmpeg_process.stdin.close()
    stderr_output = ffmpeg_process.stderr.read().decode()
    ffmpeg_process.wait()
    cap.release()

    if ffmpeg_process.returncode != 0:
        print("\n   ❌ FFmpeg frame processing failed.")
        print("   Stderr:", stderr_output)
        return False, []

    print("\n   🔊 Step 5: Extracting audio...")
    audio_extract_command = [
        'ffmpeg', '-y', '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
    ]
    try:
        subprocess.run(audio_extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("\n   ❌ Audio extraction failed (maybe no audio?). Proceeding without audio.")
        pass

    print("\n   ✨ Step 6: Merging...")
    if os.path.exists(temp_audio_output):
        merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output, '-i', temp_audio_output,
            '-c:v', 'copy', '-c:a', 'copy', *METADATA_SCRUB,
            '-movflags', '+faststart', final_output_video
        ]
    else:
         merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output,
            '-c:v', 'copy', *METADATA_SCRUB,
            '-movflags', '+faststart', final_output_video
        ]
        
    try:
        subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"   ✅ Clip saved to {final_output_video}")
    except subprocess.CalledProcessError as e:
        print("\n   ❌ Final merge failed.")
        print("   Stderr:", e.stderr.decode())
        return False, []

    # Clean up temp files
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(temp_audio_output): os.remove(temp_audio_output)

    return True, []


def _generate_source_artifacts(source_video, output_dir, frame_count=12):
    """Extract stills + a low-bitrate preview proxy for the Source section.

    URL jobs delete the full-quality original after processing to save disk —
    but the dashboard's Source panel needs something to show. Generate the
    lightweight artifacts (JPEG stills + a ~480p faststart proxy) BEFORE the
    original is removed; the /api/source endpoints then serve these without
    needing the source. Fail-open: any failure only loses the Source panel,
    never the job.
    """
    try:
        frames_dir = os.path.join(output_dir, "source_frames")
        os.makedirs(frames_dir, exist_ok=True)
        if len(glob.glob(os.path.join(frames_dir, "source_*.jpg"))) < frame_count:
            duration = 0.0
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "csv=p=0", source_video],
                    capture_output=True, text=True, timeout=30)
                duration = float(probe.stdout.strip() or 0)
            except Exception:
                duration = 0.0
            if duration > 0:
                # Fast extraction: one keyframe-seek per still (-ss BEFORE -i
                # seeks without decoding the whole source — the fps-filter
                # approach decoded the full 46-min video and timed out).
                for idx in range(frame_count):
                    t = duration * (idx + 0.5) / frame_count
                    subprocess.run(
                        ["ffmpeg", "-y", "-loglevel", "error",
                         "-ss", f"{t:.2f}", "-i", source_video,
                         "-frames:v", "1",
                         "-vf", "scale='min(480,iw)':-2",
                         os.path.join(frames_dir, f"source_{idx + 1:02d}.jpg")],
                        check=True, timeout=60)
        preview = os.path.join(output_dir, "source_preview.mp4")
        if not os.path.exists(preview):
            tmp = os.path.join(output_dir, ".source_preview.tmp.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", source_video,
                 "-t", "90", "-vf", "scale='min(720,iw)':-2",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "30",
                 "-an", "-movflags", "+faststart", tmp],
                check=True, timeout=300)
            os.replace(tmp, preview)
        print("🖼️  Source artifacts ready (stills + preview proxy)")
    except Exception as e:
        print(f"⚠️  Source artifacts failed ({type(e).__name__}: {e}) — "
              f"Source panel will be unavailable for this job")


def transcribe_video(video_path):
    print("🎙️  Transcribing video...")
    from transcribe_backends import transcribe_media

    transcript = transcribe_media(video_path)

    print(f"   Detected language '{transcript['language']}', "
          f"{len(transcript['segments'])} segments")
    for segment in transcript['segments']:
        # Print progress to keep user informed (and prevent timeouts feeling)
        print(f"   [{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

    return transcript

def _run_gemini_stage(client, model_name, prompt, schema):
    """One schema-enforced Gemini call with transient-error backoff.
    Returns (parsed_dict, cost_analysis)."""
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        safety_settings=gemini_worker.RELAXED_SAFETY_SETTINGS,
    )
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = gemini_pool.generate_with_fallback(
                client, model_name, prompt, config=config, max_attempts=1,
                log=lambda msg: print(msg))
            # Policy blocks are deterministic — retrying only burns quota and
            # time, and the user deserves the real reason instead of a generic
            # "empty response" (prod 23-jul: PROHIBITED_CONTENT on every try).
            gemini_worker.raise_if_blocked(response)
            # Parsing lives inside the retry loop on purpose: Gemini sometimes
            # returns 200 with an empty body, which raises here rather than at
            # the call. Retrying that recovered every occurrence seen in prod
            # (22-jul-2026) — the same payload succeeds on the next attempt.
            parsed_obj = getattr(response, "parsed", None)
            if parsed_obj is not None:
                parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
            else:
                parsed = gemini_worker._parse_json_response_text(
                    gemini_worker._get_response_text(response))
            return parsed, gemini_worker._calculate_cost_analysis(response, model_name)
        except gemini_worker.GeminiBlockedError:
            raise  # deterministic policy block — never retry
        except Exception as e:
            msg = str(e)
            transient = any(tok in msg for tok in (
                '503', 'UNAVAILABLE', '429', 'RESOURCE_EXHAUSTED',
                '500', 'INTERNAL', 'overloaded', 'Deadline',
                'empty response body', 'did not contain a JSON object',
                'Failed to parse Gemini JSON response'))
            if attempt == max_attempts or not transient:
                raise
            wait = 5 * (2 ** (attempt - 1))
            print(f"⚠️ Gemini transient error (attempt {attempt}/{max_attempts}), retrying in {wait}s: {msg[:150]}")
            time.sleep(wait)


def _rough_cut_candidate(source_video_path, start, end, video_duration, pad=2.0):
    """ffmpeg-extracts [start-pad, end+pad] (clamped to the source) to a temp
    file for Gemini Vision to review — a little padding on each side gives
    the model actual room to suggest a boundary shift, not just confirm the
    exact span it was handed."""
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="vision_review_")
    os.close(fd)
    s = max(0.0, start - pad)
    e = min(float(video_duration), end + pad)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
           "-i", source_video_path, "-c", "copy", path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
    return path


def _upload_and_generate(client, model_name, video_path, prompt, schema):
    """Shared upload -> poll-until-ACTIVE -> generate -> cleanup. Same
    mechanism get_visual_clips() uses on a full source video, reused here on
    a short rough-cut candidate clip instead."""
    file_upload = None
    try:
        file_upload = client.files.upload(file=video_path)
        deadline = time.time() + 180
        while True:
            info = client.files.get(name=file_upload.name)
            state = str(getattr(getattr(info, "state", info), "name", "")).upper()
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Gemini could not process the rough-cut clip.")
            if time.time() > deadline:
                raise TimeoutError("Gemini rough-cut processing timed out.")
            time.sleep(2)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema,
            safety_settings=gemini_worker.RELAXED_SAFETY_SETTINGS)
        response = gemini_pool.generate_with_fallback(
            client, model_name, [file_upload, prompt], config=config,
            max_attempts=1, log=lambda msg: print(msg))
        gemini_worker.raise_if_blocked(response)
        parsed_obj = getattr(response, "parsed", None)
        if parsed_obj is not None:
            return parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
        return gemini_worker._parse_json_response_text(gemini_worker._get_response_text(response))
    finally:
        if file_upload is not None:
            try:
                client.files.delete(name=file_upload.name)
            except Exception:
                pass


def _transcript_excerpt(transcript_result, start, end, proposed_start=None):
    """Return a transcript excerpt for vision review.

    When ``proposed_start`` falls inside the excerpt, mark its exact location
    so the context reviewer can distinguish evidence in the pre-roll from the
    words that will actually open the final short. This is essential for a
    cold-open review: a reviewer with only the selected span can diagnose a
    missing setup but cannot see which nearby sentence fixes it.
    """
    parts = []
    marker_added = False
    for seg in transcript_result.get("segments", []):
        seg_start, seg_end = seg.get("start", 0), seg.get("end", 0)
        if seg_end <= start or seg_start >= end:
            continue
        if (proposed_start is not None and not marker_added
                and seg_end > proposed_start):
            parts.append("[PROPOSED CLIP START]")
            marker_added = True
        text = str(seg.get("text", "")).strip()
        if text:
            parts.append(text)
    if proposed_start is not None and not marker_added:
        parts.append("[PROPOSED CLIP START]")
    return " ".join(parts)


def _context_check_once(pool, model_name, source_video_path, candidate,
                        video_duration, transcript_result, max_duration_ceiling):
    """One context/sync check call: rough-cut -> upload -> generate -> cleanup.
    Returns the parsed response dict, or None if no key was available or the
    call failed (caller decides how to treat that). Factored out of
    confirm_clip_with_vision so it can be reused for the bounded retry below.
    """
    key = pool.acquire()
    if not key:
        return None
    start, end = candidate["start"], candidate["end"]
    context_preroll = min(30.0, start)
    context_cut = None
    try:
        context_cut = _rough_cut_candidate(
            source_video_path, start - context_preroll, end, video_duration, pad=0.0)
        client = genai.Client(api_key=key)
        # A long-context segment is a different product from a tight short, so
        # it gets a different reviewer. Judging it with the short prompt was why
        # requesting long clips still produced short ones: rule 3 there demands
        # a self-contained punchy claim in the opening seconds, which a full arc
        # (someone walking in, a round starting) never has — so the reviewer
        # either pulled the start later to manufacture a hook, or dropped the
        # candidate outright. `clip_type` was being written by the selector and
        # then never read by anything downstream.
        is_long = candidate.get("clip_type") == "long_context"
        template = (gemini_worker.VISION_LONG_CONTEXT_CHECK_PROMPT_TEMPLATE
                    if is_long else gemini_worker.VISION_CONTEXT_CHECK_PROMPT_TEMPLATE)
        prompt = template.format(
            narrative_summary=candidate.get("narrative_summary", ""),
            candidate_start_offset=context_preroll,
            transcript_excerpt=_transcript_excerpt(
                transcript_result, start - context_preroll, end + 2.0,
                proposed_start=start),
            max_duration_ceiling=max_duration_ceiling)
        return _upload_and_generate(
            client, model_name, context_cut, prompt, gemini_worker.VisionContextCheckResponse)
    except gemini_worker.GeminiBlockedError:
        raise
    except Exception as e:
        pool.mark_bad(key)
        print(f"⚠️ Vision context/sync check failed: {e}")
        return None
    finally:
        if context_cut:
            try:
                os.remove(context_cut)
            except OSError:
                pass


def _clip_relative_transcript_excerpt(transcript_result, clip_start, clip_end):
    """Transcript lines re-based to clip-relative seconds, so the scene-context
    director's timestamps come back on the same clock the renderer uses."""
    parts = []
    for seg in (transcript_result or {}).get("segments", []):
        seg_start, seg_end = seg.get("start", 0), seg.get("end", 0)
        if seg_end <= clip_start or seg_start >= clip_end:
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        rel = max(0.0, seg_start - clip_start)
        speaker = seg.get("speaker")
        tag = f"[{rel:.1f}s{'' if not speaker else ' ' + str(speaker)}] "
        parts.append(tag + text)
    return "\n".join(parts)


def analyze_scene_context(pool, model_name, clip_path, clip_duration,
                          transcript_result=None, clip_start=0.0, clip_end=None):
    """Third verification layer: after a clip has been selected and its
    boundaries confirmed, have Gemini watch the FINAL streamlined cut and
    direct the camera — who should be on screen when, and why.

    This is the only stage that understands meaning rather than pixels. The
    reframe engine can detect faces and motion but has no way to know that a
    guy grunting and clowning is the REASON everyone is laughing, so it framed
    the people laughing instead of the person causing it (ground-truthed
    31-jul-2026). See gemini_worker.VISION_SCENE_CONTEXT_PROMPT_TEMPLATE for
    the universal rule this encodes.

    Returns a list of focus-directive dicts (clip-relative seconds), or []
    when unavailable — the renderer treats an empty list as "fall back to
    pure heuristic tracking", so this stays a quality layer, never a hard
    dependency.
    """
    if not pool or not clip_duration:
        return []
    key = pool.acquire()
    if not key:
        return []
    try:
        client = genai.Client(api_key=key)
        excerpt = _clip_relative_transcript_excerpt(
            transcript_result, clip_start,
            clip_end if clip_end is not None else clip_start + clip_duration)
        # ~1 directive per 2s of footage, matching the 1.5-3s shot cadence
        # measured in RESEARCH_viral_clip_patterns.md.
        target_count = max(3, int(round(clip_duration / 2.0)))
        prompt = gemini_worker.VISION_SCENE_CONTEXT_PROMPT_TEMPLATE.format(
            transcript_excerpt=excerpt or "(no transcript available)",
            clip_duration=clip_duration,
            target_count=target_count)
        parsed = _upload_and_generate(
            client, model_name, clip_path, prompt,
            gemini_worker.SceneContextResponse)
    except gemini_worker.GeminiBlockedError:
        return []
    except Exception as e:
        pool.mark_bad(key)
        print(f"⚠️ Scene-context direction failed: {e}")
        return []

    directives = (parsed or {}).get("directives") or []
    cleaned = []
    for d in directives:
        try:
            s = max(0.0, float(d.get("start", 0)))
            e = min(float(clip_duration), float(d.get("end", 0)))
            x = float(d.get("x_position", 0.5))
        except (TypeError, ValueError):
            continue
        if e - s < 0.2:
            continue
        cleaned.append({
            "start": s,
            "end": e,
            "subject": str(d.get("subject", ""))[:120],
            # Clamped, not dropped: an out-of-range position is still a usable
            # "far left"/"far right" signal.
            "x_position": min(max(x, 0.0), 1.0),
            "reason": str(d.get("reason", "speaking")),
            "intensity": min(max(float(d.get("intensity", 0.5) or 0.5), 0.0), 1.0),
        })
    cleaned.sort(key=lambda d: d["start"])
    try:
        prim_x = float((parsed or {}).get("primary_subject_x"))
        prim_x = min(max(prim_x, 0.0), 1.0)
    except (TypeError, ValueError):
        prim_x = None
    if cleaned:
        causes = sum(1 for d in cleaned if d["reason"] == "causing_reaction")
        refs = sum(1 for d in cleaned if d["reason"] == "referenced")
        print(f"   🎬 Scene direction: {len(cleaned)} shot(s) "
              f"({causes} cause-of-reaction, {refs} referenced) — "
              f"{(parsed or {}).get('summary', '')[:80]}"
              + (f" | key subject x={prim_x:.2f}" if prim_x is not None else ""))
    return {
        "directives": cleaned,
        "primary_subject": str((parsed or {}).get("primary_subject", ""))[:120],
        "primary_subject_x": prim_x,
    }


def confirm_clip_with_vision(pool, model_name, source_video_path, candidate,
                             video_duration, transcript_result,
                             max_duration_ceiling=180.0):
    """Runs the visual/audio check and the context/sync check for one
    DeepSeek-selected candidate, in parallel across the Gemini key pool
    (each check uses its own key so the two calls don't wait on each other's
    rate limit), and folds any approved boundary deltas back into the
    candidate's start/end in place.

    Fails OPEN (returns True, no boundary change) on any error or when no
    pool is configured — vision confirmation is a quality layer on top of
    DeepSeek's own selection, not a hard gate; a transient API hiccup or a
    self-hoster who hasn't added a Gemini pool yet shouldn't zero out a run's
    clips. Returns False only when a check that DID run explicitly rejected
    the candidate.
    """
    if not pool:
        return True

    start, end = candidate["start"], candidate["end"]
    rough_cut = None
    try:
        rough_cut = _rough_cut_candidate(source_video_path, start, end, video_duration)
    except Exception as e:
        print(f"⚠️ Vision confirm: rough-cut failed ({e}) — skipping confirmation for this candidate")
        return True

    results = {}

    def _run_visual():
        key = pool.acquire()
        if not key:
            return
        try:
            client = genai.Client(api_key=key)
            prompt = gemini_worker.VISION_VISUAL_CHECK_PROMPT_TEMPLATE.format(
                narrative_summary=candidate.get("narrative_summary", ""),
                candidate_start_offset=min(2.0, start))
            results["visual"] = _upload_and_generate(
                client, model_name, rough_cut, prompt, gemini_worker.VisionVisualCheckResponse)
        except gemini_worker.GeminiBlockedError:
            raise
        except Exception as e:
            pool.mark_bad(key)
            print(f"⚠️ Vision visual/audio check failed: {e}")

    def _run_context():
        results["context"] = _context_check_once(
            pool, model_name, source_video_path, candidate,
            video_duration, transcript_result, max_duration_ceiling)

    try:
        threads = [threading.Thread(target=_run_visual), threading.Thread(target=_run_context)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        try:
            os.remove(rough_cut)
        except OSError:
            pass

    visual = results.get("visual")
    context = results.get("context")
    if visual is None and context is None:
        return True  # neither check ran (no free keys / both errored) — fail open

    orig_start, orig_end = candidate["start"], candidate["end"]
    approved = True
    reasons = []

    if visual is not None:
        if not visual.get("approved", True):
            approved = False
            reasons.append(f"visual: {visual.get('reason', '(no reason given)')}")
        delta = max(-3.0, min(3.0, float(visual.get("suggested_start_delta", 0) or 0)))
        candidate["start"] = max(0.0, candidate["start"] + delta)

    if context is not None:
        if not context.get("approved", True) or not context.get("narrative_resolved", True):
            approved = False
            reasons.append(f"context: {context.get('reason', '(no reason given)')}")
        if not context.get("has_real_hook", True):
            approved = False
            reasons.append(f"hook: {context.get('reason', '(no reason given)')}")
        # Hook rescue: pull the start earlier (never later) to include the
        # setup a cold viewer needs — bounded so a "no hook" verdict can't
        # silently balloon the clip. Applied before the end-delta duration
        # check below so the ceiling accounts for the wider span.
        start_delta = min(0.0, float(context.get("suggested_start_delta", 0) or 0))
        start_delta = max(-30.0, start_delta)
        candidate["start"] = max(0.0, candidate["start"] + start_delta)

        delta = max(0.0, float(context.get("suggested_end_delta", 0) or 0))
        new_end = candidate["end"] + delta
        if new_end - candidate["start"] <= max_duration_ceiling:
            candidate["end"] = min(float(video_duration), new_end)

    boundaries_moved = candidate["start"] != orig_start or candidate["end"] != orig_end

    # A context/hook rejection always comes with a fix suggestion baked into
    # `has_real_hook: false` — the check prompt tells the model to propose
    # the delta needed rather than silently assume it works. We apply that
    # delta above, but never actually verified it fixed anything; on real
    # content this meant genuinely-rescuable candidates (e.g. one segment
    # pulled back to include the missing question) were dropped anyway,
    # because the ORIGINAL verdict never got revisited (confirmed 30-jul-2026,
    # a rescued candidate was still logged "still rejected" for the exact
    # problem the rescue fixed). One bounded retry — re-run just the context
    # check on the corrected boundaries — actually collects on the fix
    # instead of computing it and throwing it away.
    if not approved and context is not None and boundaries_moved:
        retry = _context_check_once(
            pool, model_name, source_video_path, candidate,
            video_duration, transcript_result, max_duration_ceiling)
        if retry is not None:
            retry_ok = (retry.get("approved", True) and retry.get("narrative_resolved", True)
                        and retry.get("has_real_hook", True))
            print(f"   🔁 Retry context check on rescued boundaries "
                  f"[{candidate['start']:.1f}s-{candidate['end']:.1f}s]: "
                  f"{'approved' if retry_ok else 'rejected — ' + retry.get('reason', '(no reason given)')}")
            if retry_ok:
                approved = True
                candidate.pop("_rejection_reason", None)
                # A second, smaller fix may still be worth taking (e.g. the
                # opening is now fine but the payoff needs a touch more room).
                delta = max(0.0, float(retry.get("suggested_end_delta", 0) or 0))
                new_end = candidate["end"] + delta
                if new_end - candidate["start"] <= max_duration_ceiling:
                    candidate["end"] = min(float(video_duration), new_end)
            else:
                reasons.append(f"retry: {retry.get('reason', '(no reason given)')}")

    if not approved:
        candidate["_rejection_reason"] = "; ".join(reasons) if reasons else "(no reason given)"

    if boundaries_moved:
        print(f"   🔧 Vision adjusted [{orig_start:.1f}s-{orig_end:.1f}s] -> "
              f"[{candidate['start']:.1f}s-{candidate['end']:.1f}s] "
              f"({'approved' if approved else 'still rejected'})")

    _extend_keep_spans_to_cover_boundaries(candidate)

    return approved


def _extend_keep_spans_to_cover_boundaries(candidate):
    """Vision confirmation can push start earlier / end later as a rescue
    (the has_real_hook backstop, or a narrative_resolved extension) — if
    the candidate carries DeepSeek's keep_spans (the jump-cut plan), those
    spans must grow to cover the rescued region too, or the jump-cutter
    would silently drop exactly the content the rescue was for (confirmed
    31-jul-2026: a hook rescue pulled start 1.2s earlier to include the
    missing question, but keep_spans still started at the OLD boundary —
    that 1.2s would never have made it into the rendered clip). No-op when
    the candidate carries no keep_spans.
    """
    keep_spans = candidate.get("keep_spans")
    if not keep_spans:
        return
    start, end = candidate["start"], candidate["end"]
    span_start = min(float(s.get("start", start)) for s in keep_spans)
    span_end = max(float(s.get("end", end)) for s in keep_spans)
    if start < span_start:
        keep_spans.append({"start": start, "end": span_start})
    if end > span_end:
        keep_spans.append({"start": span_end, "end": end})


def _snap_keep_spans_to_words(keep_spans, words, clip_start, clip_end, min_span_duration=1.0):
    """Snap each DeepSeek-proposed keep_span's boundaries onto real word
    edges — same reasoning as clip_selection.snap_clip_to_words (LLMs are
    bad at millisecond arithmetic, word timestamps are ground truth) —
    then clamp to [clip_start, clip_end], drop spans that collapse below
    min_span_duration once snapped, and merge spans left touching/
    overlapping after snapping. Returns a list of [start, end] pairs,
    sorted, non-overlapping. Falls back to unsnapped (but still clamped)
    boundaries when no nearby words exist.
    """
    if not keep_spans:
        return []

    relevant_words = [w for w in words if w.get('e', 0) > clip_start and w.get('s', 0) < clip_end]

    def _nearest_start(t):
        return min(relevant_words, key=lambda w: abs(w['s'] - t))['s'] if relevant_words else t

    def _nearest_end(t):
        return min(relevant_words, key=lambda w: abs(w['e'] - t))['e'] if relevant_words else t

    snapped = []
    for span in keep_spans:
        s = max(clip_start, min(clip_end, float(span.get('start', clip_start))))
        e = max(clip_start, min(clip_end, float(span.get('end', clip_end))))
        if e <= s:
            continue
        s = max(clip_start, _nearest_start(s))
        e = min(clip_end, _nearest_end(e))
        if e - s >= min_span_duration:
            snapped.append([s, e])

    if not snapped:
        return []

    snapped.sort(key=lambda sp: sp[0])
    merged = [snapped[0]]
    for s, e in snapped[1:]:
        if s <= merged[-1][1] + 0.05:  # touching/overlapping once snapped
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _remap_transcript_onto_jump_cut(transcript_result, keep_spans):
    """Build a transcript whose segment/word timestamps are shifted onto
    the JUMP-CUT video's own (shorter) timeline instead of the source
    video's — required for speaker_change_frames() and future captions to
    stay in sync once dead-air/cuttable stretches have been physically
    removed from the footage, not just skipped in metadata. Content
    outside every keep_span doesn't exist in the output video, so it's
    dropped, not just hidden.
    """
    offsets = []
    cursor = 0.0
    for s, e in keep_spans:
        offsets.append(cursor)
        cursor += (e - s)

    def _shift(t, s, offset):
        return t - s + offset

    remapped_segments = []
    for seg in transcript_result.get('segments', []):
        seg_start, seg_end = seg.get('start', 0), seg.get('end', 0)
        for (s, e), offset in zip(keep_spans, offsets):
            if seg_end <= s or seg_start >= e:
                continue
            new_words = []
            for w in seg.get('words', []):
                w_start = w.get('start', w.get('s', 0))
                w_end = w.get('end', w.get('e', 0))
                if w_end <= s or w_start >= e:
                    continue
                new_word = dict(w)
                ws = _shift(max(w_start, s), s, offset)
                we = _shift(min(w_end, e), s, offset)
                for key in ('start', 's'):
                    if key in new_word:
                        new_word[key] = ws
                for key in ('end', 'e'):
                    if key in new_word:
                        new_word[key] = we
                new_words.append(new_word)
            new_seg = dict(seg)
            new_seg['start'] = _shift(max(seg_start, s), s, offset)
            new_seg['end'] = _shift(min(seg_end, e), s, offset)
            new_seg['words'] = new_words
            remapped_segments.append(new_seg)

    remapped_transcript = dict(transcript_result)
    remapped_transcript['segments'] = remapped_segments
    return remapped_transcript, cursor


def _build_jump_cut_source(source_video_path, keep_spans, workdir):
    """Extract and concatenate only the keep_spans from the source video —
    the actual jump cut, removing dead air/cuttable stretches physically
    from the footage rather than just picking [start, end] boundaries.
    Each span is cut with the same encode params so the concat demuxer can
    stream-copy them together with no quality loss on the join. Returns
    the combined video's path.
    """
    segment_paths = []
    for idx, (s, e) in enumerate(keep_spans):
        seg_path = os.path.join(workdir, f"keep_{idx:03d}.mp4")
        cmd = ['ffmpeg', '-y', '-ss', f'{s:.3f}', '-to', f'{e:.3f}', '-i', source_video_path,
               *source_logo_crop_vf_args(),
               *video_encode_args(QUALITY_FAST), *audio_encode_args(), seg_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
        segment_paths.append(seg_path)

    list_path = os.path.join(workdir, 'jump_cut_concat.txt')
    with open(list_path, 'w') as f:
        for p in segment_paths:
            f.write(f"file '{p}'\n")

    combined_path = os.path.join(workdir, 'jump_cut_combined.mp4')
    subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path,
                     '-c', 'copy', combined_path], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
    return combined_path


def _apply_term_corrections(transcript_result, corrections):
    """Fix ASR mishearings of proper nouns/brand terms (DeepSeek-flagged,
    e.g. "Clod" -> "Claude") in place, across both segment text and
    word-level tokens — the latter is what captions actually render, so a
    text-only fix would leave the burned-in captions wrong.

    Whole-word, case-insensitive matching so a substring like "cloud" isn't
    clobbered while fixing "clod". Word-level replacement preserves each
    original word's leading-space/punctuation shell (main.py's word dicts
    carry those inside the 'word' string) by only swapping the inner
    alphabetic run.
    """
    for corr in corrections:
        wrong, correct = corr.get("wrong", "").strip(), corr.get("correct", "").strip()
        if not wrong or not correct:
            continue
        pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
        for segment in transcript_result.get('segments', []):
            if segment.get('text'):
                segment['text'] = pattern.sub(correct, segment['text'])
            for word in segment.get('words', []):
                if word.get('word'):
                    word['word'] = pattern.sub(correct, word['word'])


_QUESTION_START_RE = re.compile(
    r'^(who|what|when|where|why|how|do|does|did|are|is|can|could|'
    r'would|will|have|has|tell|walk|so)\b', re.IGNORECASE)
_LEADING_CONJUNCTION_RE = re.compile(
    r'^(and|but|so|because|which|or|yet|then)\b', re.IGNORECASE)
_DEPENDENT_OPENING_RE = re.compile(
    r'^(he|she|it|they|we|you|this|that|i|i\'m|i\'d|i\'ll|it\'s|that\'s|'
    r'daily|weekly|monthly|yearly|once|twice|thrice|three|two|one|four|five|'
    r'six|seven|eight|nine|ten|yes|no|yeah|yep|nope|nah|probably|maybe|'
    r'about|like|minimum|maximum|every|each|always|usually|sometimes)\b',
    re.IGNORECASE)


def _is_question_like(segment_text):
    """Round-5 spec 1.3: a segment is question-like when it ends in a literal
    '?' or starts with an interrogative — catches unpunctuated spoken
    questions ("Tell me how often") that the old endswith('?') check missed."""
    text = str(segment_text or '').strip()
    if not text:
        return False
    return text.endswith('?') or bool(_QUESTION_START_RE.match(text))


def _is_dependent_opening(opening_text):
    """True when a clip opening is a grammatical continuation that NEEDS the
    preceding question to make sense: leading conjunction, bare pronoun, or a
    dangling bare value (number / frequency / rating / yes-no). Self-contained
    claims ("If you don't understand…") must NOT match — that's the
    over-correction regression guard."""
    text = str(opening_text or '').strip()
    if not text:
        return False
    return bool(_LEADING_CONJUNCTION_RE.match(text)
                or _DEPENDENT_OPENING_RE.match(text))


def _sentence_start_at_or_before(transcript_result, t, max_backtrack=6.0):
    """Start of the sentence containing time ``t``, from word punctuation.

    Falls back to ``t`` when no sentence boundary is close enough (unpunctuated
    transcript, or the sentence runs longer than ``max_backtrack``), so this can
    only ever move a boundary EARLIER, never later, and never strands a caller.
    """
    words = []
    for seg in transcript_result.get('segments', []):
        for w in seg.get('words', []) or []:
            words.append({'w': w.get('word', ''), 's': w.get('start', 0),
                          'e': w.get('end', 0)})
    if not words:
        return t
    try:
        from clip_selection import sentence_boundaries
        starts, _ends = sentence_boundaries(words)
    except Exception:
        return t
    earlier = [x for x in starts if x <= t + 0.05 and t - x <= max_backtrack]
    return max(earlier) if earlier else t


def _extend_start_for_preceding_question(candidate, transcript_result,
                                         max_gap=6.0, max_segments=3,
                                         max_prepend=8.0):
    """Deterministic backstop for reply-only / context-dependent openings
    ("Deal breakers? I don't have any", "three times what?").

    Round-5 spec 1.3 — the old version only checked the immediately preceding
    segment for a literal '?'. Real misses: the question split across segments,
    unpunctuated spoken questions, and dangling-answer openings ("Daily, I'd
    say minimum 3 times a day") with no visible question. This version:

    - only rewinds when the OPENING is a dependent shape (leading conjunction,
      bare pronoun, or dangling bare value) — self-contained claims are never
      touched;
    - searches backward up to ``max_segments`` segments for the NEAREST
      question-like segment (ends in '?' or starts with an interrogative);
    - hard-caps the amount of setup prepended (``max_prepend``).

    Records ``candidate['_context_start']`` = the question's start so the word/
    sentence snapper can lock the boundary (clip_selection.snap_clip_to_words'
    ``context_start``), instead of re-truncating it (spec 1.2).
    """
    segments = sorted(
        (s for s in transcript_result.get('segments', []) if str(s.get('text', '')).strip()),
        key=lambda s: s.get('start', 0))
    if not segments:
        return
    start = float(candidate['start'])

    # Opening text: the segment actually being spoken at the candidate start
    # (its end is after the start). The question that ENDS just before the
    # start must not be mistaken for the opening.
    opening = None
    for seg in segments:
        if seg.get('end', 0) > start:
            opening = str(seg.get('text', '')).strip()
            break
    if not _is_dependent_opening(opening):
        return

    # Walk backward from the segment before the start, up to max_segments, and
    # rewind to the NEAREST question-like segment that is close enough.
    before_start = [s for s in segments if s.get('end', 0) <= start]
    for seg in reversed(before_start[-max_segments:]):
        gap = start - seg.get('end', 0)
        if gap < 0 or gap > max_gap:
            continue
        if _is_question_like(seg.get('text', '')):
            question_start = max(0.0, seg.get('start', start))
            # Rewind to the SENTENCE the question belongs to, not the segment.
            # Segments are utterance chunks and routinely split a sentence in
            # half, so "the segment ending in '?'" is often only the question's
            # TAIL. Ground truth from the shipped clip: the question is one
            # sentence at 833.72 ("When you say I can please you more than the
            # weekend, just so he knows, how often do you like it weekly?") but
            # it is split across segments, so rewinding to the segment start
            # (838.14) opened the clip on "like it weekly?" — which tells a
            # cold viewer nothing about what is weekly. The whole point of the
            # rewind is a question the viewer can actually understand.
            question_start = _sentence_start_at_or_before(
                transcript_result, question_start)
            if start - question_start > max_prepend:
                return  # too much setup to drag in
            candidate['start'] = question_start
            candidate['_context_start'] = question_start
            return


def get_viral_clips(transcript_result, video_duration, source_video_path=None,
                    clip_count=None, long_context_count=0, style_variant="balanced"):
    """Narrative-arc-aware clip selection.

    Tries DeepSeek first (deepseek_worker.deepseek_select_narrative_clips) if
    DEEPSEEK_API_KEY is configured: a single-pass, full-transcript read that
    traces a story to its actual resolution instead of scoring independent
    90s windows — the old approach here had no concept of narrative closure
    at all, so a hook spanning a window edge was silently truncated. Each
    DeepSeek candidate then gets a Gemini Vision confirmation pass (if a
    Gemini key pool is available) before word-snapping commits the final
    boundaries — see confirm_clip_with_vision. Falls back to the original
    Gemini 2-pass (score windows, then detail the shortlist) when DeepSeek
    isn't configured or its call fails, so a self-hoster who hasn't set up
    DeepSeek yet keeps working exactly as before. Cuts are snapped to word
    boundaries either way so clips don't start/end mid-word.
    """
    deepseek_result = deepseek_worker.deepseek_select_narrative_clips(
        transcript_result, video_duration, clip_count=clip_count,
        long_context_count=long_context_count, style_variant=style_variant)
    if deepseek_result and deepseek_result.get("term_corrections"):
        _apply_term_corrections(transcript_result, deepseek_result["term_corrections"])

    # Full word list — ground truth for snapping cut points, needed by both
    # paths. Built AFTER term corrections so captions/snapping see the fixed
    # spelling, not AssemblyAI's raw mishearing.
    words = []
    for segment in transcript_result['segments']:
        for word in segment.get('words', []):
            words.append({'w': word['word'], 's': word['start'], 'e': word['end']})

    if deepseek_result and deepseek_result.get("clips"):
        shorts = deepseek_result["clips"]
        for s in shorts:
            _extend_start_for_preceding_question(s, transcript_result)

        if source_video_path:
            pool = gemini_pool.pool_from_env()
            if pool:
                model_name = os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'
                print(f"👁️  Vision-confirming {len(shorts)} candidate(s) across a pool of {len(pool)} key(s)...")
                approved_shorts = []
                for s in shorts:
                    try:
                        approved = confirm_clip_with_vision(
                            pool, model_name, source_video_path, s, video_duration, transcript_result)
                    except gemini_worker.GeminiBlockedError as e:
                        print(f"🚫 Vision confirm blocked: {e} — dropping this candidate")
                        approved = False
                    if approved:
                        approved_shorts.append(s)
                    else:
                        print(f"   ✗ dropped candidate [{s.get('start', 0):.1f}s-{s.get('end', 0):.1f}s]: "
                              f"{s.get('_rejection_reason', 'vision confirmation rejected it')}")
                if not approved_shorts:
                    if os.environ.get("VISION_CONFIRM_FALLBACK", "1").strip().lower() in ("0", "false", "no"):
                        print("❌ Vision confirmation rejected every candidate — no clip had both a clean "
                              "opening and an actual narrative payoff. Returning no clips for this video "
                              "rather than one we already know is bad.")
                        return None
                    print("⚠️ Vision confirmation rejected every candidate, but VISION_CONFIRM_FALLBACK=1 "
                          "— shipping the narrative picks anyway. The vision review stays in the log for "
                          "inspection; the user explicitly prefers clips over a zero-clip failure.")
                else:
                    shorts = approved_shorts

        for s in shorts:
            # Long-context segments get a higher floor. With the shared 15s
            # minimum, a full-arc candidate that lost a little at the edges
            # could legally snap down into short territory — which is the
            # "I asked for long clips and got short ones" complaint arriving
            # by a second route, after the reviewer fix above.
            is_long = s.get("clip_type") == "long_context"
            ns, ne = snap_clip_to_words(
                s.get("start", 0), s.get("end", 0), words, video_duration,
                min_duration=45.0 if is_long else 15.0,
                context_start=s.get("_context_start"))
            s["start"], s["end"] = ns, ne
            s.pop("_context_start", None)
        result = {"shorts": shorts}
        if deepseek_result.get("cost_analysis"):
            result["cost_analysis"] = deepseek_result["cost_analysis"]
        return result

    print("\U0001f916  Analyzing with Gemini (2-pass: score → detail)...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        return None

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'
    language = str(transcript_result.get('language') or 'unknown')
    print(f"\U0001f916  Model: {model_name} | language: {language}")

    try:
        windows = build_transcript_windows(transcript_result, video_duration)
        print(f"   Built {len(windows)} scoring window(s).")
        costs = []

        # --- Pass 1: score windows in batches, keep the highest-scoring ---
        scored = []
        SCORE_BATCH = 8
        for b in range(0, len(windows), SCORE_BATCH):
            batch = windows[b:b + SCORE_BATCH]
            payload = [{"id": w["id"], "start": w["start"], "end": w["end"], "text": w["text"]} for w in batch]
            prompt = gemini_worker.SCORE_PROMPT_TEMPLATE.format(
                video_duration=video_duration, language=language,
                windows_json=json.dumps(payload, ensure_ascii=False))
            parsed, cost = _run_gemini_stage(client, model_name, prompt, gemini_worker.ScoreResponse)
            if cost:
                costs.append(cost)
            scored.extend(parsed.get("windows") or [])

        # Shortlist the top windows; scale with duration so long videos surface
        # more candidates without exploding the detail call.
        scored.sort(key=lambda w: w.get("score", 0), reverse=True)
        target = max(3, min(10, int(video_duration // 90) + 2))
        by_id = {w["id"]: w for w in windows}
        shortlist = [by_id[w["id"]] for w in scored[:target] if w.get("id") in by_id]
        if not shortlist:
            shortlist = windows[:target]  # scoring returned nothing usable
        print(f"   Shortlisted {len(shortlist)} window(s) for detail.")

        # --- Pass 2: detailed clip extraction on the shortlist ---
        payload = [{"id": w["id"], "start": w["start"], "end": w["end"], "text": w["text"]} for w in shortlist]
        prompt = gemini_worker.DETAIL_PROMPT_TEMPLATE.format(
            video_duration=video_duration, language=language,
            windows_json=json.dumps(payload, ensure_ascii=False))
        detail, cost = _run_gemini_stage(client, model_name, prompt, gemini_worker.DetailResponse)
        if cost:
            costs.append(cost)

        shorts = detail.get("shorts") or []
        # Snap each proposed clip onto real word boundaries (+ a bit of silence).
        for s in shorts:
            ns, ne = snap_clip_to_words(
                s.get("start", 0), s.get("end", 0), words, video_duration,
                context_start=s.get("_context_start"))
            s["start"], s["end"] = ns, ne
            s.pop("_context_start", None)

        # Aggregate cost across both passes.
        cost_analysis = None
        if costs:
            cost_analysis = {
                "input_tokens": sum(c.get("input_tokens", 0) for c in costs),
                "output_tokens": sum(c.get("output_tokens", 0) for c in costs),
                "total_cost": sum(c.get("total_cost", 0) for c in costs),
                "model": model_name,
            }
            print(f"\U0001f4b0 Total cost ({model_name}, 2-pass, {len(costs)} calls): ${cost_analysis['total_cost']:.6f}")

        if not shorts:
            print("⚠️ 2-pass returned no clips.")
            return None

        result = {"shorts": shorts}
        if cost_analysis:
            result["cost_analysis"] = cost_analysis
        return result
    except gemini_worker.GeminiBlockedError as e:
        # Content-policy rejection: propagate so the job fails with the real
        # reason instead of a generic "no clips found".
        print(f"🚫 {e}")
        raise
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return None


def get_visual_clips(video_path, video_duration, language="en"):
    """Clip a SILENT video by vision: Gemini watches the footage and picks the
    most engaging visual moments (no transcript). Returns the same
    {"shorts", "cost_analysis"} shape as get_viral_clips, or None."""
    print("🎥  Silent video — analyzing with Gemini vision (no transcript)...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found.")
        return None
    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'
    print(f"🎥  Model: {model_name} | uploading {os.path.basename(video_path)}…")

    file_upload = None
    try:
        file_upload = client.files.upload(file=video_path)
        deadline = time.time() + 180
        while True:
            info = client.files.get(name=file_upload.name)
            state = str(getattr(getattr(info, "state", info), "name", "")).upper()
            if state == "ACTIVE":
                break
            if state == "FAILED":
                print("❌ Gemini could not process the video.")
                return None
            if time.time() > deadline:
                print("❌ Gemini video processing timed out.")
                return None
            time.sleep(2)

        prompt = gemini_worker.VISUAL_PROMPT_TEMPLATE.format(
            video_duration=video_duration, language=language)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_worker.VisualResponse,
            safety_settings=gemini_worker.RELAXED_SAFETY_SETTINGS,
        )
        response = gemini_pool.generate_with_fallback(
            client, model_name, [file_upload, prompt], config=config,
            max_attempts=1, log=lambda msg: print(msg))
        gemini_worker.raise_if_blocked(response)
        parsed = json.loads(response.text)
        shorts = parsed.get("shorts") or []
        # Clamp to the real duration; drop anything degenerate.
        clean = []
        for s in shorts:
            s["start"] = max(0.0, float(s.get("start", 0)))
            s["end"] = min(float(video_duration), float(s.get("end", 0)))
            if s["end"] - s["start"] >= 1.0:
                clean.append(s)
        if not clean:
            print("⚠️ Vision pass returned no usable clips.")
            return None

        cost = gemini_worker._calculate_cost_analysis(response, model_name)
        if cost:
            print(f"💰 Vision cost ({model_name}): ${cost.get('total_cost', 0):.6f}")
        result = {"shorts": clean}
        if cost:
            result["cost_analysis"] = cost
        return result
    except gemini_worker.GeminiBlockedError as e:
        print(f"🚫 {e}")
        raise
    except Exception as e:
        print(f"❌ Gemini vision error: {e}")
        return None
    finally:
        if file_upload is not None:
            try:
                client.files.delete(name=file_upload.name)
            except Exception:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoCrop-Vertical with Viral Clip Detection.")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help="Path to the input video file.")
    input_group.add_argument('-u', '--url', type=str, help="YouTube URL to download and process.")
    
    parser.add_argument('-o', '--output', type=str, help="Output directory or file (if processing whole video).")
    parser.add_argument('--keep-original', action='store_true', help="Keep the downloaded YouTube video.")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip AI analysis and convert the whole video.")
    parser.add_argument('--clip-count', type=int, default=None,
                        help="Hard target clip count (1-40): narrative is built to "
                             "that number rather than the duration-based auto floor.")
    parser.add_argument('--long-context-clips', type=int, default=0,
                        help="Request up to N full-arc 1-3 min long-context clips "
                             "alongside the tight shorts (0 = off).")
    parser.add_argument('--format', type=str, default="auto", choices=["auto", "vertical", "horizontal", "square", "custom"],
                        help="Output aspect: vertical/auto (9:16), horizontal (keep 16:9), square (1:1).")
    parser.add_argument('--custom-width', type=int, default=None,
                        help="Custom output width (with --format custom); pairs with --custom-height.")
    parser.add_argument('--custom-height', type=int, default=None,
                        help="Custom output height (with --format custom); pairs with --custom-width.")
    parser.add_argument('--style-variant', type=str, default="balanced",
                        choices=["balanced", "high_energy", "story_driven"],
                        help="Narrative-prompt style variant (AI Preferences, round 3 item 6).")
    parser.add_argument('--remove-background-audio', type=str, default="",
                        choices=["", "auto", "isolate", "denoise"],
                        help="Strip background audio from delivered clips. "
                             "'isolate' = Demucs voice separation (removes music); "
                             "'denoise' = fast FFmpeg speech chain (removes hiss/room "
                             "tone only); 'auto' = Demucs when installed, else denoise.")

    args = parser.parse_args()
    remove_background_audio = (args.remove_background_audio or "").strip() or None
    output_format = args.format
    # Custom aspect ratio drives the reframe crop shape end-to-end; the
    # pipeline still scales to the delivery floor for quality, same as 9:16.
    output_aspect = None
    if output_format == "custom":
        if not (args.custom_width and args.custom_height):
            print("❌ --format custom requires --custom-width and --custom-height")
            exit(1)
        output_aspect = args.custom_width / args.custom_height

    script_start_time = time.time()
    # Per-stage wall-clock tracking for the ETA estimate (plan round 2,
    # item 3): durations are flushed to stage_durations.json at job end.
    _stage_t0 = time.time()
    _stage_durations = {}

    def _ensure_dir(path: str) -> str:
        """Create directory if missing and return the same path."""
        if path:
            os.makedirs(path, exist_ok=True)
        return path
    
    # 1. Get Input Video
    if args.url:
        # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
        # For whole-video runs (--skip-analysis), --output can be a file path.
        if args.output and not args.skip_analysis:
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default "."
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or "."
            else:
                output_dir = "."
        
        input_video, video_title = download_youtube_video(args.url, output_dir)
        _write_progress(output_dir, "download", note="source downloaded")
        _stage_durations["download"] = time.time() - _stage_t0
        _stage_t0 = time.time()
    else:
        input_video = args.input
        video_title = os.path.splitext(os.path.basename(input_video))[0]
        
        if args.output and not args.skip_analysis:
            # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default to input dir.
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or os.path.dirname(input_video)
            else:
                output_dir = os.path.dirname(input_video)

    if not os.path.exists(input_video):
        print(f"❌ Input file not found: {input_video}")
        exit(1)

    # 2. Decision: Analyze clips or process whole?
    if args.skip_analysis:
        print("⏩ Skipping analysis, processing entire video...")
        output_file = args.output if args.output else os.path.join(output_dir, f"{video_title}_vertical.mp4")
        render_clip(input_video, output_file, output_format,
                    custom_aspect=output_aspect)
    else:
        # Get duration (needed by both the transcript and the vision path).
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        # 3. Transcribe — unless the video has no audio, in which case fall back
        # to Gemini vision (picks clips from the imagery instead of the speech).
        from transcribe_backends import NoAudioError
        transcript = None
        try:
            _write_progress(output_dir, "transcribe", note="transcribing audio",
                            duration_seconds=duration)
            transcript = transcribe_video(input_video)
        except NoAudioError as e:
            print(f"🔇 {e} — switching to visual analysis.")
        _stage_durations["transcribe"] = time.time() - _stage_t0
        _stage_t0 = time.time()

        # 4. Gemini Analysis (transcript-driven, or vision for silent videos)
        _write_progress(output_dir, "analyze", note="analyzing narrative arcs")
        if transcript is not None:
            clips_data = get_viral_clips(transcript, duration,
                                         source_video_path=input_video,
                                         clip_count=args.clip_count,
                                         long_context_count=args.long_context_clips,
                                         style_variant=args.style_variant)
        else:
            clips_data = get_visual_clips(input_video, duration)
        _stage_durations["analyze"] = time.time() - _stage_t0
        _stage_t0 = time.time()

        if not clips_data or 'shorts' not in clips_data:
            # Deliberately fail instead of reframing the whole video: that path
            # wrote no metadata.json, so app.py marked the job failed anyway
            # (app.py:1087) after burning GPU on a render nobody could see.
            raise RuntimeError(
                "Clip detection failed — Gemini did not return usable clips for this video.")
        else:
            print(f"🔥 Found {len(clips_data['shorts'])} clips!")

            # Save metadata. Silent videos have no transcript → no subtitles,
            # which is correct (there's no speech to caption).
            clips_data['transcript'] = transcript or {"language": "none", "segments": []}
            # Source identity for same-source job reuse (round-5 feature: "if
            # the clip already exists, just map it") — /api/process checks this
            # before starting a fresh run of the same URL.
            clips_data['source_url'] = args.url or ""
            clips_data['source_file'] = os.path.basename(input_video)
            metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
            # Round-5 spec 4.1: write atomically (tmp + os.replace) so a crash
            # mid-write never leaves a truncated metadata file that every
            # consumer treats as authoritative. .ready markers gate the poll
            # loop; only ORPHAN markers (whose clip file no longer exists) are
            # purged — a marker whose file is present is a genuinely completed
            # clip and lets this run RESUME it instead of re-rendering
            # (_process_one_clip's skip-existing path).
            for marker in glob.glob(os.path.join(output_dir, "*.ready")):
                clip_file = marker[:-len(".ready")]
                if not os.path.exists(clip_file):
                    try:
                        os.remove(marker)
                    except OSError:
                        pass
            _meta_tmp = metadata_file + ".tmp"
            with open(_meta_tmp, 'w') as f:
                json.dump(clips_data, f, indent=2)
            os.replace(_meta_tmp, metadata_file)
            print(f"   Saved metadata to {metadata_file}")
            _write_progress(output_dir, "render", 0, len(clips_data['shorts']),
                            note="rendering clips")

            # Ground-truth word list for keep_span snapping — same source
            # get_viral_clips() uses for boundary snapping, rebuilt here
            # since it's cheap pure-Python and not passed out of that call.
            _all_words = []
            if transcript:
                for segment in transcript.get('segments', []):
                    for word in segment.get('words', []):
                        _all_words.append({'w': word.get('word', ''),
                                           's': word.get('start', 0), 'e': word.get('end', 0)})

            # Gemini key pool for the scene-context direction layer, built
            # once and shared across clip workers (it's internally safe to
            # acquire from several threads — that's what it exists for).
            # Named distinctly because `pool` below is the ThreadPoolExecutor.
            vision_pool = gemini_pool.pool_from_env()
            vision_model = os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'

            # 5. Process clips in parallel: each worker cuts + renders one
            # clip. Renders are mostly ffmpeg subprocesses (parallelize well);
            # detector inference is serialized internally via DETECT_LOCK.
            def _process_one_clip(i, clip):
                start = clip['start']
                end = clip['end']
                print(f"\n🎬 Processing Clip {i+1}: {start}s - {end}s")
                print(f"   Title: {clip.get('video_title_for_youtube_short', 'No Title')}")

                clip_filename = f"{video_title}_clip_{i+1}.mp4"
                clip_temp_path = os.path.join(output_dir, f"temp_{clip_filename}")
                clip_final_path = os.path.join(output_dir, clip_filename)
                jump_cut_workdir = None

                # Resume: if this clip's final file AND its .ready marker both
                # already exist (a previous run completed this exact clip), map
                # to it instead of re-rendering — the poll loop will surface it
                # immediately. Saves GPU time on re-runs of the same source.
                if (os.path.exists(clip_final_path)
                        and os.path.getsize(clip_final_path) > 0
                        and os.path.exists(
                            os.path.join(output_dir, f"{clip_filename}.ready"))):
                    print(f"   ♻️ Clip {i+1} already exists — mapping to "
                          f"{clip_filename} (no re-render)")
                    return True

                # keep_spans marks the sub-ranges DeepSeek judged essential —
                # everything else in [start, end] is dead air/filler to jump-
                # cut out, not just boundary trim (see RESEARCH_viral_clip_
                # patterns.md §4 — 6/6 real published shorts studied do this,
                # the single most consistent editing pattern found).
                keep_spans = _snap_keep_spans_to_words(
                    clip.get('keep_spans') or [], _all_words, start, end)
                total_kept = sum(e - s for s, e in keep_spans)
                # If DeepSeek returned nothing usable, or the kept spans
                # already cover ~all of [start, end], there's nothing to cut
                # — skip the extra re-encode/concat pass entirely.
                do_jump_cut = bool(keep_spans) and total_kept < (end - start) * 0.97

                try:
                    if do_jump_cut:
                        jump_cut_workdir = tempfile.mkdtemp(prefix=f"jumpcut_{i}_")
                        removed = (end - start) - total_kept
                        print(f"   ✂️  Jump-cutting {len(keep_spans)} span(s), "
                              f"removing {removed:.1f}s of dead air/filler")
                        combined_path = _build_jump_cut_source(input_video, keep_spans, jump_cut_workdir)
                        clip_transcript, new_duration = _remap_transcript_onto_jump_cut(
                            transcript or {"segments": []}, keep_spans)
                        # os.replace requires same-filesystem; the jump-cut
                        # workdir (mkdtemp -> /tmp) and output_dir (a bind
                        # mount) are different devices in this container,
                        # so a plain rename fails with EXDEV (confirmed in
                        # prod, 31-jul-2026). shutil.move falls back to
                        # copy+delete across devices.
                        shutil.move(combined_path, clip_temp_path)
                        render_clip_start, render_clip_end = 0.0, new_duration
                    else:
                        # ffmpeg cut — re-encoding for precision on strict seconds
                        cut_command = [
                            'ffmpeg', '-y',
                            '-ss', str(start),
                            '-to', str(end),
                            '-i', input_video,
                            *source_logo_crop_vf_args(),
                            *video_encode_args(QUALITY_FAST),
                            *audio_encode_args(),
                            clip_temp_path
                        ]
                        subprocess.run(cut_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        clip_transcript = transcript
                        render_clip_start, render_clip_end = start, end

                    # Third verification layer, on the STREAMLINED cut (post
                    # jump-cut) so its timestamps land on the same clock the
                    # renderer uses: Gemini watches the finished clip and
                    # directs the camera — who to be on, and why. See
                    # analyze_scene_context.
                    scene_ctx = analyze_scene_context(
                        vision_pool, vision_model, clip_temp_path,
                        render_clip_end - render_clip_start,
                        transcript_result=clip_transcript,
                        clip_start=render_clip_start, clip_end=render_clip_end)
                    focus_directives = scene_ctx["directives"]
                    primary_subject_x = scene_ctx.get("primary_subject_x")

                    success, general_ranges = render_clip(
                        clip_temp_path, clip_final_path, output_format,
                        transcript=clip_transcript, clip_start=render_clip_start,
                        clip_end=render_clip_end,
                        focus_directives=focus_directives,
                        primary_subject_x=primary_subject_x,
                        custom_aspect=output_aspect)
                    if success and os.environ.get("WATERMARK") == "1":
                        apply_watermark(clip_final_path)
                    if success:
                        # Captions last, so they sit on top of the watermark and
                        # the canonical file stays clean for re-styling.
                        auto_caption_clip(clip_final_path, clip_transcript, render_clip_start,
                                         render_clip_end, general_ranges=general_ranges)
                        # Optional background-audio removal, last before the
                        # ready marker. Copies the video stream, so isolating
                        # the voice never costs a generation of video quality.
                        # Fails OPEN: a separation error must ship the clip with
                        # its original audio, never a silent or missing track.
                        if remove_background_audio:
                            try:
                                import audio_cleanup
                                cleaned = clip_final_path + ".voice.mp4"
                                audio_cleanup.clean_audio(
                                    clip_final_path, cleaned,
                                    mode=remove_background_audio)
                                shutil.move(cleaned, clip_final_path)
                                print(f"   🔇 Background audio removed ({remove_background_audio}).")
                            except Exception as e:
                                print(f"   ⚠️ Background-audio removal failed ({e}); "
                                      "keeping the original audio.")
                                try:
                                    os.remove(clip_final_path + ".voice.mp4")
                                except OSError:
                                    pass
                        # Only now — captions burned (or deliberately skipped) and
                        # the file fully written — is the clip safe to surface.
                        # app.py's poll loop gates on this marker.
                        _mark_clip_ready(output_dir, clip_filename)
                        print(f"   ✅ Clip {i+1} ready: {clip_final_path}")
                    return success
                finally:
                    if os.path.exists(clip_temp_path):
                        os.remove(clip_temp_path)
                    if jump_cut_workdir:
                        shutil.rmtree(jump_cut_workdir, ignore_errors=True)

            # 5 on GPU hosts (T4 verification showed idle encode capacity);
            # fewer on CPU hosts, where parallel ffmpeg/whisper threads
            # oversubscribe a small box (see hardware_defaults). An explicit
            # CLIP_WORKERS env var always wins.
            clip_workers = max(
                int(os.environ.get("CLIP_WORKERS") or default_clip_workers()), 1)
            shorts = clips_data['shorts']
            _progress_lock = threading.Lock()
            _rendered_count = [0]
            with ThreadPoolExecutor(max_workers=min(clip_workers, len(shorts))) as pool:
                futures = {pool.submit(_process_one_clip, i, clip): i
                           for i, clip in enumerate(shorts)}
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        ok = future.result()
                    except Exception as e:
                        ok = False
                        print(f"   ❌ Clip {i+1} failed: {type(e).__name__}: {e}")
                    if ok:
                        with _progress_lock:
                            _rendered_count[0] += 1
                            _write_progress(output_dir, "render",
                                            _rendered_count[0], len(shorts))
            _write_progress(output_dir, "finalize", len(shorts), len(shorts),
                            note="job complete")
            _stage_durations["render"] = time.time() - _stage_t0
            _stage_t0 = time.time()
            _stage_durations["finalize"] = time.time() - _stage_t0

    # Clean up original if requested
    if args.url and not args.keep_original and os.path.exists(input_video):
        # Source panel artifacts (stills + preview proxy) must survive the
        # cleanup — generate them while the original is still on disk.
        _generate_source_artifacts(input_video, output_dir)
        os.remove(input_video)
        print(f"🗑️  Cleaned up downloaded video.")

    # Feed the rolling per-stage averages used by the dashboard's ETA.
    if _stage_durations:
        record_stage_durations(output_dir, _stage_durations)

    total_time = time.time() - script_start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
