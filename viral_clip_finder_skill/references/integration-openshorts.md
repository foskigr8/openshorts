# Integration Guide — Wiring viral-clip-finder into an Existing Pipeline

This reference covers how to replace the "find viral moments" step in an existing clip-generation pipeline (OpenShorts, OpusClip-style architecture, or custom) with the viral-clip-finder skill. It also covers how to add a face-identification layer (DeepFace or similar) that enriches the skill's input with named speakers.

## Table of contents

1. [Architectural fit](#1-architectural-fit)
2. [OpenShorts Stage 3 replacement](#2-openshorts-stage-3-replacement)
3. [Adding face identification (DeepFace)](#3-adding-face-identification-deepface)
4. [Input format contract](#4-input-format-contract)
5. [Output format conversion](#5-output-format-conversion)
6. [Code-level integration points](#6-code-level-integration-points)
7. [Performance and cost](#7-performance-and-cost)
8. [Honest assessment: do you actually need face ID?](#8-honest-assessment-do-you-actually-need-face-id)

---

## 1. Architectural fit

Most AI clip-finders (OpenShorts, OpusClip, Vidyo, Munch, Klap, Vizard) share the same 7-stage pipeline:

```
1. Ingest (yt-dlp / upload)
2. Transcribe (faster-whisper / AssemblyAI)
3. Find viral moments  ← THIS IS WHAT WE'RE REPLACING
4. Cut (FFmpeg precision cut)
5. Reframe 9:16 (MediaPipe + YOLOv8 + cameraman logic)
6. Subtitle + effects
7. Publish
```

Stage 3 is the judgment layer. It's where the pipeline decides which 15–60 second windows of a multi-hour source are worth cutting. In OpenShorts specifically, this is currently a DeepSeek call that returns 3–15 candidates with `keep_spans` for jump-cutting, followed by a Gemini vision confirmation pass.

The viral-clip-finder skill replaces the *judgment* in Stage 3 — not the surrounding infrastructure. The transcription, cutting, reframing, subtitling, and publishing stages all stay. Only the "decide what's viral" step swaps out.

### What stays the same

- Stage 1 (ingest), Stage 2 (transcription), Stage 4 (cutting), Stage 5 (reframing), Stage 6 (subtitle/effects), Stage 7 (publish)
- The existing `keep_spans` jump-cut logic (the skill's trim suggestions can be converted to keep_spans)
- The existing Gemini vision confirmation pass (useful for verifying the skill's picks actually land on screen)
- The existing scene detection, identity tracking, and reframing logic

### What changes

- The DeepSeek call that produces 3–15 candidate clips → replaced by the skill's 6-step methodology
- The candidates come back scored (0–100) and ranked, with named patterns, anti-pattern rejections, and cut briefs
- Optionally: the input to the skill is enriched with named-speaker identity (from face ID)

---

## 2. OpenShorts Stage 3 replacement

The current OpenShorts Stage 3 (in `get_viral_clips` at `main.py:2835`) does this:

1. Sends the entire transcript to DeepSeek with a prompt asking for 3–15 viral moments
2. DeepSeek returns candidates with `keep_spans` (sub-ranges for jump-cutting) and `term_corrections`
3. Each candidate gets a Gemini vision confirmation pass (upload rough cut, check if hook lands and payoff is real)
4. Cut points snap to word boundaries

To replace this with the viral-clip-finder skill:

### Step 2a: Construct the skill's input

Take the transcript from Stage 2 and format it as the skill expects:

```python
def format_transcript_for_skill(transcript_words, diarization=None, face_identities=None):
    """
    Convert OpenShorts' internal transcript format (list of {w, s, e, speaker?})
    into the timecoded transcript format the viral-clip-finder skill expects.
    """
    lines = []
    current_speaker = None
    current_line_words = []
    current_line_start = None

    for word in transcript_words:
        speaker = word.get("speaker", "Speaker")
        # If face identities are available, replace anonymous speaker with name
        if face_identities and speaker in face_identities:
            speaker = face_identities[speaker]["name"]

        if speaker != current_speaker or current_line_start is None:
            # Flush previous line
            if current_line_words:
                timestamp = format_timestamp(current_line_start)
                text = " ".join(current_line_words)
                lines.append(f"[{timestamp}] {current_speaker}: {text}")
            # Start new line
            current_speaker = speaker
            current_line_words = [word["w"]]
            current_line_start = word["s"]
        else:
            current_line_words.append(word["w"])

    # Flush last line
    if current_line_words:
        timestamp = format_timestamp(current_line_start)
        text = " ".join(current_line_words)
        lines.append(f"[{timestamp}] {current_speaker}: {text}")

    return "\n".join(lines)


def format_timestamp(seconds):
    """Convert 1234.567 seconds to '00:20:34' format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
```

### Step 2b: Call the skill

The skill is invoked via a prompt to your LLM (the same LLM that powers your agent, e.g. Claude, GPT-4o, Gemini, or GLM). Construct the prompt as:

```python
SKILL_PATH = "/path/to/viral-clip-finder/SKILL.md"

def call_viral_clip_finder(transcript_text, target_platforms=None, target_clip_count=3):
    """
    Invoke the viral-clip-finder skill via the LLM.
    Returns the skill's Markdown report.
    """
    skill_content = open(SKILL_PATH).read()

    # Load only the relevant reference files (progressive disclosure)
    # The skill body itself contains guidance on when to load each reference
    # In practice, load all references up front if context allows
    references = {}
    for ref in ["niches", "hook-library", "anti-patterns", "scoring-rubric"]:
        ref_path = f"/path/to/viral-clip-finder/references/{ref}.md"
        references[ref] = open(ref_path).read()

    system_prompt = f"""You are using the viral-clip-finder skill to find viral clips in a transcript.

SKILL INSTRUCTIONS:
{skill_content}

REFERENCE: HOOK LIBRARY
{references['hook-library']}

REFERENCE: ANTI-PATTERNS
{references['anti-patterns']}

REFERENCE: NICHES
{references['niches']}

REFERENCE: SCORING RUBRIC
{references['scoring-rubric']}
"""

    user_prompt = f"""Find me the top {target_clip_count} viral clips from this transcript.

Target platforms: {target_platforms or "TikTok, Reels, Shorts, X"}
Output format: follow the skill's output contract exactly.

TRANSCRIPT:
{transcript_text}
"""

    # Call your LLM here (Claude, GPT-4o, Gemini, GLM, etc.)
    response = your_llm_client.messages.create(
        model="your-model",
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=8000,
    )

    return response.content[0].text
```

### Step 2c: Parse the skill's output

The skill returns a Markdown report. Parse it back into OpenShorts' internal format:

```python
import re
import json

def parse_skill_output(markdown_report):
    """
    Parse the viral-clip-finder skill's Markdown report into OpenShorts'
    internal clip format (compatible with the existing Stage 4+ pipeline).
    """
    clips = []

    # Find all clip sections
    clip_sections = re.split(r'^## Clip #', markdown_report, flags=re.MULTILINE)[1:]

    for section in clip_sections:
        clip = {}

        # Score
        score_match = re.search(r'^(\d+)/100', section, re.MULTILINE)
        clip["score"] = int(score_match.group(1)) if score_match else 0

        # Timestamps - need to convert HH:MM:SS back to seconds
        start_match = re.search(r'\| Start \| (\d{2}:\d{2}:\d{2}) \|', section)
        end_match = re.search(r'\| End \| (\d{2}:\d{2}:\d{2}) \|', section)
        clip["start"] = parse_timestamp(start_match.group(1)) if start_match else 0
        clip["end"] = parse_timestamp(end_match.group(1)) if end_match else 0

        # Patterns
        primary_match = re.search(r'\| Primary pattern \| (.+?) \|', section)
        clip["primary_pattern"] = primary_match.group(1) if primary_match else None

        # Extract keep_spans from trim suggestion
        # (Convert "Cut from 00:14:34" into keep_span starting at that timestamp)
        keep_spans = extract_keep_spans(section, clip["start"], clip["end"])
        clip["keep_spans"] = keep_spans

        # Extract cut brief fields for downstream use
        clip["cold_open"] = extract_field(section, "Cold open line")
        clip["caption_text"] = extract_field(section, "Caption text")
        clip["b_roll_cues"] = extract_field(section, "B-roll cues")
        clip["cta_placement"] = extract_field(section, "CTA placement")
        clip["sound_design"] = extract_field(section, "Sound design")
        clip["platform_fit"] = extract_field(section, "Platform fit")
        clip["reaction_cam"] = extract_field(section, "Reaction cam")  # only if enriched input

        clips.append(clip)

    # Sort by score descending
    clips.sort(key=lambda c: c["score"], reverse=True)

    return clips


def parse_timestamp(hhmmss):
    h, m, s = hhmmss.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def extract_keep_spans(section, clip_start, clip_end):
    """
    Parse the trim suggestion to derive keep_spans.
    If the brief says "Cut from 00:14:34, drop the lead-in",
    keep_span = [[00:14:34, clip_end]]
    """
    trim_match = re.search(r'\*\*Trim suggestion:\*\*\s*(.+?)(?:\n-|\n\n|\Z)', section, re.DOTALL)
    if not trim_match:
        return [[clip_start, clip_end]]  # no trim suggestion = keep whole clip

    trim_text = trim_match.group(1)
    # Look for "Cut from HH:MM:SS" or "Start at HH:MM:SS"
    start_match = re.search(r'(?:Cut from|Start at|Start hard on.*?at)\s+(\d{2}:\d{2}:\d{2})', trim_text)
    if start_match:
        keep_start = parse_timestamp(start_match.group(1))
    else:
        keep_start = clip_start

    # Look for "End at HH:MM:SS" or "Cut to HH:MM:SS" for end trim
    end_match = re.search(r'(?:End at|Cut to|Stop at)\s+(\d{2}:\d{2}:\d{2})', trim_text)
    if end_match:
        keep_end = parse_timestamp(end_match.group(1))
    else:
        keep_end = clip_end

    return [[keep_start, keep_end]]
```

### Step 2d: Wire into the existing pipeline

In `main.py`, replace the DeepSeek call in `get_viral_clips` with a call to your skill wrapper:

```python
# BEFORE (OpenShorts Stage 3, ~line 2835):
def get_viral_clips(transcript, video_duration, ...):
    # DeepSeek call
    candidates = deepseek_find_clips(transcript, ...)
    # Gemini vision confirmation
    confirmed = gemini_vision_confirm(candidates, ...)
    return confirmed

# AFTER:
def get_viral_clips(transcript, video_duration, face_identities=None, ...):
    # Format transcript for the skill
    formatted = format_transcript_for_skill(transcript, face_identities=face_identities)

    # Call the viral-clip-finder skill
    skill_report = call_viral_clip_finder(formatted, target_clip_count=5)

    # Parse back to OpenShorts format
    clips = parse_skill_output(skill_report)

    # Keep the existing Gemini vision confirmation pass
    # (It's still useful — verifies the skill's picks actually land on screen)
    confirmed = gemini_vision_confirm(clips, ...)

    return confirmed
```

The downstream Stage 4 (cutting) consumes the same `start`, `end`, and `keep_spans` fields — no changes needed there. The extra fields the skill produces (`cold_open`, `caption_text`, `b_roll_cues`, etc.) flow through to Stage 6 (subtitles/effects) where they can be used to drive hook overlays and B-roll insertion.

---

## 3. Adding face identification (DeepFace)

Your existing OpenShorts code already has face detection (MediaPipe BlazeFace), face tracking across frames (ByteTrack/BoT-SORT in `identity_tracker.py`), and lip-sync speaker ID (LR-ASD). What it lacks is **naming** those tracked identities — your system knows "Person 1 is talking" but not "Person 1 is Joe Rogan."

DeepFace (or InsightFace, Facenet-PyTorch, ArcFace) fills this gap. It compares detected faces against a database of known faces and returns identity with confidence.

### Step 3a: Choose your library

| Library | Models | Speed | Accuracy | License |
|---|---|---|---|---|
| **DeepFace** | VGG-Face, Facenet, Facenet512, OpenFace, ArcFace, SFace | Medium | High | MIT |
| **InsightFace** | ArcFace, Buffalo models | Fast (GPU-optimized) | Very high | Apache 2.0 (non-commercial restrictions on some models) |
| **Facenet-PyTorch** | InceptionResnet | Fast | High | MIT |
| **Compreface** (self-hosted service) | Multiple | Medium | High | Apache 2.0 |

**Recommendation:** For your dual-GPU setup, **InsightFace** is the best choice. It's the fastest on GPU, the ArcFace model is the current state-of-the-art for face recognition, and it integrates cleanly with the existing OpenShorts stack (which already uses YOLOv8 — InsightFace uses similar ONNX/TensorRT pipelines).

If you want maximum model variety and don't mind slower inference, DeepFace is fine. It supports more models and is more "Pythonic" to integrate.

### Step 3b: Build the known-faces database

For podcasts/streams, you have two options:

**Option A: Per-show database (recommended for episodic content)**
- For each show (e.g. "Joe Rogan Experience"), maintain a small DB of recurring faces: the host, regular guests, recurring characters
- Database can be as simple as a folder of `{name}.jpg` files, one per person
- InsightFace/DeepFace handles the embedding generation and comparison

**Option B: Celebrity database (for one-off viral clips)**
- Use a public celebrity face database (e.g. Celebrity Face Dataset)
- Useful for general-purpose identification but heavier to maintain
- Many public figures won't be in the DB; falls back to "unrecognized"

For most use cases, Option A is sufficient and far more accurate for the shows you actually process.

```python
# Example: Build a known-faces DB from a folder of headshots
import insightface
import numpy as np
from insightface.utils import face_align
import cv2
import os

class KnownFacesDB:
    def __init__(self, db_path, model_name="buffalo_l"):
        self.app = insightface.app.FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for GPU 0
        self.embeddings = {}  # name -> np.array
        self._load_db(db_path)

    def _load_db(self, db_path):
        """Load all .jpg/.png files named {person_name}.jpg from db_path."""
        for fname in os.listdir(db_path):
            if not fname.lower().endswith(('.jpg', '.png')):
                continue
            name = os.path.splitext(fname)[0]
            img = cv2.imread(os.path.join(db_path, fname))
            faces = self.app.get(img)
            if len(faces) > 0:
                # Use the largest face
                largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                self.embeddings[name] = largest.normed_embedding

    def identify(self, face_embedding, threshold=0.4):
        """
        Compare a face embedding against the DB.
        Returns (name, confidence) or (None, 0) if no match.
        """
        best_name = None
        best_sim = 0
        for name, db_emb in self.embeddings.items():
            sim = float(np.dot(face_embedding, db_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= threshold:
            return best_name, best_sim
        return None, 0.0
```

### Step 3c: Run face ID over the video

Sample frames at a reasonable rate (every 0.5–1 second is enough for talking-head content — don't waste GPU on every frame):

```python
def identify_faces_in_video(video_path, known_faces_db, sample_fps=2):
    """
    Run face identification over the video.
    Returns: list of (timestamp, name, confidence) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = int(fps / sample_fps)
    identifications = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            faces = known_faces_db.app.get(frame)
            for face in faces:
                name, conf = known_faces_db.identify(face.normed_embedding)
                if name:
                    identifications.append({
                        "timestamp": timestamp,
                        "name": name,
                        "confidence": conf,
                        "bbox": face.bbox.tolist()
                    })

        frame_idx += 1

    cap.release()
    return identifications
```

### Step 3d: Merge face ID with the existing identity tracker

Your existing `identity_tracker.py` produces anonymous track IDs (Person 1, Person 2, etc.) that persist across frames. The face ID produces (name, timestamp) pairs. Merge them:

```python
def merge_face_id_with_tracker(tracker_tracks, face_identifications):
    """
    Map anonymous track IDs to named identities using face ID results.

    tracker_tracks: list of {track_id, frames: [{frame_idx, bbox}]}
    face_identifications: list of {timestamp, name, confidence, bbox}

    Returns: dict {track_id -> {"name": str, "confidence": float, "on_screen": [[start, end], ...]}}
    """
    track_to_name = {}  # track_id -> {name: vote_count}
    track_to_screen_time = {}  # track_id -> list of (start, end) ranges

    # For each face identification, find the track whose bbox overlaps at that timestamp
    for ident in face_identifications:
        matching_track = find_track_at_timestamp(
            tracker_tracks, ident["timestamp"], ident["bbox"]
        )
        if matching_track is None:
            continue
        if matching_track not in track_to_name:
            track_to_name[matching_track] = {}
        track_to_name[matching_track][ident["name"]] = \
            track_to_name[matching_track].get(ident["name"], 0) + 1

    # Pick the name with the most votes for each track
    result = {}
    for track_id, votes in track_to_name.items():
        if not votes:
            continue
        best_name = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        confidence = votes[best_name] / total_votes
        if confidence >= 0.6:  # only commit if 60%+ of identifications agree
            result[track_id] = {
                "name": best_name,
                "confidence": confidence,
                "on_screen": track_to_screen_time.get(track_id, [])
            }

    return result


def find_track_at_timestamp(tracks, timestamp, bbox, iou_threshold=0.3):
    """Find which track is active at this timestamp with bbox overlap."""
    for track in tracks:
        for frame in track["frames"]:
            if abs(frame["timestamp"] - timestamp) < 0.5:  # within 500ms
                iou = compute_iou(frame["bbox"], bbox)
                if iou >= iou_threshold:
                    return track["track_id"]
    return None
```

### Step 3e: Merge with diarization

The final piece: combine the named face tracks with the voice diarization. This gives you a unified identity for each speaker turn:

```python
def merge_with_diarization(transcript_with_diarization, face_identities):
    """
    Replace anonymous speaker labels in the transcript with named identities.
    Uses LR-ASD lip-sync results to match voice to face.

    transcript_with_diarization: list of {w, s, e, speaker: "S1"|"S2"|...}
    face_identities: dict {track_id -> {name, confidence, on_screen}}

    Returns: same transcript but with speaker labels replaced by names where confident.
    """
    # Build speaker_id -> name mapping using lip-sync results
    # (Assumes you've already correlated diarization speaker IDs to face track IDs
    # via LR-ASD, which the existing pipeline does)
    speaker_to_name = {}
    for speaker_id, track_id in speaker_to_track_mapping.items():
        if track_id in face_identities:
            speaker_to_name[speaker_id] = face_identities[track_id]["name"]

    # Apply to transcript
    for word in transcript_with_diarization:
        if word["speaker"] in speaker_to_name:
            word["speaker"] = speaker_to_name[word["speaker"]]

    return transcript_with_diarization
```

The output of this is a transcript where speaker labels are actual names ("Joe Rogan", "Elon Musk") instead of anonymous IDs ("Speaker 1", "Speaker 2"). This is the enriched input format the skill consumes.

### Step 3f: Where to insert this in the pipeline

The face ID stage should run **after** transcription + diarization but **before** the viral moment selection:

```
1. Ingest
2. Transcribe (faster-whisper) + Diarize (AssemblyAI or pyannote)
   2.5. Face ID (DeepFace/InsightFace) ← NEW
        - Run face detection on sampled frames
        - Identify faces against known-faces DB
        - Merge with diarization via lip-sync (LR-ASD)
        - Output: transcript with named speakers
3. Find viral moments (viral-clip-finder skill) ← REPLACED
4. Cut
5. Reframe (uses existing identity_tracker, which now has named IDs)
6. Subtitle + effects
7. Publish
```

With your dual GPU, dedicate **GPU 0** to the existing pipeline (Whisper, YOLOv8, MediaPipe, LR-ASD, FFmpeg) and **GPU 1** to face ID (InsightFace). They run in parallel; face ID adds maybe 5–15 minutes to a 2-hour video, which is acceptable since transcription typically takes 5–10 minutes anyway.

---

## 4. Input format contract

The skill accepts three input tiers, in order of richness:

### Tier 1: Plain transcript (minimum viable)

```
[00:00:00] Welcome back to the show, today we have...
[00:00:14] Thanks for having me.
[00:00:16] So I want to start with something controversial...
```

No speaker labels. The skill can find viral clips but loses speaker attribution.

### Tier 2: Diarized transcript (default)

```
[00:00:00] Speaker A: Welcome back to the show, today we have...
[00:00:14] Speaker B: Thanks for having me.
[00:00:16] Speaker A: So I want to start with something controversial...
```

Anonymous speaker labels. The skill can attribute clips to speakers but doesn't know who they are. This is what most pipelines (including OpenShorts without face ID) produce.

### Tier 3: Named-speaker transcript (enriched)

```
[00:00:00] Joe Rogan: Welcome back to the show, today we have...
[00:00:14] Elon Musk: Thanks for having me.
[00:00:16] Joe Rogan: So I want to start with something controversial...
```

Named speakers. The skill can detect Status Play patterns, produce named-speaker cut briefs, recommend reaction cam cuts, and route to niches based on speaker identity.

Optionally accompanied by a face trajectory sidecar:

```json
{
  "identities": [
    {
      "name": "Joe Rogan",
      "on_screen": [["00:00:00", "00:14:32"], ["00:15:01", "00:31:44"]],
      "confidence": 0.94
    },
    {
      "name": "Elon Musk",
      "on_screen": [["00:00:08", "00:47:22"]],
      "confidence": 0.91
    }
  ]
}
```

This lets the skill recommend "cut to Joe Rogan's reaction at 00:14:38" in the cut brief — the most valuable single capability face ID unlocks.

### What to send your LLM

When calling the skill, pass the transcript text in the user message and the face trajectory (if available) as JSON in the same message. The skill's prompt template handles both:

```
SYSTEM: [skill content + reference files]

USER:
Find me the top 3 viral clips from this transcript.

TRANSCRIPT:
[00:00:00] Joe Rogan: Welcome back to the show...
[00:00:14] Elon Musk: Thanks for having me.
...

FACE TRAJECTORY (optional):
{"identities": [{"name": "Joe Rogan", "on_screen": [...]}, ...]}

Target platforms: TikTok, Reels, Shorts, X
Output format: follow the skill's output contract exactly.
```

---

## 5. Output format conversion

The skill outputs a Markdown report. To convert to OpenShorts' internal format, see the parser in section 2c. The key fields map as follows:

| Skill output field | OpenShorts internal field | Stage that consumes it |
|---|---|---|
| Start timestamp | `clip.start` | Stage 4 (cutting) |
| End timestamp | `clip.end` | Stage 4 (cutting) |
| Trim suggestion | `clip.keep_spans` | Stage 4 (jump-cut pass) |
| Caption text | `clip.hook_text` | Stage 6 (hook overlay) |
| B-roll cues | `clip.b_roll_cues` | Stage 6 (effects) |
| Sound design | `clip.sound_design` | Stage 6 (audio effects) |
| CTA placement | `clip.cta` | Stage 6 (end card) |
| Reaction cam | `clip.reaction_cues` | Stage 5 (reframing decision layer) |
| Score | `clip.virality_score` | Dashboard display, ranking |
| Primary pattern | `clip.pattern_tag` | Dashboard display, filtering |
| Risk flags | `clip.risk_flags` | Dashboard display, review queue |
| Rejected candidates | `clip.reject_reasons` | Dashboard display, override queue |

The most important field for downstream stages is `keep_spans` — derived from the trim suggestion, this drives the jump-cut pass that removes filler and dead air. The skill's "Cold open line" + "Trim suggestion" together specify exactly which sub-range of the candidate clip is the actual content.

---

## 6. Code-level integration points

For OpenShorts specifically, here are the exact files and functions to modify:

| File | Function | Change |
|---|---|---|
| `main.py` | `get_viral_clips` (line 2835) | Replace DeepSeek call with skill wrapper |
| `main.py` | New: `format_transcript_for_skill` | Add helper to format transcript |
| `main.py` | New: `call_viral_clip_finder` | Add LLM client wrapper |
| `main.py` | New: `parse_skill_output` | Add Markdown report parser |
| `main.py` | New: `identify_faces_in_video` | Add face ID pass (Stage 2.5) |
| `main.py` | New: `merge_face_id_with_tracker` | Add merger with identity_tracker |
| `main.py` | New: `merge_with_diarization` | Add merger with diarization |
| `identity_tracker.py` | `track_identities` | Output named IDs instead of anonymous when face ID available |
| `reframe_v2.py` | `_analyze_trajectory` (line 1806) | Consume `clip.reaction_cues` from skill output |
| `app.py` | `edit_clip` (line 2565) | Pass skill's B-roll cues to Gemini for edit plan |
| `subtitles.py` | `generate_subtitles` (line 416) | Use skill's caption text as the hook overlay |
| `hooks.py` | `render_hook` (line 160) | Render skill's caption text as the visual hook |

The changes are localized — mostly additions, not modifications. The existing `keep_spans` jump-cut logic in Stage 4 doesn't need to change because the skill's trim-suggestion parsing produces the same `keep_spans` format.

---

## 7. Performance and cost

### Latency impact

Adding face ID + skill-based judgment to the existing pipeline:

| Stage | Existing (OpenShorts) | With skill + face ID | Delta |
|---|---|---|---|
| 1. Ingest | ~30s for 8-min video | ~30s | 0 |
| 2. Transcribe | ~50s (GPU) | ~50s | 0 |
| 2.5. Face ID | — | ~30s (dual GPU, sampled frames) | +30s |
| 3. Viral moments | ~5s (DeepSeek) | ~60-180s (skill via LLM) | +55-175s |
| 4. Cut | ~5s per clip | ~5s per clip | 0 |
| 5. Reframe | ~30s per clip | ~30s per clip | 0 |
| 6. Subtitle + effects | ~10s per clip | ~10s per clip | 0 |
| **Total for 8-min source, 5 clips** | ~3-4 min | ~5-8 min | +2-4 min |

The skill adds 1–3 minutes per source. For most use cases this is acceptable — the existing pipeline already takes 3–4 minutes, and the quality improvement justifies the latency.

### Cost impact

| Stage | Existing cost | With skill + face ID |
|---|---|---|
| Transcription | Whisper: $0 (self-host) | Same |
| Face ID | — | $0 (self-hosted, GPU) |
| Viral moment detection | DeepSeek: ~$0.001 per call | LLM (Claude/GPT/Gemini): ~$0.05–0.15 per call |
| Gemini vision confirm | ~$0.005 per clip | Same (keep this pass) |

The skill costs ~$0.05–0.15 per source video in LLM calls (depending on the model and transcript length). For a podcast network processing 100 episodes/day, that's $5–$15/day in additional LLM cost. The face ID stage is free if self-hosted on your dual GPU.

### Quality impact

Based on the benchmark (see `viral-clip-finder-benchmark.html`):

- **Discard rate reduction**: typical AI clip-finders have ~40% discard rate (users throw away 40% of suggested clips). The skill's structured methodology + anti-pattern detection should reduce this to 10–15%. That means less wasted editor time.
- **Hit rate improvement**: the skill's per-clip scores actually correlate with virality (because they're anchored on the 8-axis rubric, not "LLM vibes"). OpusClip's scores have been independently reported as uncorrelated with actual performance.
- **Cut brief leverage**: the cold-open line, trim suggestion, and reaction cam cues save 5–10 minutes of editor time per clip. For 5 clips per source, that's 25–50 minutes saved per episode.

---

## 8. Honest assessment: do you actually need face ID?

The PRIMARY reason to add face ID is **speaker-attribution correctness** — ensuring the right face is shown when someone is talking. This is a production-quality issue, not a celebrity-identification issue. Let me break down the value proposition honestly.

### The problem face ID solves (the real one)

Your existing pipeline has three signals about who's speaking at any moment:

1. **Diarization** (from AssemblyAI or pyannote) — voice fingerprinting, tells you "Speaker A is talking"
2. **LR-ASD lip-sync** — mouth-movement-to-audio correlation, tells you "Face B is producing audio"
3. **identity_tracker** (ByteTrack/BoT-SORT) — visual face tracking, tells you "Face B has been on screen for the last 30 seconds"

These signals often **disagree**, especially on:
- Code-switched speech (speaker switches language mid-sentence)
- Heavy accents (ASR misattributes)
- Overlapping audio (two people talking at once)
- Quick back-and-forth (speaker changes every 1–2 seconds)
- Same-guest-multiple-seats (two guests look similar, tracker conflates them)

When they disagree, the camera gets confused. The diarization says "Speaker A is talking" so the camera frames Face A, but the lip-sync says "Face B is producing audio" — the viewer sees Face A's mouth not moving while Face B's mouth moves in the background. **This is the single most common production-quality failure in AI reframing**, and it's the problem face ID solves.

**Face ID is the ground truth that breaks the tie.** When diarization and lip-sync disagree, the face ID signal (which tracked face has been most consistently correlated with the active audio) wins. The camera frames that face. The speaker label in the transcript gets corrected to that face's name.

### Where your collaborator is right

**Face ID is NOT needed for the baseline judgment layer.** The viral-clip-finder skill works on transcripts. You can find viral clips without knowing who anyone is. The 15 named frameworks (Open Loop, Pattern Interrupt, Pop-the-Balloon, etc.) are content patterns, not identity patterns. You can detect "this is a pattern interrupt" without knowing who's interrupting.

**Face ID adds computational cost.** Even with dual GPUs, you're adding 30 seconds to 5 minutes per video (depending on length and sampling rate). For a high-volume pipeline, this matters.

**Face ID adds a maintenance burden.** You need to maintain the known-faces database. New guests need to be added. Misidentifications need to be flagged and corrected. This is real ongoing work.

### Where your collaborator is wrong

**Face ID is not "not needed" — it's the ground-truth layer for speaker attribution.** Without it, your pipeline has no way to break ties when diarization and lip-sync disagree. The result is visible camera mistakes: wrong face framed, mouth not matching audio, speaker label wrong in the transcript. These mistakes are the difference between a clip that looks professional and a clip that looks AI-generated.

**The "camera decisions don't need names" argument misses the point.** Camera decisions don't need names — but they DO need to know which face is producing audio. Face ID is how you confirm that. The identity_tracker follows faces; face ID confirms which followed face is the speaker.

**Face ID unlocks speaker-attributed cut briefs.** Without face ID, the skill's cut brief can say "frame the host during the cold open." With face ID, it can say "frame Joe Rogan during the cold open, then cut to Elon's reaction at 00:14:38." Specific names → specific camera directions → fewer editor mistakes.

**Face ID is the redundant signal that saves you when diarization fails.** Your existing LR-ASD lip-sync is great, but it fails on code-switched speech, heavy accents, and overlapping audio. Face ID is a completely independent signal — when diarization and face ID agree, you're confident; when they disagree, you know to flag the segment for manual review. This is more robust than either signal alone.

**Status Play detection (secondary benefit).** When face ID identifies a recognized public figure, the skill can flag clips where that figure's status is elevated or taken down. This boosts share-trigger scores for drama/controversy and business niches. But this is a bonus, not the reason to add face ID.

### My recommendation

**Add face ID, as an ENRICHMENT layer on top of the existing identity_tracker.** The pipeline should be:

1. Existing identity_tracker produces anonymous track IDs (Person 1, Person 2) — **keep this, it's essential for reframing**
2. Face ID identifies those tracks against a known-faces DB — **adds names where confident**
3. Face ID + lip-sync vote on who's actively speaking — **breaks ties when diarization and lip-sync disagree**
4. Named identities + speaking ranges flow into the skill's input — **enables speaker-attributed cut briefs, reaction cam direction**
5. The skill's output flows back into the existing pipeline — **cut briefs can reference named individuals**

This gives you the best of both worlds. The baseline pipeline works without face ID (graceful degradation). When face ID is available, the skill produces richer output AND the camera decisions are more reliable.

### When NOT to add face ID

- If you're processing one-off viral clips from random sources where the people aren't recurring — face ID adds little value, just cost
- If you're in a jurisdiction with strict biometric privacy laws (Illinois BIPA, EU GDPR biometric provisions) — the legal cost may outweigh the benefit
- If your known-faces DB would be empty (you have no recurring guests) — face ID can't identify anyone without a DB. But you can still use it for anonymous-but-consistent identities (Face A, Face B) which solves the speaker-attribution problem without needing names
- If your content is mostly animation, voice-only podcasts, or faceless creators — there's nothing to identify

For your case — long-form podcasts with recurring guests, dual T4 GPU, existing face infrastructure — **add it.** The speaker-attribution correctness alone justifies the cost.

### A note on the "don't add face ID" position

If your collaborator's argument is "the existing identity_tracker is enough because the camera decisions don't need names" — they're right that camera decisions don't need names, but wrong that identity_tracker alone is sufficient. identity_tracker follows faces but doesn't confirm which face is producing audio. Face ID + lip-sync is the confirmation layer.

If your collaborator's argument is "face ID is unreliable and produces false positives that hurt quality" — this was true 5 years ago but isn't true now. ArcFace (in InsightFace) achieves 99.7%+ accuracy on LFW. With a confidence threshold of 0.4–0.6 and the voting scheme in section 3d, false positives are rare. The bigger risk is false negatives (failing to identify someone) — which just degrades gracefully to the existing anonymous-ID behavior.

If your collaborator's argument is "it's not worth the engineering time" — that's a legitimate prioritization call. Face ID is a multi-day integration project. If you have higher-priority work, defer it. But don't defer it because it's "not needed" — defer it because you have better things to do first.

---

## 9. Face ID library recommendations for T4 dual GPU

You're running dual NVIDIA T4 GPUs (16GB VRAM each, Turing architecture). This is a capable but older setup — T4s are not as fast as A10/A100 but they're solid for inference workloads. Here's what works best on T4.

### Recommended: InsightFace with buffalo_l model

**This is the best choice for your hardware.** Here's why:

| Criterion | InsightFace (buffalo_l) | DeepFace (Facenet512) | dlib (CNN) |
|---|---|---|---|
| T4 GPU support | ✅ Native ONNX Runtime GPU | ✅ TF/Keras GPU | ⚠️ CPU-only or partial GPU |
| Inference speed (T4) | ~15ms per face | ~80ms per face | ~300ms per face |
| VRAM usage | ~600MB | ~1.2GB | ~200MB (CPU) |
| Accuracy (LFW) | 99.77% | 99.20% | 99.38% |
| Detection + recognition in one package | ✅ | ❌ (needs separate detector) | ❌ |
| Tracker included | ✅ (sort.py) | ❌ | ❌ |
| Install complexity | `pip install insightface onnxruntime-gpu` | `pip install deepface tensorflow-gpu` | Compile dlib with CUDA |
| Integration with your YOLOv8 stack | Excellent (same ONNX ecosystem) | Awkward (TF/Keras vs PyTorch) | Poor |

**Installation:**

```bash
pip install insightface onnxruntime-gpu
# The buffalo_l model downloads automatically on first use
```

**Why buffalo_l (not buffalo_s or buffalo_m):**
- buffalo_l is the largest model, best accuracy (99.77% LFW)
- On T4, inference is still fast enough (~15ms/face) — you won't bottleneck
- For dual GPU, dedicate GPU 1 to InsightFace while GPU 0 runs the rest of the pipeline

**Why ONNX Runtime GPU (not TensorRT):**
- ONNX Runtime is easier to install and works out of the box on T4
- TensorRT is faster but requires building engines, which is a maintenance burden
- The speed difference on T4 is ~30%, not worth the complexity for most use cases

### Alternative: DeepFace with Facenet512 model

Use this if you need maximum model variety (DeepFace supports VGG-Face, Facenet, Facenet512, OpenFace, ArcFace, SFace, DeepID). For most use cases, InsightFace is better.

**Installation:**

```bash
pip install deepface tensorflow-gpu
```

**Why this is worse than InsightFace on T4:**
- Slower inference (~80ms/face vs 15ms)
- Requires TensorFlow GPU setup, which can conflict with your PyTorch stack
- Doesn't include a face detector — you'd use your existing MediaPipe BlazeFace for detection, then DeepFace for recognition, which is more code to maintain
- Lower accuracy than InsightFace's ArcFace

### Not recommended: dlib with CNN face detector

dlib is great for small projects but doesn't scale to production pipelines on T4:
- CNN face detector is CPU-only in default builds (you'd need to compile dlib with CUDA support, which is fragile)
- Even with CUDA, inference is ~300ms/face — 20× slower than InsightFace
- No built-in tracker, no built-in alignment
- The accuracy isn't better than InsightFace to justify the cost

### Not recommended: cloud APIs (AWS Rekognition, Azure Face, Google Vision)

These work but:
- Send face data to a third party (defeats the self-hosted privacy advantage of OpenShorts)
- Cost per inference ($0.001 per face) adds up fast for 2-hour podcasts sampled at 2fps
- Latency (200–500ms per call) is worse than local inference
- Don't integrate with your existing identity_tracker

### T4 dual GPU setup recipe

```python
import insightface
import numpy as np

# Detect available GPUs
import onnxruntime as ort
available_providers = ort.get_available_providers()
# Should include 'CUDAExecutionProvider'

# Initialize InsightFace on GPU 1 (reserve GPU 0 for the rest of the pipeline)
app = insightface.app.FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=1, det_size=(640, 640))  # ctx_id=1 = GPU 1

# For very long videos, you can split the workload:
# - GPU 0: faster-whisper transcription + YOLOv8 + MediaPipe + LR-ASD
# - GPU 1: InsightFace face ID
# Both run in parallel; face ID adds ~30s to a 2-hour video

# For high-volume processing, you can run TWO instances:
# - GPU 0: face ID for video A
# - GPU 1: face ID for video B
# This doubles throughput but requires careful queue management
```

### Performance expectations on T4

For a 2-hour podcast sampled at 2fps (14,400 frames):

| Step | Time on T4 | Notes |
|---|---|---|
| Frame sampling (read 14,400 frames) | ~30s | I/O bound, not GPU |
| Face detection (InsightFace) | ~3.5 min | ~15ms × 14,400 frames |
| Face recognition (embedding + comparison) | ~2 min | Faster than detection, fewer faces |
| Total face ID stage | ~6 min | Acceptable for 2-hour source |

This adds ~6 minutes to your existing ~3-4 minute pipeline, so total processing time goes from ~4 min to ~10 min per 2-hour source. For a podcast network processing 5 episodes/day, that's an extra 30 minutes of compute per day — easily absorbed by a dual T4 setup.

### Known T4 issues and fixes

**Issue: ONNX Runtime falls back to CPU**
- Symptom: Inference is 10× slower than expected
- Fix: Verify `CUDAExecutionProvider` is in `ort.get_available_providers()`. If not, install `onnxruntime-gpu` (not `onnxruntime`). The CPU-only package shadows the GPU one if both are installed.

**Issue: VRAM OOM with concurrent pipelines**
- Symptom: CUDA OOM error when running InsightFace alongside the rest of the pipeline
- Fix: T4 has 16GB VRAM. InsightFace uses ~600MB, faster-whisper uses ~2GB, YOLOv8 uses ~1.5GB, LR-ASD uses ~500MB. Total ~4.6GB — well within 16GB. But if you're running multiple videos in parallel, you'll OOM. Limit concurrent jobs per GPU.

**Issue: First inference is slow (model loading)**
- Symptom: First face ID call takes 30s, subsequent calls take 15ms
- Fix: This is normal — the model loads into VRAM on first use. Initialize the FaceAnalysis object once at pipeline startup, not per-video.

**Issue: T4 doesn't support FP16 natively**
- Symptom: Trying to use FP16 inference gives an error or no speedup
- Fix: T4 (Turing) supports FP16 but it's not faster than FP32 for most models. Stick with FP32. If you upgrade to A10/A100 (Ampere), FP16 gives a 2× speedup.

### Monitoring and observability

Add these metrics to your pipeline dashboard:

- `face_id_inference_ms` — per-face recognition time
- `face_id_match_rate` — % of detected faces that matched the known-faces DB
- `face_id_confidence_avg` — average confidence of matches
- `face_id_confidence_low_count` — number of matches below 0.5 confidence (these need review)
- `speaker_attribution_disagreements` — number of times diarization and lip-sync disagreed, and face ID broke the tie

The last metric is the most important — it tells you how often face ID is actually doing useful work. If it's 0, your diarization and lip-sync are always agreeing and face ID is just confirmation. If it's >5% of speaker turns, face ID is actively correcting mistakes.

### Fallback behavior

If face ID fails (model not loaded, OOM, etc.), the pipeline should gracefully degrade:
- Skip the face ID stage
- Pass anonymous speaker IDs to the skill
- Log the failure for debugging
- Continue with the rest of the pipeline

The skill handles anonymous input fine — it just loses the speaker-attribution correctness layer. The pipeline should never fail entirely because face ID failed.
