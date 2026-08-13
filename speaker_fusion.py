"""Phase 3 of the framing-engine rebuild: global speaker<->track fusion.

WHY THIS EXISTS
---------------
The v2 engine's binding (`reframe_v2._apply_asd_speaker_boost`, now removed)
matched LR-ASD's speaker box to a candidate box EVERY FRAME, with a
decisiveness gate (`ASD_MATCH_MARGIN`) that discarded the match outright when
two candidates looked similarly close. Discarded evidence fell through to
"hold whoever the camera already had" — and because the match was re-attempted
independently every single frame, a momentary ambiguity (a reacting listener
leaning in, two faces briefly at similar distance) could flip the binding
mid-turn even though the SAME diarized speaker was still talking. That
per-frame flip was the actual mechanism behind "long stares at the wrong
person" and the "audio matches but the frame doesn't" symptom (see the
rebuild plan, §1.1).

This module replaces per-frame matching with ONE decision per clip: for each
diarized speaker label, look at ALL the evidence across the whole clip (every
second where ASD pointed at a track while that label was talking), and bind
the label to whichever track the evidence most consistently supports. Once
bound, a speaker's track cannot flip mid-turn, because there is no more
per-frame re-decision left to make — the shot planner (Phase 4) just looks
the binding up.

A speaker that never accumulates confident evidence is left UNBOUND rather
than guessed at (see resolve_speaker_bindings) — Phase 4 owns the fallback
tier for that case (hold / size / whatever the shot-planning rules decide),
the same "no strong evidence means don't invent a decision" philosophy the
old subject-policy tiers enforced.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# ASD box -> face_spine track (position lookup against KNOWN tracks, not a
# live disambiguation between two unrelated id spaces)
# ---------------------------------------------------------------------------

def _box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _nearest_box_at(track: dict, timestamp: float, time_tolerance: float = 0.5):
    """The track's box from the detection nearest `timestamp`, or None if
    the nearest detection is further than `time_tolerance` away (the track
    was not on screen at this moment).
    """
    frames = track.get("frames") or []
    boxes = track.get("boxes") or []
    if not frames:
        return None
    best_i, best_dt = None, None
    for i, t in enumerate(frames):
        dt = abs(t - timestamp)
        if best_dt is None or dt < best_dt:
            best_i, best_dt = i, dt
    if best_i is None or best_dt > time_tolerance:
        return None
    return boxes[best_i]


def match_box_to_track(spine_tracks: Dict[int, dict], timestamp: float, box,
                       iou_threshold: float = 0.3,
                       time_tolerance: float = 0.5) -> Optional[int]:
    """Which face_spine track a predicted (timestamp, box) belongs to.

    This is a LOOKUP against known tracks (Phase 1's already-resolved
    identities), not the live disambiguation between two unrelated id spaces
    that caused the original binding bug — the tracks here are the ground
    truth for "who is on screen," so there is nothing ambiguous left to
    reconcile, only "which known box is this."

    Matching is two-tier, because plain best-IoU picks the wrong person in
    exactly the scene that matters most — a crowd. An ASD box is a SPEAKER
    box: it can be large enough to swallow two or three neighbouring faces,
    and then IoU rewards whichever face happens to be biggest or most
    centred rather than the one the box was drawn around. The head sits at
    the TOP of a speaker box, so:

      1. HEAD REGION (preferred): among tracks whose face centre falls
         inside the ASD box, take the one whose face top-centre is nearest
         the box's top-centre. One box covering a talker and the listener
         beside/below them resolves to the talker.
      2. IOU (fallback): when no face centre is inside the box at all, fall
         back to the original best-IoU-at-or-above-`iou_threshold` match.

    Returns None when neither tier finds anything (the predicted box does
    not correspond to any known person — e.g. a background false positive).
    """
    bx, by, bw, bh = box
    box_top_cx = bx + bw / 2.0
    best_head, best_head_dist = None, None
    best_iou_id, best_iou = None, iou_threshold
    for track_id, track in spine_tracks.items():
        candidate = _nearest_box_at(track, timestamp, time_tolerance)
        if candidate is None:
            continue
        fx, fy, fw, fh = candidate
        fcx, fcy = fx + fw / 2.0, fy + fh / 2.0
        if bx <= fcx <= bx + bw and by <= fcy <= by + bh:
            dist = math.hypot(fx + fw / 2.0 - box_top_cx, fy - by)
            if best_head_dist is None or dist < best_head_dist:
                best_head, best_head_dist = track_id, dist
        iou = _box_iou(box, candidate)
        if iou >= best_iou:
            best_iou_id, best_iou = track_id, iou
    return best_head if best_head is not None else best_iou_id


def asd_predicted_track_per_second(asd_per_second_boxes: List[Optional[tuple]],
                                   spine_tracks: Dict[int, dict],
                                   iou_threshold: float = 0.3) -> List[Optional[int]]:
    """An ASD model's raw per-second output -> per-second track id, via the
    face spine. `asd_per_second_boxes[i]` is the model's predicted
    active-speaker box at second i, or None where the model made no call.
    """
    result = []
    for t, box in enumerate(asd_per_second_boxes):
        if box is None:
            result.append(None)
            continue
        result.append(match_box_to_track(spine_tracks, float(t), box, iou_threshold))
    return result


def smooth_track_sequence(seq: List[Optional[int]],
                          window: int = 3) -> List[Optional[int]]:
    """Majority-filter a per-second predicted-track sequence.

    A single bad ASD second (box flicker between two faces in a crowd —
    the "framed the reacting listener instead of the talker" failure) must
    not create a fake speaker change: each second's value becomes the
    majority of its `window`-second neighborhood, with ties resolved toward
    the current value (temporal persistence). None seconds cast no vote; a
    neighborhood with no votes at all stays None. A real change survives
    because a sustained run of the new track outvotes the old one after
    ~window/2 seconds — sub-3s alternations collapse, which is exactly the
    flicker the split planner should not treat as a genuine exchange.
    """
    radius = max(1, window // 2)
    out = []
    for i in range(len(seq)):
        lo = max(0, i - radius)
        hi = min(len(seq), i + radius + 1)
        votes = [v for v in seq[lo:hi] if v is not None]
        if not votes:
            out.append(None)
            continue
        counts = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        top = max(counts.values())
        leaders = sorted(v for v, c in counts.items() if c == top)
        if len(leaders) == 1:
            out.append(leaders[0])
        elif seq[i] is not None and seq[i] in leaders:
            out.append(seq[i])
        else:
            out.append(leaders[0])
    return out


# ---------------------------------------------------------------------------
# Diarized transcript -> per-second speaker label
# ---------------------------------------------------------------------------

def per_second_speaker_label(segments, clip_start: float, clip_end: float,
                             speaker_names: Optional[Dict[str, str]] = None
                             ) -> List[Optional[str]]:
    """Per-second diarized speaker label for one clip span.

    segments: transcript_result["segments"], each with 'start', 'end',
        'speaker' (assemblyai backend; see transcribe_backends.py) — reads
        the SAME transcript the rest of the pipeline uses, not a parallel one.
    speaker_names: optional {"A": "host", ...} to resolve diarization letters
        to names; unresolved labels pass through as their raw letter.

    Returns a list, index = second offset from clip_start, value = the
    speaker label active that second, or None where no segment covers it
    (silence, cross-talk gap) — those seconds contribute no fusion evidence.
    """
    speaker_names = speaker_names or {}
    duration = int(clip_end - clip_start)
    result = [None] * max(0, duration)
    for seg in segments or []:
        s, e = float(seg.get("start", 0)), float(seg.get("end", 0))
        speaker = seg.get("speaker")
        if speaker is None or e <= clip_start or s >= clip_end:
            continue
        label = speaker_names.get(speaker, speaker)
        lo = max(0, int(s - clip_start))
        hi = min(duration, math.ceil(e - clip_start))
        for i in range(lo, hi):
            result[i] = label
    return result


# ---------------------------------------------------------------------------
# The fusion itself — one binding table per clip
# ---------------------------------------------------------------------------

DEFAULT_DECISIVE_MARGIN = 0.10
DEFAULT_REBIND_SECONDS = 4
DEFAULT_REBIND_WINDOW = 8


def decisive_seconds(asd_per_second_margin: Optional[List[Optional[float]]],
                     decisive_margin: float = DEFAULT_DECISIVE_MARGIN
                     ) -> Optional[List[bool]]:
    """Which seconds of LR-ASD output are strong enough to vote.

    `asd_per_second_margin[i]` is `asd_worker.score_clip`'s per-second lead
    of the winning face over the next best on-screen face (None where no
    face scored positive at all). A second where two faces were within
    `decisive_margin` of each other is the model saying "it could be either
    of them" — the reacting-listener ambiguity. Those seconds used to vote
    anyway, and enough of them could bind a speaker to the wrong face for a
    whole clip. Here they are simply dropped from the evidence.

    Returns None when no margin data was supplied (LR-ASD unavailable, or an
    older caller), which every consumer reads as "no gate, count every
    second" — the pre-existing behaviour.
    """
    if asd_per_second_margin is None:
        return None
    return [m is not None and m >= decisive_margin
            for m in asd_per_second_margin]


def resolve_speaker_bindings(per_second_speaker: List[Optional[str]],
                             predicted_track_ps: List[Optional[int]],
                             min_agreement: float = 0.6,
                             min_seconds: float = 1.0,
                             decisive_ps: Optional[List[bool]] = None
                             ) -> Dict[str, int]:
    """The core Phase 3 fusion: ONE speaker_label -> track_id binding per
    label, decided from ALL the evidence across the whole clip at once.

    For every second where BOTH a diarized speaker is talking AND ASD
    pointed at a resolvable track — and, when `decisive_ps` is supplied,
    where ASD was DECISIVE about it (see `decisive_seconds`) — that track
    gets one vote toward that speaker's binding. A label binds to its
    top-voted track only when:
      - that track has at least `min_agreement` of the label's total votes
        (a speaker whose evidence is split roughly evenly between two
        tracks — e.g. ASD flickering between the speaker and a reacting
        listener — should NOT bind confidently to either), and
      - the label accumulated at least `min_seconds` of evidence at all
        (a label that barely appears has too little signal to trust).

    Labels that do not clear both bars are left OUT of the returned dict
    entirely — never bound to a guess. The caller (Phase 4's shot planner)
    is responsible for the fallback behavior when a speaker has no binding.

    This function's whole reason to exist: once computed, a binding cannot
    flip mid-turn, because nothing re-evaluates it per frame anymore.
    """
    n = min(len(per_second_speaker), len(predicted_track_ps))
    votes: Dict[str, Dict[int, int]] = {}
    for i in range(n):
        label = per_second_speaker[i]
        track = predicted_track_ps[i]
        if label is None or track is None:
            continue
        if decisive_ps is not None and not (
                i < len(decisive_ps) and decisive_ps[i]):
            continue
        votes.setdefault(label, {})
        votes[label][track] = votes[label].get(track, 0) + 1

    bindings: Dict[str, int] = {}
    for label, track_votes in votes.items():
        total = sum(track_votes.values())
        if total < min_seconds:
            continue
        best_track = max(track_votes, key=track_votes.get)
        agreement = track_votes[best_track] / total
        if agreement >= min_agreement:
            bindings[label] = best_track
    return bindings


def per_second_active_track(per_second_speaker: List[Optional[str]],
                            bindings: Dict[str, int],
                            predicted_track_ps: Optional[List[Optional[int]]] = None
                            ) -> List[Optional[int]]:
    """Expand the binding table across the clip: for each second, whichever
    track is bound to the speaker active that second.

    When a second's diarized speaker never got a confident binding (the
    "framed the host instead of the talker" failure — diarization can lag or
    miss, but ASD watched the faces), fall back to the LR-ASD predicted track
    for that second if one exists: the model that directly marks who is
    speaking beats a silent guess. None only where BOTH signals are absent.
    """
    out = []
    for i, label in enumerate(per_second_speaker):
        if label is not None and label in bindings:
            out.append(bindings[label])
        elif predicted_track_ps is not None and i < len(predicted_track_ps):
            out.append(predicted_track_ps[i])
        else:
            out.append(None)
    return out


def active_from_identity_map(per_second_speaker: List[Optional[str]],
                             identity_map: Dict[str, int],
                             ) -> List[Optional[int]]:
    """ASR-first active track: the diarized transcript decides WHO, and the
    confirmed identity map decides WHICH FACE.

    This is the director-v2 path: once a speaker label has been mapped to a
    face track (Gemini confirmation, see identity_confirm.py), each second's
    active track is simply that label's confirmed track — no LR-ASD
    per-second voting, no host-default. Seconds whose label is unconfirmed
    (absent from the map) are None: the caller must NOT guess — wide/hold.
    """
    out = []
    for i, label in enumerate(per_second_speaker):
        if label is not None and label in identity_map:
            out.append(identity_map[label])
        else:
            out.append(None)
    return out


def apply_gated_rebinding(per_second_speaker: List[Optional[str]],
                          predicted_track_ps: List[Optional[int]],
                          bindings: Dict[str, int],
                          active: List[Optional[int]],
                          decisive_ps: Optional[List[bool]] = None,
                          rebind_seconds: int = DEFAULT_REBIND_SECONDS,
                          rebind_window: int = DEFAULT_REBIND_WINDOW,
                          ) -> tuple:
    """Let a WRONG binding be corrected mid-clip — but only under duress.

    One binding per clip killed the per-frame flip-flop, and that property is
    worth keeping. Its cost was the opposite failure: when the single decision
    came out wrong, the camera stayed on the wrong person for the ENTIRE clip
    ("held on the host while she wasn't talking"), with no mechanism left to
    notice. This adds exactly one such mechanism, gated hard enough that
    ordinary noise can never reach it:

      - only DECISIVE seconds (see `decisive_seconds`) are admitted as
        evidence at all;
      - a second where the bound label is talking and decisive ASD points at
        a DIFFERENT track is a contradiction, recorded against that
        candidate track;
      - contradictions older than `rebind_window` seconds fall out of the
        window, and a second that AGREES with the current binding decays the
        oldest contradiction against every candidate;
      - only when one candidate holds `rebind_seconds` live contradictions
        does the label re-bind to it, from that second to the end of the clip.

    So a one-second blip cannot move anything (it needs sustained
    disagreement), and after a re-bind the counters reset — flipping back
    demands a fresh, equally sustained case. Seconds whose label is unbound
    keep whatever `active` already resolved them to (the per-second ASD
    fallback); this only ever revises BOUND labels.

    Returns `(bindings_at_clip_end, corrected_active)`; neither input is
    mutated. `rebind_seconds <= 0` disables the whole mechanism.
    """
    corrected = list(active)
    if rebind_seconds <= 0:
        return dict(bindings), corrected

    current = dict(bindings)
    # {label: {candidate_track: [second, ...]}} — live contradictions only.
    pending: Dict[str, Dict[int, List[int]]] = {}
    n = min(len(per_second_speaker), len(predicted_track_ps))
    for i in range(n):
        label = per_second_speaker[i]
        if label is None or label not in current:
            continue
        track = predicted_track_ps[i]
        decisive = (decisive_ps is None
                    or (i < len(decisive_ps) and decisive_ps[i]))
        if track is not None and decisive:
            counters = pending.setdefault(label, {})
            for candidate in list(counters):
                # Recency decay: drop contradictions that have aged out of
                # the window, so a slow drip of disagreement spread over a
                # minute never accumulates into a re-bind.
                counters[candidate] = [s for s in counters[candidate]
                                       if i - s < rebind_window]
            if track == current[label]:
                for candidate in list(counters):
                    # Agreement decays the case against the binding rather
                    # than wiping it: an alternating signal is ambiguity, and
                    # ambiguity should stall the re-bind, not reset it.
                    if counters[candidate]:
                        counters[candidate].pop(0)
            else:
                counters.setdefault(track, []).append(i)
                if len(counters[track]) >= rebind_seconds:
                    current[label] = track
                    pending[label] = {}
        if i < len(corrected):
            corrected[i] = current[label]
    return current, corrected


def fuse_speaker_tracks(asd_per_second_boxes: List[Optional[tuple]],
                        spine_tracks: Dict[int, dict],
                        segments, clip_start: float, clip_end: float,
                        speaker_names: Optional[Dict[str, str]] = None,
                        iou_threshold: float = 0.3,
                        min_agreement: float = 0.6,
                        min_seconds: float = 1.0,
                        smooth_window: int = 3,
                        asd_per_second_margin: Optional[List[Optional[float]]] = None,
                        decisive_margin: float = DEFAULT_DECISIVE_MARGIN,
                        rebind_seconds: int = DEFAULT_REBIND_SECONDS,
                        rebind_window: int = DEFAULT_REBIND_WINDOW,
                        ) -> tuple:
    """Full Phase 3 pipeline for one clip: ASD boxes + the Phase 1 face
    spine + the diarized transcript -> (bindings, per_second_active_track).

    This is the plan's Pass 2 deliverable: "who is speaking, and which
    track is them" — the input Phase 4's shot planner reads, replacing the
    v2 engine's per-frame speaker-match with one decision per clip.

    `asd_per_second_margin` is `asd_worker.score_clip`'s per-second
    decisiveness lead. Supplying it gates both the binding vote and the
    re-binding evidence to seconds where LR-ASD was actually sure; omitting
    it (no ASD, older caller) leaves every second counting, as before.
    """
    predicted_track_ps = asd_predicted_track_per_second(
        asd_per_second_boxes, spine_tracks, iou_threshold)
    if smooth_window and smooth_window > 1:
        # Stabilize the noisy per-second ASD signal BEFORE it votes for
        # speaker bindings or falls through as the per-second fallback: a
        # crowd flicker must neither bind a speaker to the wrong track nor
        # send the planner hopping between faces.
        predicted_track_ps = smooth_track_sequence(
            predicted_track_ps, window=smooth_window)
    speaker_ps = per_second_speaker_label(
        segments, clip_start, clip_end, speaker_names)
    decisive_ps = decisive_seconds(asd_per_second_margin, decisive_margin)
    bindings = resolve_speaker_bindings(
        speaker_ps, predicted_track_ps, min_agreement, min_seconds,
        decisive_ps=decisive_ps)
    active = per_second_active_track(speaker_ps, bindings, predicted_track_ps)
    # A binding that came out wrong is corrected mid-clip under sustained,
    # decisive contradiction — the only escape hatch from "wrong person for
    # the whole clip". Returns the binding table as of the clip's end.
    bindings, active = apply_gated_rebinding(
        speaker_ps, predicted_track_ps, bindings, active,
        decisive_ps=decisive_ps, rebind_seconds=rebind_seconds,
        rebind_window=rebind_window)
    return bindings, active
