"""Who is on screen — decided by EVIDENCE, not by who is biggest.

WHY THIS EXISTS
---------------
Target selection used to be a single accumulating score. Every candidate's
score started as its box AREA; each signal (lip-sync, diarization, mouth
motion, reaction motion, director directive) multiplied that area; the
tracker summed the result per identity with a 0.85 decay and picked the
argmax, with a 1.6-2.2x "sticky" bonus for whoever was already framed.

Three things are wrong with that, all measured on the Blind Dating source
(20s, 9 shots, 121 detection frames) rather than argued:

1. **Area dominates.** A decaying sum reaches ~6.7x its per-sample input in
   steady state, so an incumbent whose face is merely 2x larger sits at
   ~13x while a challenger's single 4x lip-sync boost reaches ~4x. Raising
   the lip-sync boost from 4x to 1000x (250x!) moved framing accuracy only
   61% -> 70%. The boost knobs were never the lever anyone thought.

2. **Hysteresis damps signal and noise equally.** Removing the sticky
   factors, the switch cooldown and the minimum shot hold took accuracy
   70% -> 97%. Those guards were suppressing the best evidence in the
   system just as hard as they suppressed detector jitter.

3. **Fighting a bad signal is what actually caused the jitter.** Simulated
   over the same frames: selecting by size produces 15 camera jumps at
   44.9% accuracy, while following lip-sync directly produces **9** jumps
   -- fewer -- at 100%. The source itself has 9 shots. When the camera is
   on the wrong person the evidence keeps pulling at it and it oscillates;
   when it is on the right person it simply sits there. Hysteresis was
   treating the symptom and feeding the cause.

THE MODEL
---------
Evidence is TIERED, and hysteresis is applied in proportion to how weak the
evidence is -- never uniformly:

    tier 1  lip-sync (LR-ASD)      switch immediately
    tier 2  diarized speaker id    switch immediately
    tier 3  director directive     switch, but respect the shot-hold floor
    tier 4  mouth motion           switch after `confirm` agreeing samples
    tier 5  hold whoever is framed (evidence absent, subject still visible)
    tier 6  biggest / most central (nothing else to go on)

A strong tier can always interrupt a weaker one. A weak tier can never
interrupt a strong one inside the shot-hold floor. Source scene cuts reset
everything -- the director already cut, so re-deciding there is free.

This module is pure: no ffmpeg, no detection, no tracker state. It takes
candidates plus evidence and returns a choice, which makes every rule above
testable without rendering a frame.
"""
import os

# Evidence tiers, strongest first. Exposed as module constants so callers and
# tests can name them instead of hardcoding integers.
TIER_ASD = 1
TIER_DIARIZED = 2
TIER_DIRECTIVE = 3
TIER_MOUTH = 4
TIER_HOLD = 5
TIER_SIZE = 6
TIER_REACTION = 7
TIER_JCUT = 8
TIER_FATIGUE = 9

# Tiers considered STRONG: a positive identification of who is speaking.
# They bypass the WEAK-evidence floor (MIN_SHOT_HOLD_SECONDS — gating them
# is what cost 27 points of accuracy) but still wait out the absolute
# minimum shot length (ABSOLUTE_MIN_SHOT_SECONDS, the owner's strict
# 1.5-2s no-shot-shorter rule, 4-aug-2026).
STRONG_TIERS = (TIER_ASD, TIER_DIARIZED, TIER_JCUT)

# Agreeing consecutive detections required before a WEAK signal may move the
# camera. Strong signals use 1 (measured optimal: 100% accuracy, 9 switches).
MOUTH_CONFIRM_SAMPLES = int(os.environ.get("MOUTH_CONFIRM_SAMPLES", "3"))
SIZE_CONFIRM_SAMPLES = int(os.environ.get("SIZE_CONFIRM_SAMPLES", "4"))

# Minimum time a shot must hold before WEAK evidence (tier 3 and below) may
# cut away from it. Does not apply to tiers 1-2, and never applies across a
# source scene cut.
MIN_SHOT_HOLD_SECONDS = float(os.environ.get("POLICY_MIN_SHOT_HOLD", "0.9"))

# The shortest shot the camera may ever produce, whatever the evidence says.
#
# Distinct from MIN_SHOT_HOLD_SECONDS above, which protects a shot from WEAK
# evidence. This one binds even lip-sync, because a signal can be both
# correct and unusable: in a 7-person lineup the LR-ASD box is roughly
# equidistant from several faces, and the identification hops between them
# every 0.17s. Following that faithfully is accurate and unwatchable —
# reference edits in this genre run 1-4s per shot. A floor here is not
# hysteresis in the old sense: it never changes WHO is chosen, only how soon
# the change may land, so it costs coverage rather than correctness.
ABSOLUTE_MIN_SHOT_SECONDS = float(
    os.environ.get("POLICY_ABSOLUTE_MIN_SHOT", "1.5"))

# Reaction shots ("show the shocked face"). Measured on Pop The Balloon span
# 2 (4 Aug 2026): a non-speaker's mouth_activity runs ~0.07 median but
# reaction spikes (shock/laugh) reach 0.18-0.32, while the framed speaker's
# own mouth drops below ~0.08 during the beat that follows a line. So: when
# the framed person is quiet and another face spikes hard, give that face a
# SHORT bounded shot, then let the speaker reclaim. Thresholds are env-tunable
# and everything degrades to no-reactions when candidates carry no mouth data.
REACTION_MOUTH_ACTIVITY = float(
    os.environ.get("REACTION_MOUTH_ACTIVITY", "0.18"))
REACTION_SPEAKER_QUIET = float(
    os.environ.get("REACTION_SPEAKER_QUIET", "0.08"))
# A reaction may only interrupt a shot that has held this long (anti-flicker).
REACTION_MIN_HOLD_SECONDS = float(
    os.environ.get("REACTION_MIN_HOLD", "0.45"))
# The identified speaker must have been QUIET this long before a reaction may
# fire. Filters the word gaps inside a sentence (measured 4-aug-2026: the
# first version cut to a reaction mid-clause because the speaker's mouth dips
# between words; the harsh-editor review flagged exactly that: 'cuts a split
# second before the man finishes his sentence').
REACTION_MIN_QUIET_SECONDS = float(
    os.environ.get("REACTION_MIN_QUIET", "0.35"))
# A reaction may also fire DURING the speaker's line (the listener is
# reacting to what is being said — owner spec, 4-aug-2026: the woman's face
# must show while the man talks down her lifestyle) once the current shot is
# this old. A fresh shot (< this) still requires the quiet-beat trigger, so a
# punchline delivery is never interrupted mid-word.
REACTION_DURING_SPEECH_AGE_SECONDS = float(
    os.environ.get("REACTION_DURING_SPEECH_AGE", "1e9"))
# Never hold a reaction longer than this — reference edits cap reaction shots
# at ~1.5s and use them as punctuation, not scenes (see
# RESEARCH_pop_the_balloon_shorts.md).
REACTION_MAX_HOLD_SECONDS = float(
    os.environ.get("REACTION_MAX_HOLD", "2.0"))
# Gap required between two reaction shots, so a laughing group doesn't chain
# the camera from face to face.
REACTION_COOLDOWN_SECONDS = float(
    os.environ.get("REACTION_COOLDOWN", "2.2"))
# A reactor must be at least this fraction of the largest face's area —
# a distant background face spiking on a laugh must not hijack the lineup.
REACTION_MIN_RELATIVE_AREA = float(
    os.environ.get("REACTION_MIN_RELATIVE_AREA", "0.25"))

# Fatigue cut: after the camera has been locked on one speaker this long,
# force a short cutaway to another visible person even if they are silent —
# breaks the monotony of a long monologue (owner spec: 7-10s, 4-aug-2026).
FATIGUE_CUT_SECONDS = float(
    os.environ.get("FATIGUE_CUT_SECONDS", "8.0"))

# Cutaways are OFF by default (4-aug-2026). Both were added to add rhythm and
# both were measured, by the owner watching the render, to do the opposite:
# they take the frame away from the person talking, which is the one thing the
# clip exists to show. Turn them on per-deployment only after the speaker
# framing itself is judged good.
JCUT_ENABLED = os.environ.get("JCUT_PREROLL", "0").strip() not in ("0", "false", "no")
REACTION_ENABLED = os.environ.get("REACTION_CUTS", "0").strip() not in ("0", "false", "no")
FATIGUE_ENABLED = os.environ.get("FATIGUE_CUTS", "0").strip() not in ("0", "false", "no")

# Box overlap at which two detections are considered THE SAME PERSON,
# regardless of what id they carry.
#
# This is load-bearing, not a nicety. Identity ids in this pipeline are not
# stable: the tracker hands every unconfirmed detection a fresh negative
# provisional id, and confirmed ids are effectively scene-scoped on
# fast-cut multicam source. Traced on the Blind Dating clip, one motionless
# person produced the id sequence 0, -1, 0, -3, 0, -4, 0 across seven
# consecutive detections. A policy that equates "new id" with "new subject"
# reads that as six cuts; it measured 25 subject switches in 20 seconds
# where the source has 9. Deciding on overlap instead of on the label makes
# the camera immune to relabelling from any source.
SAME_SUBJECT_IOU = float(os.environ.get("SAME_SUBJECT_IOU", "0.4"))

# Overlap alone is not enough, because detection supplies TWO box shapes for
# the same person: a MediaPipe face (e.g. 135x135) and a YOLO body (624x384).
# Traced on the same clip, one person alternating between the two produced
# IoU 0.08 — indistinguishable from a cut to someone else. So a face sitting
# inside a body at the same horizontal position also counts as the same
# subject, provided the two are vertically aligned enough for one to be the
# other's head.
SAME_SUBJECT_CX_FRAC = float(os.environ.get("SAME_SUBJECT_CX_FRAC", "0.05"))

# How far the framing box moves toward a new detection of the SAME subject
# per detection sample. 1.0 reproduces the old snap-to-detection behaviour.
BOX_BLEND = float(os.environ.get("POLICY_BOX_BLEND", "0.35"))


def iou(a, b):
    """Intersection-over-union of two (x, y, w, h) boxes."""
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _contains_centre(outer, inner):
    ox, oy, ow, oh = outer
    cx, cy = inner[0] + inner[2] / 2.0, inner[1] + inner[3] / 2.0
    return ox <= cx <= ox + ow and oy <= cy <= oy + oh


def stabilize_box(new_box, prev_box, blend=None):
    """Ease a subject's framing box toward a new detection instead of
    snapping to it.

    Detection hands back two different shapes for the same person: when
    MediaPipe sees the face it is a ~135px square; when the face is turned,
    masked or motion-blurred, only YOLO fires and the box becomes the
    person's whole head-and-chest (~624x384). The subject has not moved and
    must not be re-cut — but feeding that box straight to the camera moves
    the crop centre and changes the zoom on every alternation, which reads
    as jitter even though nothing was mis-framed.

    Applies only while the subject is unchanged; a real cut passes
    prev_box=None and lands exactly on the new detection.
    """
    if prev_box is None:
        return tuple(new_box)
    k = BOX_BLEND if blend is None else blend
    return tuple(p + (n - p) * k for p, n in zip(prev_box, new_box))


def same_subject(a, b, frame_width=None):
    """Are these two boxes the same person, whatever ids they carry?

    Either they overlap substantially, or one is plainly a detail of the
    other (a face box inside a body box) at the same horizontal position.
    """
    if a is None or b is None:
        return False
    if iou(a, b) >= SAME_SUBJECT_IOU:
        return True
    if not (_contains_centre(a, b) or _contains_centre(b, a)):
        return False
    if not frame_width:
        return True
    acx = a[0] + a[2] / 2.0
    bcx = b[0] + b[2] / 2.0
    return abs(acx - bcx) <= SAME_SUBJECT_CX_FRAC * frame_width


class Evidence:
    """What we know about this detection frame.

    Every field is optional; the policy degrades tier by tier as signals go
    missing, which is what makes it safe to run with LR-ASD off, with no
    diarization, or with no scene-context directives at all.
    """

    __slots__ = ("asd_id", "diarized_id", "directive_id", "directive_reason",
                 "mouth_id", "scene_changed", "jcut_id")

    def __init__(self, asd_id=None, diarized_id=None, directive_id=None,
                 directive_reason=None, mouth_id=None, scene_changed=False,
                 jcut_id=None):
        self.asd_id = asd_id
        self.diarized_id = diarized_id
        self.directive_id = directive_id
        self.directive_reason = directive_reason
        self.mouth_id = mouth_id
        self.scene_changed = scene_changed
        self.jcut_id = jcut_id

    def proposal(self):
        """(candidate_id, tier) for the strongest evidence present, or
        (None, None). Lip-sync outranks diarization because it names a face
        on screen; diarization only names an audio label that still has to
        be bound to one, and that binding chain is where identity goes
        wrong (see reframe_v2._resolve_speaker_binding).

        A directive ranks below both on purpose: it is Gemini's reading of
        the shot, and it was measured pointing at the wrong person often
        enough that the transcript must win when the two disagree.
        """
        # A J-cut names the NEXT speaker up to 0.5s before their audio starts
        # (owner spec, 4-aug-2026: "execute the hard cut to Speaker B 0.5
        # seconds before their audio waveform actually begins"). It outranks
        # lip-sync for the pre-roll window, otherwise the current speaker's
        # mouth would win and the anticipation would never reach the screen.
        # DISABLED BY DEFAULT (4-aug-2026). Verified on the Pop The Balloon
        # sample: because this outranks lip-sync, the j-cut lookahead pulled
        # the camera off Solomon mid-sentence onto the listener eight separate
        # times in 32s ("Camera on woman with red hair while Solomon speaks",
        # harsh-editor review). A pre-roll is a real technique, but only when
        # the outgoing speaker has actually finished; ranking it above live
        # lip-sync makes it a cutaway generator. Owner spec is unambiguous:
        # "just make sure whoever's talking gets framed."
        if JCUT_ENABLED and self.jcut_id is not None:
            return self.jcut_id, TIER_JCUT
        if self.asd_id is not None:
            return self.asd_id, TIER_ASD
        if self.diarized_id is not None:
            return self.diarized_id, TIER_DIARIZED
        if self.directive_id is not None:
            return self.directive_id, TIER_DIRECTIVE
        if self.mouth_id is not None:
            return self.mouth_id, TIER_MOUTH
        return None, None


def _by_id(candidates, cid):
    if cid is None:
        return None
    for c in candidates:
        if c.get("id") == cid:
            return c
    return None


def _biggest(candidates, frame_width=None):
    """Largest face, tie-broken toward frame centre. The weakest possible
    basis for a decision, used only when literally nothing else is known."""
    if not candidates:
        return None

    def rank(c):
        x, y, w, h = c["box"]
        area = w * h
        if not frame_width:
            return area
        # Centre proximity as a mild tie-break (never a primary criterion):
        # scales area by at most ~10%.
        off = abs((x + w / 2.0) - frame_width / 2.0) / float(frame_width)
        return area * (1.0 - 0.2 * min(off, 0.5))

    return max(candidates, key=rank)


class SubjectPolicy:
    """Stateful per-clip decision maker. One instance per render pass.

    Holds only what hysteresis genuinely needs: who is framed, when that
    shot started, and how many consecutive samples a challenger has had.
    """

    def __init__(self, fps, min_shot_hold_seconds=None,
                 mouth_confirm=None, size_confirm=None,
                 absolute_min_shot_seconds=None,
                 reaction_enabled=None, fatigue_enabled=None):
        self.fps = float(fps) or 25.0
        hold = (MIN_SHOT_HOLD_SECONDS if min_shot_hold_seconds is None
                else min_shot_hold_seconds)
        self.min_hold_frames = max(0, int(hold * self.fps))
        floor = (ABSOLUTE_MIN_SHOT_SECONDS if absolute_min_shot_seconds is None
                 else absolute_min_shot_seconds)
        self.absolute_min_frames = max(0, int(floor * self.fps))
        self.mouth_confirm = (MOUTH_CONFIRM_SAMPLES if mouth_confirm is None
                              else mouth_confirm)
        self.size_confirm = (SIZE_CONFIRM_SAMPLES if size_confirm is None
                             else size_confirm)
        self.reaction_min_frames = max(0, int(REACTION_MIN_HOLD_SECONDS * self.fps))
        self.reaction_min_quiet_frames = max(0, int(REACTION_MIN_QUIET_SECONDS * self.fps))
        self.reaction_during_speech_frames = max(
            0, int(REACTION_DURING_SPEECH_AGE_SECONDS * self.fps))
        self.reaction_max_frames = max(1, int(REACTION_MAX_HOLD_SECONDS * self.fps))
        self.reaction_cooldown_frames = max(0, int(REACTION_COOLDOWN_SECONDS * self.fps))
        self.fatigue_frames = max(1, int(FATIGUE_CUT_SECONDS * self.fps))
        self.reaction_enabled = (REACTION_ENABLED if reaction_enabled is None
                                 else reaction_enabled)
        self.fatigue_enabled = (FATIGUE_ENABLED if fatigue_enabled is None
                                else fatigue_enabled)
        self.target_id = None
        self.target_tier = None
        # Where the framed subject was last seen. Identity of a SUBJECT is
        # this box, not self.target_id — see SAME_SUBJECT_IOU.
        self.target_box = None
        self.shot_started = -10 ** 9
        self._pending_id = None
        self._pending_n = 0
        self._reaction_start = None
        self._reaction_cooldown_until = -10 ** 9
        self._fatigue_start = None
        self._fatigue_cooldown_until = -10 ** 9
        # True while the current reaction was triggered DURING the speaker's
        # line (the speaker never stopped mouthing). Such reactions hold their
        # full window — the early-end only applies to quiet-beat reactions.
        self._reaction_during_speech = False
        # The box of the last DIFFERENT subject we framed (the conversational
        # partner). Reactions/cutaways prefer this person over bystanders.
        self._last_other_box = None
        # When the current SPEAKER hold began. Reactions/cutaways do not reset
        # this, so a long monologue still reaches the fatigue threshold even
        # with reaction cuts in between.
        self._speaker_hold_start = -10 ** 9
        # First frame of the current "speaker is quiet" beat (None while the
        # identified speaker is actively mouthing). A reaction may only start
        # once this beat has lasted REACTION_MIN_QUIET_SECONDS.
        self._speaker_quiet_since = None
        # Diagnostics: how many decisions each tier accounted for. Rendered
        # into the render log so a regression is visible without a bisect.
        self.tier_counts = {}

    # -- internals ---------------------------------------------------------

    def _confirm_needed(self, tier):
        if tier in STRONG_TIERS:
            return 1
        if tier == TIER_DIRECTIVE:
            return 1
        if tier == TIER_MOUTH:
            return self.mouth_confirm
        return self.size_confirm

    def _may_interrupt(self, tier, frame_number):
        """Strong evidence always may. Everything else waits out the floor,
        which is what keeps a size flicker or a marginal mouth reading from
        chopping a shot in half."""
        held = frame_number - self.shot_started
        if tier in STRONG_TIERS:
            return held >= self.absolute_min_frames
        return held >= max(self.min_hold_frames, self.absolute_min_frames)

    def _commit(self, cand, tier, frame_number, new_shot=True):
        if new_shot:
            if (self.target_box is not None
                    and not same_subject(tuple(cand["box"]), self.target_box)):
                self._last_other_box = self.target_box
            self.shot_started = frame_number
            if tier in STRONG_TIERS or tier == TIER_DIRECTIVE:
                self._speaker_hold_start = frame_number
        self.target_id = cand.get("id")
        self.target_box = tuple(cand["box"])
        self.target_tier = tier
        self._pending_id = None
        self._pending_n = 0

    def _refresh(self, cand, tier):
        """Update where the framed subject is (and under which id) without
        touching the shot clock or a challenger's confirmation streak.
        Holding a subject must not reset a weak challenger's progress —
        otherwise mouth/size evidence can never reach its threshold.
        """
        self.target_id = cand.get("id")
        self.target_box = tuple(cand["box"])
        self.target_tier = tier

    def _same_subject(self, cand, frame_width=None):
        """Is this candidate the person we are already framing, wearing a
        different id? Geometry answers that; the id cannot."""
        return same_subject(tuple(cand["box"]), self.target_box, frame_width)

    def _find_held(self, candidates, frame_width=None):
        """The framed subject in this frame's candidates: by id when it
        survived, otherwise by overlap. The fallback is what keeps the
        camera on a person whose id was just reassigned instead of dropping
        to the biggest face."""
        held = _by_id(candidates, self.target_id)
        if held is not None:
            return held
        if self.target_box is None:
            return None
        for c in candidates:
            if same_subject(tuple(c["box"]), self.target_box, frame_width):
                return c
        return None

    def _reactors(self, candidates, evidence, frame_width=None):
        """Candidates with a strong expression spike who are NOT the person
        we are already framing and NOT the identified speaker (lip-sync or
        diarized). The speaker mid-sentence is not a reaction — measured on
        Pop The Balloon span 2 (4 Aug 2026): a callout cut to the speaker
        himself because his mouth spike looked like a reaction. Missing mouth
        data yields [] — fail open."""
        max_area = max((c["box"][2] * c["box"][3] for c in candidates),
                       default=0.0)
        speaker_ids = set()
        if evidence is not None:
            for label in (evidence.asd_id, evidence.diarized_id):
                if label is not None:
                    speaker_ids.add(label)
        out = []
        for c in candidates:
            if c.get("mouth_activity", 0.0) < REACTION_MOUTH_ACTIVITY:
                continue
            if c.get("id") in speaker_ids:
                continue  # the identified speaker is not a reactor
            x, y, w, h = c["box"]
            if max_area and w * h < max_area * REACTION_MIN_RELATIVE_AREA:
                continue  # background-small — not a lineup reaction
            if not self._same_subject(c, frame_width):
                out.append(c)
        return out

    def _best_reactor(self, reactors):
        """Prefer the conversational partner (the last OTHER subject we
        framed) over a random bystander who happens to be mouthing — owner
        spec: 'any reaction shots must be of Speaker B, not random
        bystanders in the wide shot'."""
        if self._last_other_box is not None:
            for c in reactors:
                if same_subject(tuple(c["box"]), self._last_other_box):
                    return c
        return max(reactors, key=lambda c: c.get("mouth_activity", 0.0))

    def _maybe_fatigue(self, candidates, frame_number, evidence=None):
        """Forced short cutaway after a long lock-on (owner spec, 4-aug-2026:
        camera on one speaker for 7-10s -> cut to another visible person for
        ~2s even if they are silent, to break visual monotony).

        NEVER fires while the framed person is the identified speaker. Owner
        spec, 4-aug-2026: a monologue is not a defect to be broken up — taking
        the frame off someone mid-sentence to show a silent listener is the
        "why is it showing the host and the other person's reactions" failure.
        Monotony is only worth breaking when nobody is positively speaking.

        Disabled unless explicitly enabled — see FATIGUE_ENABLED."""
        if not self.fatigue_enabled or not candidates:
            return None, None
        if evidence is not None and self.target_tier != TIER_FATIGUE:
            spk, spk_tier = evidence.proposal()
            if spk_tier in STRONG_TIERS and spk == self.target_id:
                return None, None
        if self.target_tier == TIER_FATIGUE:
            if (self._fatigue_start is not None
                    and frame_number - self._fatigue_start < self.reaction_max_frames):
                held = self._find_held(candidates)
                if held is not None:
                    self._refresh(held, TIER_FATIGUE)
                    return held, TIER_FATIGUE
            self._fatigue_cooldown_until = frame_number + self.reaction_cooldown_frames
            return None, None
        if frame_number < self._fatigue_cooldown_until:
            return None, None
        if (self._speaker_hold_start < 0
                or frame_number - self._speaker_hold_start < self.fatigue_frames):
            return None, None
        if self.target_tier not in STRONG_TIERS and self.target_tier != TIER_DIRECTIVE:
            return None, None  # only a real speaker hold gets the fatigue cut
        held = self._find_held(candidates)
        if held is None:
            return None, None
        max_area = max((c["box"][2] * c["box"][3] for c in candidates),
                       default=0.0)
        others = [c for c in candidates
                  if not self._same_subject(c)
                  and c["box"][2] * c["box"][3] >= max_area * REACTION_MIN_RELATIVE_AREA]
        if not others:
            return None, None
        # Active Participant Priority: the fatigue cutaway should land on the
        # conversational partner, not a random bystander (owner spec).
        best = None
        if self._last_other_box is not None:
            best = next((c for c in others
                         if same_subject(tuple(c["box"]), self._last_other_box)),
                        None)
        if best is None:
            best = max(others, key=lambda c: c["box"][2] * c["box"][3])
        self._fatigue_start = frame_number
        self._commit(best, TIER_FATIGUE, frame_number)
        return best, TIER_FATIGUE

    def _maybe_reaction(self, candidates, evidence, frame_number):
        """The "show the shocked face" edit. Returns (cand, tier) when the
        camera should cut to a reactor this frame, else (None, None).

        Trigger (measured, not assumed): the framed person is quiet (mouth
        below REACTION_SPEAKER_QUIET — a beat after a line) while another
        face spikes at or above REACTION_MOUTH_ACTIVITY. The reactor gets a
        bounded shot (REACTION_MIN..MAX_HOLD), then the normal evidence flow
        reclaims — a strong speaker proposal switches back immediately.

        Disabled unless explicitly enabled — see REACTION_ENABLED.
        """
        if not self.reaction_enabled:
            return None, None
        if not candidates:
            return None, None
        held = self._find_held(candidates)
        if held is None or self.target_box is None:
            return None, None  # no established subject to deviate from

        if self.target_tier == TIER_REACTION:
            # Already showing a reaction: hold it through its bounded window,
            # then hand back to the normal evidence flow (speaker reclaims).
            # End EARLY when the identified speaker resumes mouthing — but
            # only for quiet-beat reactions. A reaction triggered DURING the
            # speaker's line must hold its window: the speaker never stopped
            # mouthing, so early-ending would make it a single-frame flash.
            speaker_back = False
            if evidence is not None and not self._reaction_during_speech:
                want_id, want_tier = evidence.proposal()
                if want_tier in STRONG_TIERS:
                    want = _by_id(candidates, want_id)
                    if (want is not None and not self._same_subject(want)
                            and want.get("mouth_activity", 0.0) >= REACTION_SPEAKER_QUIET):
                        speaker_back = True
            if not speaker_back and (self._reaction_start is not None
                                     and frame_number - self._reaction_start
                                     < self.reaction_max_frames):
                self._refresh(held, TIER_REACTION)
                return held, TIER_REACTION
            self._reaction_cooldown_until = frame_number + self.reaction_cooldown_frames
            return None, None

        # Track the speaker's quiet beat. "Speaker" = the evidence-named
        # strong speaker when the label binds to a candidate this frame,
        # otherwise the person we are framing. Not updated during a reaction
        # shot (the reactor's own laughing mouth would reset the beat).
        speaker_mouth = None
        if evidence is not None:
            want_id, want_tier = evidence.proposal()
            if want_tier in STRONG_TIERS:
                want = _by_id(candidates, want_id)
                if want is not None:
                    speaker_mouth = want.get("mouth_activity", 0.0)
        if speaker_mouth is None:
            speaker_mouth = held.get("mouth_activity", 0.0)
        if speaker_mouth >= REACTION_SPEAKER_QUIET:
            self._speaker_quiet_since = None
        elif self._speaker_quiet_since is None:
            self._speaker_quiet_since = frame_number

        if frame_number - self.shot_started < self.reaction_min_frames:
            return None, None  # shot too fresh to interrupt
        if frame_number < self._reaction_cooldown_until:
            return None, None  # just had a reaction — no chaining
        target_mouth = held.get("mouth_activity", 0.0)
        if target_mouth >= REACTION_SPEAKER_QUIET:
            # Speaker actively mouthing: the listener's face is allowed once
            # the shot has aged past the punchline window (owner spec: show
            # the woman while he talks down her lifestyle), unless a genuine
            # quiet beat already qualified.
            quiet_beat = (self._speaker_quiet_since is not None
                          and frame_number - self._speaker_quiet_since
                          >= self.reaction_min_quiet_frames)
            if (not quiet_beat
                    and frame_number - self.shot_started
                    < self.reaction_during_speech_frames):
                return None, None  # fresh punchline — do not interrupt
            self._reaction_during_speech = not quiet_beat
        else:
            if (self._speaker_quiet_since is None
                    or frame_number - self._speaker_quiet_since
                    < self.reaction_min_quiet_frames):
                return None, None  # a word gap, not a beat after a line
            self._reaction_during_speech = False
        reactors = self._reactors(candidates, evidence)
        if not reactors:
            return None, None
        best = self._best_reactor(reactors)
        self._reaction_start = frame_number
        self._commit(best, TIER_REACTION, frame_number)
        return best, TIER_REACTION

    # -- the decision ------------------------------------------------------

    def decide(self, candidates, evidence, frame_number, frame_width=None):
        """Returns (box, id, tier). ``box`` is the chosen candidate's box —
        the same list object the caller passed in, so identity-based
        bookkeeping downstream (boosted sets, split cells) still works.
        Returns (None, None, None) only when there is nothing to frame.
        """
        if not candidates:
            return None, None, None

        # A source cut is the director's own decision to change subject.
        # Re-deciding here is free: there is no shot to protect.
        if evidence is not None and evidence.scene_changed:
            self.shot_started = -10 ** 9
            self._pending_id = None
            self._pending_n = 0

        want_id, tier = (evidence.proposal() if evidence is not None
                         else (None, None))
        want = _by_id(candidates, want_id)
        if want is None:
            want_id, tier = None, None

        # THE SPEAKER OUTRANKS EVERY CUTAWAY. Owner spec, 4-aug-2026:
        # "It's supposed to frame the main subject that is talking. Prioritize
        # him because it's him that is being shown. Instead you're prioritizing
        # the host and every other person before him."
        #
        # Cutaways are punctuation, not content. Both are now OFF by default
        # (REACTION_ENABLED / FATIGUE_ENABLED): a fatigue cut fired every 8s to
        # a SILENT person, and a reaction could interrupt a line after 2s,
        # which together produced "showing the host, then showing the
        # reactions of the other person he's talking to."
        #
        # When enabled, a reaction may still only fire on a QUIET beat (the
        # framed speaker's own mouth has stopped) — never over a live line.
        react_cand, react_tier = self._maybe_reaction(
            candidates, evidence, frame_number)
        if react_cand is not None:
            self.tier_counts[react_tier] = self.tier_counts.get(react_tier, 0) + 1
            return react_cand["box"], react_cand.get("id"), react_tier

        fatigue_cand, fatigue_tier = self._maybe_fatigue(
            candidates, frame_number, evidence)
        if fatigue_cand is not None:
            self.tier_counts[fatigue_tier] = self.tier_counts.get(fatigue_tier, 0) + 1
            return fatigue_cand["box"], fatigue_cand.get("id"), fatigue_tier

        chosen = None
        if want is not None:
            if want_id == self.target_id or self._same_subject(want, frame_width):
                # Already on this person (possibly under a new id). Refresh
                # the box and tier — a shot that STARTED on weak evidence and
                # is now confirmed by lip-sync becomes protected as strong —
                # but do NOT restart the shot clock: nothing cut.
                self._commit(want, tier, frame_number, new_shot=False)
                chosen = want
            elif self._may_interrupt(tier, frame_number):
                need = self._confirm_needed(tier)
                self._pending_n = (self._pending_n + 1
                                   if self._pending_id == want_id else 1)
                self._pending_id = want_id
                if self._pending_n >= need:
                    self._commit(want, tier, frame_number)
                    chosen = want
            # else: floor not served — fall through and hold.

        if chosen is None:
            # Tier 5: hold whoever is framed, as long as they are visible.
            held = self._find_held(candidates, frame_width)
            if held is not None:
                self._refresh(held, TIER_HOLD)
                chosen, tier = held, TIER_HOLD
            else:
                # Tier 6: nothing known, nobody held. Size decides, but only
                # after size_confirm agreeing samples, so a detector blink
                # cannot jump the camera across the frame.
                big = _biggest(candidates, frame_width)
                if big is None:
                    return None, None, None
                bid = big.get("id")
                if self.target_id is None:
                    self._commit(big, TIER_SIZE, frame_number)
                else:
                    self._pending_n = (self._pending_n + 1
                                       if self._pending_id == bid else 1)
                    self._pending_id = bid
                    if self._pending_n >= self.size_confirm:
                        self._commit(big, TIER_SIZE, frame_number)
                    else:
                        # Hold the (currently invisible) subject's last known
                        # framing rather than jumping on one weak sample.
                        return None, None, None
                chosen, tier = big, TIER_SIZE

        self.tier_counts[tier] = self.tier_counts.get(tier, 0) + 1
        return chosen["box"], chosen.get("id"), tier

    def summary(self):
        """Human-readable tier breakdown for the render log."""
        names = {TIER_ASD: "lip-sync", TIER_DIARIZED: "diarized",
                 TIER_DIRECTIVE: "directed", TIER_MOUTH: "mouth",
                 TIER_HOLD: "held", TIER_SIZE: "size",
                 TIER_REACTION: "reaction", TIER_JCUT: "j-cut",
                 TIER_FATIGUE: "fatigue"}
        total = sum(self.tier_counts.values()) or 1
        parts = [f"{names[t]} {100.0 * n / total:.0f}%"
                 for t, n in sorted(self.tier_counts.items()) if n]
        return ", ".join(parts)
