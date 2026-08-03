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

# Tiers considered STRONG: a positive identification of who is speaking.
# These bypass the shot-hold floor entirely (see the measurement in the
# module docstring -- gating them is what cost 27 points of accuracy).
STRONG_TIERS = (TIER_ASD, TIER_DIARIZED)

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
    os.environ.get("POLICY_ABSOLUTE_MIN_SHOT", "0.5"))

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
                 "mouth_id", "scene_changed")

    def __init__(self, asd_id=None, diarized_id=None, directive_id=None,
                 directive_reason=None, mouth_id=None, scene_changed=False):
        self.asd_id = asd_id
        self.diarized_id = diarized_id
        self.directive_id = directive_id
        self.directive_reason = directive_reason
        self.mouth_id = mouth_id
        self.scene_changed = scene_changed

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
                 absolute_min_shot_seconds=None):
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
        self.target_id = None
        self.target_tier = None
        # Where the framed subject was last seen. Identity of a SUBJECT is
        # this box, not self.target_id — see SAME_SUBJECT_IOU.
        self.target_box = None
        self.shot_started = -10 ** 9
        self._pending_id = None
        self._pending_n = 0
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
            self.shot_started = frame_number
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
                 TIER_HOLD: "held", TIER_SIZE: "size"}
        total = sum(self.tier_counts.values()) or 1
        parts = [f"{names[t]} {100.0 * n / total:.0f}%"
                 for t, n in sorted(self.tier_counts.items()) if n]
        return ", ".join(parts)
