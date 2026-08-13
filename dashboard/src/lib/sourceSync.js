/**
 * Playback bus between a 9:16 result clip and the 16:9 source preview.
 *
 * The rule the owner asked for: while you watch a clip, the ORIGINAL video
 * plays alongside it at the same moment in time, so the framing of the two
 * can be compared side by side. That needs three things the old prop-drilled
 * version never gave:
 *
 *   1. The source must not be pausable by the singleton-player rule. It is a
 *      companion view, always muted — pausing it whenever another player
 *      claims audio is exactly what made the preview sit still. It therefore
 *      never registers in playerSync.
 *   2. Only the clip that OWNS the sync may stop it. Playing clip B pauses
 *      clip A, and A's `pause` event used to arrive after B's `play` and
 *      immediately switch the source back off (a race that read as "the
 *      preview just doesn't work").
 *   3. Continuous drift correction, not a single seek at play time: the clip
 *      publishes its position as it runs and the source nudges itself back
 *      whenever the two slip apart.
 *
 * A module-level bus (rather than React state) keeps per-second position
 * updates off the App's render path — the whole dashboard re-rendering once a
 * second while a clip plays is its own kind of jank.
 */

const listeners = new Set();

let state = {
  owner: null,      // id of the clip currently driving the source
  time: 0,          // position in the SOURCE's timeline, seconds
  playing: false,
  seq: 0,           // bumps on every play/seek → forces a hard seek
  clip: null,       // {url, title, jobId, index} for the compare view
};

function emit() {
  for (const fn of listeners) {
    try {
      fn(state);
    } catch (_) { /* a bad subscriber must not break playback */ }
  }
}

export function subscribeSourceSync(fn) {
  listeners.add(fn);
  fn(state);
  return () => listeners.delete(fn);
}

export function getSourceSync() {
  return state;
}

/** A clip started playing (or seeked): hard-seek the source and roll. */
export function sourceSyncPlay(owner, time, clip = null) {
  state = {
    owner,
    time: Number.isFinite(time) ? Math.max(0, time) : 0,
    playing: true,
    seq: state.seq + 1,
    clip: clip || (state.owner === owner ? state.clip : null),
  };
  emit();
}

/** Position update while playing — no seq bump, so it only corrects drift. */
export function sourceSyncTime(owner, time) {
  if (state.owner !== owner || !state.playing) return;
  if (!Number.isFinite(time)) return;
  state = { ...state, time: Math.max(0, time) };
  emit();
}

/** Only the owning clip can stop the source. */
export function sourceSyncStop(owner) {
  if (state.owner !== owner) return;
  state = { ...state, playing: false };
  emit();
}

/** Show a clip beside the source without driving playback (history rail). */
export function setCompareClip(clip) {
  state = { ...state, clip };
  emit();
}

export function clearCompareClip() {
  state = { ...state, clip: null, playing: false, owner: null };
  emit();
}
