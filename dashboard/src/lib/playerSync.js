/**
 * Singleton playback across the whole dashboard (plan round 2, item 2).
 *
 * Result clips render in a grid of independent <video> elements; with no
 * coordination, playing clip B while clip A is still looping produces two
 * unmuted streams at once — the "another sound bleeding in" complaint.
 * Module-level registry (no React context needed): every mounted player
 * registers here, and any play() claims exclusive audio by pausing the rest.
 */

const activePlayers = new Set();

export function pauseAllOtherPlayers(el) {
  for (const other of activePlayers) {
    if (other !== el) {
      try {
        other.pause();
      } catch (_) { /* element may be unmounted */ }
    }
  }
}

export function registerPlayer(el) {
  if (el) activePlayers.add(el);
  return () => { if (el) activePlayers.delete(el); };
}

export function pauseAllPlayers() {
  for (const other of activePlayers) {
    try {
      other.pause();
    } catch (_) { /* ignore */ }
  }
}
