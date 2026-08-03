import { Clock, Loader2 } from 'lucide-react';
import ProgressRing from './ProgressRing';

/**
 * One slot in the live clip grid — the only place a user watches their actual
 * output appear in real time, so it is never allowed to be a blank box.
 *
 * Each slot is in exactly one of two pre-finished states (the third, "done",
 * is a real <ResultCard/> rendered by the parent once the clip's .ready marker
 * lands, so the grid fills in incrementally rather than staying empty until
 * the whole job finishes):
 *
 *   queued    — muted placeholder + clock icon      → "not started yet"
 *   rendering — shimmer skeleton + "Rendering clip N of M…"
 *
 * Laid out as a compact row matching the history rail's card shape, so a
 * pending slot and a finished clip read as the same kind of object.
 */
export default function ClipSlotPlaceholder({ index, total = null, state = 'rendering' }) {
  const rendering = state === 'rendering';

  return (
    <div
      className={`rounded-input border overflow-hidden transition-colors ${
        rendering ? 'border-brass/30' : 'border-rule'
      }`}
      style={{
        background: 'linear-gradient(180deg, rgba(255,255,255,0.03) 0%, transparent 45%), var(--color-paper)',
      }}
    >
      <div className="flex items-center gap-3 p-3">
        {/* Thumbnail well — fixed size so a pending slot never balloons to a
            full-width 9:16 box the way an unsized placeholder used to. */}
        <div
          className={`w-16 h-24 rounded-md shrink-0 relative overflow-hidden flex items-center justify-center bg-paper3 ${
            rendering ? 'animate-pulse' : ''
          }`}
        >
          {rendering
            ? <Loader2 size={20} className="animate-spin text-brass" />
            : <Clock size={18} className="text-muted" />}
          <span className="absolute top-1 left-1 readout text-[8px] text-muted/70">
            {String(index + 1).padStart(2, '0')}
          </span>
        </div>

        <div className="min-w-0 flex-1">
          <p className={`text-sm truncate ${rendering ? 'text-ink2' : 'text-muted'}`}>
            {rendering
              ? `Rendering clip ${index + 1}${total ? ` of ${total}` : ''}…`
              : `Clip ${index + 1} — queued`}
          </p>
          {/* Skeleton lines stand in for the title/meta that will land here. */}
          <div className="mt-2 space-y-1.5">
            <div className={`h-2.5 w-3/4 rounded bg-paper3 ${rendering ? 'animate-pulse' : 'opacity-50'}`} />
            <div className={`h-2 w-1/2 rounded bg-paper3 ${rendering ? 'animate-pulse' : 'opacity-50'}`} />
          </div>
        </div>

        {/* Same ring component used everywhere else — indeterminate while
            rendering (there is no real per-clip percentage), muted when the
            slot hasn't started. Never a fake number. */}
        <ProgressRing
          size={30}
          stroke={3}
          state={rendering ? 'processing' : 'pending'}
          indeterminate={rendering}
          pct={0}
          label={null}
        />
      </div>
    </div>
  );
}
