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
    // Same geometry as a finished clip's card (ClipRow): one strip of
    // identically sized objects, some done and some not — a pending slot that
    // is 20px taller makes the row look broken.
    <div
      className={`rounded-input border overflow-hidden transition-colors h-[104px] ${
        rendering ? 'border-brass/30' : 'border-rule'
      }`}
      style={{
        background: 'linear-gradient(180deg, rgba(255,255,255,0.03) 0%, transparent 45%), var(--color-paper)',
      }}
    >
      <div className="flex items-stretch gap-2.5 p-2.5 h-full">
        <div
          className={`w-[54px] shrink-0 rounded-md relative overflow-hidden flex items-center justify-center bg-paper3 ${
            rendering ? 'animate-pulse' : ''
          }`}
        >
          {rendering
            ? <Loader2 size={16} className="animate-spin text-brass" />
            : <Clock size={15} className="text-muted" />}
          <span className="absolute top-1 left-1 readout text-[8px] text-muted/70">
            {String(index + 1).padStart(2, '0')}
          </span>
        </div>

        <div className="min-w-0 flex-1 flex flex-col">
          <p className={`text-[13px] leading-snug truncate ${rendering ? 'text-ink2' : 'text-muted'}`}>
            {rendering
              ? `Rendering clip ${index + 1}${total ? ` of ${total}` : ''}…`
              : `Clip ${index + 1}`}
          </p>
          <div className="mt-1.5 space-y-1.5">
            <div className={`h-2 w-3/4 rounded bg-paper3 ${rendering ? 'animate-pulse' : 'opacity-50'}`} />
            <div className={`h-2 w-1/2 rounded bg-paper3 ${rendering ? 'animate-pulse' : 'opacity-50'}`} />
          </div>
          <div className="mt-auto flex items-center">
            <span className="readout text-[9px] uppercase tracking-wider text-muted">
              {rendering ? 'rendering' : 'queued'}
            </span>
            <span className="flex-1" />
            <ProgressRing
              size={18}
              stroke={2}
              state={rendering ? 'processing' : 'pending'}
              indeterminate={rendering}
              pct={0}
              label={null}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
