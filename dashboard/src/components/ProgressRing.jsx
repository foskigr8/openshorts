/**
 * The single circular progress/completion component for the whole app.
 *
 * There is exactly ONE of these — the job-status panel, the history rail and
 * the clip slots all render this same component, so "show completion state"
 * can never drift into a ring in one place and a bare checkmark in another.
 *
 * `pct` is real state (0-100) and drives BOTH the stroke fill and the number
 * in the middle, so those two can never disagree. `state` decides the colour
 * and the centre glyph:
 *   processing — accent (red), glowing, animated fill  → "alive, working"
 *   complete   — green, calm, no glow                  → "finished"
 *   failed     — flat dark red + X, no glow            → "stopped, broken"
 *   pending    — muted grey track only                 → "not started yet"
 *
 * The glow is an inset SVG drop-shadow sized to fit INSIDE the viewBox (the
 * stroke is inset by `pad`), so the component never paints outside its own
 * width/height box and can't be clipped by an `overflow-hidden` parent.
 */
export default function ProgressRing({
  pct = 0,
  size = 96,
  stroke = 7,
  label = null,
  state = 'processing',
  indeterminate = false,
  className = '',
}) {
  const clamped = Math.min(100, Math.max(0, Number(pct) || 0));
  // Inset the ring so the glow renders inside the box instead of bleeding out.
  const pad = Math.max(3, Math.round(stroke * 0.7));
  const r = (size - stroke) / 2 - pad;
  const c = 2 * Math.PI * r;
  const center = size / 2;

  const isFailed = state === 'failed';
  const isComplete = state === 'complete' || (state !== 'failed' && clamped >= 100);
  const isPending = state === 'pending';

  // A finished job should read as settled, not celebratory. Full-strength
  // --color-ok next to this red/black palette looked like a neon sticker, so
  // the completed ring is deliberately desaturated.
  const color = isFailed
    ? 'var(--color-danger)'
    : isComplete
      ? 'color-mix(in oklab, var(--color-ok) 72%, transparent)'
      : 'var(--color-accent)';

  // Glow only while genuinely alive. Terminal states (done/failed) stay flat —
  // a finished job should look calm, a broken one contained.
  const glow = state === 'processing' && !isComplete && !isPending;

  return (
    <div
      className={`relative shrink-0 ${className}`}
      style={{ width: size, height: size }}
      title={label || undefined}
      role="img"
      aria-label={
        isFailed ? 'job failed'
          : isPending ? 'not started'
            : `${Math.round(clamped)} percent complete`
      }
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90 block">
        <circle
          cx={center}
          cy={center}
          r={r}
          fill="none"
          stroke="var(--color-rule-2)"
          strokeWidth={stroke}
        />
        {!isPending && (
          <circle
            cx={center}
            cy={center}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeDasharray={indeterminate ? `${c * 0.28} ${c}` : c}
            strokeDashoffset={indeterminate ? 0 : c * (1 - clamped / 100)}
            strokeLinecap="round"
            // Smooth ~0.6s fill so a percentage change reads as continuous
            // progress rather than a number snapping to a new value.
            className={`transition-[stroke-dashoffset] duration-[600ms] ease-out ${
              indeterminate ? 'origin-center animate-spin' : ''
            }`}
            style={{
              ...(glow ? { filter: `drop-shadow(0 0 ${Math.round(stroke * 0.9)}px ${color})` } : null),
              ...(indeterminate ? { animationDuration: '1.4s' } : null),
            }}
          />
        )}
      </svg>

      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none px-1">
        {isFailed ? (
          <svg
            width={size * 0.34}
            height={size * 0.34}
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--color-danger)"
            strokeWidth="2.6"
            strokeLinecap="round"
            aria-hidden="true"
          >
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        ) : isComplete && size < 64 ? (
          // Too small for a legible "100%" — the check carries the same meaning.
          <svg
            width={size * 0.42}
            height={size * 0.42}
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--color-ok)"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M20 6 9 17l-5-5" />
          </svg>
        ) : indeterminate || size < 34 ? (
          // No real percentage exists — show the moving arc alone rather than
          // a "0%" that would be a fabricated readout.
          null
        ) : (
          <span
            className={`font-semibold leading-none tabular-nums ${
              size >= 96 ? 'text-xl' : size >= 64 ? 'text-sm' : 'text-[10px]'
            }`}
            style={{ color: isComplete ? 'color-mix(in oklab, var(--color-ok) 85%, var(--color-ink))' : 'var(--color-ink)' }}
          >
            {isPending ? '—' : `${Math.round(clamped)}%`}
          </span>
        )}
        {label && size >= 64 && (
          <span
            className="readout uppercase tracking-wider mt-1 leading-none truncate max-w-full text-center"
            style={{
              fontSize: size >= 96 ? 9 : 8,
              color: isFailed ? 'var(--color-danger)' : isComplete ? 'var(--color-ok)' : 'var(--color-muted)',
            }}
          >
            {label}
          </span>
        )}
      </div>
    </div>
  );
}
