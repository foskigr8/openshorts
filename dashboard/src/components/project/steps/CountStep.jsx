import { useEffect } from 'react';
import { ArrowLeft, ArrowRight } from 'lucide-react';

/**
 * Step 3 — how many clips.
 *
 * A dial: the number is the hero, the arc around it is the value, and the
 * whole thing is draggable (a range input laid transparently over the arc) as
 * well as arrow-key nudgeable. The quick picks below cover the counts people
 * actually choose without making them aim.
 *
 * 1–40 is the backend's own range, and the count is a hard target the picker
 * fulfils exactly — so it deserves to be stated once, clearly.
 */
const MIN = 1;
const MAX = 40;
const QUICK = [3, 5, 8, 12, 20];

const SIZE = 190;
const R = 78;
const CIRC = 2 * Math.PI * R;
// Three-quarter dial: a gap at the bottom so start and end are distinguishable.
const SWEEP = 0.75;

export default function CountStep({ value, onChange, onNext, onBack }) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowUp') onChange(Math.min(MAX, value + 1));
      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') onChange(Math.max(MIN, value - 1));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [value, onChange]);

  const frac = (value - MIN) / (MAX - MIN);

  return (
    <div className="w-full max-w-xl mx-auto text-center space-y-6">
      <div className="space-y-2">
        <h2 className="font-display lowercase text-3xl text-ink">how many clips?</h2>
        <p className="text-sm text-muted">A hard target — you get exactly this many.</p>
      </div>

      <div className="relative mx-auto" style={{ width: SIZE, height: SIZE }}>
        <svg width={SIZE} height={SIZE} className="block -rotate-[225deg]">
          <circle
            cx={SIZE / 2} cy={SIZE / 2} r={R}
            fill="none" stroke="var(--color-rule-2)" strokeWidth="10" strokeLinecap="round"
            strokeDasharray={`${CIRC * SWEEP} ${CIRC}`}
          />
          <circle
            cx={SIZE / 2} cy={SIZE / 2} r={R}
            fill="none" stroke="var(--color-accent)" strokeWidth="10" strokeLinecap="round"
            strokeDasharray={`${CIRC * SWEEP * frac} ${CIRC}`}
            style={{
              filter: 'drop-shadow(0 0 8px var(--color-glow))',
              transition: 'stroke-dasharray 260ms cubic-bezier(0.16,1,0.3,1)',
            }}
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <span className="text-6xl font-semibold text-ink tabular-nums leading-none">{value}</span>
          <span className="readout text-[10px] uppercase tracking-[0.18em] text-muted mt-2">
            clip{value === 1 ? '' : 's'}
          </span>
        </div>

        {/* The real control, invisible over the dial. */}
        <input
          type="range"
          min={MIN}
          max={MAX}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          aria-label="number of clips"
          className="absolute left-0 right-0 bottom-1 w-full cursor-pointer opacity-0 h-10"
        />
      </div>

      <div className="flex flex-wrap items-center justify-center gap-2">
        {QUICK.map((n) => (
          <button
            key={n}
            type="button"
            onClick={() => onChange(n)}
            className={`px-3.5 py-1.5 rounded-full text-xs border transition-colors ${
              value === n
                ? 'border-brass/60 bg-brass/10 text-ink'
                : 'border-rule text-muted hover:border-rule2'
            }`}
          >
            {n}
          </button>
        ))}
        <span className="readout text-[10px] text-muted/70 ml-1">drag the dial for any number</span>
      </div>

      <div className="flex items-center justify-center gap-2">
        <button type="button" onClick={onBack} className="btn-ghost px-4 py-2.5 text-sm">
          <ArrowLeft size={15} /> back
        </button>
        <button type="button" onClick={onNext} className="btn-primary px-6 py-2.5">
          continue <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}
