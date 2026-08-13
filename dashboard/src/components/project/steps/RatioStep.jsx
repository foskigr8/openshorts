import { useEffect, useRef } from 'react';
import { ArrowLeft, ArrowRight } from 'lucide-react';

/**
 * Step 2 — the shape of the output.
 *
 * A reel of frames you move through rather than a grid of radio tiles: the
 * selected frame is the one in the middle, at full size, with its real aspect
 * ratio drawn to scale. Arrow keys work, because picking a format is exactly
 * the kind of choice that wants to be flicked through.
 *
 * The values are the backend's own (`output_format`), unchanged.
 */
const RATIOS = [
  { value: 'vertical', label: '9:16', hint: 'shorts · reels · tiktok', w: 54, h: 96 },
  { value: 'square', label: '1:1', hint: 'feed posts', w: 84, h: 84 },
  { value: 'horizontal', label: '16:9', hint: 'landscape', w: 108, h: 61 },
  { value: 'custom', label: 'custom', hint: 'set your own size', w: 72, h: 88 },
];

export default function RatioStep({ value, onChange, custom, onCustomChange, onNext, onBack }) {
  const idx = Math.max(0, RATIOS.findIndex((r) => r.value === value));
  const railRef = useRef(null);

  useEffect(() => {
    const el = railRef.current?.children?.[idx];
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
  }, [idx]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'ArrowRight') onChange(RATIOS[Math.min(RATIOS.length - 1, idx + 1)].value);
      if (e.key === 'ArrowLeft') onChange(RATIOS[Math.max(0, idx - 1)].value);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [idx, onChange]);

  return (
    <div className="w-full max-w-2xl mx-auto text-center space-y-6">
      <div className="space-y-2">
        <h2 className="font-display lowercase text-3xl text-ink">what shape?</h2>
        <p className="text-sm text-muted">Use ← → to flick through.</p>
      </div>

      <div
        ref={railRef}
        className="flex items-center justify-start sm:justify-center gap-4 overflow-x-auto custom-scrollbar px-6 py-4 snap-x"
      >
        {RATIOS.map((r) => {
          const active = r.value === value;
          return (
            <button
              key={r.value}
              type="button"
              onClick={() => onChange(r.value)}
              className="shrink-0 snap-center flex flex-col items-center gap-2.5 px-3 outline-none group"
              style={{
                transform: active ? 'scale(1)' : 'scale(0.82)',
                opacity: active ? 1 : 0.5,
                transition: 'transform 320ms cubic-bezier(0.16,1,0.3,1), opacity 320ms',
              }}
            >
              <span
                className="rounded-md border-2 flex items-center justify-center"
                style={{
                  width: r.w, height: r.h,
                  borderColor: active ? 'var(--color-accent)' : 'var(--color-rule-2)',
                  background: active
                    ? 'linear-gradient(180deg, color-mix(in oklab, var(--color-accent) 20%, transparent), transparent)'
                    : 'transparent',
                  boxShadow: active ? '0 0 26px -6px var(--color-glow)' : 'none',
                  transition: 'all 320ms cubic-bezier(0.16,1,0.3,1)',
                }}
              >
                <span className={`font-mono text-xs ${active ? 'text-ink' : 'text-muted'}`}>{r.label}</span>
              </span>
              <span className={`text-[11px] lowercase ${active ? 'text-ink2' : 'text-muted'}`}>{r.hint}</span>
            </button>
          );
        })}
      </div>

      {value === 'custom' && (
        <div className="flex items-center justify-center gap-3 animate-fade">
          <label className="flex items-center gap-2 text-xs text-muted">
            width
            <input
              type="number" min="100" max="4000" value={custom.width}
              onChange={(e) => onCustomChange({
                ...custom,
                width: Math.max(100, Math.min(4000, Number(e.target.value) || 1080)),
              })}
              className="input-field py-1.5 px-2 w-24 text-sm"
            />
          </label>
          <span className="text-muted">×</span>
          <label className="flex items-center gap-2 text-xs text-muted">
            height
            <input
              type="number" min="100" max="4000" value={custom.height}
              onChange={(e) => onCustomChange({
                ...custom,
                height: Math.max(100, Math.min(4000, Number(e.target.value) || 1920)),
              })}
              className="input-field py-1.5 px-2 w-24 text-sm"
            />
          </label>
        </div>
      )}

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
