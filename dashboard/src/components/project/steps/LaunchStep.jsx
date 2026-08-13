import { useState } from 'react';
import {
  ArrowLeft, ChevronDown, Rocket, Loader2, Link2, FileVideo, Crop, Layers,
} from 'lucide-react';

/**
 * Step 4 — confirm and go.
 *
 * The summary is three chips (source, shape, count) you can click to jump
 * back to that step, then one control that launches. Everything else the
 * pipeline accepts lives in the "more options" drawer, already set to the
 * defaults the previous form used — nothing was dropped, it just stopped
 * demanding a decision from you every single time.
 */
export default function LaunchStep({
  draft, onPatch, onBack, onJump, onLaunch, starting, error,
}) {
  const [open, setOpen] = useState(false);

  const sourceLabel = draft.source?.type === 'file'
    ? draft.source.payload?.name
    : draft.source?.payload;

  const shapeLabel = {
    vertical: '9:16', square: '1:1', horizontal: '16:9',
    custom: `${draft.custom.width}×${draft.custom.height}`,
  }[draft.outputFormat] || draft.outputFormat;

  const chip = (Icon, text, step) => (
    <button
      type="button"
      onClick={() => onJump(step)}
      title="change this"
      className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-rule bg-paper2
                 text-xs text-ink2 hover:border-brass/50 transition-colors max-w-full"
    >
      <Icon size={13} className="text-muted shrink-0" />
      <span className="truncate max-w-[220px]">{text}</span>
    </button>
  );

  return (
    <div className="w-full max-w-xl mx-auto text-center space-y-6">
      <div className="space-y-2">
        <h2 className="font-display lowercase text-3xl text-ink">ready</h2>
        <p className="text-sm text-muted">Change anything by tapping a chip.</p>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-2">
        {chip(draft.source?.type === 'file' ? FileVideo : Link2, sourceLabel || 'source', 0)}
        {chip(Crop, shapeLabel, 1)}
        {chip(Layers, `${draft.clipCount} clips`, 2)}
      </div>

      <div>
        <button
          type="button"
          onClick={onLaunch}
          disabled={starting}
          className="btn-primary px-8 py-3.5 text-base disabled:opacity-60"
        >
          {starting
            ? <><Loader2 size={18} className="animate-spin" /> starting…</>
            : <><Rocket size={18} /> generate</>}
        </button>
        <p className="text-[11px] text-muted mt-3 max-w-md mx-auto leading-relaxed">
          By generating you confirm you own this content or have the rights to process it.{' '}
          <a
            href="/#legal"
            target="_blank"
            rel="noopener noreferrer"
            className="underline underline-offset-2 hover:text-brass transition-colors"
          >
            Terms &amp; privacy
          </a>
        </p>
        {error && <p className="text-xs text-danger mt-2">{error}</p>}
      </div>

      {/* Everything the old form asked up front, kept and pre-set. */}
      <div className="text-left">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-1.5 mx-auto text-xs lowercase text-muted hover:text-ink transition-colors"
        >
          fine-tune the output
          <ChevronDown size={13} className={`transition-transform duration-300 ${open ? 'rotate-180' : ''}`} />
        </button>

        {open && (
          <div className="mt-4 rounded-card border border-rule bg-paper2 p-4 space-y-4 animate-fade">
            <Row
              label="Captions"
              hint="burned-in word timing"
              control={(
                <Toggle
                  on={draft.captions}
                  onChange={(v) => onPatch({ captions: v })}
                />
              )}
            />
            <Row
              label="Caption position"
              hint="where the words sit"
              dim={!draft.captions}
              control={(
                <select
                  value={draft.captionPlacement}
                  disabled={!draft.captions}
                  onChange={(e) => onPatch({ captionPlacement: e.target.value })}
                  className="bg-paper3 border border-rule rounded px-2 py-1 text-xs text-ink2 disabled:opacity-50"
                >
                  <option value="bottom">Bottom</option>
                  <option value="raised">Bottom, raised</option>
                  <option value="middle">Middle</option>
                </select>
              )}
            />
            <Row
              label="Remove background audio"
              hint="isolates the voice · ~30s per clip"
              control={(
                <Toggle
                  on={draft.removeBgAudio}
                  onChange={(v) => onPatch({ removeBgAudio: v })}
                />
              )}
            />
            <p className="readout text-[10px] text-muted/70">
              highlight detection is always on · framing and zoom are decided per scene
            </p>
          </div>
        )}
      </div>

      <button type="button" onClick={onBack} className="btn-ghost px-4 py-2 text-sm">
        <ArrowLeft size={15} /> back
      </button>
    </div>
  );
}

function Row({ label, hint, control, dim = false }) {
  return (
    <div className={`flex items-center justify-between gap-4 ${dim ? 'opacity-45' : ''}`}>
      <div className="min-w-0">
        <p className="text-xs text-ink2">{label}</p>
        <p className="text-[10px] text-muted lowercase truncate">{hint}</p>
      </div>
      <div className="shrink-0">{control}</div>
    </div>
  );
}

/** A switch, not a checkbox — the owner's word for those was "crunchy". */
function Toggle({ on, onChange }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      onClick={() => onChange(!on)}
      className="w-11 h-6 rounded-full border transition-colors relative shrink-0"
      style={{
        borderColor: on ? 'color-mix(in oklab, var(--color-accent) 55%, transparent)' : 'var(--color-rule-2)',
        background: on
          ? 'linear-gradient(90deg, color-mix(in oklab, var(--color-accent) 34%, transparent), color-mix(in oklab, var(--color-accent) 16%, transparent))'
          : 'var(--color-paper-3)',
      }}
    >
      <span
        className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full transition-all"
        style={{
          left: on ? 'calc(100% - 1.25rem)' : '0.25rem',
          background: on ? 'var(--color-accent)' : 'var(--color-muted)',
          boxShadow: on ? '0 0 10px var(--color-glow)' : 'none',
        }}
      />
    </button>
  );
}
