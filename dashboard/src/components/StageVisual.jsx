import {
  Download, AudioLines, ScanSearch, Scissors, Clapperboard, CheckCircle2,
  AlertTriangle, FileVideo, Sparkles, Type,
} from 'lucide-react';

/**
 * <StageVisual/> — "what is the machine doing right now", as a picture.
 *
 * This replaces the second text panel (the simplified log), which said the
 * same words as the log next to it in a smaller font. Each stage gets its own
 * animation built from what that stage actually does: the file landing while
 * it downloads, a live waveform while it transcribes, a sweeping frame while
 * it looks for moments, a moving film strip while it renders. The numbers on
 * it are the backend's own (clips done, measured stage durations, the real
 * speed multiplier) — nothing here is decorative data.
 */

const STAGES = ['download', 'transcribe', 'analyze', 'render', 'finalize'];

const STAGE_COPY = {
  download: { Icon: Download, title: 'pulling the source video', sub: 'streaming the highest-quality version we can get' },
  transcribe: { Icon: AudioLines, title: 'listening to every word', sub: 'transcribing with speaker + timing detail' },
  analyze: { Icon: ScanSearch, title: 'hunting for the moments', sub: 'reading the whole transcript for what will travel' },
  render: { Icon: Scissors, title: 'cutting and reframing', sub: 'tracking faces, choosing shots, encoding on the GPU' },
  finalize: { Icon: Clapperboard, title: 'wrapping up', sub: 'writing metadata and backing the clips up' },
};

function DownloadScene() {
  return (
    <div className="relative w-full h-full flex items-end justify-center pb-6">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="absolute top-0 text-brass"
          style={{
            left: `${34 + i * 16}%`,
            animation: `drop-in 2.1s ${i * 0.45}s cubic-bezier(0.4, 0, 0.2, 1) infinite`,
          }}
        >
          <FileVideo size={20} />
        </span>
      ))}
      <div className="w-28 h-1.5 rounded-full bg-brass/30 relative overflow-hidden">
        <span
          className="absolute inset-y-0 left-0 w-1/3 rounded-full bg-brass"
          style={{ animation: 'shimmer 1.6s linear infinite' }}
        />
      </div>
    </div>
  );
}

function TranscribeScene() {
  const bars = [0.35, 0.7, 0.45, 1, 0.6, 0.85, 0.4, 0.95, 0.55, 0.75, 0.3, 0.65];
  return (
    <div className="w-full h-full flex flex-col items-center justify-center gap-3">
      <div className="flex items-end gap-1 h-12">
        {bars.map((h, i) => (
          <span
            key={i}
            className="w-1.5 rounded-full bg-brass origin-bottom"
            style={{
              height: `${h * 100}%`,
              animation: `eq-bounce ${0.7 + (i % 4) * 0.18}s ${i * 0.06}s ease-in-out infinite`,
            }}
          />
        ))}
      </div>
      <div className="flex items-center gap-1.5 text-muted">
        <Type size={12} />
        <span className="readout text-[9px] uppercase tracking-[0.14em]">audio → words</span>
      </div>
    </div>
  );
}

function AnalyzeScene() {
  return (
    <div className="w-full h-full flex items-center justify-center">
      <div className="relative w-28 h-16 rounded-md border border-brass/40 overflow-hidden bg-paper2">
        <div className="scan-fx">
          <span className="scan-band" />
          <span className="scan-edge" />
        </div>
        {[12, 44, 76].map((x, i) => (
          <Sparkles
            key={x}
            size={12}
            className="absolute text-brass"
            style={{
              left: x, top: 8 + (i % 2) * 26,
              animation: `filament-pulse ${1.2 + i * 0.3}s ease-in-out infinite`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

function RenderScene({ done, total }) {
  const frames = Array.from({ length: 12 });
  return (
    <div className="w-full h-full flex flex-col items-center justify-center gap-3">
      <div className="relative w-full max-w-[240px] overflow-hidden">
        <div className="flex gap-1.5 w-[200%]" style={{ animation: 'strip-slide 6s linear infinite' }}>
          {frames.map((_, i) => (
            <span
              key={i}
              className={`h-10 w-8 shrink-0 rounded-sm border ${
                total > 0 && i % 6 < Math.min(6, Math.round((done / Math.max(total, 1)) * 6))
                  ? 'border-brass/60 bg-brass/20'
                  : 'border-rule bg-paper2'
              }`}
            />
          ))}
        </div>
      </div>
      {total > 0 && (
        <p className="readout text-[10px] text-brass tabular-nums">
          clip {Math.min(done + 1, total)} of {total}
        </p>
      )}
    </div>
  );
}

function FinalizeScene({ ok = true }) {
  return (
    <div className="w-full h-full flex items-center justify-center">
      <span className={ok ? 'text-ok' : 'text-danger'}>
        {ok ? <CheckCircle2 size={40} strokeWidth={1.5} /> : <AlertTriangle size={40} strokeWidth={1.5} />}
      </span>
    </div>
  );
}

export default function StageVisual({
  stage,
  status,
  progress = null,
  stageDurations = null,
  speedMultiplier = null,
}) {
  const failed = status === 'error' || status === 'cancelled';
  const complete = status === 'complete';
  const active = complete ? 'finalize' : (stage || 'download');
  const copy = STAGE_COPY[active] || STAGE_COPY.download;
  const { Icon } = copy;
  const done = progress?.clips_done ?? 0;
  const total = progress?.clips_total ?? 0;
  const activeIdx = STAGES.indexOf(active);

  const scene = failed
    ? <FinalizeScene ok={false} />
    : complete
      ? <FinalizeScene />
      : active === 'download' ? <DownloadScene />
        : active === 'transcribe' ? <TranscribeScene />
          : active === 'analyze' ? <AnalyzeScene />
            : active === 'render' ? <RenderScene done={done} total={total} />
              : <FinalizeScene />;

  const headline = failed
    ? 'the run stopped'
    : complete
      ? 'all clips are ready'
      : copy.title;
  const sub = failed
    ? 'open the log to see the last thing it did'
    : complete
      ? 'play any clip — the source preview follows it'
      : (progress?.step || copy.sub);

  return (
    <div className="rounded-card border border-rule bg-paper overflow-hidden flex flex-col min-h-0">
      <div className="px-4 py-3 flex items-center justify-between gap-2 border-b border-rule shrink-0">
        <p className="text-xs text-ink2 lowercase font-medium">what's happening</p>
        <span className="readout text-[11px] text-brass">
          {speedMultiplier
            ? `${speedMultiplier}x faster than realtime`
            : status === 'processing' ? 'measuring…' : ''}
        </span>
      </div>

      <div className="flex-1 min-h-[128px] px-4 py-3 flex items-center gap-4">
        <span
          className={`icon-chip !w-11 !h-11 shrink-0 ${
            failed ? '!border-danger/30 text-danger'
              : complete ? '!border-ok/30 text-ok'
                : '!border-brass/40 text-brass animate-pulse'
          }`}
        >
          {failed ? <AlertTriangle size={20} /> : complete ? <CheckCircle2 size={20} /> : <Icon size={20} />}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-ink lowercase truncate" title={headline}>{headline}</p>
          <p className="text-[11px] text-muted leading-snug line-clamp-2" title={sub}>{sub}</p>
        </div>
        <div className="w-[42%] max-w-[240px] h-[88px] shrink-0">{scene}</div>
      </div>

      {/* Real per-stage timing — measured, never invented. */}
      <div className="px-4 pb-3 shrink-0 flex items-center gap-1.5">
        {STAGES.map((s, i) => {
          const v = Number(stageDurations?.[s]) || 0;
          const state = complete || i < activeIdx ? 'done' : i === activeIdx ? 'active' : 'todo';
          return (
            <div key={s} className="flex-1 min-w-0">
              <div
                className="h-1 rounded-full transition-colors"
                style={{
                  background: state === 'active'
                    ? 'var(--grad-accent)'
                    : state === 'done'
                      ? 'color-mix(in oklab, var(--color-ok) 55%, transparent)'
                      : 'var(--color-rule-2)',
                  boxShadow: state === 'active' && !failed ? '0 0 8px var(--color-glow)' : 'none',
                }}
              />
              <p className={`readout text-[8px] uppercase tracking-wider mt-1 truncate ${
                state === 'active' ? 'text-brass' : state === 'done' ? 'text-ink2' : 'text-muted/60'
              }`}
              >
                {s}
                {v > 0 && <span className="text-muted"> · {v < 10 ? v.toFixed(1) : Math.round(v)}s</span>}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
