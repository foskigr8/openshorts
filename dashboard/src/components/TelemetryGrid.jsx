import { useState } from 'react';
import {
  Activity, Sparkles, Play, CheckCircle2, AlertTriangle, ChevronDown,
  Download, AudioLines, ScanSearch, Scissors, Clapperboard,
} from 'lucide-react';
import StageVisual from './StageVisual';

/**
 * <TelemetryGrid/> — the two instrument panels under the preview.
 *
 * Left: the log, COMPACT by default. It shows the one thing that is true
 * right now — the stage, what it is doing, and a live percentage — instead of
 * dumping every line the pipeline has ever printed at you. The full stream is
 * one click away (and still copyable in raw form for pasting into a chat),
 * but it is opt-in rather than the resting state.
 *
 * Right: <StageVisual/>, the picture of the same moment.
 */

const STAGE_META = {
  download: { Icon: Download, label: 'downloading the source' },
  transcribe: { Icon: AudioLines, label: 'transcribing the audio' },
  analyze: { Icon: ScanSearch, label: 'finding the viral moments' },
  render: { Icon: Scissors, label: 'cutting + rendering clips' },
  finalize: { Icon: Clapperboard, label: 'finalizing' },
};

function fmtClock(ts) {
  if (ts == null) return '';
  return new Date(ts * 1000).toLocaleTimeString(undefined, { hour12: false });
}

function LogIcon({ text }) {
  if (/❌|error|failed/i.test(text)) return <AlertTriangle size={12} className="text-danger shrink-0" />;
  if (/✅|ready|finished/i.test(text)) return <CheckCircle2 size={12} className="text-ok shrink-0" />;
  if (/clip \d+ ready/i.test(text)) return <Play size={12} className="text-brass shrink-0" />;
  if (/found \d+/i.test(text)) return <Sparkles size={12} className="text-brass shrink-0" />;
  return <Activity size={12} className="text-muted shrink-0" />;
}

const asText = (l) => (typeof l === 'string' ? l : l?.text || '');

// The most recent line that says something a person would want to read —
// the compact view's one-line answer to "where are we".
function headlineLine(logs) {
  const texts = (logs || []).map(asText).filter(Boolean);
  for (let i = texts.length - 1; i >= 0; i -= 1) {
    const t = texts[i].trim();
    if (t.length > 3 && !/^[-=_.\s]+$/.test(t)) return t;
  }
  return '';
}

function LogPanel({ logs, status, raw, onRawToggle, progress }) {
  const [view, setView] = useState('compact');
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const failed = status === 'error' || status === 'cancelled';
  const complete = status === 'complete';
  const stage = progress?.stage || (complete ? 'finalize' : 'download');
  const meta = STAGE_META[stage] || STAGE_META.download;
  const StageIcon = meta.Icon;
  const pct = Math.round(progress?.overall_pct ?? (complete ? 100 : 0));
  const stepPct = typeof progress?.step_pct === 'number' ? progress.step_pct : null;
  const done = progress?.clips_done ?? 0;
  const total = progress?.clips_total ?? 0;

  const stepText = failed
    ? 'stopped — open the full log for the last output'
    : complete
      ? 'every clip rendered'
      : (progress?.step || progress?.note || meta.label);

  const copyLogs = async () => {
    const text = (logs || [])
      .map((l) => (typeof l === 'string' ? l : `${fmtClock(l.ts)} ${l.text || ''}`))
      .join('\n');
    try {
      await navigator.clipboard.writeText(text || '(no logs yet)');
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { /* clipboard unavailable — ignore */ }
  };

  const fullStream = (
    <div className="flex-1 min-h-[140px] max-h-[260px] overflow-y-auto custom-scrollbar p-3 space-y-1">
      {(logs || []).length === 0 ? (
        <p className="text-xs text-muted text-center py-6 lowercase">waiting for output…</p>
      ) : (
        (logs || []).map((l, i) => (
          <div key={i} className="flex items-start gap-2 text-[11px] leading-relaxed">
            <span className="readout text-muted/70 shrink-0 pt-px">
              {fmtClock(typeof l === 'string' ? null : l.ts)}
            </span>
            <LogIcon text={asText(l)} />
            <span className="text-ink2 break-all min-w-0">{asText(l)}</span>
          </div>
        ))
      )}
    </div>
  );

  return (
    <div className="rounded-card border border-rule bg-paper overflow-hidden flex flex-col min-h-0">
      <div className="px-4 py-2.5 flex items-center justify-between gap-2 shrink-0 border-b border-rule">
        <div className="flex gap-1">
          {[{ id: 'compact', label: 'Compact' }, { id: 'raw', label: 'Raw' }].map((t) => (
            <button
              key={t.id}
              onClick={() => setView(t.id)}
              className={`px-2.5 py-1 rounded-full text-[11px] lowercase border transition-colors ${
                view === t.id
                  ? 'border-brass/50 bg-brass/10 text-ink'
                  : 'border-rule text-muted hover:border-rule2'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {view === 'raw' && (
            <button
              onClick={onRawToggle}
              className={`text-[10px] readout px-2 py-1 rounded-full border transition-colors ${
                raw ? 'border-brass/50 text-brass' : 'border-rule text-muted hover:border-rule2'
              }`}
              title="Include the unfiltered pipeline stream"
            >
              unfiltered
            </button>
          )}
          <button
            onClick={copyLogs}
            className={`text-[10px] readout px-2 py-1 rounded-full border transition-colors ${
              copied ? 'border-ok/50 text-ok' : 'border-rule text-muted hover:border-rule2'
            }`}
            title="Copy the whole log (for pasting into a chat)"
          >
            {copied ? 'copied ✓' : 'copy'}
          </button>
        </div>
      </div>

      {view === 'raw' ? fullStream : (
        <div className="flex flex-col min-h-0">
          <div className="px-4 pt-3.5 pb-3">
            <div className="flex items-center gap-3">
              <span
                className={`icon-chip !w-10 !h-10 shrink-0 ${
                  failed ? '!border-danger/30 text-danger'
                    : complete ? '!border-ok/30 text-ok'
                      : '!border-brass/40 text-brass animate-pulse'
                }`}
              >
                {failed ? <AlertTriangle size={18} />
                  : complete ? <CheckCircle2 size={18} />
                    : <StageIcon size={18} />}
              </span>
              <div className="min-w-0 flex-1">
                <p className="readout text-[9px] uppercase tracking-wider text-muted">
                  {failed ? 'stopped at' : complete ? 'finished' : 'now'} · {stage}
                </p>
                <p className="text-sm font-semibold text-ink truncate leading-snug" title={stepText}>
                  {stepText}
                </p>
              </div>
              {/* The digital readout the whole panel exists for. */}
              <div className="text-right shrink-0">
                <p
                  className="text-3xl font-semibold tabular-nums leading-none"
                  style={{ color: failed ? 'var(--color-danger)' : complete ? 'var(--color-ok)' : 'var(--color-ink)' }}
                >
                  {pct}<span className="text-base text-muted">%</span>
                </p>
                {stepPct != null && !complete && !failed && (
                  <p className="readout text-[10px] text-brass tabular-nums mt-1">step {stepPct}%</p>
                )}
              </div>
            </div>

            <div className="mt-3 h-1.5 rounded-full bg-paper3 overflow-hidden">
              <div
                className="h-full rounded-full transition-[width] duration-500"
                style={{
                  width: `${Math.max(pct, 1)}%`,
                  background: failed ? 'var(--color-danger)'
                    : complete ? 'color-mix(in oklab, var(--color-ok) 60%, transparent)'
                      : 'var(--grad-accent)',
                  boxShadow: failed || complete ? 'none' : '0 0 8px var(--color-glow)',
                }}
              />
            </div>

            <div className="mt-2.5 flex items-center gap-2">
              {total > 0 && (
                <span className="readout text-[10px] px-2 py-0.5 rounded-full border border-rule text-muted tabular-nums">
                  {done}/{total} clips
                </span>
              )}
              <span className="text-[10px] text-muted/80 truncate min-w-0 flex-1" title={headlineLine(logs)}>
                {headlineLine(logs)}
              </span>
              <button
                onClick={() => setExpanded((v) => !v)}
                className="shrink-0 flex items-center gap-1 text-[10px] lowercase text-muted hover:text-ink transition-colors"
              >
                {expanded ? 'collapse' : 'expand'}
                <ChevronDown size={11} className={`transition-transform ${expanded ? 'rotate-180' : ''}`} />
              </button>
            </div>
          </div>

          {expanded && <div className="border-t border-rule">{fullStream}</div>}
        </div>
      )}
    </div>
  );
}

export default function TelemetryGrid({
  logs, status, raw, onRawToggle, stageDurations, speedMultiplier,
  currentStage = null, progress = null,
}) {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
      <LogPanel
        logs={logs}
        status={status}
        raw={raw}
        onRawToggle={onRawToggle}
        progress={progress}
      />
      <StageVisual
        stage={currentStage}
        status={status}
        progress={progress}
        stageDurations={stageDurations}
        speedMultiplier={speedMultiplier}
      />
    </div>
  );
}
