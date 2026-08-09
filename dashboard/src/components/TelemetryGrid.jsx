import { useState } from 'react';
import {
  Activity, Sparkles, Play, CheckCircle2, AlertTriangle, Search, UploadCloud, Eye,
} from 'lucide-react';

/**
 * <TelemetryGrid/> (round 3, item 3): replaces the raw log dump with two
 * equal cards — Parsed Logs (Live Logs / AI Insights tabs, real per-line
 * server timestamps, self-host noise filter with an explicit raw toggle) and
 * a Performance chart driven by REAL stage durations + the real speed
 * multiplier. No fabricated numbers anywhere.
 */

const STAGE_ICONS = {
  download: UploadCloud,
  transcribe: Activity,
  analyze: Search,
  render: Play,
  finalize: CheckCircle2,
};

const STAGES = ['download', 'transcribe', 'analyze', 'render', 'finalize'];

function fmtClock(ts) {
  if (ts == null) return '';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString(undefined, { hour12: false });
}

function LogIcon({ text }) {
  if (/❌|error|failed/i.test(text)) return <AlertTriangle size={12} className="text-danger shrink-0" />;
  if (/✅|ready|finished/i.test(text)) return <CheckCircle2 size={12} className="text-ok shrink-0" />;
  if (/clip \d+ ready/i.test(text)) return <Play size={12} className="text-brass shrink-0" />;
  if (/found \d+/i.test(text)) return <Sparkles size={12} className="text-brass shrink-0" />;
  return <Activity size={12} className="text-muted shrink-0" />;
}

function ParsedLogs({ logs, status, raw, onRawToggle }) {
  const [tab, setTab] = useState('logs');
  const healthy = status === 'processing' || status === 'complete';
  const texts = (logs || []).map((l) => (typeof l === 'string' ? l : l.text || ''));

  const insights = (logs || [])
    .map((l) => (typeof l === 'string' ? l : l.text || ''))
    .map((t) => {
      const m = t.match(/Found (\d+) viral clips/i);
      if (m) return `Selected ${m[1]} candidate clips from the transcript`;
      const c = t.match(/Processing Clip (\d+)/i);
      if (c) return `Rendering clip ${c[1]}…`;
      const r = t.match(/Clip (\d+) ready/i);
      if (r) return `Clip ${r[1]} finished rendering`;
      if (/transcrib/i.test(t)) return 'Transcribing and chunking the audio';
      if (/scene/i.test(t)) return 'Detecting scenes for framing decisions';
      if (/encoder/i.test(t)) return 'Initializing the video encoder';
      if (/analyzing|gemini|picker|narrative/i.test(t)) return 'Finding the viral moments';
      if (/error|❌|failed/i.test(t)) return `⚠ ${t}`;
      return null;
    })
    .filter(Boolean);

  return (
    <div className="rounded-card border border-rule bg-paper overflow-hidden flex flex-col min-h-0">
      <div className="px-4 pt-3 pb-0 flex items-center justify-between gap-2 shrink-0">
        <div className="flex gap-1">
          {[{ id: 'logs', label: 'Live Logs' }, { id: 'insights', label: 'AI Insights' }].map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-3 py-1.5 rounded-t-md text-xs transition-colors border-b-2 ${
                tab === t.id
                  ? 'text-ink font-semibold border-brass'
                  : 'text-muted border-transparent hover:text-ink2'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <span className="flex items-center gap-1.5 text-[10px] readout text-muted">
            <span className={`w-1.5 h-1.5 rounded-full ${healthy ? 'bg-ok' : 'bg-danger'}`} />
            {healthy ? 'System healthy' : 'System degraded'}
          </span>
          <button
            onClick={onRawToggle}
            className={`text-[10px] readout px-2 py-1 rounded-full border transition-colors ${
              raw ? 'border-brass/50 text-brass' : 'border-rule text-muted hover:border-rule2'
            }`}
            title="Toggle the raw unfiltered pipeline stream"
          >
            raw
          </button>
        </div>
      </div>
      <div className="flex-1 min-h-[140px] max-h-[260px] overflow-y-auto custom-scrollbar p-3 space-y-1">
        {tab === 'logs' ? (
          (logs || []).length === 0 ? (
            <p className="text-xs text-muted text-center py-6 lowercase">waiting for output…</p>
          ) : (
            (logs || []).map((l, i) => (
              <div key={i} className="flex items-start gap-2 text-[11px] leading-relaxed">
                <span className="readout text-muted/70 shrink-0 pt-px">
                  {fmtClock(typeof l === 'string' ? null : l.ts)}
                </span>
                <LogIcon text={typeof l === 'string' ? l : l.text} />
                <span className="text-ink2 break-all min-w-0">
                  {typeof l === 'string' ? l : l.text}
                </span>
              </div>
            ))
          )
        ) : insights.length === 0 ? (
          <p className="text-xs text-muted text-center py-6 lowercase">no insights yet — waiting for pipeline events…</p>
        ) : (
          insights.slice(-20).map((t, i) => (
            <div key={i} className="flex items-start gap-2 text-[11px] leading-relaxed">
              <Eye size={12} className="text-muted shrink-0 mt-px" />
              <span className="text-ink2">{t}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

const STAGE_LABEL = {
  download: 'Fetching source',
  transcribe: 'Transcribing audio',
  analyze: 'Finding viral moments',
  render: 'Rendering clips',
  finalize: 'Finalising',
};

/**
 * Per-stage bars you can read at a glance instead of parsing log lines.
 *
 * Each stage is a bar whose length is its real measured mean duration. The
 * stage currently running is filled with the accent and carries a travelling
 * shimmer, so the panel answers "what is it doing right now" visually — which
 * the old static line chart never did (it plotted five averages and looked
 * identical whether the job was on step 1 or step 5).
 */
function PerformanceChart({ stageDurations, speedMultiplier, currentStage, status }) {
  const values = STAGES.map((s) => Number(stageDurations?.[s]) || 0);
  const max = Math.max(...values, 1);
  const hasData = values.some((v) => v > 0);
  const activeIdx = STAGES.indexOf(currentStage);
  const live = status === 'processing';

  return (
    <div className="rounded-card border border-rule bg-paper overflow-hidden flex flex-col min-h-0">
      <div className="px-4 py-3 flex items-center justify-between gap-2 border-b border-rule shrink-0">
        <p className="text-xs text-ink2 lowercase font-medium">Processing Performance</p>
        <span className="readout text-[11px] text-brass">
          {speedMultiplier
            ? `${speedMultiplier}x faster than realtime`
            : live
              ? 'measuring…'
              : <span className="inline-block w-24 h-2.5 rounded-full bg-paper3 animate-pulse align-middle" />}
        </span>
      </div>

      <div className="flex-1 min-h-[110px] p-3 space-y-1.5">
        {STAGES.map((s, i) => {
          const v = values[i];
          const isActive = live && i === activeIdx;
          const isDone = activeIdx > i || (!live && v > 0);
          // With no measurement yet, an active stage still shows a moving
          // indeterminate bar; an unreached stage stays a thin muted rail.
          const width = v > 0 ? Math.max(6, (v / max) * 100) : (isActive ? 100 : 6);
          return (
            <div key={s} className="flex items-center gap-2">
              <span
                className={`readout text-[8px] uppercase tracking-wider w-14 shrink-0 truncate transition-colors ${
                  isActive ? 'text-brass' : isDone ? 'text-ink2' : 'text-muted/60'
                }`}
              >
                {s}
              </span>
              <div className="flex-1 h-2 rounded-full bg-paper3 overflow-hidden relative">
                <div
                  className="h-full rounded-full transition-[width] duration-500 ease-out relative overflow-hidden"
                  style={{
                    width: `${width}%`,
                    background: isActive
                      ? 'var(--grad-accent)'
                      : isDone
                        ? 'color-mix(in oklab, var(--color-ok) 55%, transparent)'
                        : 'var(--color-rule-2)',
                    boxShadow: isActive ? '0 0 10px var(--color-glow)' : 'none',
                  }}
                >
                  {isActive && (
                    // Travelling sheen = "this stage is working right now".
                    <span
                      className="absolute inset-y-0 w-1/3 animate-[shimmer_1.4s_linear_infinite]"
                      style={{
                        background:
                          'linear-gradient(90deg, transparent, rgba(255,255,255,0.45), transparent)',
                      }}
                    />
                  )}
                </div>
              </div>
              <span className="readout text-[9px] text-muted w-9 text-right shrink-0 tabular-nums">
                {v > 0 ? `${v < 10 ? v.toFixed(1) : Math.round(v)}s` : isActive ? '…' : '—'}
              </span>
            </div>
          );
        })}
      </div>

      <div className="px-4 pb-3 shrink-0">
        <p className="text-[11px] text-muted truncate">
          {live && activeIdx >= 0
            ? <><span className="text-brass">▸</span> {STAGE_LABEL[currentStage] || currentStage}…</>
            : status === 'complete'
              ? 'All stages finished.'
              : status === 'error'
                ? 'Stopped — see the logs for the failure.'
                : hasData ? 'Averages from this job.' : 'Timing appears once the first stage completes.'}
        </p>
      </div>
    </div>
  );
}

export default function TelemetryGrid({ logs, status, raw, onRawToggle, stageDurations, speedMultiplier, currentStage = null }) {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
      <ParsedLogs logs={logs} status={status} raw={raw} onRawToggle={onRawToggle} />
      <PerformanceChart
        stageDurations={stageDurations}
        speedMultiplier={speedMultiplier}
        currentStage={currentStage}
        status={status}
      />
    </div>
  );
}
