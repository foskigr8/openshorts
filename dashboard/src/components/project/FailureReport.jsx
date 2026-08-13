import { useState } from 'react';
import { AlertTriangle, Copy, Check, RotateCcw } from 'lucide-react';

/**
 * What actually went wrong.
 *
 * The old panel said "We couldn't process this video" and hid the real line
 * behind a "raw error" disclosure — so the one fact worth having was the one
 * thing you had to go looking for, while the word FAILED was repeated in four
 * places around it. This says the failing line first, in plain type, with the
 * pipeline's own words underneath it and the stage it died at. One failure
 * marker, one message, and a copy button for when it needs to go somewhere
 * else.
 */

const asText = (l) => (typeof l === 'string' ? l : l?.text || '');

// The line that actually explains it, working backwards from the end. The
// last log line is often a stack-trace tail or a bare exit code; the message
// people need is usually the last one carrying an error marker.
function realError(logs) {
  const texts = (logs || []).map(asText).map((t) => t.trim()).filter(Boolean);
  const scored = [];
  for (let i = texts.length - 1; i >= 0 && scored.length < 3; i -= 1) {
    const t = texts[i];
    if (/^error: /i.test(t)) scored.push(t.replace(/^error:\s*/i, ''));
    else if (/❌|\berror\b|\bfailed\b|exception|traceback|refused|denied|timed? out|not found/i.test(t)) {
      scored.push(t);
    }
  }
  return scored[0] || texts[texts.length - 1] || '';
}

// A one-line human framing of the failure, chosen from what the log shows.
// This sits UNDER the real message, never instead of it.
function context(logs) {
  const joined = (logs || []).map(asText).join(' ');
  if (/cookie|sign in to confirm|bot/i.test(joined)) {
    return 'YouTube asked the server to prove it is not a bot. Refreshing the cookies usually clears this.';
  }
  if (/download|yt-dlp|403|unavailable|private|age.?restrict/i.test(joined)) {
    return 'The source could not be fetched — age-restricted, private, region-locked or removed.';
  }
  if (/quota|429|resource_exhausted|api key|unauthor/i.test(joined)) {
    return 'An API key was rejected or out of quota. Check the keys in Settings.';
  }
  if (/transcrib|whisper|assemblyai/i.test(joined)) {
    return 'Transcription failed, so there was nothing to pick clips from.';
  }
  if (/cuda|nvenc|nvidia|gpu|onnxruntime/i.test(joined)) {
    return 'The GPU pipeline could not start. Check the bootstrap output on the host.';
  }
  if (/ffmpeg|encode|reframe|codec/i.test(joined)) {
    return 'Rendering failed while cutting the clips — the source may use an unusual codec.';
  }
  if (/no space|disk/i.test(joined)) return 'The machine ran out of disk space.';
  return null;
}

export default function FailureReport({ logs, stage, cancelled = false, onRetry, onNewSource }) {
  const [copied, setCopied] = useState(false);
  const message = realError(logs);
  const hint = context(logs);
  const tail = (logs || []).map(asText).filter(Boolean).slice(-12);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(tail.join('\n') || message);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (_) { /* clipboard unavailable */ }
  };

  return (
    <div className="px-1 py-6 max-w-2xl mx-auto space-y-4">
      <div className="flex items-start gap-3">
        <span
          className="w-9 h-9 rounded-input flex items-center justify-center shrink-0"
          style={{
            background: 'color-mix(in oklab, var(--color-danger) 14%, transparent)',
            color: 'var(--color-danger)',
          }}
        >
          <AlertTriangle size={18} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-ink">
            {cancelled ? 'You stopped this run' : 'The run stopped'}
            {stage && !cancelled && <span className="text-muted font-normal"> during {stage}</span>}
          </p>
          {hint && <p className="text-xs text-muted mt-1 leading-relaxed">{hint}</p>}
        </div>
      </div>

      {message && (
        <div className="rounded-input border border-danger/30 bg-danger/5 p-3">
          <p className="readout text-[9px] uppercase tracking-wider text-danger/80 mb-1.5">
            what the pipeline said
          </p>
          <p className="font-mono text-[11px] text-ink2 break-words leading-relaxed">{message}</p>
        </div>
      )}

      {tail.length > 1 && (
        <details className="group">
          <summary className="cursor-pointer select-none text-[11px] lowercase text-muted hover:text-ink transition-colors">
            the last {tail.length} lines
          </summary>
          <div className="mt-2 rounded-input border border-rule bg-paper2 p-3 max-h-56 overflow-y-auto custom-scrollbar space-y-0.5">
            {tail.map((line, i) => (
              <p key={i} className="font-mono text-[10px] text-muted break-words leading-relaxed">{line}</p>
            ))}
          </div>
        </details>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {onRetry && (
          <button onClick={onRetry} className="btn-primary px-4 py-2 text-xs">
            <RotateCcw size={13} /> try again
          </button>
        )}
        {onNewSource && (
          <button onClick={onNewSource} className="btn-ghost px-4 py-2 text-xs">
            use a different source
          </button>
        )}
        <button onClick={copy} className="btn-ghost px-4 py-2 text-xs">
          {copied ? <><Check size={13} /> copied</> : <><Copy size={13} /> copy the log</>}
        </button>
      </div>
    </div>
  );
}
