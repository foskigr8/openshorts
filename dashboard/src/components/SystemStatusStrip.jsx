import { Server, Cookie, Cpu } from 'lucide-react';

/**
 * Live system status in the top bar.
 *
 * Design rule: this is ambient telemetry, not a call to action. It sits next to
 * the workspace all day, so it must recede until something is actually wrong.
 * The previous version painted each pill as a big saturated green/red block
 * with a white glyph, which read as a toy toolbar and fought the app's
 * red-on-black palette for attention it hadn't earned.
 *
 * Now: one dark hairline pill per signal, a muted monochrome icon, and a single
 * small status dot carrying the colour. Only the dot changes hue, so a healthy
 * bar is quiet and a degraded one is genuinely noticeable. Every pill shares one
 * shape so the row reads as a system rather than a pile of stickers.
 */

function formatAge(seconds) {
  if (seconds == null) return '';
  const mins = Math.floor(seconds / 60);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  const rem = mins % 60;
  return rem ? `${hours}h ${rem}m ago` : `${hours}h ago`;
}

const TONES = {
  ok: 'var(--color-ok)',
  warn: 'var(--color-warn)',
  err: 'var(--color-danger)',
};

function Pill({ label, value, tone = 'ok', title, Icon, pulse = false }) {
  const dot = TONES[tone] || TONES.ok;
  const degraded = tone !== 'ok';
  return (
    <div
      title={title}
      className="group flex items-center gap-2 h-9 pl-2.5 pr-3 rounded-lg border bg-paper2/80 min-w-0
                 transition-colors duration-200 hover:border-rule2"
      style={{
        // Only a degraded pill tints its own border; a healthy one stays neutral.
        borderColor: degraded
          ? `color-mix(in oklab, ${dot} 34%, var(--color-rule))`
          : 'var(--color-rule)',
      }}
    >
      {Icon && (
        <Icon
          size={14}
          strokeWidth={1.75}
          className="shrink-0 text-muted transition-colors duration-200 group-hover:text-ink2"
        />
      )}
      <span className="min-w-0 leading-none">
        <span className="block readout text-[8px] uppercase tracking-[0.12em] text-muted/80">
          {label}
        </span>
        <span className="block text-[11px] font-medium text-ink2 truncate mt-0.5">
          {value}
        </span>
      </span>
      <span className="relative flex w-1.5 h-1.5 shrink-0 ml-0.5">
        {/* The ping only runs on a healthy, live signal — a warning shouldn't
            animate, or the eye reads motion as "working" instead of "check me". */}
        {pulse && !degraded && (
          <span
            className="absolute inline-flex h-full w-full rounded-full animate-ping opacity-50"
            style={{ background: dot }}
          />
        )}
        <span className="relative inline-flex rounded-full w-1.5 h-1.5" style={{ background: dot }} />
      </span>
    </div>
  );
}

export default function SystemStatusStrip({ status }) {
  if (!status) {
    // Honest skeleton rather than pills implying a healthy system we haven't
    // actually confirmed yet.
    return (
      <div className="flex items-center gap-2">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-9 w-28 rounded-lg border border-rule bg-paper2/60 animate-pulse" />
        ))}
      </div>
    );
  }

  const backendOk = status.backend !== false;
  const cookies = status.cookies || {};
  const gpu = status.gpu || {};

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Pill
        label="Backend"
        value={backendOk ? 'Online' : 'Unreachable'}
        tone={backendOk ? 'ok' : 'err'}
        title={backendOk ? 'API reachable' : 'The API is not responding'}
        Icon={Server}
        pulse={backendOk}
      />
      <Pill
        label="Cookies"
        value={cookies.present
          ? (cookies.age_seconds != null ? formatAge(cookies.age_seconds) : 'Present')
          : 'Not set'}
        tone={cookies.present ? 'ok' : 'warn'}
        title={cookies.refreshed_at
          ? `YouTube cookies last refreshed ${cookies.refreshed_at}`
          : 'No cookies file on the server — YouTube downloads will hit the bot check'}
        Icon={Cookie}
      />
      <Pill
        label="GPU"
        value={gpu.detected ? gpu.name : 'CPU only'}
        tone={gpu.detected ? 'ok' : 'warn'}
        title={gpu.detected
          ? 'CUDA/NVENC host detected — hardware encoding available'
          : 'No GPU detected; rendering runs on CPU and will be slower'}
        Icon={Cpu}
      />
    </div>
  );
}
