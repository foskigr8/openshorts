import { useEffect, useRef, useState } from 'react';
import {
  Activity, Cpu, HardDrive, MemoryStick, Zap, AlertTriangle, Cookie, Server, X, Film,
} from 'lucide-react';
import { apiFetch } from '../lib/api';

/**
 * The machine, on demand.
 *
 * Three status pills used to sit in the header all day telling you the
 * backend was up — space spent on something that is almost always true. This
 * is one chip instead: a live pulse of what the box is doing, which opens
 * into the readings that actually matter while a render is running (CPU, RAM,
 * GPU utilisation, VRAM, disk, job slots), the way a notebook shows them.
 *
 * Polling only happens while the panel is open, plus a slow heartbeat for the
 * chip itself, so an idle tab is not asking the server anything twice a
 * second. Every number is measured server-side; nothing here is decorative.
 */

const HISTORY = 40;   // ~80s of trail at the open-panel poll rate

function Spark({ points, tone = 'var(--color-accent)' }) {
  if (!points.length) return null;
  const w = 100;
  const h = 26;
  const max = Math.max(100, ...points);
  const step = w / Math.max(1, HISTORY - 1);
  const d = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${(i * step).toFixed(1)} ${(h - (p / max) * h).toFixed(1)}`)
    .join(' ');
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="w-full h-6 block">
      <path d={d} fill="none" stroke={tone} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Meter({ Icon, label, value, sub, pct, tone = 'var(--color-accent)', trail, wide = false }) {
  const shown = Math.max(0, Math.min(100, pct ?? 0));
  return (
    <div className={`rounded-input border border-rule bg-paper2 px-3 py-2.5 min-w-0 ${wide ? 'col-span-2' : ''}`}>
      <div className="flex items-baseline gap-2">
        <Icon size={12} className="text-muted shrink-0 self-center" />
        <span className="readout text-[8px] uppercase tracking-wider text-muted truncate flex-1">{label}</span>
      </div>
      <p className="text-lg font-semibold text-ink tabular-nums leading-none mt-1.5 truncate" title={String(value)}>
        {value}
      </p>
      <div className="mt-1.5 h-1 rounded-full bg-paper3 overflow-hidden">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{
            width: `${Math.max(shown, 1)}%`,
            background: shown >= 88 ? 'var(--color-danger)' : tone,
          }}
        />
      </div>
      <div className="flex items-end justify-between gap-2 mt-1">
        <span className="readout text-[9px] text-muted/80 truncate">{sub}</span>
        {trail && trail.length > 3 && (
          <span className="w-14 shrink-0 opacity-70"><Spark points={trail} tone={tone} /></span>
        )}
      </div>
    </div>
  );
}

export default function SystemMonitor({ status }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const [m, setM] = useState(null);
  const [failed, setFailed] = useState(false);
  const trails = useRef({ cpu: [], mem: [], gpu: [] });

  useEffect(() => {
    let cancelled = false;
    const read = async () => {
      try {
        const res = await apiFetch('/api/metrics');
        if (!res.ok) throw new Error('metrics');
        const data = await res.json();
        if (cancelled) return;
        const t = trails.current;
        const push = (k, v) => {
          if (typeof v !== 'number') return;
          t[k] = [...t[k], v].slice(-HISTORY);
        };
        push('cpu', data.cpu_pct);
        push('mem', data.memory?.pct);
        push('gpu', data.gpus?.[0]?.util_pct);
        setM(data);
        setFailed(false);
      } catch (_) {
        if (!cancelled) setFailed(true);
      }
    };
    read();
    // Open: a live trace. Closed: a slow heartbeat, just enough to keep the
    // chip honest without polling a machine nobody is looking at.
    const timer = setInterval(read, open ? 2000 : 15000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [open]);

  // A full-screen click-catcher used to sit under this panel, which meant
  // that with the meters open you could not scroll the page or touch a clip —
  // it swallowed everything. Listening on the document instead leaves the
  // workspace completely usable while you watch the load.
  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => {
      if (!rootRef.current?.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const gpu = m?.gpus?.[0] || null;
  const cpu = typeof m?.cpu_pct === 'number' ? m.cpu_pct : null;
  const mem = m?.memory || null;
  // The chip shows the busiest thing on the box — that is the number worth
  // one glance.
  const headline = gpu?.util_pct != null ? gpu.util_pct : cpu;
  const headlineLabel = gpu?.util_pct != null ? 'gpu' : 'cpu';

  // Warnings from /api/system keep their meaning, they just live in here now.
  const cookieAge = status?.cookies?.age_seconds;
  const warnings = [];
  if (status && status.backend === false) warnings.push('backend unreachable');
  if (status?.cookies?.present && cookieAge != null && cookieAge > 60 * 60 * 12) {
    warnings.push('youtube cookies are stale');
  }
  if (m?.disk && m.disk.pct >= 90) warnings.push('disk almost full');

  return (
    <div className="relative" ref={rootRef}>
      <button
        onClick={() => setOpen((v) => !v)}
        title="machine load"
        className="group flex items-center gap-2 h-9 pl-2.5 pr-3 rounded-lg border bg-paper2/80
                   transition-colors duration-200 hover:bg-paper3"
        style={{
          borderColor: warnings.length
            ? 'color-mix(in oklab, var(--color-warn) 38%, var(--color-rule))'
            : 'var(--color-rule)',
        }}
      >
        {warnings.length
          ? <AlertTriangle size={14} className="text-warn shrink-0" />
          : <Activity size={14} className="text-muted shrink-0" />}
        <span className="min-w-0 leading-none text-left">
          <span className="block readout text-[8px] uppercase tracking-[0.12em] text-muted/80">
            {warnings.length ? 'attention' : headlineLabel}
          </span>
          <span className="block text-[11px] font-medium text-ink2 truncate mt-0.5 tabular-nums">
            {warnings.length
              ? warnings[0]
              : headline != null ? `${Math.round(headline)}%` : failed ? 'offline' : '—'}
          </span>
        </span>
        {m?.jobs_active > 0 && (
          <span className="readout text-[9px] text-brass shrink-0 tabular-nums">
            {m.jobs_active}/{m.job_slots}
          </span>
        )}
      </button>

      {open && (
        <>
          <div className="absolute right-0 top-11 z-40 w-[min(92vw,460px)] card p-3 space-y-2.5 animate-fade">
            <div className="flex items-center justify-between px-1">
              <p className="readout text-[9px] uppercase tracking-wider text-muted">machine</p>
              <div className="flex items-center gap-2">
                {m?.job_slots != null && (
                  <span className="readout text-[9px] text-muted tabular-nums">
                    {m.jobs_active || 0}/{m.job_slots} slots
                  </span>
                )}
                <button
                  onClick={() => setOpen(false)}
                  className="text-muted hover:text-ink transition-colors"
                  aria-label="close"
                >
                  <X size={13} />
                </button>
              </div>
            </div>

            {failed && (
              <p className="text-[11px] text-danger px-1">Can't read the machine right now.</p>
            )}

            {/* Two across. A column of full-width bars made four numbers take
                the height of a paragraph and read like a list of nothing. */}
            <div className="grid grid-cols-2 gap-2">
              <Meter
                Icon={Cpu}
                label={`cpu${m?.cpu_count ? ` · ${m.cpu_count} cores` : ''}`}
                value={cpu != null ? `${Math.round(cpu)}%` : '—'}
                sub={m?.jobs_active ? `${m.jobs_active} running` : 'idle'}
                pct={cpu}
                trail={trails.current.cpu}
              />
              <Meter
                Icon={MemoryStick}
                label="ram"
                value={mem ? `${mem.used_gb}` : '—'}
                sub={mem ? `of ${mem.total_gb} GB · ${mem.pct}%` : 'unavailable'}
                pct={mem?.pct}
                tone="var(--color-warn)"
                trail={trails.current.mem}
              />
            </div>

            {m?.gpus?.length ? m.gpus.map((g, i) => (
              <div key={i} className="rounded-input border border-rule bg-paper2 p-2.5">
                <div className="flex items-center gap-2 mb-2">
                  <Zap size={12} className="text-brass shrink-0" />
                  <span className="text-[11px] text-ink2 truncate flex-1" title={g.name}>
                    {g.name || `gpu ${i}`}
                  </span>
                  {g.temp_c != null && (
                    <span className="readout text-[9px] text-muted tabular-nums">
                      {Math.round(g.temp_c)}°c
                    </span>
                  )}
                </div>
                {/* utilisation and vram as one row of two gauges */}
                <div className="grid grid-cols-2 gap-2">
                  <Gauge
                    label="load"
                    value={g.util_pct != null ? `${Math.round(g.util_pct)}%` : '—'}
                    pct={g.util_pct}
                    tone="var(--color-accent)"
                    trail={i === 0 ? trails.current.gpu : null}
                  />
                  <Gauge
                    label="vram"
                    value={g.vram_total_gb ? `${g.vram_used_gb} / ${g.vram_total_gb} GB` : '—'}
                    pct={g.vram_pct}
                    tone="var(--color-ok)"
                  />
                </div>
              </div>
            )) : (
              <div className="rounded-input border border-rule bg-paper2 px-3 py-2.5 flex items-center gap-2">
                <Zap size={13} className="text-muted" />
                <span className="text-[11px] text-muted lowercase">no gpu on this host</span>
              </div>
            )}

            <div className="grid grid-cols-2 gap-2">
              <Meter
                Icon={HardDrive}
                label="disk"
                value={m?.disk ? `${m.disk.used_gb} GB` : '—'}
                sub={m?.disk ? `of ${m.disk.total_gb} GB` : ''}
                pct={m?.disk?.pct}
                tone="var(--color-muted)"
              />
              <Meter
                Icon={Film}
                label="clips on disk"
                value={status?.storage ? `${status.storage.used_gb} GB` : '—'}
                sub={status?.storage ? `cap ${status.storage.cap_gb} GB` : ''}
                pct={status?.storage?.pct}
                tone="var(--color-muted)"
              />
            </div>

            {/* The old header pills, now where they cost nothing. */}
            <div className="flex flex-wrap items-center gap-2 pt-0.5 px-1">
              <Chip Icon={Server} ok={status?.backend !== false} label="backend" />
              <Chip
                Icon={Cookie}
                ok={!!status?.cookies?.present}
                warn={cookieAge != null && cookieAge > 60 * 60 * 12}
                label="cookies"
              />
              {status?.storage_backend?.configured && (
                <Chip Icon={HardDrive} ok label="storage" />
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/** Half-width readout used inside a GPU card. */
function Gauge({ label, value, pct, tone, trail = null }) {
  const shown = Math.max(0, Math.min(100, pct ?? 0));
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between gap-2">
        <span className="readout text-[8px] uppercase tracking-wider text-muted">{label}</span>
        <span className="text-[11px] font-semibold text-ink tabular-nums truncate">{value}</span>
      </div>
      <div className="mt-1 h-1.5 rounded-full bg-paper3 overflow-hidden">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{
            width: `${Math.max(shown, 1)}%`,
            background: shown >= 88 ? 'var(--color-danger)' : tone,
            boxShadow: shown > 4 ? '0 0 8px var(--color-glow)' : 'none',
          }}
        />
      </div>
      {trail && trail.length > 3 && (
        <span className="block mt-0.5 opacity-70"><Spark points={trail} tone={tone} /></span>
      )}
    </div>
  );
}

function Chip({ Icon, ok, warn = false, label }) {
  const tone = warn ? 'var(--color-warn)' : ok ? 'var(--color-ok)' : 'var(--color-danger)';
  return (
    <span className="flex items-center gap-1.5 px-2 py-1 rounded-full border border-rule text-[10px] text-muted">
      <Icon size={11} />
      {label}
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: tone }} />
    </span>
  );
}
