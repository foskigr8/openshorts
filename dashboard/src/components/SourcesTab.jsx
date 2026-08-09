import React, { useState, useEffect } from 'react';
import { Loader2, HardDrive, FileVideo, X, ExternalLink, CheckCircle2, FileText, Sparkles } from 'lucide-react';
import { apiJson } from '../lib/api';
import { getApiUrl } from '../config';

/**
 * <SourcesTab/> — the saved source library. Every URL that has been
 * processed once is cached in source_store (the persistent `sources/` dir),
 * so re-runs skip the download/transcription/context calls. This tab lists
 * them like History: cards with a real first-frame preview, size, date and
 * what's cached, and a full streaming viewer for the original video.
 */

function fmtSize(bytes) {
  const b = Number(bytes);
  if (!Number.isFinite(b) || b <= 0) return null;
  if (b < 1024 * 1024) return `${Math.round(b / 1024)}KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(1)}MB`;
  return `${(b / (1024 * 1024 * 1024)).toFixed(2)}GB`;
}

function fmtDate(ts) {
  if (!ts) return 'Unknown';
  const d = new Date(ts * 1000);
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

function fmtClock(ts) {
  if (!ts) return '';
  return new Date(ts * 1000).toLocaleTimeString(undefined, { hour12: false });
}

export default function SourcesTab() {
  const [sources, setSources] = useState(null);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    apiJson('/api/sources')
      .then((d) => setSources(d.sources || []))
      .catch(() => setError('Could not load the saved sources.'));
  }, []);

  if (error) {
    return (
      <div className="card p-8 text-center">
        <p className="text-sm text-danger">{error}</p>
      </div>
    );
  }
  if (!sources) {
    return (
      <div className="flex items-center justify-center py-24 text-muted gap-2">
        <Loader2 size={18} className="animate-spin text-brass" /> loading sources…
      </div>
    );
  }

  const totalBytes = (sources || []).reduce((s, x) => s + (x.size_bytes || 0), 0);

  return (
    <div className="animate-fade">
      <div className="flex items-end justify-between gap-3 mb-6">
        <div>
          <p className="eyebrow">Saved Sources</p>
          <h1 className="font-display lowercase text-2xl text-ink tracking-tight">your video library</h1>
        </div>
        <span className="readout text-[11px] text-muted flex items-center gap-1.5">
          <HardDrive size={13} className="text-brass" />
          {fmtSize(totalBytes) || '0MB'} on disk · {sources.length} source{sources.length === 1 ? '' : 's'}
        </span>
      </div>

      {sources.length === 0 ? (
        <div className="card p-10 text-center space-y-2">
          <HardDrive size={22} className="mx-auto text-muted" strokeWidth={1.5} />
          <p className="text-sm text-ink2 lowercase">no saved sources yet</p>
          <p className="text-xs text-muted lowercase">
            every link you process once gets cached here — re-runs skip the download, transcription and context calls
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {sources.map((s) => (
            <button
              key={s.key}
              onClick={() => setSelected(s)}
              className="card p-3 text-left transition-all hover:border-brass/40 hover:shadow-lg group"
            >
              <div className="relative aspect-video w-full rounded-input overflow-hidden bg-black border border-rule mb-3">
                <video
                  src={getApiUrl(`/api/sources/${s.key}/video`)}
                  preload="metadata"
                  muted
                  playsInline
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/40">
                  <span className="icon-chip !w-10 !h-10">
                    <FileVideo size={18} />
                  </span>
                </div>
              </div>
              <p className="text-sm font-semibold text-ink truncate" title={s.title}>{s.title}</p>
              <div className="flex items-center gap-2 mt-1.5 text-[10px] readout text-muted">
                <span>{fmtSize(s.size_bytes) || '—'}</span>
                <span>·</span>
                <span>{fmtDate(s.updated_at)} {fmtClock(s.updated_at)}</span>
              </div>
              <div className="flex flex-wrap gap-1.5 mt-2">
                {s.has_transcript && (
                  <span className="px-2 py-0.5 rounded-full border border-rule text-[9px] readout text-ok flex items-center gap-1">
                    <FileText size={9} /> transcript
                  </span>
                )}
                {s.has_context && (
                  <span className="px-2 py-0.5 rounded-full border border-rule text-[9px] readout text-brass flex items-center gap-1">
                    <Sparkles size={9} /> context
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      )}

      {selected && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4 sm:p-8"
          onClick={() => setSelected(null)}
        >
          <div
            className="card p-5 w-full max-w-4xl max-h-full overflow-y-auto custom-scrollbar animate-fade"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-3 mb-4">
              <div className="min-w-0">
                <p className="eyebrow">Source video</p>
                <h2 className="font-display lowercase text-lg text-ink truncate" title={selected.title}>
                  {selected.title}
                </h2>
                <p className="text-[11px] text-muted mt-0.5 break-all">
                  {selected.filename} · {fmtSize(selected.size_bytes)} · {fmtDate(selected.updated_at)}
                </p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                {selected.url && (
                  <a
                    href={selected.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="icon-chip !w-8 !h-8 text-muted hover:text-brass"
                    title="Open the original link"
                  >
                    <ExternalLink size={14} />
                  </a>
                )}
                <button
                  onClick={() => setSelected(null)}
                  className="icon-chip !w-8 !h-8 text-muted hover:text-ink"
                  aria-label="close source viewer"
                >
                  <X size={15} />
                </button>
              </div>
            </div>

            <video
              controls
              autoPlay
              src={getApiUrl(`/api/sources/${selected.key}/video`)}
              className="w-full aspect-video rounded-input bg-black border border-rule"
            />

            <div className="flex flex-wrap gap-1.5 mt-4">
              {selected.has_transcript && (
                <span className="px-2 py-1 rounded-full border border-rule text-[10px] readout text-ok flex items-center gap-1.5">
                  <CheckCircle2 size={11} /> transcript cached — re-runs skip transcription
                </span>
              )}
              {selected.has_context && (
                <span className="px-2 py-1 rounded-full border border-rule text-[10px] readout text-brass flex items-center gap-1.5">
                  <Sparkles size={11} /> context cached — re-runs skip the Gemini link call
                </span>
              )}
              {!selected.has_transcript && (
                <span className="px-2 py-1 rounded-full border border-rule text-[10px] readout text-muted">
                  no transcript cached yet
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
