import { useEffect, useRef, useState } from 'react';
import { Film, ChevronRight, ChevronLeft, HardDrive, Play, X, Clock, Database, Trash2, FolderOpen, Loader2, PanelRightClose, PanelRightOpen } from 'lucide-react';
import { apiFetch } from '../lib/api';
import { pauseAllOtherPlayers, registerPlayer } from '../lib/playerSync';
import ProgressRing from './ProgressRing';
import { usePanelState } from './ui/CollapsiblePanel';

/**
 * Persistent right rail on the Home view (reference UI): Generated Shorts /
 * History with status filter chips, plus the storage-used bar — the OUTPUT_MAX_GB
 * cap made visible for the first time.
 */
const FILTERS = ['all', 'completed', 'processing', 'failed'];

// Workspace status → the vocabulary /api/history speaks.
const railStatus = (s) => (
  s === 'complete' ? 'completed'
    : s === 'error' || s === 'cancelled' ? 'failed'
      : s === 'processing' || s === 'queued' ? 'processing'
        : 'completed');

function fmtDur(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s) || s <= 0) return null;
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

function fmtSize(bytes) {
  const b = Number(bytes);
  if (!Number.isFinite(b) || b <= 0) return null;
  if (b < 1024 * 1024) return `${Math.round(b / 1024)}KB`;
  return `${(b / (1024 * 1024)).toFixed(1)}MB`;
}

// Every completion indicator in the app is the SAME <ProgressRing/> — one
// component for "show completion state", never a ring here and a bare check
// there. When we have a real percentage for a running job we pass it; when we
// don't, the ring spins indeterminate rather than inventing a number.
function StatusRing({ status, pct = null }) {
  if (status === 'completed') {
    return <ProgressRing size={30} stroke={3} pct={100} state="complete" />;
  }
  if (status === 'failed') {
    return <ProgressRing size={30} stroke={3} pct={pct ?? 0} state="failed" />;
  }
  return (
    <ProgressRing
      size={30}
      stroke={3}
      state="processing"
      pct={pct ?? 0}
      indeterminate={pct == null}
    />
  );
}

export default function HomeRail({ onViewAll, search = '', projects = [], activeId = null, onOpenProject = null, onCompare = null, comparedId = null }) {
  const [videos, setVideos] = useState(null);
  const [storage, setStorage] = useState(null);
  const [filter, setFilter] = useState('all');
  const [playingId, setPlayingId] = useState(null);
  const [railOpen, toggleRail] = usePanelState('rail', false);
  const [deleting, setDeleting] = useState(null);
  const [deleteError, setDeleteError] = useState('');
  const [stale, setStale] = useState(false);
  const [loadError, setLoadError] = useState('');
  const itemRefs = useRef({});

  // Any project running anywhere keeps the rail live — not just the one on
  // screen, because several can be rendering at once now.
  const jobRunning = projects.some((p) => p.status === 'processing' || p.status === 'queued');

  // The rail is a live view of the same state the main panel shows, not a
  // one-time snapshot: it re-reads history while a job runs (so finished clips
  // appear as they land) and once more when the job reaches a terminal state.
  // A poll that fails must NEVER blank the list. /api/history walks the whole
  // output directory (and can touch remote storage), so under load it times
  // out now and then — and `setVideos(d?.videos || [])` turned every one of
  // those into "all your projects just disappeared". The last good answer
  // stands until a better one arrives.
  useEffect(() => {
    let cancelled = false;
    const load = () => {
      apiFetch('/api/history')
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (cancelled || !d || !Array.isArray(d.videos)) {
            if (!cancelled && d === null) setStale(true);
            return;
          }
          setVideos(d.videos);
          setStale(false);
        })
        .catch((e) => {
          if (!cancelled) { setStale(true); setLoadError(e?.message || ''); }
        });
      apiFetch('/api/system')
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { if (!cancelled && d?.storage) setStorage(d.storage); })
        .catch(() => {});
    };
    load();
    const interval = jobRunning ? setInterval(load, 5000) : null;
    return () => { cancelled = true; if (interval) clearInterval(interval); };
  }, [jobRunning, projects.length]);

  // The playing clip's row snaps to the TOP of the visible scroll area
  // (its own top edge aligned with the container's top) — but it never
  // changes position in the underlying list. So scrolling further UP from
  // there still reveals whatever was already above it (e.g. the clip you
  // just finished watching), instead of that clip being reordered away or
  // lost. (An earlier version re-sorted the playing clip to array index 0,
  // which physically moved clips past each other — removed; only the
  // viewport scrolls, the list order never does.)
  useEffect(() => {
    if (!playingId) return;
    const el = itemRefs.current[playingId];
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [playingId]);

  // No hard cap here — the rail scrolls (overflow-y-auto below) specifically
  // so older/already-watched clips stay reachable by scrolling down, not
  // truncated out of the list. "view all" still exists for jumping to the
  // full History page.
  const handleDelete = async (v) => {
    if (deleting) return;
    const label = v.title || 'this project';
    if (!window.confirm(
      `Delete "${label}" and everything on disk for it${fmtSize(v.size_bytes) ? ` (${fmtSize(v.size_bytes)})` : ''}?\n\nThis removes the clips and the downloaded source from the server. It cannot be undone.`
    )) return;
    setDeleting(v.job_id);
    setDeleteError('');
    try {
      const res = await apiFetch(`/api/history/${v.job_id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // Drop every clip belonging to that job, not just the row clicked.
      setVideos((prev) => (prev || []).filter((x) => x.job_id !== v.job_id));
      if (playingId && String(playingId).startsWith(v.job_id)) setPlayingId(null);
      // Storage bar is now stale — re-read it so the freed space shows up.
      apiFetch('/api/system')
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => setStorage(d?.storage || null))
        .catch(() => {});
    } catch (e) {
      setDeleteError('Could not delete that project. Please try again.');
    } finally {
      setDeleting(null);
    }
  };

  const q = search.trim().toLowerCase();

  // A job that was just submitted isn't on disk yet, so a plain history read
  // would show nothing and the user would think the project failed to create.
  // Merge the locally-known running job in at the top until the backend
  // catches up and reports it itself.
  // Newest-to-oldest at all times in the side panel — never rely on the
  // server's array order (a freshly finished clip must lead its job).
  const fetched = (videos || []).slice().sort(
    (a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0));

  // History rows carry ids like "<job_id>_3", so matching them against the
  // job id itself never hit — the running job was pinned at the top a SECOND
  // time next to its own disk entry, and the disk entry's stale status (a
  // job mid-download has no metadata yet) is what made a live run read as
  // "Failed" in this panel. Match on job_id, and let the locally-known
  // status win for every row of the job this tab is running.
  // The workspace's own view of each project beats the disk scan: a job
  // mid-download has no metadata and no progress file yet, and the scan used
  // to call that "failed" while it was plainly running.
  const localById = new Map(projects.map((p) => [p.id, p]));
  const reconciled = fetched.map((v) => {
    const local = localById.get(v.job_id);
    if (!local) return v;
    const s = railStatus(local.status);
    if (s === 'processing' && v.status !== 'completed') return { ...v, status: 'processing', pct: local.pct };
    if (s === 'failed' && v.status === 'processing') return { ...v, status: 'failed' };
    return v;
  });
  // A project the backend hasn't written to disk yet still belongs in the
  // list — otherwise starting a run looks like nothing happened.
  const unseen = projects
    .filter((p) => !fetched.some((v) => v.job_id === p.id))
    .map((p) => ({
      id: p.id,
      job_id: p.id,
      title: p.title || 'New project',
      status: railStatus(p.status),
      pct: p.pct,
      pending: true,
      created_at: new Date(p.createdAt || Date.now()).toISOString(),
    }));
  const merged = [...unseen, ...reconciled];

  const shown = merged
    .filter((v) => filter === 'all' || (v.status || 'completed') === filter)
    .filter((v) => !q || (v.title || '').toLowerCase().includes(q));
  const storagePct = storage ? Math.min(100, Math.max(0, storage.pct || 0)) : 0;

  // Folded away, the rail is a 12px spine you can push back open — the
  // workspace gets the width, and the choice survives a reload.
  if (!railOpen) {
    const running = merged.filter((v) => v.status === 'processing').length;
    const done = merged.length - running;
    return (
      <aside className="w-12 shrink-0 h-full border-l border-rule bg-paper2 flex flex-col items-center py-3 gap-3">
        <button
          onClick={toggleRail}
          className="w-9 h-9 rounded-lg border border-rule text-muted hover:text-ink hover:border-rule2
                     hover:bg-paper3 transition-colors flex items-center justify-center relative"
          title="open your library"
          aria-label="open your library"
        >
          <PanelRightOpen size={16} />
          {running > 0 && (
            <span
              className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-brass"
              style={{ boxShadow: '0 0 8px var(--color-glow)' }}
            />
          )}
        </button>
        <button
          onClick={toggleRail}
          className="flex-1 flex flex-col items-center gap-3 group"
          title="open your library"
        >
          <span
            className="readout text-[9px] uppercase tracking-[0.22em] text-muted group-hover:text-ink2
                       transition-colors whitespace-nowrap"
            style={{ writingMode: 'vertical-rl' }}
          >
            library
          </span>
          {(running > 0 || done > 0) && (
            <span className="readout text-[9px] text-muted tabular-nums" style={{ writingMode: 'vertical-rl' }}>
              {running > 0 ? `${running} running` : `${done}`}
            </span>
          )}
        </button>
      </aside>
    );
  }

  return (
    <aside className="w-full lg:w-[380px] shrink-0 h-full flex flex-col overflow-hidden border-l border-rule bg-paper2">
      <div className="px-4 py-3.5 border-b border-rule shrink-0">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-display lowercase text-base text-ink">Projects</h3>
          <div className="flex items-center gap-1">
            <button
              onClick={onViewAll}
              className="text-xs lowercase text-muted hover:text-brass transition-colors flex items-center gap-0.5"
            >
              view all <ChevronRight size={13} />
            </button>
            <button
              onClick={toggleRail}
              className="p-1.5 rounded-md text-muted hover:text-ink hover:bg-paper3 transition-colors"
              title="hide this panel"
              aria-label="hide projects panel"
            >
              <PanelRightClose size={15} />
            </button>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {FILTERS.map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded-full text-xs lowercase transition-colors border ${
                filter === f ? 'border-brass/50 bg-brass/10 text-ink' : 'border-rule text-muted hover:border-rule2'
              }`}
            >
              {f === 'all' ? 'All' : f[0].toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar p-3 space-y-3">
        {videos === null ? (
          <p className="text-xs text-muted px-3 py-6 text-center lowercase">loading…</p>
        ) : shown.length === 0 ? (
          <div className="flex flex-col items-center gap-2 text-muted py-10">
            <span className="icon-chip-muted !w-9 !h-9"><Film size={17} /></span>
            <p className="text-xs lowercase">no clips yet</p>
          </div>
        ) : (
          shown.map((v) => {
            // With a preview panel on screen, a clip opens THERE — big, next
            // to the 16:9 source, which is the comparison the rail's ~200px
            // card could never give. Without one (the home view) it still
            // expands and plays in place.
            const isPlaying = !onCompare && playingId === v.id;
            const isCompared = !!onCompare && comparedId === v.id;
            const isOpen = activeId && v.job_id === activeId;
            return (
              <div
                key={v.id}
                ref={(el) => { itemRefs.current[v.id] = el; }}
                className={`group rounded-input border overflow-hidden transition-all duration-200 ${
                  isPlaying || isCompared || isOpen
                    ? 'border-brass/60'
                    : 'border-rule hover:border-[color:color-mix(in_oklab,var(--color-accent)_35%,var(--color-rule-2))] hover:-translate-y-0.5'
                }`}
                style={{
                  background: 'linear-gradient(180deg, rgba(255,255,255,0.035) 0%, transparent 42%), var(--color-paper)',
                  boxShadow: isPlaying || isCompared ? '0 0 0 1px rgba(239,68,68,0.10), 0 16px 40px -20px rgba(239,68,68,0.4)' : undefined,
                }}
              >
                {isPlaying ? (
                  // Expanded: a tiny thumbnail can't actually be watched —
                  // playing takes over the full card width at real 9:16
                  // size instead of trying to play inside a ~60px chip.
                  <div className="p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs text-ink font-medium truncate pr-2" title={v.title}>{v.title}</p>
                      <button
                        onClick={() => setPlayingId(null)}
                        className="text-muted hover:text-ink shrink-0"
                        aria-label="close"
                      >
                        <X size={15} />
                      </button>
                    </div>
                    <div className="w-full max-w-[220px] mx-auto aspect-[9/16] bg-black rounded-md overflow-hidden">
                      <video
                        src={v.view_url}
                        autoPlay
                        controls
                        playsInline
                        className="w-full h-full object-contain"
                        ref={(el) => {
                          if (!el) return;
                          registerPlayer(el);
                          pauseAllOtherPlayers(el);
                          // NOTE: pausing intentionally does NOT collapse the
                          // card (round 3, item 0) — pausing to inspect a
                          // frame must keep the player expanded in place.
                          el.onended = () => setPlayingId((cur) => (cur === v.id ? null : cur));
                        }}
                      />
                    </div>
                  </div>
                ) : (
                  <button
                    onClick={() => (onCompare ? onCompare(v) : setPlayingId(v.id))}
                    disabled={!v.view_url}
                    className="w-full flex items-center gap-3 p-3 text-left disabled:cursor-not-allowed"
                  >
                    <div className="w-16 h-24 rounded-md overflow-hidden bg-black shrink-0 relative group">
                      {v.view_url ? (
                        <>
                          <video src={v.view_url} muted preload="metadata" className="w-full h-full object-cover" />
                          {fmtDur(v.duration) && (
                            <span className="absolute bottom-1 left-1 px-1 py-px rounded bg-black/70 readout text-[8px] text-ink2">
                              {fmtDur(v.duration)}
                            </span>
                          )}
                          <div className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-black/40 transition-colors">
                            <Play size={20} className="text-white drop-shadow" fill="white" />
                          </div>
                        </>
                      ) : (
                        <div className="w-full h-full flex items-center justify-center bg-paper3 text-danger">
                          <Film size={16} />
                        </div>
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm text-ink truncate" title={v.title}>{v.title}</p>
                      <div className="flex flex-wrap gap-1 mt-1.5">
                        {fmtDur(v.duration) && (
                          <span className="px-1.5 py-px rounded bg-paper3 border border-rule text-[9px] readout text-muted">
                            {fmtDur(v.duration)}
                          </span>
                        )}
                        {/* An unfinished job says so outright — a card that
                            shows only "AI Generated" and dashes reads as a
                            finished clip that lost its data. */}
                        {isCompared && (
                          <span className="px-1.5 py-px rounded border border-brass/50 bg-brass/15 text-[9px] readout text-brass">
                            In preview
                          </span>
                        )}
                        {v.status === 'processing' ? (
                          <span className="px-1.5 py-px rounded border border-brass/40 bg-brass/10 text-[9px] readout text-brass">
                            Rendering…
                          </span>
                        ) : v.status === 'failed' ? (
                          <span className="px-1.5 py-px rounded border border-danger/40 bg-danger/10 text-[9px] readout text-danger">
                            Failed
                          </span>
                        ) : (
                          <span className="px-1.5 py-px rounded bg-paper3 border border-rule text-[9px] readout text-muted">
                            AI Generated
                          </span>
                        )}
                      </div>
                      {v.status === 'completed' && (
                        <div className="flex items-center gap-3 mt-1.5 text-[10px] text-muted">
                          <span className="flex items-center gap-1">
                            <Clock size={10} /> {fmtDur(v.duration) || '—'}
                          </span>
                          <span className="flex items-center gap-1">
                            <Database size={10} /> {fmtSize(v.size_bytes) || '—'}
                          </span>
                        </div>
                      )}
                    </div>
                    <StatusRing status={v.status} pct={v.pct ?? null} />
                  </button>
                )}

                {/* Per-project actions. Open works for ANY job — completed or
                    failed — so a failed run can be inspected instead of being
                    a dead tile. Delete removes the job on the backend (clips,
                    source and all), which is the only way disk usage ever
                    comes back down. */}
                {!isPlaying && !v.pending && (
                  <div className="flex items-center gap-1 px-3 pb-2 -mt-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                    <button
                      onClick={() => onOpenProject?.(v)}
                      className="text-[10px] lowercase px-2 py-1 rounded-md text-muted hover:text-ink hover:bg-paper3 transition-colors flex items-center gap-1"
                      title="Open this project"
                    >
                      <FolderOpen size={11} /> open
                    </button>
                    <button
                      onClick={() => handleDelete(v)}
                      disabled={deleting === v.job_id}
                      className="text-[10px] lowercase px-2 py-1 rounded-md text-muted hover:text-danger hover:bg-danger/10 transition-colors flex items-center gap-1 disabled:opacity-50"
                      title="Delete this project and free its disk space"
                    >
                      {deleting === v.job_id
                        ? <><Loader2 size={11} className="animate-spin" /> deleting…</>
                        : <><Trash2 size={11} /> delete</>}
                    </button>
                    {fmtSize(v.size_bytes) && (
                      <span className="ml-auto readout text-[9px] text-muted/70">
                        {fmtSize(v.size_bytes)}
                      </span>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {deleteError && (
        <p className="px-4 py-2 text-[11px] text-danger shrink-0">{deleteError}</p>
      )}
      {stale && videos !== null && (
        <p className="px-4 py-1.5 text-[10px] text-muted shrink-0 lowercase truncate"
          title={loadError}
        >
          couldn't refresh{loadError ? ` — ${loadError}` : ''}
        </p>
      )}

      {storage && (
        <div className="p-5 border-t border-rule shrink-0">
          <div className="flex items-center justify-between text-[11px] text-muted mb-1.5">
            <span className="flex items-center gap-1.5 readout uppercase tracking-wider">
              <span className="icon-chip-muted !w-6 !h-6"><HardDrive size={12} /></span> storage
            </span>
            <span className="readout">{storage.used_gb} / {storage.cap_gb} GB</span>
          </div>
          <div className="h-1.5 rounded-full bg-paper3 overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${Math.max(storagePct, 1.5)}%`,
                background: storagePct >= 85 ? 'var(--color-danger)' : 'var(--grad-accent)',
                boxShadow: '0 0 8px var(--color-glow)',
              }}
            />
          </div>
        </div>
      )}
    </aside>
  );
}
