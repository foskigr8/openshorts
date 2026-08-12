import React, { useState, useEffect, useMemo } from 'react';
import { Loader2, Download, Film, FolderOpen, Trash2, AlertTriangle } from 'lucide-react';
import { apiJson, apiFetch } from '../lib/api';
import { pauseAllOtherPlayers, registerPlayer } from '../lib/playerSync';

// The signed-in user's saved video library (stored in R2). Private, signed links.
// Videos are grouped by project (job); re-openable projects get a "reopen"
// action that restores the whole job for further editing in the Clip Generator.
export default function HistoryTab({ onReopenProject, search = '' }) {
  const [videos, setVideos] = useState(null);
  const [projects, setProjects] = useState({});
  const [system, setSystem] = useState(null);
  const [filter, setFilter] = useState('all');
  const [reopening, setReopening] = useState(null);
  const [reopenError, setReopenError] = useState('');
  const [error, setError] = useState('');
  const [deleting, setDeleting] = useState(null);
  const [showDataFiles, setShowDataFiles] = useState(false);
  const [dataFiles, setDataFiles] = useState(null);
  const [deletingData, setDeletingData] = useState(false);

  useEffect(() => {
    apiJson('/api/history')
      .then((d) => setVideos(d.videos || []))
      .catch(() => setError('Could not load your library.'));
    apiJson('/api/projects')
      .then((d) => {
        const map = {};
        for (const p of d.projects || []) map[p.job_id] = p;
        setProjects(map);
      })
      .catch(() => {});
    apiFetch('/api/system')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setSystem(d))
      .catch(() => {});
  }, []);

  // Group videos by job. Newest-to-oldest AT ALL TIMES: each job's clips
  // are sorted by their own recency (a clip finished a minute ago leads its
  // job), then jobs by their newest clip — never relies on server order.
  const groups = useMemo(() => {
    const byJob = new Map();
    for (const v of videos || []) {
      const key = v.job_id || v.id;
      if (!byJob.has(key)) byJob.set(key, []);
      byJob.get(key).push(v);
    }
    let entries = [...byJob.entries()];
    for (const [, vids] of entries) {
      vids.sort((a, b) =>
        new Date(b.created_at || 0) - new Date(a.created_at || 0));
    }
    entries.sort(([, a], [, b]) =>
      new Date(b[0]?.created_at || 0) - new Date(a[0]?.created_at || 0));
    if (filter !== 'all') {
      entries = entries.filter(([, vids]) => (vids[0]?.status || 'completed') === filter);
    }
    const q = search.trim().toLowerCase();
    if (q) {
      entries = entries.filter(([jobId, vids]) => {
        const project = projects[jobId];
        const title = project?.title || vids[0]?.title || '';
        return title.toLowerCase().includes(q) || vids.some((v) => (v.title || '').toLowerCase().includes(q));
      });
    }
    return entries;
  }, [videos, filter, search, projects]);

  // Day-bucketed section headers ("Today", "Yesterday", then a date) so a
  // library with weeks of runs reads as a timeline instead of one long,
  // undifferentiated grid — the ask was "organized with time", not just
  // newest-first, which /api/history already guaranteed on its own.
  const dayLabel = (iso) => {
    if (!iso) return 'Undated';
    const d = new Date(iso);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(today.getDate() - 1);
    const sameDay = (a, b) => a.toDateString() === b.toDateString();
    if (sameDay(d, today)) return 'Today';
    if (sameDay(d, yesterday)) return 'Yesterday';
    return d.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
  };
  const dayGroups = useMemo(() => {
    const buckets = [];
    let last = null;
    for (const entry of groups) {
      const label = dayLabel(entry[1][0]?.created_at);
      if (label !== last) {
        buckets.push({ label, entries: [] });
        last = label;
      }
      buckets[buckets.length - 1].entries.push(entry);
    }
    return buckets;
  }, [groups]);

  const FILTERS = ['all', 'completed', 'processing', 'failed'];
  const storage = system?.storage;
  const storagePct = storage ? Math.min(100, Math.max(0, storage.pct || 0)) : null;

  const handleReopen = async (jobId) => {
    if (!onReopenProject || reopening) return;
    setReopening(jobId);
    setReopenError('');
    try {
      await onReopenProject(jobId);
    } catch (e) {
      setReopenError('Could not reopen this project. Please try again.');
      setReopening(null);
    }
  };

  // 6-aug-2026 (PART 5.1): the owner wants date AND time so two runs of the
  // same link are distinguishable at a glance, not just the date.
  const fmtDate = (iso) => (iso
    ? new Date(iso).toLocaleString(undefined, {
        year: 'numeric', month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit',
      })
    : '');

  const fmtSize = (bytes) => {
    const b = Number(bytes);
    if (!Number.isFinite(b) || b <= 0) return '0B';
    if (b < 1024 * 1024) return `${Math.round(b / 1024)}KB`;
    if (b < 1024 * 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(1)}MB`;
    return `${(b / (1024 * 1024 * 1024)).toFixed(2)}GB`;
  };

  const toggleDataFiles = () => {
    const next = !showDataFiles;
    setShowDataFiles(next);
    if (next && dataFiles === null) {
      apiJson('/api/history/meta')
        .then((d) => setDataFiles(d || { files: [], total_bytes: 0 }))
        .catch(() => setDataFiles({ files: [], total_bytes: 0 }));
    }
  };

  const handleDeleteAllData = async () => {
    if (deletingData) return;
    const count = dataFiles?.files?.length || 0;
    if (!window.confirm(
      `Delete ALL metadata JSON files (${count} file${count === 1 ? '' : 's'}, ~${fmtSize(dataFiles?.total_bytes)})?\n\nClips stay on disk, but deleting a job's metadata removes it from this library. This can't be undone.`)) return;
    setDeletingData(true);
    try {
      await apiFetch('/api/history/meta', { method: 'DELETE' });
      setDataFiles({ files: [], total_bytes: 0 });
      setVideos([]); // metadata is gone → the library list is empty now
    } catch (e) {
      setError('Could not delete the data files.');
    } finally {
      setDeletingData(false);
    }
  };

  const handleDelete = async (jobId, title) => {
    if (deleting) return;
    if (!window.confirm(`Delete "${title || 'this project'}" and all its clips? This can't be undone.`)) return;
    setDeleting(jobId);
    try {
      await apiFetch(`/api/history/${jobId}`, { method: 'DELETE' });
      setVideos((prev) => (prev || []).filter((v) => v.job_id !== jobId));
    } catch (e) {
      setError('Could not delete this project. Please try again.');
    } finally {
      setDeleting(null);
    }
  };

  if (videos === null && !error) {
    return <div className="flex justify-center py-20"><Loader2 className="animate-spin text-brass" /></div>;
  }

  return (
    <div className="h-full overflow-y-auto p-8 max-w-5xl mx-auto animate-fade">
      <p className="eyebrow mb-1.5">06 · HISTORY</p>
      <h1 className="font-display lowercase text-2xl text-ink mb-2">Your library</h1>
      <p className="text-muted text-sm mb-8 lowercase">
        Every short you've generated, kept until you delete it. Reopen a project to keep editing its clips.
      </p>

      {storage && (
        <div className="mb-6">
          <div className="flex items-center justify-between text-xs text-muted mb-1.5">
            <span className="readout uppercase tracking-wider">storage</span>
            <span className="readout">{storage.used_gb} GB / {storage.cap_gb} GB</span>
          </div>
          <div className="h-2 rounded-full bg-paper3 overflow-hidden">
            <div
              className={`h-full rounded-full ${storagePct >= 85 ? 'bg-danger' : 'bg-brass'}`}
              style={{ width: `${storagePct}%` }}
            />
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 mb-6">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-full text-xs lowercase transition-colors border ${
              filter === f
                ? 'bg-brass/15 border-brass/40 text-ink'
                : 'border-rule text-muted hover:border-rule2'
            }`}
          >
            {f}
          </button>
        ))}
        {filter !== 'all' && (
          <span className="readout text-[10px] text-muted">
            {groups.length} job{groups.length === 1 ? '' : 's'}
          </span>
        )}
      </div>

      {error && <p className="text-danger text-sm">{error}</p>}
      {reopenError && <p className="text-danger text-sm mb-4">{reopenError}</p>}

      {videos && videos.length === 0 && (
        <div className="text-center py-20 text-muted">
          <span className="icon-chip-muted !w-12 !h-12 mx-auto mb-4"><Film size={20} /></span>
          <p className="lowercase">No videos yet. Generate your first short from the Clip Generator.</p>
        </div>
      )}

      <div className="space-y-12">
        {dayGroups.map(({ label, entries }) => (
        <div key={label}>
          <h2 className="readout uppercase tracking-wider text-muted mb-4 pb-1.5 border-b border-rule">
            {label}
          </h2>
          <div className="space-y-10">
        {entries.map(([jobId, vids]) => {
          const project = projects[jobId];
          return (
            <section key={jobId}>
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-2 border-b border-rule">
                <div className="min-w-0">
                  <p className="text-sm text-ink font-medium truncate" title={project?.title || vids[0]?.title}>
                    {project?.title || vids[0]?.title || 'Project'}
                  </p>
                  <p className="readout mt-0.5">
                    {fmtDate(vids[0]?.created_at)} · {vids.length} clip{vids.length === 1 ? '' : 's'}
                    {vids[0]?.status && vids[0].status !== 'completed' && (
                      <span className={`ml-2 px-2 py-0.5 rounded-full text-[10px] ${
                        vids[0].status === 'failed' ? 'bg-danger/10 text-danger' : 'bg-brass/15 text-brass'
                      }`}>
                        {vids[0].status}
                      </span>
                    )}
                  </p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <a
                    href={`/api/jobs/${jobId}/logs`}
                    className="btn-ghost px-3 py-2 text-xs"
                    title="Download this run's full log as a text file"
                  >
                    <span className="icon-chip-muted !w-6 !h-6"><Download size={13} /></span> logs
                  </a>
                  {project && onReopenProject && (
                    <button
                      onClick={() => handleReopen(jobId)}
                      disabled={!!reopening}
                      className="btn-ghost px-3 py-2 text-xs"
                      title="Restore this project in the Clip Generator to keep editing subtitles, hooks, effects and dubbing"
                    >
                      {reopening === jobId
                        ? <><Loader2 size={14} className="animate-spin" /> reopening…</>
                        : <><span className="icon-chip-muted !w-6 !h-6"><FolderOpen size={13} /></span> reopen project</>}
                    </button>
                  )}
                  <button
                    onClick={() => handleDelete(jobId, project?.title || vids[0]?.title)}
                    disabled={deleting === jobId}
                    className="btn-ghost px-3 py-2 text-xs text-danger hover:text-danger"
                    title="Permanently delete this project and all its clips"
                  >
                    {deleting === jobId
                      ? <><Loader2 size={14} className="animate-spin" /> deleting…</>
                      : <><span className="icon-chip-muted !w-6 !h-6"><Trash2 size={13} /></span> delete</>}
                  </button>
                </div>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5">
                {vids.map((v, vi) => (
                  <div key={v.id} className="card card-hover overflow-hidden group">
                    {v.view_url ? (
                      <div className="aspect-[9/16] bg-black">
                        <video
                          src={v.view_url}
                          poster={`/api/thumbnails/${jobId}/${vi}`}
                          controls
                          preload="metadata"
                          className="w-full h-full object-contain"
                          ref={(el) => { if (el && !el.dataset.synced) { el.dataset.synced = '1'; registerPlayer(el); } }}
                          onPlay={(e) => pauseAllOtherPlayers(e.currentTarget)}
                        />
                      </div>
                    ) : v.status === 'processing' || vids[0]?.status === 'processing' ? (
                      <div className="aspect-[9/16] bg-paper3 flex flex-col items-center justify-center gap-2 text-brass relative overflow-hidden">
                        <div className="absolute inset-0 bg-gradient-to-b from-brass/5 via-brass/20 to-brass/5 animate-pulse" />
                        <Loader2 size={24} className="animate-spin relative z-10 text-brass" />
                        <span className="readout text-[10px] uppercase relative z-10 text-ink">rendering clip…</span>
                      </div>
                    ) : (
                      <div className="aspect-[9/16] bg-paper3 flex flex-col items-center justify-center gap-2 text-muted">
                        <AlertTriangle size={22} />
                        <span className="readout text-[10px] uppercase">failed · no clips</span>
                      </div>
                    )}
                    <div className="p-3">
                      <p className="text-sm text-ink font-medium line-clamp-2 mb-1" title={v.title}>{v.title || 'Short'}</p>
                      <div className="flex items-center justify-between">
                        <span className="readout">{fmtDate(v.created_at)}</span>
                        <a href={v.download_url} className="text-micro font-mono uppercase text-brass hover:text-ink flex items-center gap-1 transition-colors" title="Download">
                          <span className="icon-chip-muted !w-6 !h-6"><Download size={13} /></span> Download
                        </a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          );
        })}
          </div>
        </div>
        ))}
      </div>

      {/* Data files — hidden by default (owner rule: JSON only when asked).
          A collapsible subsection to inspect or bulk-delete the job
          metadata files, never surfaced as videos. */}
      <div className="mt-12 border-t border-rule pt-6">
        <button
          onClick={toggleDataFiles}
          className="flex items-center justify-between w-full text-left"
        >
          <span className="readout uppercase tracking-wider text-muted">
            storage · data files
          </span>
          <span className="readout text-[10px] text-muted">
            {showDataFiles ? 'hide' : 'show'}
          </span>
        </button>
        {showDataFiles && (
          <div className="mt-4">
            {dataFiles === null ? (
              <div className="flex justify-center py-6 text-muted">
                <Loader2 size={16} className="animate-spin text-brass" />
              </div>
            ) : (
              <>
                <div className="flex items-center justify-between mb-3">
                  <p className="readout text-[10px] text-muted">
                    {dataFiles.files?.length || 0} metadata file(s) · {fmtSize(dataFiles.total_bytes)}
                  </p>
                  <button
                    onClick={handleDeleteAllData}
                    disabled={deletingData || !(dataFiles.files?.length)}
                    className="btn-ghost px-3 py-1.5 text-xs text-danger hover:text-danger disabled:opacity-40"
                  >
                    {deletingData
                      ? <><Loader2 size={12} className="animate-spin inline mr-1" />deleting…</>
                      : <><Trash2 size={12} className="inline mr-1" />delete all</>}
                  </button>
                </div>
                {dataFiles.files?.length === 0 ? (
                  <p className="text-xs text-muted">no metadata files on disk</p>
                ) : (
                  <ul className="max-h-64 overflow-y-auto custom-scrollbar space-y-1">
                    {dataFiles.files.map((f) => (
                      <li key={f.path} className="flex items-center justify-between gap-3 px-3 py-2 rounded-input border border-rule bg-paper">
                        <div className="min-w-0">
                          <p className="text-xs text-ink truncate" title={f.filename}>{f.filename}</p>
                          <p className="readout text-[10px] text-muted truncate">
                            {f.job_id} · {fmtSize(f.size_bytes)} · {fmtDate(f.modified)}
                          </p>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
