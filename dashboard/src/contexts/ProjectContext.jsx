/**
 * Every job the user has on the go, in one place.
 *
 * The app used to hold exactly one run in App.jsx state — one `jobId`, one
 * `status`, one `results`. Three consequences the owner felt daily:
 *   · "New Project" created nothing you could come back to,
 *   · leaving the tab felt like losing the work,
 *   · and a second video could not be started, even though the backend has
 *     been willing to run five at once the whole time (MAX_CONCURRENT_JOBS).
 *
 * So a project — not a job — is the unit of state here. The provider owns a
 * map of them, polls EVERY live one on a single interval (one timer, not one
 * per project), and persists the map, so switching tabs or reloading the page
 * resumes exactly where you were, with every run still ticking.
 *
 * Components stay presentational: they read a project and render it. Adding a
 * field to a project means touching this file and the component that shows it,
 * nothing in between.
 */
import {
  createContext, useContext, useState, useEffect, useCallback, useRef, useMemo,
} from 'react';
import { apiFetch, apiJson } from '../lib/api';

const ProjectContext = createContext(null);
export const useProjects = () => useContext(ProjectContext);

const STORE_KEY = 'os_projects_v2';
const LEGACY_SESSION_KEY = 'openshorts_session';
const POLL_MS = 2000;
const MAX_PERSISTED_LOGS = 200;
// Old projects are worth keeping in the drawer, but not forever and not at the
// cost of a bloated localStorage entry — the backend's History is the durable
// record, this is just the working set.
const MAX_PERSISTED_PROJECTS = 12;

const LIVE = new Set(['queued', 'processing']);
export const isLive = (p) => LIVE.has(p?.status);

/** A brand-new project record. Everything the workspace needs to render one. */
function makeProject(jobId, patch = {}) {
  return {
    id: jobId,
    title: 'New project',
    source: null,            // {type:'url'|'file'|'server', payload}
    status: 'queued',
    progress: null,
    logs: [],
    results: null,
    stageDurations: null,
    projectState: null,
    durableClips: {},
    requestedClipCount: null,
    submittedFormat: 'auto',
    noSource: false,
    parentId: null,          // set when this run adds clips to an earlier one
    createdAt: Date.now(),
    ...patch,
  };
}

/** A File payload can't survive a reload — point the preview at the server. */
function persistable(projects) {
  const out = {};
  const ordered = Object.values(projects)
    .sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0))
    .slice(0, MAX_PERSISTED_PROJECTS);
  for (const p of ordered) {
    out[p.id] = {
      ...p,
      source: p.source?.type === 'url'
        ? p.source
        : p.source
          ? { type: 'server', payload: `/api/source/${p.id}` }
          : null,
      logs: (p.logs || []).slice(-MAX_PERSISTED_LOGS),
    };
  }
  return out;
}

function loadPersisted() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) return JSON.parse(raw) || {};
  } catch (_) { /* corrupt entry — start clean rather than crash the app */ }
  // One-time migration from the single-session key this replaces.
  try {
    const legacy = JSON.parse(localStorage.getItem(LEGACY_SESSION_KEY) || 'null');
    localStorage.removeItem(LEGACY_SESSION_KEY);
    if (legacy?.jobId && legacy.status && legacy.status !== 'idle') {
      return {
        [legacy.jobId]: makeProject(legacy.jobId, {
          status: legacy.status === 'processing' ? 'processing' : legacy.status,
          results: legacy.results || null,
          source: legacy.processingMedia || null,
          projectState: legacy.projectState || null,
          noSource: !!legacy.noSource,
          title: legacy.processingMedia?.payload || 'Recovered project',
        }),
      };
    }
  } catch (_) { /* nothing to migrate */ }
  return {};
}

export function ProjectProvider({ children }) {
  const [projects, setProjects] = useState(loadPersisted);
  const [activeId, setActiveId] = useState(() => {
    try { return localStorage.getItem(`${STORE_KEY}:active`) || null; } catch (_) { return null; }
  });

  // --- persistence --------------------------------------------------------
  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(persistable(projects)));
    } catch (_) { /* quota — the backend is the durable record anyway */ }
  }, [projects]);

  useEffect(() => {
    try {
      if (activeId) localStorage.setItem(`${STORE_KEY}:active`, activeId);
      else localStorage.removeItem(`${STORE_KEY}:active`);
    } catch (_) { /* ignore */ }
  }, [activeId]);

  const patch = useCallback((id, changes) => {
    setProjects((prev) => {
      const cur = prev[id];
      if (!cur) return prev;
      const next = typeof changes === 'function' ? changes(cur) : changes;
      return { ...prev, [id]: { ...cur, ...next } };
    });
  }, []);

  // --- the one poll loop, for every live project --------------------------
  // Keyed on the SET of live ids so starting or finishing a run re-arms the
  // timer, but a progress tick (which changes project contents, not the set)
  // does not tear it down and rebuild it four times a second.
  const liveIds = useMemo(
    () => Object.values(projects).filter(isLive).map((p) => p.id).sort().join(','),
    [projects],
  );
  const rawLogs = useRef({});   // id → whether to ask for the unfiltered stream
  const missed = useRef({});    // id → consecutive failed polls
  // A restored session can hold a job the server has since forgotten (a
  // restart, a purge). Rather than spinning "processing…" forever — the exact
  // kind of lie this app has been called out for — give it a grace window and
  // then say plainly that the run is no longer tracked.
  const MISS_LIMIT = 15;        // × POLL_MS = 30s

  useEffect(() => {
    if (!liveIds) return undefined;
    const ids = liveIds.split(',');
    let cancelled = false;

    const pollOne = async (id) => {
      try {
        const raw = rawLogs.current[id] ? '?raw=1' : '';
        const data = await apiJson(`/api/status/${id}${raw}`);
        if (cancelled) return;
        missed.current[id] = 0;
        patch(id, (cur) => {
          const next = {};
          if (data.result) next.results = data.result;
          if (data.progress) next.progress = data.progress;
          if (data.stage_durations) next.stageDurations = data.stage_durations;
          if (data.logs) next.logs = data.logs;
          if (data.status === 'completed') next.status = 'complete';
          else if (data.status === 'failed') {
            next.status = 'error';
            const last = data.logs?.[data.logs.length - 1];
            const lastText = typeof last === 'string' ? last : last?.text || '';
            const message = data.error || lastText || 'Process failed';
            const prevLast = cur.logs?.[cur.logs.length - 1];
            const prevText = typeof prevLast === 'string' ? prevLast : prevLast?.text || '';
            // Appended once, not on every poll — the old loop re-added the
            // same "Error:" line every two seconds.
            if (!prevText.startsWith('Error: ')) {
              next.logs = [...(next.logs || cur.logs || []), `Error: ${message}`];
            }
          } else if (data.status === 'cancelled') next.status = 'cancelled';
          else if (data.status) next.status = data.status;
          return next;
        });
      } catch (_) {
        if (cancelled) return;
        const n = (missed.current[id] || 0) + 1;
        missed.current[id] = n;
        if (n >= MISS_LIMIT) {
          missed.current[id] = 0;
          patch(id, (cur) => ({
            status: 'error',
            logs: [...(cur.logs || []),
              'The server is no longer tracking this run (it was restarted or the job expired).'],
          }));
        }
      }
    };

    const tick = () => ids.forEach(pollOne);
    tick();
    const timer = setInterval(tick, POLL_MS);
    return () => { cancelled = true; clearInterval(timer); };
  }, [liveIds, patch]);

  // --- durable (HF/R2) clip urls for managed accounts ---------------------
  // Fetched once per project completion so a preview still plays after the
  // ephemeral local file has been cleaned up.
  const hydrateDurable = useCallback(async (id) => {
    try {
      const d = await apiJson('/api/history');
      const map = {};
      for (const v of d.videos || []) {
        if (v.job_id === id && v.clip_index != null) map[v.clip_index] = v.view_url;
      }
      patch(id, { durableClips: map });
    } catch (_) { /* optional enhancement */ }
  }, [patch]);

  // --- actions ------------------------------------------------------------

  /**
   * Submit a new run. Returns {jobId} on success, or {needsConfirmation,
   * qualityCheck} when the pre-flight gate wants an answer first. QuotaError
   * propagates so the caller can show the top-up path.
   */
  const startProject = useCallback(async (form, { headers = {}, force = false, parentId = null } = {}) => {
    let body;
    const h = { ...headers };
    if (form.type === 'url') {
      h['Content-Type'] = 'application/json';
      body = JSON.stringify({
        url: form.payload,
        acknowledged: !!form.acknowledged,
        output_format: form.outputFormat || 'auto',
        force_low_quality: force,
        clip_count: form.clipCount ?? 8,
        long_context_clips: form.longContextClips || 0,
        remove_background_audio: form.removeBackgroundAudio || '',
        force_new: !!form.forceNew,
        custom_width: form.outputFormat === 'custom' ? form.customWidth : null,
        custom_height: form.outputFormat === 'custom' ? form.customHeight : null,
        captions: form.captions !== false,
        zoom_mode: form.zoomMode || 'auto',
        style_variant: form.styleVariant || 'balanced',
        caption_position: form.captionPosition || 'bottom',
        caption_margin: form.captionMargin ?? null,
      });
    } else {
      const fd = new FormData();
      fd.append('file', form.payload);
      fd.append('acknowledged', form.acknowledged ? 'true' : 'false');
      fd.append('output_format', form.outputFormat || 'auto');
      if (form.clipCount != null) fd.append('clip_count', String(form.clipCount));
      if (form.longContextClips > 0) fd.append('long_context_clips', String(form.longContextClips));
      if (form.removeBackgroundAudio) fd.append('remove_background_audio', form.removeBackgroundAudio);
      if (form.outputFormat === 'custom') {
        fd.append('custom_width', String(form.customWidth || 1080));
        fd.append('custom_height', String(form.customHeight || 1920));
      }
      fd.append('captions', form.captions !== false ? 'true' : 'false');
      fd.append('zoom_mode', form.zoomMode || 'auto');
      fd.append('style_variant', form.styleVariant || 'balanced');
      fd.append('caption_position', form.captionPosition || 'bottom');
      if (form.captionMargin != null) fd.append('caption_margin', String(form.captionMargin));
      body = fd;
    }

    const res = await apiFetch('/api/process', { method: 'POST', headers: h, body });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if (data.needs_confirmation) {
      return { needsConfirmation: true, qualityCheck: data.quality_check };
    }

    const jobId = data.job_id;
    const title = form.type === 'file'
      ? (form.payload?.name || 'Uploaded video')
      : form.payload;
    setProjects((prev) => ({
      ...prev,
      [jobId]: makeProject(jobId, {
        title,
        source: { type: form.type, payload: form.payload },
        status: 'processing',
        logs: ['Starting process…'],
        requestedClipCount: form.clipCount ?? 8,
        submittedFormat: form.outputFormat || 'auto',
        parentId,
        form,               // remembered so "more clips from this source" can replay it
      }),
    }));
    setActiveId(jobId);
    return { jobId };
  }, []);

  /** Reopen an archived project; the backend restores its files from storage. */
  const openProject = useCallback(async (jobId, meta = {}) => {
    const data = await apiJson(`/api/projects/${jobId}/restore`, { method: 'POST' });
    setProjects((prev) => ({
      ...prev,
      [jobId]: makeProject(jobId, {
        ...(prev[jobId] || {}),
        id: jobId,
        title: data.title || meta.title || prev[jobId]?.title || 'Project',
        status: 'complete',
        results: data.result || null,
        projectState: data.project_state || null,
        logs: ['♻️ Project restored from your library.'],
        // The source may or may not still be on disk; the preview degrades to
        // "source no longer on disk" on its own if it isn't.
        source: { type: 'server', payload: `/api/source/${jobId}` },
        noSource: true,
        createdAt: meta.createdAt ? new Date(meta.createdAt).getTime() : Date.now(),
      }),
    }));
    setActiveId(jobId);
    return data;
  }, []);

  /**
   * Open a job we can't restore (a failed run, or one whose files are gone)
   * read-only, with whatever logs the server still has. Better than a dead
   * tile you can only delete.
   */
  const inspectProject = useCallback(async (jobId, meta = {}) => {
    setProjects((prev) => ({
      ...prev,
      [jobId]: makeProject(jobId, {
        ...(prev[jobId] || {}),
        id: jobId,
        title: meta.title || prev[jobId]?.title || jobId,
        status: meta.status === 'processing' ? 'processing' : 'error',
        noSource: true,
        source: { type: 'server', payload: `/api/source/${jobId}` },
        createdAt: meta.createdAt ? new Date(meta.createdAt).getTime() : Date.now(),
      }),
    }));
    setActiveId(jobId);
    try {
      const data = await apiJson(`/api/status/${jobId}`);
      patch(jobId, {
        logs: data.logs?.length ? data.logs : ['No logs retained for this run.'],
        ...(data.progress ? { progress: data.progress } : {}),
        ...(data.result ? { results: data.result } : {}),
        ...(data.status
          ? { status: data.status === 'processing' ? 'processing' : data.status === 'completed' ? 'complete' : data.status }
          : {}),
      });
    } catch (_) {
      patch(jobId, {
        logs: [
          `Opened "${meta.title || jobId}" from your library.`,
          'This run kept no logs on the server.',
        ],
      });
    }
  }, [patch]);

  const cancelProject = useCallback(async (jobId) => {
    await apiFetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
    patch(jobId, { status: 'cancelled' });
  }, [patch]);

  /** Drop from the workspace. Does NOT delete anything on the server. */
  const closeProject = useCallback((jobId) => {
    setProjects((prev) => {
      const next = { ...prev };
      delete next[jobId];
      return next;
    });
    setActiveId((cur) => (cur === jobId ? null : cur));
  }, []);

  /** Remove from the workspace AND the server (clips, source, storage copy). */
  const deleteProject = useCallback(async (jobId) => {
    await apiFetch(`/api/history/${jobId}`, { method: 'DELETE' });
    closeProject(jobId);
  }, [closeProject]);

  // --- per-clip edit state, debounced to the backend ----------------------
  const clipSync = useRef({ pending: {}, timer: null });

  const flushClipState = useCallback(() => {
    const s = clipSync.current;
    if (s.timer) { clearTimeout(s.timer); s.timer = null; }
    for (const [jobId, byIndex] of Object.entries(s.pending)) {
      const clips = Object.entries(byIndex).map(([i, v]) => ({
        index: Number(i),
        active_layers: v.activeLayers,
        server_file: v.serverVideoFile,
      }));
      if (clips.length) {
        apiFetch(`/api/projects/${jobId}/state`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ clips }),
        }).catch(() => {});
      }
    }
    s.pending = {};
  }, []);

  const updateClipState = useCallback((jobId, index, state) => {
    const s = clipSync.current;
    s.pending[jobId] = { ...(s.pending[jobId] || {}), [index]: state };
    if (s.timer) clearTimeout(s.timer);
    s.timer = setTimeout(flushClipState, 2000);
  }, [flushClipState]);

  // Flush pending edits when the tab goes away, so a close never loses them.
  useEffect(() => {
    const onHide = () => flushClipState();
    window.addEventListener('pagehide', onHide);
    return () => window.removeEventListener('pagehide', onHide);
  }, [flushClipState]);

  const list = useMemo(
    () => Object.values(projects).sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0)),
    [projects],
  );
  const active = activeId ? projects[activeId] || null : null;
  const liveCount = useMemo(() => list.filter(isLive).length, [list]);

  const value = {
    projects, list, active, activeId, liveCount,
    setActiveId, patch,
    startProject, openProject, inspectProject, cancelProject,
    closeProject, deleteProject,
    updateClipState, flushClipState, hydrateDurable,
    setRawLogs: (id, on) => { rawLogs.current[id] = on; },
  };

  return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>;
}

export default ProjectContext;
