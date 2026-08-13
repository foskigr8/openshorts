import React, { useState, useEffect, useRef } from 'react';
import { Sparkles, Youtube, Instagram, Share2, ChevronDown, Check, LayoutDashboard, Settings, Plus, History, X, Shield, LayoutGrid, Image, Globe, Calendar, AlertTriangle, KeyRound, Bot, Loader2, Download, Search, Flame, TrendingUp, ChevronRight, ChevronLeft, Film, HardDrive } from 'lucide-react';
import KeyInput from './components/KeyInput';
import KeyListInput from './components/KeyListInput';
import Sidebar from './components/Sidebar';
import ProjectLauncher from './components/project/ProjectLauncher';
import StoragePanel from './components/StoragePanel';
import ClipRow from './components/project/ClipRow';
import FailureReport from './components/project/FailureReport';
import CollapsiblePanel, { usePanelState, useStoredChoice } from './components/ui/CollapsiblePanel';
import ResultCard from './components/ResultCard';
import ProcessingAnimation from './components/ProcessingAnimation';
import ClipSlotPlaceholder from './components/ClipSlotPlaceholder';
import SystemMonitor from './components/SystemMonitor';
import StageTracker from './components/StageTracker';
import HomeRail from './components/HomeRail';
import TelemetryGrid from './components/TelemetryGrid';
import SourcePanel from './components/SourcePanel';
import ThumbnailStudio from './components/ThumbnailStudio';
import ScheduleWeekModal from './components/ScheduleWeekModal';
import UsageMeter from './components/UsageMeter';
import SourcesTab from './components/SourcesTab';
import TopUpModal from './components/TopUpModal';
import PlanChoiceModal from './components/PlanChoiceModal';
import TrialUpgradeModal from './components/TrialUpgradeModal';
import LoginModal from './components/LoginModal';
import TrialGate from './components/TrialGate';
import HistoryTab from './components/HistoryTab';
import ProfileMenu from './components/ProfileMenu';
import Modal from './components/ui/Modal';
import { useAuth } from './contexts/AuthContext';
import { useProjects } from './contexts/ProjectContext';
import { apiFetch, apiJson, QuotaError } from './lib/api';
import { clearCompareClip, setCompareClip, subscribeSourceSync } from './lib/sourceSync';
import { getApiUrl } from './config';
import { BRAND } from './brand';

// Enhanced "Encryption" using XOR + Base64 with a Salt
// This is better than plain Base64 but still client-side.
const SECRET_KEY = import.meta.env.VITE_ENCRYPTION_KEY || "OpenShorts-Static-Salt-Change-Me";
const ENCRYPTION_PREFIX = "ENC:";

const encrypt = (text) => {
  if (!text) return '';
  try {
    const xor = text.split('').map((c, i) =>
      String.fromCharCode(c.charCodeAt(0) ^ SECRET_KEY.charCodeAt(i % SECRET_KEY.length))
    ).join('');
    return ENCRYPTION_PREFIX + btoa(xor);
  } catch (e) {
    console.error("Encryption failed", e);
    return text;
  }
};

const decrypt = (text) => {
  if (!text) return '';
  if (text.startsWith(ENCRYPTION_PREFIX)) {
    try {
      const raw = text.slice(ENCRYPTION_PREFIX.length);
      // Check if it's plain base64 or our custom XOR (simple try)
      const xor = atob(raw);
      const result = xor.split('').map((c, i) =>
        String.fromCharCode(c.charCodeAt(0) ^ SECRET_KEY.charCodeAt(i % SECRET_KEY.length))
      ).join('');
      return result;
    } catch (e) {
      // Fallback if decryption fails (might be old plain text)
      return '';
    }
  }
  // Backward compatibility: If no prefix, assume old plain text (or return empty if you want to force re-login)
  // For migration: Return text as is, so it populates the field, and next save will encrypt it.
  return text;
};

// Simple TikTok icon sine Lucide might not have it or it varies
const TikTokIcon = ({ size = 16, className = "" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M19.589 6.686a4.793 4.793 0 0 1-3.77-4.245V2h-3.445v13.672a2.896 2.896 0 0 1-5.201 1.743l-.002-.001.002.001a2.895 2.895 0 0 1 3.183-4.51v-3.5a6.329 6.329 0 0 0-5.394 10.692 6.33 6.33 0 0 0 10.857-4.424V8.687a8.182 8.182 0 0 0 4.773 1.526V6.79a4.831 4.831 0 0 1-1.003-.104z" />
  </svg>
);

// Cloud accounts get an auto-generated opaque id (os_<hash>) as username —
// meaningless to the user, so the selector shows connected networks instead.
const isAutoProfileId = (username) => /^os_[0-9a-f]/i.test(username || "");

const ProfileNetworkIcons = ({ profile, size = 12 }) => (
  <span className="flex items-center gap-1.5">
    <span className={profile?.connected?.includes('tiktok') ? 'text-ink' : 'text-muted opacity-40'}>
      <TikTokIcon size={size} />
    </span>
    <span className={profile?.connected?.includes('instagram') ? 'text-ink' : 'text-muted opacity-40'}>
      <Instagram size={size} />
    </span>
    <span className={profile?.connected?.includes('youtube') ? 'text-ink' : 'text-muted opacity-40'}>
      <Youtube size={size} />
    </span>
  </span>
);

const UserProfileSelector = ({ profiles, selectedUserId, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false);

  if (!profiles || profiles.length === 0) return null;

  const selectedProfile = profiles.find(p => p.username === selectedUserId) || profiles[0];
  const autoId = isAutoProfileId(selectedProfile?.username);

  return (
    <div className="relative z-50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between bg-paper2 border border-rule2 rounded-input px-3 py-2 text-sm text-ink2 hover:bg-paper3 transition-colors min-w-[180px]"
      >
        <span className="flex items-center gap-2">
          <div className="w-5 h-5 rounded-full bg-paper3 border border-rule flex items-center justify-center font-mono text-micro text-brass">
            {autoId ? "S" : (selectedProfile?.username?.substring(0, 1).toUpperCase() || "U")}
          </div>
          {autoId ? (
            <ProfileNetworkIcons profile={selectedProfile} size={13} />
          ) : (
            <span className="font-medium text-ink truncate max-w-[100px]">{selectedProfile?.username || "Select User"}</span>
          )}
        </span>
        <ChevronDown size={14} className={`text-muted transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute top-full mt-2 right-0 w-64 card overflow-hidden">
          <div className="max-h-60 overflow-y-auto custom-scrollbar">
            {profiles.map((profile) => (
              <button
                key={profile.username}
                onClick={() => {
                  onSelect(profile.username);
                  setIsOpen(false);
                }}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-paper3 transition-colors text-left group border-b border-rule last:border-0"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-paper3 flex items-center justify-center font-mono text-micro text-ink border border-rule shrink-0">
                    {isAutoProfileId(profile.username) ? "S" : profile.username.substring(0, 2).toUpperCase()}
                  </div>
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-ink2 group-hover:text-ink transition-colors truncate">
                      {isAutoProfileId(profile.username)
                        ? `Social profile ${profiles.indexOf(profile) + 1}`
                        : profile.username}
                    </div>
                    <div className="flex gap-2 mt-0.5">
                      {/* Status indicators */}
                      <div className={`flex items-center gap-1 ${profile.connected.includes('tiktok') ? 'text-ink2' : 'text-muted opacity-40'}`}>
                        <TikTokIcon size={10} />
                      </div>
                      <div className={`flex items-center gap-1 ${profile.connected.includes('instagram') ? 'text-ink2' : 'text-muted opacity-40'}`}>
                        <Instagram size={10} />
                      </div>
                      <div className={`flex items-center gap-1 ${profile.connected.includes('youtube') ? 'text-ink2' : 'text-muted opacity-40'}`}>
                        <Youtube size={10} />
                      </div>
                    </div>
                  </div>
                </div>
                {selectedUserId === profile.username && <Check size={14} className="text-brass shrink-0" />}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Human "2m 30s" style readout for the ETA returned by /api/status.
const formatEta = (seconds) => {
  const s = Number(seconds);
  if (!Number.isFinite(s) || s < 0) return null;
  if (s < 60) return `${Math.max(1, Math.round(s))}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return rem ? `${m}m ${rem}s` : `${m}m`;
};

function App() {
  // Cloud auth/billing session (inert when billing is disabled).
  const { billingEnabled, isManaged, isSignedIn, me, plan, refreshMe } = useAuth();
  const [showLogin, setShowLogin] = useState(false);
  const [showTopUp, setShowTopUp] = useState(false);
  const [showPlanChoice, setShowPlanChoice] = useState(false);
  const [showTrialUpgrade, setShowTrialUpgrade] = useState(false);
  const [topUpInfo, setTopUpInfo] = useState({});

  // Header search: filters the Generated Shorts rail / History library by
  // title (real, client-side substring match on what /api/history already
  // returns — not a placeholder box that does nothing when you type in it).
  const [historySearch, setHistorySearch] = useState('');
  const searchInputRef = useRef(null);
  useEffect(() => {
    const onKeydown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', onKeydown);
    return () => window.removeEventListener('keydown', onKeydown);
  }, []);

  const [apiKey, setApiKey] = useState(localStorage.getItem('gemini_key') || '');
  // Social API State - Load encrypted or plain
  const [uploadPostKey, setUploadPostKey] = useState(() => {
    const stored = localStorage.getItem('uploadPostKey_v3');
    if (stored) return decrypt(stored);
    return '';
  });
  // ElevenLabs API State - Load encrypted
  const [elevenLabsKey, setElevenLabsKey] = useState(() => {
    const stored = localStorage.getItem('elevenLabsKey_v1');
    if (stored) return decrypt(stored);
    return '';
  });

  // AssemblyAI API State (transcription backend) - Load encrypted
  const [assemblyaiKey, setAssemblyaiKey] = useState(() => {
    const stored = localStorage.getItem('assemblyaiKey_v1');
    if (stored) return decrypt(stored);
    return '';
  });

  // Extra Gemini keys beyond the primary one above — a small pool spread
  // across the picker + scene-direction calls so a single key's rate limit
  // doesn't become the bottleneck. Stored as encrypted JSON.
  const [geminiExtraKeys, setGeminiExtraKeys] = useState(() => {
    const stored = localStorage.getItem('geminiExtraKeys_v1');
    if (!stored) return [];
    try {
      const parsed = JSON.parse(decrypt(stored));
      return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
      return [];
    }
  });

  const [uploadUserId, setUploadUserId] = useState(() => localStorage.getItem('uploadUserId') || '');
  const [userProfiles, setUserProfiles] = useState([]); // List of {username, connected: []}
  const [showKeyModal, setShowKeyModal] = useState(false);
  // The workspace no longer owns job state — ProjectContext does, so several
  // projects can run at once and none of them is lost by switching tabs.
  const {
    list: projectList, active, activeId: jobId, liveCount, setActiveId, patch: patchProject,
    startProject, openProject: restoreProject, inspectProject, cancelProject,
    closeProject, updateClipState, flushClipState, hydrateDurable, setRawLogs,
  } = useProjects();

  const status = active?.status ?? 'idle';
  // A job the server reports as `queued` (waiting for a concurrency slot)
  // matched NEITHER the launcher (idle) nor the workspace, so the screen went
  // blank until it started running. Anything that isn't idle has a project to
  // show.
  const hasProject = status !== 'idle' && !!active;
  // Waiting for a free concurrency slot is work in progress as far as the
  // screen is concerned — the stage readout still says "queued".
  const viewStatus = status === 'queued' ? 'processing' : status;
  const results = active?.results ?? null;
  const logs = active?.logs ?? [];
  const progress = active?.progress ?? null;
  const stageDurations = active?.stageDurations ?? null;
  const projectState = active?.projectState ?? null;
  const processingMedia = active?.source ?? null;
  const requestedClipCount = active?.requestedClipCount ?? null;
  const submittedFormat = active?.submittedFormat ?? 'auto';
  const durableClips = active?.durableClips ?? {};

  // Local mutations of the active project, by the names the JSX already uses.
  const setResults = (v) => jobId && patchProject(jobId, { results: v });
  const setStatus = (v) => jobId && patchProject(jobId, { status: v });
  const setLogs = (v) => jobId && patchProject(jobId, (cur) => ({
    logs: typeof v === 'function' ? v(cur.logs || []) : v,
  }));

  // The gap between pressing generate and the server handing back a job id —
  // an upload can take a while, and the screen must not look idle meanwhile.
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState('');
  // Which clip is open as a full editor card while the job is still running.
  const [expandedClip, setExpandedClip] = useState(null);
  const [navOpen, toggleNav] = usePanelState('nav', false);
  // Bulk subtitles: apply one style to every clip of the job (triggered from
  // within a clip's subtitle modal via "apply to all").
  const [bulkSub, setBulkSub] = useState({ running: false, current: 0, total: 0, errors: 0 });
  const [downloadingAll, setDownloadingAll] = useState(false);
  // Pre-flight quality gate: { info: {max_height, min_height, cookies_invalid}, data }
  const [qualityGate, setQualityGate] = useState(null);
  // Live pipeline progress from main.py's progress.json: {stage, overall_pct,
  // clips_done, clips_total}. null until the backend reports a snapshot.
  // Telemetry: raw-log toggle + per-stage averages for the performance chart.
  const [logsRaw, setLogsRaw] = useState(false);
  // Right-panel view: 'clips' (Generated Shorts) or 'source' (original video
  // + stills — round-5 Source section).
  // How the generated shorts are laid out: 'row' slides sideways, 'column'
  // stacks down. Remembered, because it is a working preference.
  const [clipView, setClipView] = useStoredChoice('clip-view', 'row');
  // Itemized system status for the status strip: {backend, cookies, gpu} from
  // /api/system, refreshed on an interval. null = never checked yet.
  const [systemStatus, setSystemStatus] = useState(null);
  // The clip count the user requested at submit — sizes the placeholder
  // slots in the live grid before progress.json knows the total. Always an
  // explicit number; auto mode was removed.
  // Sidebar "Today" mini-stats, derived from the disk-backed history.
  const [todayStats, setTodayStats] = useState({ generated: 0, processing: 0, successRate: 100, loaded: false });
  const [activeTab, setActiveTab] = useState('dashboard'); // dashboard, settings
  // Reopened-project state (paid mode): per-clip {index, server_file, active_layers}
  // restored from the backend so ResultCards resume editing where they left off.

  const [showScheduleWeek, setShowScheduleWeek] = useState(false);

  // Silent-success "saved" states for the settings key inputs (design.md: no alert popups)
  const [elevenLabsSaved, setElevenLabsSaved] = useState(false);
  const [assemblyaiSaved, setAssemblyaiSaved] = useState(false);

  // Clip → source playback sync lives in lib/sourceSync.js, not in App
  // state: the playing clip publishes its position several times a second,
  // and routing that through a top-level setState re-rendered every result
  // card on every tick. The preview panel subscribes to the bus directly.

  // A clip picked from the right rail opens BIG next to the source preview
  // instead of playing inside a 200px card. When it belongs to the job on
  // screen we also know its offset in the source, so the two play in step.
  const [comparedId, setComparedId] = useState(null);
  // The bus is the single source of truth for what is being compared, so
  // closing the pane from the preview panel un-highlights the rail row too.
  useEffect(() => subscribeSourceSync((s) => setComparedId(s.clip?.id ?? null)), []);
  const handleCompareClip = (v) => {
    if (!v?.view_url) return;
    if (comparedId === v.id) {          // clicking it again closes it
      clearCompareClip();
      return;
    }
    const index = Number(String(v.id).split('_').pop());
    const own = v.job_id === jobId ? results?.clips?.[index] : null;
    setCompareClip({
      id: v.id,
      url: getApiUrl(v.view_url),
      title: v.title || 'clip',
      // Only a clip of the job on screen can drive the source: for any other
      // job the preview is a different video entirely, so it just plays
      // alongside rather than pretending to be in sync.
      start: own?.start ?? null,
    });
  };

  // --- Project persistence (paid mode) ---
  // The debounce + flush now live in ProjectContext, so an edit made in one
  // project can't be dropped by switching to another mid-save.
  const handleClipStateChange = (index, state) => {
    if (!isManaged || !jobId) return;
    updateClipState(jobId, index, state);
  };

  // Open any project from the rail or History — including a failed one. A
  // failed job has no restorable state (the pipeline never wrote metadata), so
  // it opens read-only with whatever the backend still knows, rather than
  // being a dead tile you can only delete.
  const handleOpenProject = async (v) => {
    const targetId = typeof v === 'string' ? v : v?.job_id;
    if (!targetId) return;
    const meta = typeof v === 'string' ? {} : v;
    setActiveTab('dashboard');
    flushClipState();
    clearCompareClip();
    setQualityGate(null);
    // Already in the workspace (a run started this session, or one opened
    // earlier): just look at it. Re-restoring a live job would stomp its
    // in-flight logs and progress with a stale snapshot.
    if (projectList.some((p) => p.id === targetId)) {
      setActiveId(targetId);
      return;
    }
    if (meta.status === undefined || meta.status === 'completed') {
      try {
        await restoreProject(targetId, meta);
        return;
      } catch (e) {
        // Not restorable — fall through and inspect it instead.
      }
    }
    await inspectProject(targetId, meta);
  };

  // Apply one subtitle style to every clip of the job, sequentially.
  const handleBulkSubtitles = async (options) => {
    const clips = results?.clips || [];
    const total = clips.length;
    if (!total) return;
    setBulkSub({ running: true, current: 0, total, errors: 0 });
    let errors = 0;
    for (let i = 0; i < total; i++) {
      setBulkSub({ running: true, current: i + 1, total, errors });
      try {
        const res = await apiFetch('/api/subtitle', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_id: jobId,
            clip_index: i,
            position: options.position,
            font_size: options.fontSize,
            font_name: options.fontName,
            font_color: options.fontColor,
            border_color: options.borderColor,
            border_width: options.borderWidth,
            bg_color: options.bgColor,
            bg_opacity: options.bgOpacity,
            style: options.style || 'classic',
            highlight_color: options.highlightColor || '#FFD700',
            effect: options.effect || 'none',
            base_opacity: options.baseOpacity ?? 1.0,
            uppercase: options.uppercase || false,
            // Chain from the clip's current server file (its video_url basename).
            input_filename: (clips[i].video_url || '').split('/').pop(),
          }),
        });
        if (!res.ok) errors++;
      } catch {
        errors++;
      }
    }
    setBulkSub({ running: false, current: total, total, errors });
    // Refresh results so each ResultCard picks up its new subtitled video_url.
    try {
      const data = await apiJson(`/api/status/${jobId}`);
      if (data.result) setResults(data.result);
    } catch { /* keep current results */ }
  };

  const [cancelling, setCancelling] = useState(false);
  const handleCancelJob = async () => {
    if (!jobId || cancelling) return;
    if (!window.confirm('Stop this job? Clips already rendered will stay, the rest will not be generated.')) return;
    setCancelling(true);
    try {
      await cancelProject(jobId);
    } catch (e) {
      console.error('Cancel failed', e);
    } finally {
      setCancelling(false);
    }
  };

  const handleDownloadAll = async () => {
    if (!jobId) return;
    setDownloadingAll(true);
    try {
      const res = await apiFetch(`/api/jobs/${jobId}/download-all`);
      if (!res.ok) throw new Error(await res.text());
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `openshorts_clips_${(jobId || '').slice(0, 8)}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert(`Download failed: ${e.message}`);
    } finally {
      setDownloadingAll(false);
    }
  };

  // Session recovery is gone: ProjectContext persists every project (and
  // keeps polling the live ones), so a reload or a tab switch resumes on its
  // own instead of restoring one job from a special-case snapshot.

  // True persistence: whenever the dashboard re-mounts a completed job
  // (tab switch, view re-mount), re-fetch the clips from the backend's disk
  // so they always come back — no disappearing-until-restart.
  useEffect(() => {
    if (activeTab === 'dashboard' && status === 'complete' && jobId) {
      apiJson(`/api/jobs/${jobId}/result`)
        .then((d) => { if (d?.clips?.length) setResults(d); })
        .catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  useEffect(() => {
    // Encrypt Gemini Key too for consistency if desired, but user asked specifically about Social integration not saving well.
    // For now keeping gemini plain for compatibility unless requested.
    if (apiKey) localStorage.setItem('gemini_key', apiKey);
  }, [apiKey]);

  useEffect(() => {
    if (uploadPostKey) {
      localStorage.setItem('uploadPostKey_v3', encrypt(uploadPostKey));
    }
    if (uploadUserId) {
      localStorage.setItem('uploadUserId', uploadUserId);
    }
  }, [uploadPostKey, uploadUserId]);

  useEffect(() => {
    if (elevenLabsKey) {
      localStorage.setItem('elevenLabsKey_v1', encrypt(elevenLabsKey));
    }
  }, [elevenLabsKey]);

  useEffect(() => {
    if ((uploadPostKey || isManaged) && userProfiles.length === 0) {
      fetchUserProfiles({ silent: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadPostKey, isManaged]);

  // Durable storage URLs for the active project's clips (managed accounts),
  // so a preview still plays once the ephemeral local file is cleaned up.
  useEffect(() => {
    if (!isManaged || !jobId || !(results?.clips?.length)) return;
    hydrateDurable(jobId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isManaged, jobId, results?.clips?.length]);

  // Polling lives in ProjectContext — one timer for EVERY live project, so a
  // second run is not a second interval and switching projects doesn't stop
  // the one you left. The raw-log preference is per project.
  useEffect(() => {
    if (jobId) setRawLogs(jobId, logsRaw);
  }, [jobId, logsRaw, setRawLogs]);

  // Clip indexes belong to one project; carrying an expansion across a switch
  // would open a card for a clip that isn't there.
  useEffect(() => { setExpandedClip(null); }, [jobId]);

  // System status strip: real, itemized checks (backend reachability, YouTube
  // cookie freshness, GPU) — refreshed every 30s while the dashboard is open.
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await apiFetch('/api/system');
        if (!cancelled && res.ok) setSystemStatus(await res.json());
      } catch (_) {
        if (!cancelled) setSystemStatus({ backend: false });
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [activeTab]);

  // Sidebar "Today" stats: clips generated today, jobs still processing, and
  // the completed-vs-failed success rate — all from /api/history.
  //
  // This used to fetch exactly once per jobId, which is why the sidebar could
  // sit at "processing 0" while a job was visibly running in the main panel:
  // the single fetch fired before the backend had written the job to history,
  // and nothing ever re-read it. It now re-polls while a job is live, and the
  // rendered value is reconciled with the locally-known job below, so the
  // sidebar and the main panel can never disagree about the same job.
  useEffect(() => {
    let cancelled = false;
    const load = () => {
      apiFetch('/api/history')
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (cancelled || !d) return;
          const videos = d.videos || [];
          const today = new Date().toDateString();
          const todayCount = videos.filter(
            (v) => new Date(v.created_at).toDateString() === today).length;
          const byStatus = {};
          for (const v of videos) {
            const s = v.status || 'completed';
            byStatus[s] = (byStatus[s] || 0) + 1;
          }
          const done = byStatus.completed || 0;
          const failed = byStatus.failed || 0;
          const rate = done + failed > 0 ? Math.round(100 * done / (done + failed)) : 100;
          setTodayStats({
            generated: todayCount,
            processing: byStatus.processing || 0,
            successRate: rate,
            loaded: true,
          });
        })
        .catch(() => {});
    };
    load();
    // Keep counts live for the duration of a run, and pick up the terminal
    // state right after it ends, without the user refreshing anything.
    const interval = liveCount > 0 ? setInterval(load, 5000) : null;
    return () => { cancelled = true; if (interval) clearInterval(interval); };
  }, [jobId, status]);


  // silent: background auto-fetch — never alert(), just log. Managed users need
  // no local key (the server resolves its own); BYOK sends the header.
  const fetchUserProfiles = async ({ silent = false } = {}) => {
    if (!uploadPostKey && !isManaged) return;
    try {
      const res = await apiFetch('/api/social/user', {
        headers: uploadPostKey ? { 'X-Upload-Post-Key': uploadPostKey } : {}
      });
      if (!res.ok) throw new Error("Failed to fetch");
      const data = await res.json();
      if (data.profiles && data.profiles.length > 0) {
        setUserProfiles(data.profiles);
        // Auto select first if none selected
        if (!uploadUserId) {
          setUploadUserId(data.profiles[0].username);
        }
      } else if (!silent) {
        alert("No profiles found for this API Key.");
      }
    } catch (e) {
      if (!silent) alert("Error fetching User Profiles. Please check key.");
      console.error(e);
    }
  };

  // Hosted is paid-only (no BYOK core). Self-host uses BYOK keys.
  // `keysMissing` means "self-host BYOK keys missing" — it never fires on hosted.
  // Upload-Post is only needed for social auto-publishing, not clip generation,
  // so it doesn't gate the main flow — just the Gemini key does.
  // The server may already hold a Gemini key (a .env on a self-host, Kaggle
  // Secrets on a Kaggle run). resolve_gemini() in app.py falls back to the
  // environment, so the backend works — the browser-side gate was the only
  // thing demanding a key, and it made a correctly-configured Kaggle host ask
  // for one it was already using. /api/system reports presence only, never a
  // value. A locally-entered key still wins as an override.
  const serverHasGeminiKey = !!systemStatus?.server_keys?.gemini;
  const keysMissing = !billingEnabled && !apiKey && !serverHasGeminiKey;
  const needsPlan = billingEnabled && !isManaged;   // hosted, signed-out or no active plan/trial

  // Fresh sign-up: show the welcome plan-choice popup once (AuthContext set the
  // flag after the auth redirect). Fires for free users too, so it's gated on
  // being signed in rather than on entitlement.
  useEffect(() => {
    if (billingEnabled && isSignedIn) {
      let flagged = false;
      try { flagged = localStorage.getItem('os_show_plan_choice') === '1'; } catch (_) { /* ignore */ }
      if (flagged) {
        setShowPlanChoice(true);
        try { localStorage.removeItem('os_show_plan_choice'); } catch (_) { /* ignore */ }
      }
    }
  }, [billingEnabled, isSignedIn]);
  // Included in the plan (fully managed, no keys): Clip Generator + YouTube Studio.
  // Advanced (bring your own fal.ai + ElevenLabs keys): AI Shorts + AI Agent.
  const INCLUDED_TOOL_TABS = ['dashboard', 'thumbnails'];
  const TOOL_NAMES = { dashboard: 'clip generation', thumbnails: 'the YouTube Studio' };
  const gateThisTab = needsPlan && INCLUDED_TOOL_TABS.includes(activeTab);      // included tool, no plan yet

  // Managed users connect their socials via Upload-Post's branded hosted page.
  const handleConnectSocials = async () => {
    try {
      const { access_url } = await apiJson('/api/social/connect', { method: 'POST' });
      // Same tab so the connect page's redirectUrl brings the user back into the app.
      if (access_url) window.location.href = access_url;
    } catch (e) {
      alert('Could not open the connection page. Please try again.');
    }
  };

  // Open the Upload-Post white-label page (which includes the scheduling calendar)
  // in a new tab, for consulting/managing scheduled posts from the dashboard.
  const handleOpenCalendar = async () => {
    try {
      const { access_url } = await apiJson('/api/social/connect', { method: 'POST' });
      if (access_url) window.open(access_url, '_blank', 'noopener');
    } catch (e) {
      alert('Could not open the calendar. Please try again.');
    }
  };

  const handleProcess = async (data, forceLowQuality = false) => {
    // Hosted: must be signed in AND on an active plan/trial. Self-host: BYOK keys.
    if (billingEnabled) {
      if (!isSignedIn) { setShowLogin(true); return; }
      if (!isManaged) { window.location.hash = '#/pricing'; return; }
    } else if (keysMissing) {
      setShowKeyModal(true);
      return;
    }
    setQualityGate(null);
    setRightTab('clips');
    setStarting(true);

    // BYOK sends the Gemini header; managed users rely on the bearer token
    // apiFetch attaches. AssemblyAI + the extra Gemini pool are optional — the
    // pipeline falls back (Whisper / a single key) when they are unset.
    const headers = apiKey ? { 'X-Gemini-Key': apiKey } : {};
    if (assemblyaiKey) headers['X-AssemblyAI-Key'] = assemblyaiKey;
    if (geminiExtraKeys.length > 0) headers['X-Gemini-Keys'] = geminiExtraKeys.join(',');

    try {
      const out = await startProject(data, {
        headers,
        force: forceLowQuality,
        parentId: data.parentId || null,
      });
      // Pre-flight gate: the source is below the minimum resolution. Ask
      // before burning twenty minutes on it; confirming resends with
      // force_low_quality.
      if (out.needsConfirmation) {
        setQualityGate({ info: out.qualityCheck, data });
        return;
      }
      setActiveTab('dashboard');
    } catch (e) {
      if (e instanceof QuotaError) {
        // Trial users hit the trial cap → offer activation (full minutes).
        // Active users → offer a top-up.
        if (me?.status === 'trialing') setShowTrialUpgrade(true);
        else {
          setTopUpInfo({ required: e.minutesRequired, remaining: e.minutesRemaining });
          setShowTopUp(true);
        }
        return;
      }
      setStartError(e.message || 'Could not start this job.');
    } finally {
      setStarting(false);
    }
  };

  // "New project" no longer throws work away: it clears the WORKSPACE
  // selection, and the project itself stays in the drawer (and on the server)
  // to come back to.
  const handleReset = () => {
    flushClipState();
    setActiveId(null);
    setRightTab('clips');
    setStartError('');
    clearCompareClip();
  };

  // Every project the workspace knows about, in the shape the rail renders.
  // Plural on purpose: more than one can be running at a time.
  const railProjects = projectList.map((p) => ({
    id: p.id,
    title: p.title,
    status: p.status,
    pct: p.progress?.overall_pct ?? null,
    createdAt: p.createdAt,
  }));

  // --- UI Components ---

  return (
    <div className="flex h-screen bg-paper overflow-hidden">
      <Sidebar
        activeTab={activeTab}
        onSelect={setActiveTab}
        billingEnabled={billingEnabled}
        isSignedIn={isSignedIn}
        todayStats={todayStats}
        liveCount={liveCount}
        open={navOpen}
        onToggle={toggleNav}
      />

      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Top Header */}
        <header className="h-14 border-b border-rule bg-paper flex items-center gap-4 px-6 shrink-0 z-10">
          {status !== 'idle' && (
            <button
              onClick={handleReset}
              className="btn-quiet px-3 py-1.5 text-xs shrink-0"
              title="Back to your projects — this one keeps running"
            >
              <Plus size={14} />
              <span className="hidden sm:inline">New project</span>
            </button>
          )}

          <div className="relative flex-1 max-w-md">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
            <input
              ref={searchInputRef}
              type="text"
              value={historySearch}
              onChange={(e) => setHistorySearch(e.target.value)}
              placeholder="Search projects, clips…"
              className="w-full bg-paper2 border border-rule rounded-full pl-9 pr-12 py-1.5 text-xs text-ink placeholder:text-muted focus:outline-none focus:border-rule2 transition-colors"
            />
            <kbd className="hidden sm:flex items-center gap-0.5 absolute right-2.5 top-1/2 -translate-y-1/2 text-[10px] text-muted font-mono border border-rule rounded px-1.5 py-0.5 pointer-events-none">
              &#8984;K
            </kbd>
          </div>

          <div className="flex-1" />

          {/* One chip that opens into the machine's live load. */}
          <div className="shrink-0">
            <SystemMonitor status={systemStatus} />
          </div>

          <div className="flex items-center gap-4">
            {userProfiles.length > 0 && (
              <UserProfileSelector
                profiles={userProfiles}
                selectedUserId={uploadUserId}
                onSelect={setUploadUserId}
              />
            )}

            {/* Cloud: minutes meter + account/sign-in. For free users the meter
                opens the upgrade modal — otherwise the only path to a plan is
                failing against the quota wall. */}
            {billingEnabled && isManaged && (
              <UsageMeter onClick={() => {
                if (plan === 'free') { setTopUpInfo({ context: 'upsell' }); setShowTopUp(true); }
                else { window.location.hash = '#/account'; }
              }} />
            )}
            {billingEnabled && isSignedIn && !isManaged && (
              <button onClick={() => setShowPlanChoice(true)}
                className="btn-primary px-4 py-2 text-xs">
                Choose a plan
              </button>
            )}
            {billingEnabled && !isSignedIn && (
              <button onClick={() => setShowLogin(true)}
                className="btn-ghost px-4 py-2 text-xs">
                Sign in
              </button>
            )}
            {billingEnabled && isSignedIn && <ProfileMenu />}

            {/* Same pill geometry as the system-status strip so the top bar
                reads as one row of instruments, not a badge glued next to
                three tiles. Actionable, so it keeps a warm border + arrow. */}
            {keysMissing && (
              <button
                onClick={() => (billingEnabled && !isSignedIn ? setShowLogin(true) : setActiveTab('settings'))}
                className="group flex items-center gap-2 h-9 pl-2.5 pr-3 rounded-lg border bg-paper2/80
                           transition-colors duration-200 hover:bg-paper3"
                style={{ borderColor: 'color-mix(in oklab, var(--color-warn) 38%, var(--color-rule))' }}
                title="Configure API keys or choose a plan"
              >
                <AlertTriangle size={14} strokeWidth={1.75} className="shrink-0 text-warn" />
                <span className="min-w-0 leading-none text-left">
                  <span className="block readout text-[8px] uppercase tracking-[0.12em] text-muted/80">
                    Gemini key
                  </span>
                  <span className="block text-[11px] font-medium text-warn truncate mt-0.5">
                    Missing — set up
                  </span>
                </span>
                <ChevronRight
                  size={13}
                  className="shrink-0 text-muted transition-transform duration-200 group-hover:translate-x-0.5"
                />
              </button>
            )}
          </div>
        </header>

        {/* Persistent Missing Keys Banner — visible on every screen */}
        {keysMissing && activeTab !== 'settings' && (
          <div className="mx-4 sm:mx-6 mt-3 px-4 py-3 bg-paper2 border border-rule rounded-card flex flex-wrap items-center justify-between gap-3 sm:gap-4 shrink-0 animate-fade">
            <div className="flex items-center gap-3 text-sm text-ink2">
              <KeyRound size={16} className="shrink-0 text-warn" />
              <div>
                <span className="font-medium text-ink">Required API keys missing.</span>{' '}
                <span className="text-muted">Set your Gemini API key to start generating.</span>
              </div>
            </div>
            <button
              onClick={() => setActiveTab('settings')}
              className="btn-quiet px-3 py-1.5 text-xs shrink-0"
            >
              Go to Settings
            </button>
          </div>
        )}

        {/* Included tools (Clip Generator, YouTube Studio): non-blocking trial prompt. */}
        {gateThisTab && <TrialGate toolName={TOOL_NAMES[activeTab] || 'this'} />}

        {/* Main Workspace */}
        <div className="flex-1 overflow-hidden relative">

          {/* View: Settings */}
          {activeTab === 'settings' && (
            <div className="h-full overflow-y-auto p-4 sm:p-8 max-w-2xl mx-auto animate-fade">
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-8">
                <div>
                  <h1 className="font-display lowercase text-2xl text-ink">Settings</h1>
                </div>
                <div className="flex items-center gap-2 text-xs text-muted mt-1">
                  <Shield size={12} className="text-ok shrink-0" /> Privacy: keys only live in your browser (sent to backend just to process)
                </div>
              </div>
              {isManaged ? (
                <div className="card p-6 mb-2">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                        <Shield size={16} className="text-brass" />
                      </div>
                      <h2 className="text-base font-medium text-ink lowercase">Included in your plan</h2>
                    </div>
                    <span className="badge-ok">Managed</span>
                  </div>
                  <p className="text-xs text-muted mb-5 leading-relaxed">
                    Your plan includes the <strong>Clip Generator</strong> and <strong>YouTube Studio</strong>,
                    fully managed — no API keys required. AI Shorts &amp; dubbing use your own fal.ai / ElevenLabs
                    keys (below). Connect your social accounts to publish directly.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <button onClick={handleConnectSocials} className="btn-primary py-2 px-4 text-sm">
                      <Share2 size={16} /> Connect social accounts
                    </button>
                    <button onClick={handleOpenCalendar} className="btn-quiet py-2 px-4 text-sm">
                      <Calendar size={16} /> Content calendar
                    </button>
                  </div>
                </div>
              ) : billingEnabled ? (
                <div className="card p-6 mb-2">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                        <Sparkles size={16} className="text-brass" />
                      </div>
                      <h2 className="text-base font-medium text-ink lowercase">Choose your plan</h2>
                    </div>
                    <span className="badge-ok">Free plan available</span>
                  </div>
                  <p className="text-xs text-muted mb-5 leading-relaxed">
                    Generate shorts with zero setup — no API keys needed. Start free with 20 min/month, or go paid from $12/mo. Cancel anytime.
                  </p>
                  <button onClick={() => setShowPlanChoice(true)} className="btn-primary py-2 px-4 text-sm">
                    <Sparkles size={16} /> Choose a plan
                  </button>
                </div>
              ) : (
                <>
              <KeyInput onKeySet={setApiKey} savedKey={apiKey} />

              <div className="card p-4 sm:p-6 mt-8">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                      <Bot size={16} className="text-brass" />
                    </div>
                    <h2 className="text-base font-medium text-ink lowercase">Clip Intelligence</h2>
                  </div>
                  <span className="readout">BETA · BYOK</span>
                </div>
                <p className="text-xs text-muted mb-6 leading-relaxed">
                  Narrative-aware clip selection: <strong>AssemblyAI</strong> transcribes, <strong>DeepSeek</strong> finds
                  moments that carry a hook through to its payoff, and Gemini Vision confirms each candidate against the
                  actual footage. All optional — the pipeline falls back to Whisper/Gemini-only if these are unset.
                </p>

                <div className="space-y-6">
                  <div>
                    <label className="block text-sm text-muted mb-2">AssemblyAI API Key</label>
                    <div className="flex flex-col sm:flex-row gap-2">
                      <input
                        type="password"
                        value={assemblyaiKey}
                        onChange={(e) => setAssemblyaiKey(e.target.value)}
                        className="input-field"
                        placeholder="assemblyai key..."
                      />
                      <button
                        onClick={() => {
                          if (assemblyaiKey) {
                            localStorage.setItem('assemblyaiKey_v1', encrypt(assemblyaiKey));
                            setAssemblyaiSaved(true);
                            setTimeout(() => setAssemblyaiSaved(false), 2000);
                          }
                        }}
                        className={assemblyaiSaved ? 'badge-ok px-4' : 'btn-quiet py-2 px-4 text-sm'}
                      >
                        {assemblyaiSaved ? <><Check size={12} /> saved</> : 'Save'}
                      </button>
                    </div>
                    <p className="mt-2 text-xs text-muted">
                      <a href="https://www.assemblyai.com/dashboard/signup" target="_blank" rel="noopener noreferrer" className="text-brass hover:underline">
                        Get your AssemblyAI API key →
                      </a>
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm text-muted mb-2">
                      Extra Gemini keys <span className="text-muted">(optional — spreads picker + scene-direction calls across a pool so one key's rate limit isn't the bottleneck)</span>
                    </label>
                    <KeyListInput
                      keys={geminiExtraKeys}
                      onSave={(keys) => {
                        localStorage.setItem('geminiExtraKeys_v1', encrypt(JSON.stringify(keys)));
                        setGeminiExtraKeys(keys);
                      }}
                    />
                  </div>
                </div>

                <p className="mt-6 text-xs text-muted leading-relaxed">
                  Keys are only stored in your browser. They are sent to the backend only to process your request, never stored server-side.
                </p>
              </div>

              <div className="card p-4 sm:p-6 mt-8">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                      <Share2 size={16} className="text-brass" />
                    </div>
                    <h2 className="text-base font-medium text-ink lowercase">Social Integration</h2>
                  </div>
                  <span className="badge-warn">Required</span>
                </div>
                <p className="text-xs text-muted mb-6 leading-relaxed">
                  Required to publish your clips to TikTok, Instagram Reels, and YouTube Shorts via <strong>Upload-Post</strong>.
                  Includes a <strong>free tier</strong> (no credit card required).
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-muted">Upload-Post API Key</label>
                  <div className="flex flex-col sm:flex-row gap-2">
                    <input
                      type="password"
                      value={uploadPostKey}
                      onChange={(e) => setUploadPostKey(e.target.value)}
                      className="input-field"
                      placeholder="ey..."
                    />
                    <button onClick={fetchUserProfiles} className="btn-quiet py-2 px-4 text-sm">
                      Connect
                    </button>
                  </div>
                  <div className="text-xs text-muted leading-relaxed">
                    Connect your Upload-Post account to enable one-click publishing.
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-2">
                      <a href="https://app.upload-post.com/login" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">1. Login</span>
                        <span className="text-xs text-muted">Register account</span>
                      </a>
                      <a href="https://app.upload-post.com/manage-users" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">2. Profiles</span>
                        <span className="text-xs text-muted">Create & Connect</span>
                      </a>
                      <a href="https://app.upload-post.com/api-keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">3. API Key</span>
                        <span className="text-xs text-muted">Generate key</span>
                      </a>
                    </div>
                    <br />
                    <span className="text-muted">
                      Keys are only stored in your browser. They are sent to the backend only to process your request, never stored server-side.
                    </span>
                  </div>
                </div>
              </div>

                </>
              )}

              <div className="card p-4 sm:p-6 mt-8">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                      <Globe size={16} className="text-brass" />
                    </div>
                    <h2 className="text-base font-medium text-ink lowercase">Video Translation</h2>
                  </div>
                  <span className="readout">BYOK</span>
                </div>
                <p className="text-xs text-muted mb-6 leading-relaxed">
                  For <strong>AI Shorts &amp; dubbing</strong> — bring your own key. Translate your clips to different
                  languages using <strong>ElevenLabs</strong> AI dubbing (billed by ElevenLabs). Not covered by your plan.
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-muted">ElevenLabs API Key</label>
                  <div className="flex flex-col sm:flex-row gap-2">
                    <input
                      type="password"
                      value={elevenLabsKey}
                      onChange={(e) => setElevenLabsKey(e.target.value)}
                      className="input-field"
                      placeholder="sk_..."
                    />
                    <button
                      onClick={() => {
                        if (elevenLabsKey) {
                          localStorage.setItem('elevenLabsKey_v1', encrypt(elevenLabsKey));
                          setElevenLabsSaved(true);
                          setTimeout(() => setElevenLabsSaved(false), 2000);
                        }
                      }}
                      className={elevenLabsSaved ? 'badge-ok px-4' : 'btn-quiet py-2 px-4 text-sm'}
                    >
                      {elevenLabsSaved ? <><Check size={12} /> saved</> : 'Save'}
                    </button>
                  </div>
                  <div className="text-xs text-muted leading-relaxed">
                    Get your API key from ElevenLabs to enable video translation.
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <a href="https://elevenlabs.io/sign-up" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">1. Sign Up</span>
                        <span className="text-xs text-muted">Create account</span>
                      </a>
                      <a href="https://elevenlabs.io/app/settings/api-keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">2. API Key</span>
                        <span className="text-xs text-muted">Generate key</span>
                      </a>
                    </div>
                    <br />
                    <span className="text-muted">
                      Keys are only stored in your browser. They are sent to the backend only to process your request, never stored server-side.
                    </span>
                  </div>
                </div>
              </div>

              {/* fal.ai lived here for AI Shorts, which no longer exists in
                  this build. Storage takes its place — the panel that decides
                  what is kept and the only thing that deletes it. */}
              <StoragePanel onWiped={handleReset} />
            </div>
          )}

          {/* View: History */}
          {activeTab === 'history' && (
            <div className="h-full overflow-y-auto custom-scrollbar animate-fade">
              <div className="max-w-6xl mx-auto p-6 md:p-8">
                <HistoryTab onReopenProject={handleOpenProject} search={historySearch} />
              </div>
            </div>
          )}

          {/* View: Sources (the persistent saved-source library) */}
          {activeTab === 'sources' && (
            <div className="h-full overflow-y-auto custom-scrollbar animate-fade">
              <div className="max-w-6xl mx-auto p-6 md:p-8">
                <SourcesTab />
              </div>
            </div>
          )}

          {activeTab === 'thumbnails' && (
            <ThumbnailStudio geminiApiKey={apiKey} uploadPostKey={uploadPostKey} uploadUserId={uploadUserId} managed={isManaged} />
          )}

          {/* View: no project open — the launcher */}
          {activeTab === 'dashboard' && !hasProject && (
            <div className="flex h-full min-h-0">
              <div className="flex-1 min-w-0 flex flex-col animate-fade">
                <ProjectLauncher
                  projects={railProjects}
                  onOpenProject={(p) => handleOpenProject(p.id)}
                  onSubmit={handleProcess}
                  starting={starting}
                  error={startError}
                />
              </div>
              {/* Folded to its spine here rather than removed: closed on
                  arrival is the point, but there still has to be a way to
                  reach your library from the home screen. */}
              <HomeRail
                onViewAll={() => setActiveTab("history")}
                search={historySearch}
                onOpenProject={handleOpenProject}
                projects={railProjects}
                activeId={jobId}
              />
            </div>
          )}

          {/* View: Processing / Results (Split View) */}
          {activeTab === 'dashboard' && hasProject && (
            <div className="flex h-full min-h-0">
              <div className="flex-1 min-w-0 flex flex-col">
                {/* Processing hero — same gradient-emphasis headline as Home */}
                <div className="px-5 pt-4 pb-1 shrink-0">
                  <p className="eyebrow">live pipeline</p>
                  <h1 className="font-display lowercase text-2xl text-ink tracking-tight">
                    making{' '}
                    <span
                      style={{
                        background: 'linear-gradient(180deg, #f87171 0%, #ef4444 60%, #991b1b 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        backgroundClip: 'text',
                      }}
                    >
                      viral shorts
                    </span>
                  </h1>
                </div>
                {/* Full-width pipeline header (reference layout) — this is the
                    real curved-path/icon-node tracker; it needs the full
                    content width to read properly, not the narrow left
                    column, which only ever showed the cramped `compact` dots
                    variant even after the wide one was built. */}
                <div className="px-5 pt-2 pb-1 shrink-0">
                  <StageTracker
                    stage={progress?.stage}
                    failed={status === 'error' || status === 'cancelled'}
                    complete={status === 'complete'}
                    pct={progress?.overall_pct ?? (status === 'complete' ? 100 : 0)}
                    durations={stageDurations}
                  />
                </div>
                {/* The job-status card spans the FULL content width, as in the
                    reference. Squeezed into a ~60% column it had roughly 250px
                    left for a 4-cell grid plus a 124px ring, so the ring wrapped
                    below the grid and fell outside the visible panel — the
                    "where is my video right now" answer was the thing pushed off
                    screen. Full width fits grid and ring side by side. */}
                <div className="flex-1 min-h-0 flex flex-col gap-4 p-4 overflow-y-auto custom-scrollbar animate-fade">
                {processingMedia && (
                  <ProcessingAnimation
                    media={processingMedia}
                    isComplete={status === 'complete'}
                    status={viewStatus}
                    progress={progress}
                    title={
                      processingMedia.type === 'file'
                        ? (processingMedia.payload?.name || 'Uploaded video')
                        : processingMedia.type === 'url'
                          ? processingMedia.payload
                          : 'Source video'
                    }
                    format={{
                      vertical: '9:16',
                      square: '1:1',
                      horizontal: '16:9',
                      auto: 'auto',
                    }[submittedFormat] || submittedFormat}
                    onCancel={viewStatus === 'processing' ? handleCancelJob : null}
                  />
                )}

              {/* Generated shorts. Collapsible like everything else, and you
                  choose the shape: a row that slides sideways (eight clips
                  without eight screens of scrolling) or a column. The Source
                  toggle that used to live here is gone — there is a dedicated
                  Sources tab for that now. */}
              <CollapsiblePanel
                id="clips"
                title="generated shorts"
                subtitle={results?.clips?.length ? `${results.clips.length} ready` : null}
                className="shrink-0"
                bodyClassName="p-4 sm:p-5 pt-0"
                right={(
                  <>
                    <div className="flex items-center rounded-full border border-rule overflow-hidden">
                      {[
                        { id: 'row', Icon: Rows3, title: 'slide sideways' },
                        { id: 'column', Icon: Columns3, title: 'stack down' },
                      ].map(({ id, Icon, title }) => (
                        <button
                          key={id}
                          onClick={() => setClipView(id)}
                          title={title}
                          className={`px-2 py-1.5 transition-colors ${
                            clipView === id ? 'bg-brass/15 text-ink' : 'text-muted hover:text-ink2'
                          }`}
                        >
                          <Icon size={13} className={id === 'row' ? 'rotate-90' : ''} />
                        </button>
                      ))}
                    </div>
                    {results?.cost_analysis && !isManaged && (
                      <span className="readout text-[10px] text-muted hidden sm:inline" title={`Input: ${results.cost_analysis.input_tokens} | Output: ${results.cost_analysis.output_tokens}`}>
                        ${results.cost_analysis.total_cost.toFixed(4)}
                      </span>
                    )}
                    {results?.clips?.length > 0 && status === 'complete' && (
                      <>
                        <button
                          onClick={handleDownloadAll}
                          disabled={downloadingAll}
                          className="btn-ghost px-2.5 py-1.5 text-xs"
                          title="Download all clips as a ZIP"
                        >
                          {downloadingAll
                            ? <Loader2 size={13} className="animate-spin" />
                            : <Download size={13} />}
                        </button>
                        {results.clips.length > 1 && (
                          <button
                            onClick={() => setShowScheduleWeek(true)}
                            className="btn-ghost px-2.5 py-1.5 text-xs"
                            title="Schedule these across the week"
                          >
                            <Calendar size={13} />
                          </button>
                        )}
                      </>
                    )}
                  </>
                )}
              >

                {/* Peak-moment upsell: they just SAW their clips — sell while
                    they're proud of the result. NOTE: no GitHub-star ask on this
                    screen in any state (spec rule) — it lives in the sidebar
                    footer instead, so promo never sits in the job-status area. */}
                {status === 'complete' && results?.clips?.length > 0 && plan === 'free' && (
                  <div className="mb-2">
                    <button
                      onClick={() => { setTopUpInfo({ context: 'upsell' }); setShowTopUp(true); }}
                      className="w-full text-left px-3 py-2.5 rounded-input bg-paper3 border border-brass/40 hover:border-brass text-sm transition-colors"
                    >
                      <span className="text-ink">Like these clips?</span>{' '}
                      <span className="text-muted">They carry a watermark and delete in 7 days.</span>{' '}
                      <span className="text-brass font-medium">Keep them forever →</span>
                    </button>
                  </div>
                )}

                <div className={clipView === 'row'
                  ? 'overflow-x-auto overflow-y-hidden custom-scrollbar pb-2'
                  : 'max-h-[70vh] overflow-y-auto custom-scrollbar p-1'}
                >
                  {(() => {
                    // Live grid: as many slots as the job will produce
                    // (progress.json's clips_total once render starts,
                    // otherwise the user's requested count). Ready clips flip
                    // their slot to a real ResultCard the moment the backend
                    // surfaces them; the rest stay "rendering…" placeholders.
                    const readyCount = results?.clips?.length || 0;
                    const slotCount = viewStatus === 'processing'
                      ? (progress?.clips_total || requestedClipCount || 0)
                      : 0;
                    const total = Math.max(readyCount, slotCount);
                    if (total > 0) {
                      // Columns must react to this container's OWN width, not
                      // the viewport's — this panel sits inside a nested
                      // flex split (sidebar + live-analysis panel + this
                      // results pane), so a viewport breakpoint like
                      // `xl:grid-cols-2` used to fire on a wide monitor even
                      // though the actual space here was ~670px, squeezing
                      // each card's action-button grid down to ~20px wide
                      // and overlapping every label. auto-fit sizes off the
                      // real container width instead.
                      // Row view lays the clips out along one line you slide
                      // sideways — eight clips stop meaning eight screens of
                      // scrolling. Column view keeps the stack, sized to this
                      // container's real width (a viewport breakpoint fires on
                      // a wide monitor even when this pane is ~670px, which is
                      // what used to squeeze each card's buttons to ~20px).
                      const layoutClass = clipView === 'row'
                        ? 'flex gap-4 pb-2 w-max'
                        : status === 'complete'
                          ? 'grid gap-4 pb-4 grid-cols-[repeat(auto-fit,minmax(440px,1fr))]'
                          : 'grid gap-3 pb-4 grid-cols-1';
                      const itemClass = clipView === 'row'
                        ? (status === 'complete' ? 'w-[440px] shrink-0' : 'w-[320px] shrink-0')
                        : '';
                      return (
                        <div className={layoutClass}>
                          {Array.from({ length: total }).map((_, i) => (
                            // While the job runs, a finished clip is a ROW —
                            // four 420px editor cards buried the pipeline the
                            // user was trying to watch. Clicking one unfolds
                            // the full editor in place; nothing is removed.
                            <div key={`slot-${jobId}-${i}`} className={itemClass}>
                            {results?.clips?.[i] && viewStatus === 'processing' && expandedClip !== i ? (
                              <ClipRow
                                key={`row-${jobId}-${i}`}
                                clip={results.clips[i]}
                                index={i}
                                jobId={jobId}
                                onExpand={setExpandedClip}
                                onPlay={setExpandedClip}
                              />
                            ) : results?.clips?.[i] ? (
                              <ResultCard
                                key={`${jobId}-${i}`}
                                clip={results.clips[i]}
                                index={i}
                                jobId={jobId}
                                initialState={projectState?.clips?.find((c) => c.index === i) || null}
                                onStateChange={handleClipStateChange}
                                durableUrl={durableClips[i]}
                                uploadPostKey={uploadPostKey}
                                uploadUserId={uploadUserId}
                                geminiApiKey={apiKey}
                                elevenLabsKey={elevenLabsKey}
                                isManaged={isManaged}
                                connectedPlatforms={(userProfiles.find((p) => p.username === uploadUserId) || userProfiles[0])?.connected ?? null}
                                onConnectSocials={isManaged ? handleConnectSocials : null}
                                onBulkSubtitle={handleBulkSubtitles}
                                clipCount={readyCount}
                                bulkProgress={bulkSub}
                              />
                            ) : (
                              // Exactly one slot renders at a time (the pipeline
                              // is sequential), so the first unfinished slot is
                              // the one actually rendering and the rest are
                              // honestly shown as still queued.
                              <ClipSlotPlaceholder
                                key={`ph-${i}`}
                                index={i}
                                total={total}
                                state={viewStatus === 'processing' && i === readyCount ? 'rendering' : 'queued'}
                              />
                            )}
                            </div>
                          ))}
                        </div>
                      );
                    }
                    if (viewStatus === 'processing') {
                      return (
                        <div className="h-full flex flex-col items-center justify-center text-muted space-y-4">
                          <Loader2 size={32} className="animate-spin text-brass" />
                          <p className="text-sm lowercase">Waiting for clips...</p>
                        </div>
                      );
                    }
                    if (status === 'error' || status === 'cancelled') {
                      return (
                        <FailureReport
                          logs={logs}
                          stage={progress?.stage}
                          cancelled={status === 'cancelled'}
                          onRetry={active?.form ? () => handleProcess(active.form) : null}
                          onNewSource={handleReset}
                        />
                      );
                    }
                    return null;
                  })()}
                </div>
              </CollapsiblePanel>

                {/* Telemetry: parsed logs (real server timestamps, noise filter
                    with raw toggle) + real performance data. It lives INSIDE the
                    single scroll container along with the status card and the
                    results panel. Previously it was a separate fixed row below a
                    nested `overflow-y-auto` region, which gave that region only
                    a few hundred pixels — so the results panel (and, on a failed
                    job, the entire "we couldn't process this video" explanation
                    plus its Try-again buttons) was silently cut off below the
                    fold with no indication anything was there. */}
                <CollapsiblePanel
                  id="telemetry"
                  title="pipeline detail"
                  subtitle="log · stage"
                  defaultOpen={false}
                  className="shrink-0 !bg-transparent !border-0"
                  bodyClassName="pt-3"
                >
                  <TelemetryGrid
                    logs={logs}
                    status={viewStatus}
                    raw={logsRaw}
                    onRawToggle={() => setLogsRaw((v) => !v)}
                    stageDurations={stageDurations}
                    speedMultiplier={progress?.speed_multiplier ?? null}
                    currentStage={progress?.stage ?? null}
                    progress={progress}
                  />
                </CollapsiblePanel>

              </div>

                {/* No paste-a-link bar on this screen. While a job runs the
                    only thing that matters here is watching it; queueing the
                    next video is what New Project in the sidebar is for, and
                    the bar was costing a strip of vertical space on every
                    run. (Removed at the owner's request, 13-aug-2026.) */}
              </div>
              <HomeRail
                onViewAll={() => setActiveTab("history")}
                search={historySearch}
                onOpenProject={handleOpenProject}
                onCompare={handleCompareClip}
                comparedId={comparedId}
                projects={railProjects}
                activeId={jobId}
              />
            </div>
          )}

        </div>

      </main>

      {/* Missing API Key Modal */}
      <Modal
        isOpen={showKeyModal}
        onClose={() => setShowKeyModal(false)}
        eyebrow="SETUP"
        title="Gemini API Key Required"
        footer={
          <div className="flex gap-3">
            <button
              onClick={() => setShowKeyModal(false)}
              className="btn-ghost flex-1 px-4 py-2 text-sm"
            >
              Cancel
            </button>
            <button
              onClick={() => { setShowKeyModal(false); setActiveTab('settings'); }}
              className="btn-primary flex-1 px-4 py-2 text-sm"
            >
              Go to Settings
            </button>
          </div>
        }
      >
        <div className="space-y-4">
          <p className="text-sm text-muted">
            {BRAND.name} needs both a <strong className="text-ink2">Gemini</strong> API key and an <strong className="text-ink2">Upload-Post</strong> API key. Both have free tiers.
          </p>

          {/* Gemini block */}
          <div className={`rounded-input p-4 space-y-2 border ${(!apiKey && !serverHasGeminiKey) ? 'border-rule2' : 'border-rule opacity-70'}`}>
            <p className="text-xs font-medium text-ink flex items-center gap-2">
              {(apiKey || serverHasGeminiKey) ? <Check size={12} className="text-ok" /> : <AlertTriangle size={12} className="text-warn" />}
              Gemini API Key {apiKey && <span className="text-ok">— set</span>}
              {!apiKey && serverHasGeminiKey && <span className="text-ok">— using the server's key</span>}
            </p>
            {/* The server holds a key of its own (a .env, or Kaggle Secrets), so
                nothing needs to be entered here. The field stays available as a
                per-browser override. */}
            {!apiKey && serverHasGeminiKey && (
              <p className="text-xs text-muted">
                This deployment is configured with its own Gemini key, so you can start a job right away.
                Enter a key below only if you want this browser to use a different one.
              </p>
            )}
            {!apiKey && (
              <>
                <ol className="text-xs text-muted space-y-1 list-decimal list-inside">
                  <li>Go to <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-brass underline">aistudio.google.com/app/apikey</a></li>
                  <li>Sign in with your Google account</li>
                  <li>Click "Create API Key"</li>
                  <li>Copy the key and paste it below</li>
                </ol>
                <input
                  type="text"
                  placeholder="Paste your Gemini API key here..."
                  className="input-field"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && e.target.value.trim()) {
                      setApiKey(e.target.value.trim());
                    }
                  }}
                />
              </>
            )}
          </div>

          {/* Upload-Post block */}
          <div className={`rounded-input p-4 space-y-2 border ${!uploadPostKey ? 'border-rule2' : 'border-rule opacity-70'}`}>
            <p className="text-xs font-medium text-ink flex items-center gap-2">
              {uploadPostKey ? <Check size={12} className="text-ok" /> : <AlertTriangle size={12} className="text-warn" />}
              Upload-Post API Key {uploadPostKey && <span className="text-ok">— set</span>}
            </p>
            {!uploadPostKey && (
              <>
                <p className="text-xs text-muted">
                  Required to publish your clips to TikTok, Instagram Reels, and YouTube Shorts. Free tier available, no credit card needed.
                </p>
                <ol className="text-xs text-muted space-y-1 list-decimal list-inside">
                  <li>Register at <a href="https://app.upload-post.com/login" target="_blank" rel="noopener noreferrer" className="text-brass underline">app.upload-post.com</a></li>
                  <li>Connect your TikTok, Instagram, or YouTube accounts</li>
                  <li>Go to <a href="https://app.upload-post.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-brass underline">API Keys</a> and generate one</li>
                  <li>Paste it below</li>
                </ol>
                <input
                  type="text"
                  placeholder="Paste your Upload-Post API key here..."
                  className="input-field"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && e.target.value.trim()) {
                      setUploadPostKey(e.target.value.trim());
                    }
                  }}
                />
              </>
            )}
          </div>
        </div>
      </Modal>

      <ScheduleWeekModal
        isOpen={showScheduleWeek}
        onClose={() => setShowScheduleWeek(false)}
        clips={results?.clips || []}
        jobId={jobId}
        uploadPostKey={uploadPostKey}
        uploadUserId={uploadUserId}
        isManaged={isManaged}
      />

      {/* Pre-flight quality gate */}
      {qualityGate && (
        <Modal isOpen={true} onClose={() => setQualityGate(null)} size="md" eyebrow="HEADS UP" title="low source quality">
          <div className="space-y-4">
            <p className="text-sm text-ink2">
              YouTube only offers <span className="text-brass font-semibold">{qualityGate.info.max_height}p</span> for this video
              (below the {qualityGate.info.min_height}p we recommend). Processing anyway will produce lower-quality clips.
            </p>
            {qualityGate.info.cookies_invalid && (
              <p className="text-xs text-muted">
                Your YouTube cookies look expired — refreshing them (export again from an incognito window) often unlocks HD.
              </p>
            )}
            <div className="flex gap-2 justify-end pt-2">
              <button onClick={() => setQualityGate(null)} className="btn-ghost">cancel</button>
              <button
                onClick={() => { const d = qualityGate.data; setQualityGate(null); handleProcess(d, true); }}
                className="btn-primary"
              >
                process anyway
              </button>
            </div>
          </div>
        </Modal>
      )}


      {showLogin && <LoginModal onClose={() => setShowLogin(false)} />}
      {showPlanChoice && <PlanChoiceModal onClose={() => setShowPlanChoice(false)} />}
      {showTopUp && (
        <TopUpModal
          onClose={() => setShowTopUp(false)}
          required={topUpInfo.required}
          remaining={topUpInfo.remaining}
          context={topUpInfo.context || 'wall'}
        />
      )}
      {showTrialUpgrade && (
        <TrialUpgradeModal
          plan={plan}
          onActivated={refreshMe}
          onClose={() => setShowTrialUpgrade(false)}
        />
      )}
    </div>
  );
}

export default App;
