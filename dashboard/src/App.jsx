import React, { useState, useEffect, useRef } from 'react';
import { Upload, Sparkles, Youtube, Instagram, Share2, ChevronDown, Check, Activity, LayoutDashboard, Settings, Plus, History, X, Terminal, Shield, LayoutGrid, Image, Globe, RotateCcw, Calendar, AlertTriangle, KeyRound, Bot, Users, Smartphone, ExternalLink, Copy, CheckCircle2, Mail, Loader2, Download, Search, Flame, TrendingUp, ChevronRight, Film } from 'lucide-react';
import KeyInput from './components/KeyInput';
import KeyListInput from './components/KeyListInput';
import MediaInput from './components/MediaInput';
import ResultCard from './components/ResultCard';
import ProcessingAnimation from './components/ProcessingAnimation';
import ClipSlotPlaceholder from './components/ClipSlotPlaceholder';
import SystemStatusStrip from './components/SystemStatusStrip';
import StageTracker from './components/StageTracker';
import HomeRail from './components/HomeRail';
import TelemetryGrid from './components/TelemetryGrid';
import FloatingInputBar from './components/FloatingInputBar';
import SourcePanel from './components/SourcePanel';
// import Gallery from './components/Gallery';
import ThumbnailStudio from './components/ThumbnailStudio';
import SaaShortsTab from './components/SaaShortsTab';
import UGCGallery from './components/UGCGallery';
import ScheduleWeekModal from './components/ScheduleWeekModal';
import UsageMeter from './components/UsageMeter';
import TopUpModal from './components/TopUpModal';
import PlanChoiceModal from './components/PlanChoiceModal';
import TrialUpgradeModal from './components/TrialUpgradeModal';
import LoginModal from './components/LoginModal';
import TrialGate from './components/TrialGate';
import AdvancedBanner from './components/AdvancedBanner';
import HistoryTab from './components/HistoryTab';
import ProfileMenu from './components/ProfileMenu';
import Modal from './components/ui/Modal';
import { useAuth } from './contexts/AuthContext';
import { apiFetch, apiJson, QuotaError } from './lib/api';

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

const SESSION_KEY = 'openshorts_session';
const SESSION_MAX_AGE = 3600000; // 1 hour (matches server job retention)

// Human "2m 30s" style readout for the ETA returned by /api/status.
const formatEta = (seconds) => {
  const s = Number(seconds);
  if (!Number.isFinite(s) || s < 0) return null;
  if (s < 60) return `${Math.max(1, Math.round(s))}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return rem ? `${m}m ${rem}s` : `${m}m`;
};

// Mock polling function
const pollJob = async (jobId, raw = false) => {
  const res = await apiFetch(`/api/status/${jobId}${raw ? '?raw=1' : ''}`);
  if (!res.ok) throw new Error('Status check failed');
  return res.json();
};

function App() {
  // Cloud auth/billing session (inert when billing is disabled).
  const { billingEnabled, isManaged, isSignedIn, me, plan, refreshMe } = useAuth();
  const [showLogin, setShowLogin] = useState(false);
  const [showTopUp, setShowTopUp] = useState(false);
  const [showPlanChoice, setShowPlanChoice] = useState(false);
  const [showTrialUpgrade, setShowTrialUpgrade] = useState(false);
  const [topUpInfo, setTopUpInfo] = useState({});
  // Durable R2 URLs (per clip index) for the current job — used as a fallback when
  // the ephemeral local /videos/ files have been cleaned up (e.g. after a reload).
  const [durableClips, setDurableClips] = useState({});

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

  // fal.ai API State - Load encrypted
  const [falKey, setFalKey] = useState(() => {
    const stored = localStorage.getItem('falKey_v1');
    if (stored) return decrypt(stored);
    return '';
  });

  // AssemblyAI API State (transcription backend) - Load encrypted
  const [assemblyaiKey, setAssemblyaiKey] = useState(() => {
    const stored = localStorage.getItem('assemblyaiKey_v1');
    if (stored) return decrypt(stored);
    return '';
  });

  // DeepSeek API State (narrative clip selection) - Load encrypted
  const [deepseekKey, setDeepseekKey] = useState(() => {
    const stored = localStorage.getItem('deepseekKey_v1');
    if (stored) return decrypt(stored);
    return '';
  });

  // Extra Gemini keys beyond the primary one above — a small pool spread
  // across concurrent vision-confirmation calls so a single key's rate limit
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
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, processing, complete, error
  const [results, setResults] = useState(null);
  // Bulk subtitles: apply one style to every clip of the job (triggered from
  // within a clip's subtitle modal via "apply to all").
  const [bulkSub, setBulkSub] = useState({ running: false, current: 0, total: 0, errors: 0 });
  const [downloadingAll, setDownloadingAll] = useState(false);
  // Pre-flight quality gate: { info: {max_height, min_height, cookies_invalid}, data }
  const [qualityGate, setQualityGate] = useState(null);
  const [logs, setLogs] = useState([]);
  // Live pipeline progress from main.py's progress.json: {stage, overall_pct,
  // clips_done, clips_total}. null until the backend reports a snapshot.
  const [progress, setProgress] = useState(null);
  // Telemetry: raw-log toggle + per-stage averages for the performance chart.
  const [logsRaw, setLogsRaw] = useState(false);
  const [stageDurations, setStageDurations] = useState(null);
  // Right-panel view: 'clips' (Generated Shorts) or 'source' (original video
  // + stills — round-5 Source section).
  const [rightTab, setRightTab] = useState('clips');
  // Itemized system status for the status strip: {backend, cookies, gpu} from
  // /api/system, refreshed on an interval. null = never checked yet.
  const [systemStatus, setSystemStatus] = useState(null);
  // The clip count the user requested at submit (or null = auto) — sizes the
  // placeholder slots in the live grid before progress.json knows the total.
  const [requestedClipCount, setRequestedClipCount] = useState(null);
  const [processingMedia, setProcessingMedia] = useState(null);
  const [submittedFormat, setSubmittedFormat] = useState('auto');
  // Sidebar "Today" mini-stats, derived from the disk-backed history.
  const [todayStats, setTodayStats] = useState({ generated: 0, processing: 0, successRate: 100, loaded: false });
  const [activeTab, setActiveTab] = useState('dashboard'); // dashboard, settings
  // Reopened-project state (paid mode): per-clip {index, server_file, active_layers}
  // restored from the backend so ResultCards resume editing where they left off.
  const [projectState, setProjectState] = useState(null);
  // True when the current job was reopened from the library: its source video
  // was never persisted, so the session must not fall back to /api/source.
  const [noSource, setNoSource] = useState(false);

  const [sessionRecovered, setSessionRecovered] = useState(false);
  const [showScheduleWeek, setShowScheduleWeek] = useState(false);

  // Silent-success "saved" states for the settings key inputs (design.md: no alert popups)
  const [elevenLabsSaved, setElevenLabsSaved] = useState(false);
  const [falSaved, setFalSaved] = useState(false);
  const [assemblyaiSaved, setAssemblyaiSaved] = useState(false);
  const [deepseekSaved, setDeepseekSaved] = useState(false);

  // Sync state for original video playback
  const [syncedTime, setSyncedTime] = useState(0);
  const [isSyncedPlaying, setIsSyncedPlaying] = useState(false);
  const [syncTrigger, setSyncTrigger] = useState(0);

  const handleClipPlay = (startTime) => {
    setSyncedTime(startTime);
    setIsSyncedPlaying(true);
    setSyncTrigger(prev => prev + 1);
  };

  const handleClipPause = () => {
    setIsSyncedPlaying(false);
  };

  // --- Project persistence (paid mode) ---
  // Debounced sync of each clip's browser-only edit state (Remotion layers +
  // current server file) to the backend, so a reopened project resumes intact.
  const clipStateSync = useRef({ jobId: null, pending: {}, timer: null });

  const flushClipState = () => {
    const s = clipStateSync.current;
    if (s.timer) { clearTimeout(s.timer); s.timer = null; }
    const entries = Object.entries(s.pending);
    if (!s.jobId || entries.length === 0) return;
    const clips = entries.map(([i, v]) => ({
      index: Number(i),
      active_layers: v.activeLayers,
      server_file: v.serverVideoFile,
    }));
    s.pending = {};
    apiFetch(`/api/projects/${s.jobId}/state`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clips }),
    }).catch(() => {});
  };

  const handleClipStateChange = (index, state) => {
    if (!isManaged || !jobId) return;
    const s = clipStateSync.current;
    if (s.jobId !== jobId) { s.pending = {}; s.jobId = jobId; }
    s.pending[index] = state;
    if (s.timer) clearTimeout(s.timer);
    s.timer = setTimeout(flushClipState, 2000);
  };

  // Reopen an archived project from the History tab: the backend re-downloads
  // its files from R2 into the server's working dir and returns the full state.
  const restoreProject = async (projectJobId) => {
    const data = await apiJson(`/api/projects/${projectJobId}/restore`, { method: 'POST' });
    flushClipState();
    setProjectState(data.project_state || null);
    setNoSource(true);
    setJobId(data.job_id);
    setResults(data.result || null);
    setLogs(['♻️ Project restored from your library.']);
    setProcessingMedia(null);
    setQualityGate(null);
    setStatus('complete');
    setActiveTab('dashboard');
  };

  // Open any project from the rail — including a failed one. A failed job has
  // no restorable project state (the pipeline never got far enough to write
  // metadata), so restoring would 404; instead it opens in the error state with
  // whatever the backend still knows, so the run can actually be inspected
  // rather than being a dead tile you can only delete.
  const openProject = async (v) => {
    const targetId = v?.job_id;
    if (!targetId) return;
    setActiveTab('dashboard');
    if (v.status === 'completed') {
      try {
        await restoreProject(targetId);
        return;
      } catch (e) {
        // fall through to the read-only view below
      }
    }
    flushClipState();
    setProjectState(null);
    setNoSource(true);
    setProcessingMedia(null);
    setQualityGate(null);
    setResults(null);
    setJobId(targetId);
    setStatus(v.status === 'processing' ? 'processing' : 'error');
    // /api/status only knows jobs still in memory; after a backend restart it
    // 404s, so say so plainly rather than leaving the log panel implying more
    // output is still coming.
    try {
      const data = await apiJson(`/api/status/${targetId}`);
      setLogs(data.logs?.length ? data.logs : ['No logs retained for this run.']);
      if (data.progress) setProgress(data.progress);
      if (data.status) setStatus(data.status === 'processing' ? 'processing' : data.status);
      if (data.result) setResults(data.result);
    } catch (e) {
      setLogs([
        `Opened "${v.title || targetId}" from your library.`,
        'This run failed and its live logs are no longer retained on the server.',
      ]);
    }
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
      const data = await pollJob(jobId);
      if (data.result) setResults(data.result);
    } catch { /* keep current results */ }
  };

  const [cancelling, setCancelling] = useState(false);
  const handleCancelJob = async () => {
    if (!jobId || cancelling) return;
    if (!window.confirm('Stop this job? Clips already rendered will stay, the rest will not be generated.')) return;
    setCancelling(true);
    try {
      await apiFetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
      setStatus('cancelled');
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

  // Session Recovery: Restore on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(SESSION_KEY);
      if (!saved) return;
      const session = JSON.parse(saved);
      if (Date.now() - session.timestamp > SESSION_MAX_AGE) {
        localStorage.removeItem(SESSION_KEY);
        return;
      }
      if (session.jobId && session.status && session.status !== 'idle') {
        setJobId(session.jobId);
        setResults(session.results || null);
        // Restore the source preview. Older sessions (or uploads) saved no
        // media, so fall back to the backend-served source for this job —
        // except for reopened projects, whose source was never persisted.
        if (session.processingMedia) setProcessingMedia(session.processingMedia);
        else if (!session.noSource) setProcessingMedia({ type: 'server', payload: `/api/source/${session.jobId}` });
        if (session.noSource) setNoSource(true);
        if (session.projectState) setProjectState(session.projectState);
        if (session.activeTab) setActiveTab(session.activeTab);
        // If was processing, resume polling; if complete/error, just show results
        setStatus(session.status === 'processing' ? 'processing' : session.status);
        setSessionRecovered(true);
        setTimeout(() => setSessionRecovered(false), 5000);
      }
    } catch (e) {
      localStorage.removeItem(SESSION_KEY);
    }
  }, []);

  // Session Recovery: Save state changes
  useEffect(() => {
    if (status === 'idle') {
      localStorage.removeItem(SESSION_KEY);
      return;
    }
    try {
      // URL (YouTube) media serializes as-is. Uploaded 'file' media is a blob
      // that can't be persisted, so point the recovered preview at the source
      // served by the backend instead of dropping it.
      let persistMedia = null;
      if (processingMedia?.type === 'url') persistMedia = processingMedia;
      else if (processingMedia && jobId) persistMedia = { type: 'server', payload: `/api/source/${jobId}` };
      const sessionData = {
        jobId,
        status,
        results,
        processingMedia: persistMedia,
        activeTab,
        noSource,
        projectState,
        timestamp: Date.now()
      };
      localStorage.setItem(SESSION_KEY, JSON.stringify(sessionData));
    } catch (e) {
      // localStorage full or serialization error - ignore
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, status, results, activeTab, noSource, projectState]);

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
    if (falKey) {
      localStorage.setItem('falKey_v1', encrypt(falKey));
    }
  }, [falKey]);

  useEffect(() => {
    if ((uploadPostKey || isManaged) && userProfiles.length === 0) {
      fetchUserProfiles({ silent: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadPostKey, isManaged]);

  // For managed users, fetch the durable R2 URLs of the current job's clips so the
  // preview can fall back to them when the local files have been cleaned up.
  useEffect(() => {
    if (!isManaged || !jobId || !(results?.clips?.length)) { setDurableClips({}); return; }
    let cancelled = false;
    apiJson('/api/history')
      .then((d) => {
        if (cancelled) return;
        const map = {};
        for (const v of (d.videos || [])) {
          if (v.job_id === jobId && v.clip_index != null) map[v.clip_index] = v.view_url;
        }
        setDurableClips(map);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [isManaged, jobId, results]);

  useEffect(() => {
    let interval;
    if ((status === 'processing' || status === 'completed') && jobId) {
      interval = setInterval(async () => {
        try {
          const data = await pollJob(jobId, logsRaw);
          console.log("Job status:", data);

          // Update results if available (real-time)
          if (data.result) {
            setResults(data.result);
          }
          // Update live pipeline progress (stage + percentage + clips done).
          if (data.progress) {
            setProgress(data.progress);
          }
          // Rolling per-stage averages + raw-log mode for the telemetry grid.
          if (data.stage_durations) setStageDurations(data.stage_durations);

          if (data.status === 'completed') {
            setStatus('complete');
            clearInterval(interval);
          } else if (data.status === 'failed') {
            setStatus('error');
            const lastLog = data.logs && data.logs.length > 0
              ? (typeof data.logs[data.logs.length - 1] === 'string'
                  ? data.logs[data.logs.length - 1]
                  : data.logs[data.logs.length - 1].text)
              : "Process failed";
            const errorMsg = data.error || lastLog;
            setLogs(prev => [...prev, "Error: " + errorMsg]);
            clearInterval(interval);
          } else if (data.status === 'cancelled') {
            setStatus('cancelled');
            clearInterval(interval);
          } else {
            // Update logs if available
            if (data.logs) setLogs(data.logs);
          }
        } catch (e) {
          console.error("Polling error", e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [status, jobId, logsRaw]);

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
    const interval = status === 'processing' ? setInterval(load, 5000) : null;
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
  const ADVANCED_TOOL_TABS = ['saasshorts', 'ai-agent'];
  const TOOL_NAMES = { dashboard: 'the Clip Generator', thumbnails: 'the YouTube Studio' };
  const gateThisTab = needsPlan && INCLUDED_TOOL_TABS.includes(activeTab);      // included tool, no plan yet
  const advancedThisTab = billingEnabled && ADVANCED_TOOL_TABS.includes(activeTab); // BYOK-notice tools

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
    setStatus('processing');
    setLogs(["Starting process..."]);
    setResults(null);
    setProgress(null);
    setRequestedClipCount(data.clipCount ?? null);
    setSubmittedFormat(data.outputFormat || 'auto');
    setRightTab('clips');
    setProcessingMedia(data);
    setQualityGate(null);
    setProjectState(null);
    setNoSource(false);

    try {
      let body;
      // BYOK sends the Gemini header; managed users rely on the bearer token
      // that apiFetch attaches automatically. AssemblyAI/DeepSeek/extra Gemini
      // pool keys are all optional — the pipeline falls back (Whisper /
      // Gemini-only 2-pass / single Gemini key) when any of these are unset.
      const headers = apiKey ? { 'X-Gemini-Key': apiKey } : {};
      if (assemblyaiKey) headers['X-AssemblyAI-Key'] = assemblyaiKey;
      if (deepseekKey) headers['X-DeepSeek-Key'] = deepseekKey;
      if (geminiExtraKeys.length > 0) headers['X-Gemini-Keys'] = geminiExtraKeys.join(',');

      if (data.type === 'url') {
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify({
          url: data.payload,
          acknowledged: !!data.acknowledged,
          output_format: data.outputFormat || 'auto',
          force_low_quality: forceLowQuality,
          clip_count: data.clipCount ?? null,
          long_context_clips: data.longContextClips || 0,
          remove_background_audio: data.removeBackgroundAudio || '',
          custom_width: data.outputFormat === 'custom' ? data.customWidth : null,
          custom_height: data.outputFormat === 'custom' ? data.customHeight : null,
          captions: data.captions !== false,
          zoom_mode: data.zoomMode || 'auto',
          style_variant: data.styleVariant || 'balanced',
          caption_position: data.captionPosition || 'bottom',
          caption_margin: data.captionMargin ?? null,
        });
      } else {
        const formData = new FormData();
        formData.append('file', data.payload);
        formData.append('acknowledged', data.acknowledged ? 'true' : 'false');
        formData.append('output_format', data.outputFormat || 'auto');
        if (data.clipCount != null) formData.append('clip_count', String(data.clipCount));
        if (data.longContextClips > 0) formData.append('long_context_clips', String(data.longContextClips));
        if (data.removeBackgroundAudio) formData.append('remove_background_audio', data.removeBackgroundAudio);
        if (data.outputFormat === 'custom') {
          formData.append('custom_width', String(data.customWidth || 1080));
          formData.append('custom_height', String(data.customHeight || 1350));
        }
        formData.append('captions', data.captions !== false ? 'true' : 'false');
        formData.append('zoom_mode', data.zoomMode || 'auto');
        formData.append('style_variant', data.styleVariant || 'balanced');
        formData.append('caption_position', data.captionPosition || 'bottom');
        if (data.captionMargin != null) formData.append('caption_margin', String(data.captionMargin));
        body = formData;
      }

      const res = await apiFetch('/api/process', { method: 'POST', headers, body });

      if (!res.ok) throw new Error(await res.text());
      const resData = await res.json();

      // Quality gate: the source is below the min resolution — ask before burning
      // 20 min on it. On confirm we resend with force_low_quality.
      if (resData.needs_confirmation) {
        setStatus('idle');
        setQualityGate({ info: resData.quality_check, data });
        return;
      }

      setJobId(resData.job_id);

    } catch (e) {
      if (e instanceof QuotaError) {
        setStatus('idle');
        // Trial users hit the trial minute cap → prompt them to activate the plan
        // now (unlocks full minutes). Active users → offer a top-up.
        if (me?.status === 'trialing') {
          setShowTrialUpgrade(true);
        } else {
          setTopUpInfo({ required: e.minutesRequired, remaining: e.minutesRemaining });
          setShowTopUp(true);
        }
        return;
      }
      setStatus('error');
      setLogs(l => [...l, `Error starting job: ${e.message}`]);
    }
  };

  const handleReset = () => {
    // Flush any pending edit-state sync before dropping the project: the clips
    // themselves are already archived to R2 as they were edited.
    flushClipState();
    setStatus('idle');
    setJobId(null);
    setResults(null);
    setLogs([]);
    setProgress(null);
    setRequestedClipCount(null);
    setRightTab('clips');
    setProcessingMedia(null);
    setProjectState(null);
    setNoSource(false);
    localStorage.removeItem(SESSION_KEY);
  };

  // --- UI Components ---

  const Sidebar = () => {
    const navItems = [
      { id: 'dashboard', ord: '01', icon: LayoutDashboard, label: 'Clip Generator' },
      { id: 'saasshorts', ord: '02', icon: Sparkles, label: 'AI Shorts', byok: true },
      { id: 'ai-agent', ord: '03', icon: Bot, label: 'AI Agent', byok: true },
      { id: 'ugc-gallery', ord: '04', icon: LayoutGrid, label: 'UGC Gallery' },
      { id: 'thumbnails', ord: '05', icon: Image, label: 'YouTube Studio' },
      // History must be reachable on self-host too: /api/history is disk-backed
      // and works without sign-in, so the tab was hidden exactly when the user
      // needed it most ("my projects disappeared" was this, not data loss).
      // Cloud mode still gates it behind sign-in (R2 library is per-account).
      ...(!billingEnabled || isSignedIn ? [{ id: 'history', ord: '06', icon: History, label: 'History' }] : []),
      { id: 'settings', ord: '07', icon: Settings, label: 'Settings' },
    ];

    return (
      <div className="w-20 lg:w-64 bg-paper2 border-r border-rule flex flex-col h-full shrink-0 transition-all duration-300">
        <a href="#landing" className="p-6 flex items-center gap-3" title="go to landing page">
          <div className="w-8 h-8 bg-paper3 rounded-input flex items-center justify-center shrink-0 overflow-hidden border border-rule">
            <img src="/logo-openshorts.png" alt="Logo" className="w-full h-full object-cover" />
          </div>
          <span className="font-display lowercase text-lg text-ink hidden lg:block">openshorts</span>
        </a>

        {/* New Project CTA (reference sidebar) */}
        <div className="px-4 pb-3">
          <button
            onClick={() => { setActiveTab('dashboard'); handleReset(); }}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-input text-sm font-medium text-white transition-all hover:brightness-110"
            style={{
              background: 'var(--grad-accent)',
              boxShadow: '0 1px 0 rgba(255,255,255,0.18) inset, var(--shadow-glow)',
            }}
          >
            <Plus size={16} /> <span className="hidden lg:inline">New Project</span>
          </button>
        </div>

        <nav className="flex-1 px-4 py-4 space-y-1">
          {navItems.map((item) => {
            const NavIcon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`relative w-full flex items-center gap-3 px-3 py-2.5 rounded-input transition-all ${isActive ? 'text-ink' : 'text-muted hover:text-ink2 hover:bg-paper3/50'}`}
                style={isActive ? {
                  background: 'linear-gradient(90deg, rgba(239,68,68,0.18) 0%, rgba(239,68,68,0.04) 65%, transparent 100%)',
                  boxShadow: 'inset 0 0 0 1px rgba(239,68,68,0.16)',
                } : undefined}
              >
                {isActive && (
                  <span
                    className="absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-full"
                    style={{ background: 'var(--color-accent)', boxShadow: '0 0 8px var(--color-glow)' }}
                    aria-hidden="true"
                  />
                )}
                <NavIcon size={18} className={`shrink-0 ${isActive ? 'text-brass glow-accent' : ''}`} />
                <span className="text-sm hidden lg:block flex-1 text-left truncate">{item.label}</span>
                {item.byok && <span className="readout hidden lg:block">BYOK</span>}
                <span className="readout hidden lg:block">{item.ord}</span>
              </button>
            );
          })}
        </nav>

        {/* Today mini-stats (reference sidebar) */}
        <div className="px-4 pb-4">
          <div className="card p-3.5">
            <div className="flex items-center justify-between mb-2.5">
              <p className="readout text-[10px] uppercase tracking-wider text-muted">today</p>
              <Flame size={13} className="text-brass glow-accent" />
            </div>
            <div className="space-y-2">
              {[
                { icon: LayoutGrid, label: 'shorts generated', value: todayStats.generated, tone: 'text-ink' },
                {
                  icon: Loader2,
                  label: 'processing',
                  // Reconciled with the job this tab is actually running: a job
                  // visible in the main panel must never coexist with a "0
                  // processing" sidebar, even in the window before the backend
                  // history has caught up.
                  value: Math.max(todayStats.processing, status === 'processing' ? 1 : 0),
                  tone: 'text-brass',
                  spin: status === 'processing',
                },
                { icon: TrendingUp, label: 'success rate', value: `${todayStats.successRate}%`, tone: 'text-ok' },
              ].map((s) => {
                const SIcon = s.icon;
                return (
                  <div key={s.label} className="flex items-center gap-2">
                    <SIcon size={13} className={`shrink-0 ${s.tone === 'text-ink' ? 'text-muted' : s.tone} ${s.spin ? 'animate-spin' : ''}`} />
                    <span className="text-[11px] lowercase text-muted flex-1 truncate hidden lg:block">{s.label}</span>
                    {/* Never show a static 0 that could be mistaken for a real
                        count while the first fetch is still in flight. */}
                    {todayStats.loaded ? (
                      <span className={`text-xs font-semibold ${s.tone}`}>{s.value}</span>
                    ) : (
                      <span className="inline-block w-6 h-3 rounded bg-paper3 animate-pulse" />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-rule space-y-1">
          <a
            href="#landing"
            className="flex items-center gap-2 px-3 py-1.5 text-xs lowercase text-muted hover:text-ink2 transition-colors"
          >
            <span className="icon-chip-muted !w-6 !h-6 shrink-0"><Globe size={13} /></span>
            <span className="hidden lg:block truncate">landing page</span>
          </a>
          <a
            href="https://github.com/mutonby/openshorts"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-3 py-1.5 text-xs lowercase text-muted hover:text-ink2 transition-colors"
          >
            <span className="icon-chip-muted !w-6 !h-6 shrink-0"><svg height="13" viewBox="0 0 16 16" version="1.1" width="13" aria-hidden="true" fill="currentColor"><path fillRule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg></span>
            <span className="hidden lg:block truncate">open source</span>
          </a>
          {billingEnabled && (
            <a
              href="#/pricing"
              className="flex items-center gap-2 px-3 py-1.5 text-xs lowercase text-muted hover:text-ink2 transition-colors"
            >
              <span className="icon-chip-muted !w-6 !h-6 shrink-0"><Sparkles size={13} /></span>
              <span className="hidden lg:block truncate">plans &amp; pricing</span>
            </a>
          )}
          <a
            href="mailto:info@openshorts.app"
            className="flex items-center gap-2 px-3 py-1.5 text-xs lowercase text-muted hover:text-ink2 transition-colors"
          >
            <span className="icon-chip-muted !w-6 !h-6 shrink-0"><Mail size={13} /></span>
            <span className="hidden lg:block truncate">info@openshorts.app</span>
          </a>
        </div>
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-paper overflow-hidden">
      <Sidebar />

      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Top Header */}
        <header className="h-14 border-b border-rule bg-paper flex items-center gap-4 px-6 shrink-0 z-10">
          {status !== 'idle' && (
            <button
              onClick={handleReset}
              className="btn-quiet px-3 py-1.5 text-xs shrink-0"
            >
              <Plus size={14} />
              <span className="hidden sm:inline">New Project</span>
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

          {/* Live system pills (reference top bar) */}
          <div className="hidden lg:block shrink-0">
            <SystemStatusStrip status={systemStatus} />
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
                <span className="text-muted">Set your Gemini API key to use OpenShorts.</span>
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

        {/* Session Recovery Banner */}
        {sessionRecovered && (
          <div className="mx-6 mt-2 px-4 py-3 bg-paper2 border border-rule rounded-card flex items-center justify-between animate-fade shrink-0">
            <div className="flex items-center gap-2 text-sm text-ink2">
              <RotateCcw size={16} className="text-brass" />
              <span className="font-medium">Session recovered</span>
              <span className="text-muted text-xs">Your previous work has been restored.</span>
            </div>
            <button onClick={() => setSessionRecovered(false)} className="text-muted hover:text-ink transition-colors">
              <X size={14} />
            </button>
          </div>
        )}

        {/* Included tools (Clip Generator, YouTube Studio): non-blocking trial prompt. */}
        {gateThisTab && <TrialGate toolName={TOOL_NAMES[activeTab] || 'this'} />}

        {/* Advanced tools (AI Shorts, AI Agent): BYOK fal.ai + ElevenLabs notice. */}
        {advancedThisTab && <AdvancedBanner needsPlan={needsPlan} onKeys={() => setActiveTab('settings')} />}

        {/* Main Workspace */}
        <div className="flex-1 overflow-hidden relative">

          {/* View: Settings */}
          {activeTab === 'settings' && (
            <div className="h-full overflow-y-auto p-4 sm:p-8 max-w-2xl mx-auto animate-fade">
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-8">
                <div>
                  <p className="eyebrow mb-1.5">07 · SETTINGS</p>
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
                    <label className="block text-sm text-muted mb-2">DeepSeek API Key</label>
                    <div className="flex flex-col sm:flex-row gap-2">
                      <input
                        type="password"
                        value={deepseekKey}
                        onChange={(e) => setDeepseekKey(e.target.value)}
                        className="input-field"
                        placeholder="sk-..."
                      />
                      <button
                        onClick={() => {
                          if (deepseekKey) {
                            localStorage.setItem('deepseekKey_v1', encrypt(deepseekKey));
                            setDeepseekSaved(true);
                            setTimeout(() => setDeepseekSaved(false), 2000);
                          }
                        }}
                        className={deepseekSaved ? 'badge-ok px-4' : 'btn-quiet py-2 px-4 text-sm'}
                      >
                        {deepseekSaved ? <><Check size={12} /> saved</> : 'Save'}
                      </button>
                    </div>
                    <p className="mt-2 text-xs text-muted">
                      <a href="https://platform.deepseek.com/api_keys" target="_blank" rel="noopener noreferrer" className="text-brass hover:underline">
                        Get your DeepSeek API key →
                      </a>
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm text-muted mb-2">
                      Extra Gemini keys <span className="text-muted">(optional — spreads vision-confirmation calls across a pool so one key's rate limit isn't the bottleneck)</span>
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

              <div className="card p-4 sm:p-6 mt-8">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                      <Sparkles size={16} className="text-brass" />
                    </div>
                    <h2 className="text-base font-medium text-ink lowercase">AI Shorts (UGC Videos)</h2>
                  </div>
                  <span className="readout">BYOK</span>
                </div>
                <p className="text-xs text-muted mb-6 leading-relaxed">
                  Generate UGC-style videos with AI actors for any product or business using <strong>fal.ai</strong>.
                  <strong> Not covered by your plan</strong> — bring your own fal.ai + ElevenLabs keys (billed by those
                  providers, ~$0.65-2 per video). Your plan still covers the AI script &amp; orchestration.
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-muted">fal.ai API Key</label>
                  <div className="flex flex-col sm:flex-row gap-2">
                    <input
                      type="password"
                      value={falKey}
                      onChange={(e) => setFalKey(e.target.value)}
                      className="input-field"
                      placeholder="fal_..."
                    />
                    <button
                      onClick={() => {
                        if (falKey) {
                          localStorage.setItem('falKey_v1', encrypt(falKey));
                          setFalSaved(true);
                          setTimeout(() => setFalSaved(false), 2000);
                        }
                      }}
                      className={falSaved ? 'badge-ok px-4' : 'btn-quiet py-2 px-4 text-sm'}
                    >
                      {falSaved ? <><Check size={12} /> saved</> : 'Save'}
                    </button>
                  </div>
                  <div className="text-xs text-muted leading-relaxed">
                    Get your API key from fal.ai to enable AI actor video generation.
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <a href="https://fal.ai/dashboard/keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">1. Sign Up</span>
                        <span className="text-xs text-muted">Create fal.ai account</span>
                      </a>
                      <a href="https://fal.ai/dashboard/keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-rule rounded-input hover:bg-paper3 transition-colors flex flex-col gap-1">
                        <span className="text-ink2 font-medium">2. API Key</span>
                        <span className="text-xs text-muted">Generate key</span>
                      </a>
                    </div>
                    <br />
                    <span className="text-muted">
                      Keys are only stored in your browser. Sent to backend only to process requests.
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* View: SaaS Shorts */}
          {activeTab === 'saasshorts' && (
            <SaaShortsTab geminiApiKey={apiKey} elevenLabsKey={elevenLabsKey} falKey={falKey} uploadPostKey={uploadPostKey} uploadUserId={uploadUserId} managed={isManaged} />
          )}

          {/* View: AI Agent */}
          {activeTab === 'ai-agent' && (
            <div className="h-full overflow-y-auto custom-scrollbar p-4 sm:p-6 md:p-10 animate-fade">
              <div className="max-w-4xl mx-auto space-y-8">

                {/* Header */}
                <div className="space-y-3">
                  <p className="eyebrow flex items-center gap-2">
                    <Bot size={12} /> 03 · AI AGENT · AUTONOMOUS SKILL
                  </p>
                  <h1 className="font-display lowercase text-3xl md:text-4xl text-ink">
                    Your Personal Clipping Team
                  </h1>
                  <p className="text-muted text-base md:text-lg leading-relaxed max-w-2xl">
                    Drop your videos in a folder and a team of AI clippers picks the viral moments, edits them, and queues them for your approval — like having a 24/7 short-form editing crew on autopilot.
                  </p>
                </div>

                {/* Mobile-format warning */}
                <div className="px-4 py-3 rounded-card border border-rule bg-paper2 flex items-start gap-3">
                  <Smartphone size={18} className="text-warn shrink-0 mt-0.5" />
                  <div className="text-sm text-ink2">
                    <p className="font-medium text-ink mb-1">Upload videos already in vertical (9:16) mobile format.</p>
                    <p className="text-muted leading-relaxed">
                      The agent does not reframe horizontal footage. Make sure every source video is shot or pre-cropped to mobile/portrait format before dropping it into the input folder.
                    </p>
                  </div>
                </div>

                {/* Workflow */}
                <div className="grid md:grid-cols-3 gap-4">
                  <div className="card p-5 space-y-2">
                    <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center">
                      <Upload size={18} className="text-brass" />
                    </div>
                    <h3 className="font-medium text-ink lowercase">1. Drop your videos</h3>
                    <p className="text-xs text-muted leading-relaxed">
                      Put your long-form vertical footage in the watched folder. The skill picks one video per run.
                    </p>
                  </div>

                  <div className="card p-5 space-y-2">
                    <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center">
                      <Users size={18} className="text-brass" />
                    </div>
                    <h3 className="font-medium text-ink lowercase">2. AI clippers work</h3>
                    <p className="text-xs text-muted leading-relaxed">
                      Whisper transcribes, Gemini 3 Flash spots viral beats, FFmpeg cuts each clip and adds a hook overlay.
                    </p>
                  </div>

                  <div className="card p-5 space-y-2">
                    <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center">
                      <CheckCircle2 size={18} className="text-brass" />
                    </div>
                    <h3 className="font-medium text-ink lowercase">3. You validate, it ships</h3>
                    <p className="text-xs text-muted leading-relaxed">
                      Approve the candidates you like and the skill auto-publishes them to TikTok, Reels and YouTube Shorts via Upload-Post.
                    </p>
                  </div>
                </div>

                {/* Repo CTA */}
                <div className="card p-6 md:p-8 space-y-5">
                  <div className="flex items-start justify-between gap-4 flex-wrap">
                    <div>
                      <h2 className="font-display lowercase text-xl text-ink mb-1">skill-autoshorts</h2>
                      <p className="text-sm text-muted">
                        The Claude Code skill that powers this workflow. Install it once and trigger it whenever you want a fresh batch of clips.
                      </p>
                    </div>
                    <a
                      href="https://github.com/mutonby/skill-autoshorts"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="btn-primary py-2 px-4 text-sm shrink-0"
                    >
                      View on GitHub <ExternalLink size={14} />
                    </a>
                  </div>

                  <div className="bg-paper border border-rule rounded-card p-4 font-mono text-xs text-ink2 flex items-center justify-between gap-3">
                    <span className="truncate">git clone https://github.com/mutonby/skill-autoshorts</span>
                    <button
                      onClick={() => navigator.clipboard.writeText('git clone https://github.com/mutonby/skill-autoshorts')}
                      className="text-muted hover:text-ink transition-colors shrink-0"
                      title="Copy"
                    >
                      <Copy size={14} />
                    </button>
                  </div>

                  <div className="grid sm:grid-cols-2 gap-3 text-sm">
                    <div className="flex items-start gap-2 text-ink2">
                      <Check size={16} className="text-brass shrink-0 mt-0.5" />
                      <span>Daily batch — picks one long video per run</span>
                    </div>
                    <div className="flex items-start gap-2 text-ink2">
                      <Check size={16} className="text-brass shrink-0 mt-0.5" />
                      <span>Whisper transcription with word-level timing</span>
                    </div>
                    <div className="flex items-start gap-2 text-ink2">
                      <Check size={16} className="text-brass shrink-0 mt-0.5" />
                      <span>Gemini 3 Flash multimodal moment detection</span>
                    </div>
                    <div className="flex items-start gap-2 text-ink2">
                      <Check size={16} className="text-brass shrink-0 mt-0.5" />
                      <span>Auto-publish to TikTok, Reels & YouTube Shorts</span>
                    </div>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* View: UGC Gallery */}
          {activeTab === 'ugc-gallery' && (
            <div className="h-full overflow-y-auto custom-scrollbar animate-fade">
              <div className="max-w-6xl mx-auto p-6 md:p-8">
                <UGCGallery />
              </div>
            </div>
          )}

          {/* View: History */}
          {activeTab === 'history' && (
            <div className="h-full overflow-y-auto custom-scrollbar animate-fade">
              <div className="max-w-6xl mx-auto p-6 md:p-8">
                <HistoryTab onReopenProject={restoreProject} search={historySearch} />
              </div>
            </div>
          )}

          {activeTab === 'thumbnails' && (
            <ThumbnailStudio geminiApiKey={apiKey} uploadPostKey={uploadPostKey} uploadUserId={uploadUserId} managed={isManaged} />
          )}

          {/* View: Gallery */}
          {/* {activeTab === 'gallery' && (
            <Gallery />
          )} */}

          {/* View: Dashboard (Idle) */}
          {activeTab === 'dashboard' && status === 'idle' && (
            <div className="flex h-full min-h-0">
              <div className="flex-1 min-w-0 flex flex-col">
                <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar animate-fade">
                  <div className="relative min-h-full flex flex-col items-center justify-center px-4 py-6 sm:p-6">
                    {/* Large, very-low-opacity OpenShorts mark behind the hero
                        (round 3, item 6) — own branding, not a copied motif */}
                    <div
                      aria-hidden="true"
                      className="pointer-events-none absolute -top-10 right-0 sm:-right-10 w-[320px] h-[320px] opacity-[0.05] -z-10 select-none"
                    >
                      <img src="/logo-openshorts.png" alt="" className="w-full h-full object-contain" />
                    </div>
                  <div className="max-w-xl w-full text-center space-y-8">
                    <div className="space-y-4 relative">
                      {/* Ambient pool behind the headline — pure CSS gradient,
                          no blur filter, so it costs nothing to composite. */}
                      <div
                        aria-hidden="true"
                        className="pointer-events-none absolute left-1/2 -translate-x-1/2 -top-24 w-[560px] h-[320px] -z-10"
                        style={{ background: 'radial-gradient(50% 50% at 50% 50%, rgba(239,68,68,0.16) 0%, transparent 70%)' }}
                      />
                      <p className="eyebrow flex items-center justify-center gap-2">
                        <Sparkles size={12} className="text-brass glow-accent" />
                        01 · CLIP GENERATOR
                      </p>
                      <h1 className="font-display lowercase text-4xl md:text-5xl text-ink tracking-tight">
                        create{' '}
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
                      <p className="text-muted text-lg">
                        Drop your long-form video below to instantly generate viral clips with AI.
                      </p>
                    </div>

                    <MediaInput onProcess={handleProcess} isProcessing={status === 'processing'} />

                    <div className="flex flex-wrap items-center justify-center gap-2.5 text-muted">
                      {[
                        { Icon: Youtube, label: 'YouTube' },
                        { Icon: Instagram, label: 'Instagram' },
                        { Icon: TikTokIcon, label: 'TikTok' },
                      ].map(({ Icon, label }) => (
                        <span
                          key={label}
                          className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-rule bg-paper2 text-xs lowercase"
                        >
                          <Icon size={14} className="text-ink2" /> {label}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
                </div>
              </div>
              <HomeRail
                onViewAll={() => setActiveTab("history")}
                search={historySearch}
                onOpenProject={openProject}
                activeJob={jobId ? {
                  id: jobId,
                  status,
                  pct: progress?.overall_pct ?? null,
                  title: processingMedia?.type === 'file'
                    ? (processingMedia.payload?.name || 'New project')
                    : processingMedia?.type === 'url'
                      ? processingMedia.payload
                      : 'New project',
                } : null}
              />
            </div>
          )}

          {/* View: Processing / Results (Split View) */}
          {activeTab === 'dashboard' && (status === 'processing' || status === 'complete' || status === 'error' || status === 'cancelled') && (
            <div className="flex h-full min-h-0">
              <div className="flex-1 min-w-0 flex flex-col">
                {/* Processing hero — same gradient-emphasis headline as Home */}
                <div className="px-5 pt-4 pb-1 shrink-0">
                  <p className="eyebrow">02 · LIVE PIPELINE</p>
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
                    syncedTime={syncedTime}
                    isSyncedPlaying={isSyncedPlaying}
                    syncTrigger={syncTrigger}
                    status={status}
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
                    onCancel={status === 'processing' ? handleCancelJob : null}
                  />
                )}

              {/* Results / live clip slots — full width beneath the status card */}
              <div className="flex flex-col card p-4 sm:p-6 shrink-0 min-h-[260px]">
                <h2 className="font-display lowercase text-xl text-ink mb-6 flex flex-wrap items-center gap-2 shrink-0">
                  <div className="flex items-center gap-1.5 mr-auto">
                    <button
                      onClick={() => setRightTab('clips')}
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs transition-colors ${
                        rightTab === 'clips'
                          ? 'border-brass/50 bg-brass/10 text-ink'
                          : 'border-rule text-muted hover:border-rule2'
                      }`}
                    >
                      <span className="icon-chip !w-6 !h-6"><Sparkles size={13} /></span>
                      Generated Shorts
                    </button>
                    <button
                      onClick={() => setRightTab('source')}
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs transition-colors ${
                        rightTab === 'source'
                          ? 'border-brass/50 bg-brass/10 text-ink'
                          : 'border-rule text-muted hover:border-rule2'
                      }`}
                    >
                      <span className="icon-chip !w-6 !h-6"><Film size={13} /></span>
                      Source
                    </button>
                  </div>
                  {results?.clips?.length > 0 && (
                    <span className="readout px-2.5 py-1 rounded-full ml-auto border border-rule bg-paper2">
                      {results.clips.length} Clips
                    </span>
                  )}
                  {results?.cost_analysis && !isManaged && (
                    <span className="readout bg-paper3 px-2.5 py-1 rounded-full ml-2" title={`Input: ${results.cost_analysis.input_tokens} | Output: ${results.cost_analysis.output_tokens}`}>
                      GEMINI · ${results.cost_analysis.total_cost.toFixed(5)}
                    </span>
                  )}
                  {results?.clips?.length > 0 && status === 'complete' && (
                    <div className="flex items-center gap-2 ml-auto">
                      <button
                        onClick={handleDownloadAll}
                        disabled={downloadingAll}
                        className="btn-ghost px-3 py-2 text-xs"
                        title="Download all clips as a ZIP"
                      >
                        {downloadingAll
                          ? <><Loader2 size={14} className="animate-spin" />zipping…</>
                          : <><Download size={14} />download all</>}
                      </button>
                      {results.clips.length > 1 && (
                        <button
                          onClick={() => setShowScheduleWeek(true)}
                          className="btn-primary px-4 py-2 text-xs"
                        >
                          <Calendar size={14} />
                          schedule week
                        </button>
                      )}
                    </div>
                  )}
                </h2>

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

                {rightTab === 'source' && jobId ? (
                  <SourcePanel jobId={jobId} />
                ) : (
                <div className="flex-1 overflow-y-auto custom-scrollbar p-1">
                  {(() => {
                    // Live grid: as many slots as the job will produce
                    // (progress.json's clips_total once render starts,
                    // otherwise the user's requested count). Ready clips flip
                    // their slot to a real ResultCard the moment the backend
                    // surfaces them; the rest stay "rendering…" placeholders.
                    const readyCount = results?.clips?.length || 0;
                    const slotCount = status === 'processing'
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
                      const gridClass = status === 'complete'
                        ? 'grid-cols-[repeat(auto-fit,minmax(440px,1fr))]'
                        : 'grid-cols-1';
                      return (
                        <div className={`grid gap-4 pb-10 ${gridClass}`}>
                          {Array.from({ length: total }).map((_, i) => (
                            results?.clips?.[i] ? (
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
                                onPlay={(time) => handleClipPlay(time)}
                                onPause={handleClipPause}
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
                                state={status === 'processing' && i === readyCount ? 'rendering' : 'queued'}
                              />
                            )
                          ))}
                        </div>
                      );
                    }
                    if (status === 'processing') {
                      return (
                        <div className="h-full flex flex-col items-center justify-center text-muted space-y-4">
                          <Loader2 size={32} className="animate-spin text-brass" />
                          <p className="text-sm lowercase">Waiting for clips...</p>
                        </div>
                      );
                    }
                    if (status === 'error') {
                      const lastLog = logs && logs.length > 0
                        ? (typeof logs[logs.length - 1] === 'string' ? logs[logs.length - 1] : logs[logs.length - 1].text)
                        : '';
                      const joined = (logs || []).map((l) => (typeof l === 'string' ? l : l.text || '')).join(' ');
                      let cause = 'Something went wrong on our side while processing your video.';
                      if (/download|yt-dlp|403|url|youtube/i.test(joined)) {
                        cause = 'We couldn\u2019t fetch this video. This can happen when it\u2019s age-restricted, private, temporarily unavailable, or the download link expired.';
                      } else if (/gemini|deepseek|api key|transcrib/i.test(joined)) {
                        cause = 'The AI step hit an error (transcription or analysis). Try again — if it keeps failing, check your API keys in Settings.';
                      } else if (/reframe|ffmpeg|encode|render|scene/i.test(joined)) {
                        cause = 'Rendering hit an error while cutting the clips. The source may be unusual (odd codec or damaged file) — try a different link or upload.';
                      }
                      return (
                        <div className="h-full flex flex-col items-center justify-center text-center px-6 space-y-3">
                          <span className="icon-chip !w-12 !h-12 !border-danger/30" style={{ background: 'color-mix(in oklab, var(--color-danger) 14%, transparent)', color: 'var(--color-danger)' }}>
                            <AlertTriangle size={22} />
                          </span>
                          {/* Plain text, not a \u-escape: this is JSX text
                              content, where escapes are literal characters. */}
                          <p className="text-sm font-semibold text-ink">We couldn’t process this video</p>
                          <p className="text-xs text-muted max-w-md leading-relaxed">{cause}</p>
                          {lastLog && (
                            <details className="text-[10px] text-muted/70 text-left max-w-md w-full">
                              <summary className="cursor-pointer select-none lowercase">raw error</summary>
                              <p className="mt-1 break-words font-mono">{lastLog}</p>
                            </details>
                          )}
                          <div className="flex flex-wrap items-center justify-center gap-2 mt-1">
                            <button onClick={handleReset} className="btn-primary px-4 py-2 text-xs">
                              Try again
                            </button>
                            <button onClick={handleReset} className="btn-ghost px-4 py-2 text-xs">
                              Use a different link
                            </button>
                            <a href="mailto:info@openshorts.app" className="btn-ghost px-4 py-2 text-xs">
                              Contact support
                            </a>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  })()}
                </div>
                )}
              </div>

                {/* Telemetry: parsed logs (real server timestamps, noise filter
                    with raw toggle) + real performance data. It lives INSIDE the
                    single scroll container along with the status card and the
                    results panel. Previously it was a separate fixed row below a
                    nested `overflow-y-auto` region, which gave that region only
                    a few hundred pixels — so the results panel (and, on a failed
                    job, the entire "we couldn't process this video" explanation
                    plus its Try-again buttons) was silently cut off below the
                    fold with no indication anything was there. */}
                <div className="shrink-0">
                  <TelemetryGrid
                    logs={logs}
                    status={status}
                    raw={logsRaw}
                    onRawToggle={() => setLogsRaw((v) => !v)}
                    stageDurations={stageDurations}
                    speedMultiplier={progress?.speed_multiplier ?? null}
                    currentStage={progress?.stage ?? null}
                  />
                </div>

              </div>

                {/* Queue-another-video bar, only while a job is actually
                    running. Once the job is finished or failed the work on this
                    screen is reviewing the result, and a persistent "generate"
                    field at the bottom just competes with that — New Project in
                    the sidebar is the real entry point for the next run. */}
                {status === 'processing' && (
                  <FloatingInputBar onProcess={handleProcess} isProcessing />
                )}
              </div>
              <HomeRail
                onViewAll={() => setActiveTab("history")}
                search={historySearch}
                onOpenProject={openProject}
                activeJob={jobId ? {
                  id: jobId,
                  status,
                  pct: progress?.overall_pct ?? null,
                  title: processingMedia?.type === 'file'
                    ? (processingMedia.payload?.name || 'New project')
                    : processingMedia?.type === 'url'
                      ? processingMedia.payload
                      : 'New project',
                } : null}
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
            OpenShorts needs both a <strong className="text-ink2">Gemini</strong> API key and an <strong className="text-ink2">Upload-Post</strong> API key. Both have free tiers.
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
