import React, { useEffect, useState, useRef } from 'react';
import { Play, Pause, MoreVertical, Youtube, Clapperboard, Clock, X, VideoOff,
  Download, AudioLines, ScanSearch, Scissors, CheckCircle2 } from 'lucide-react';
import { getApiUrl } from '../config';
import { pauseAllOtherPlayers, registerPlayer } from '../lib/playerSync';
import ProgressRing from './ProgressRing';

/**
 * <ActiveProcessingDashboard/> (round 3, item 2): replaces the old hacker-
 * terminal presentation (scanlines, HUD chips, raw thread lines) with a
 * unified dark card — 40% media preview on the left, metadata + gauge on the
 * right. The sync MECHANISM is untouched: always-muted source following the
 * 9:16 result clip via syncedTime/isSyncedPlaying/syncTrigger, YouTube
 * iframe postMessage control, exactly as before. Only the decoration changed.
 */
const ProcessingAnimation = ({
  media,
  isComplete,
  syncedTime,
  isSyncedPlaying,
  syncTrigger,
  status = 'processing',
  progress = null,
  title = '',
  format = '',
  onCancel = null,
  logs = [],
}) => {
  const [videoSrc, setVideoSrc] = useState(null);
  const [isYouTube, setIsYouTube] = useState(false);
  const [duration, setDuration] = useState(null);
  const [ambientPlaying, setAmbientPlaying] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [posterFailed, setPosterFailed] = useState(false);
  const [sourceFailed, setSourceFailed] = useState(false);
  const videoRef = useRef(null);
  const iframeRef = useRef(null);

  useEffect(() => {
    setPosterFailed(false);
    setSourceFailed(false);
    if (!media) return;
    if (media.type === 'file') {
      const url = URL.createObjectURL(media.payload);
      setIsYouTube(false);
      setVideoSrc(url);
      return () => URL.revokeObjectURL(url);
    } else if (media.type === 'server') {
      setIsYouTube(false);
      setVideoSrc(getApiUrl(media.payload));
    } else if (media.type === 'url') {
      setIsYouTube(true);
      setVideoSrc(getYouTubeId(media.payload));
    }
  }, [media]);

  // Sync playback for local video — unchanged mechanism.
  useEffect(() => {
    if (!isYouTube && videoRef.current) {
      if (isSyncedPlaying) {
        videoRef.current.currentTime = syncedTime;
        videoRef.current.play().catch(() => {});
        videoRef.current.loop = false;
        videoRef.current.muted = true;
        setAmbientPlaying(false);
      } else {
        videoRef.current.pause();
        if (isComplete) {
          videoRef.current.loop = true;
          videoRef.current.play().catch(() => {});
          setAmbientPlaying(true);
        }
      }
    }
  }, [syncedTime, isSyncedPlaying, isYouTube, isComplete, syncTrigger]);

  // Sync playback for YouTube (iframe postMessage) — unchanged.
  useEffect(() => {
    if (isYouTube && iframeRef.current && videoSrc) {
      const iframeWindow = iframeRef.current.contentWindow;
      if (isSyncedPlaying) {
        iframeWindow.postMessage(JSON.stringify({ event: 'command', func: 'seekTo', args: [syncedTime, true] }), '*');
        iframeWindow.postMessage(JSON.stringify({ event: 'command', func: 'playVideo', args: [] }), '*');
      } else {
        iframeWindow.postMessage(JSON.stringify({ event: 'command', func: 'pauseVideo', args: [] }), '*');
      }
    }
  }, [syncedTime, isSyncedPlaying, isYouTube, videoSrc, syncTrigger]);

  const getYouTubeId = (url) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = (url || '').match(regExp);
    return (match && match[2].length === 11) ? match[2] : null;
  };

  const toggleAmbient = () => {
    if (!videoRef.current) return;
    if (ambientPlaying) {
      videoRef.current.pause();
      setAmbientPlaying(false);
    } else {
      pauseAllOtherPlayers(videoRef.current);
      videoRef.current.play().catch(() => {});
      setAmbientPlaying(true);
    }
  };

  const fmtDuration = (s) => {
    if (!Number.isFinite(s) || s < 0) return null;
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  };

  const failed = status === 'error' || status === 'cancelled';
  const pct = progress?.overall_pct ?? (isComplete ? 100 : 0);
  const clipsDone = progress?.clips_done ?? 0;
  const clipsTotal = progress?.clips_total ?? 0;
  const statusText = status === 'complete' ? 'complete'
    : status === 'cancelled' ? 'cancelled'
      : status === 'error' ? 'error'
        : 'processing';
  const ringState = failed ? 'failed' : isComplete ? 'complete' : 'processing';

  // Plain-language ETA. "Calculating…" while the backend hasn't projected one
  // yet — never a bare dash, which reads as "no estimate will ever come".
  const etaText = isComplete
    ? 'Done'
    : failed
      ? '—'
      : typeof progress?.eta_seconds === 'number'
        ? (() => {
          const s = Math.max(0, Math.round(progress.eta_seconds));
          const m = Math.floor(s / 60);
          const rem = s % 60;
          return m > 0 ? `~${m}m ${rem}s` : `~${rem}s`;
        })()
        : 'Calculating…';

  // The stage word must always agree with the pipeline node above it: once a
  // job has actually failed it says "Failed", never a word implying it's live.
  const stageText = status === 'error' ? 'Failed'
    : status === 'cancelled' ? 'Cancelled'
      : isComplete ? 'Complete'
        : (progress?.stage || 'Queued');

  // ---- live "now doing" step (backend progress.json's step/step_pct/note) --
  const STAGE_GLYPH = {
    download: { Icon: Download, label: 'downloading the source video' },
    transcribe: { Icon: AudioLines, label: 'transcribing the audio' },
    analyze: { Icon: ScanSearch, label: 'finding the viral moments' },
    render: { Icon: Scissors, label: 'cutting + rendering the clips' },
    finalize: { Icon: Clapperboard, label: 'finalizing' },
  };
  const activeStage = failed ? null : isComplete ? 'finalize' : (progress?.stage || 'download');
  const glyph = STAGE_GLYPH[activeStage] || { Icon: ScanSearch, label: 'working' };
  const StepIcon = glyph.Icon;
  const stepText = failed
    ? 'Failed — see the logs below'
    : isComplete
      ? 'Done — all clips rendered'
      : (progress?.step || progress?.note || glyph.label);
  const stepPct = typeof progress?.step_pct === 'number' ? progress.step_pct : null;
  const logTail = (logs || []).slice(-4).map((l) => (typeof l === 'string' ? l : l.text || ''));

  return (
    // shrink-0 is load-bearing: this card sits in a `flex flex-col` column, so
    // without it flexbox compresses the card below its content height and the
    // card's own rounding/clipping cuts the bottom row of the metrics grid in
    // half. The card also no longer clips its own children — only the video
    // well does, so nothing inside can ever be sliced off.
    <div className="card-lit rounded-card bg-paper animate-fade shrink-0 p-4 sm:p-5">
      <div className="flex flex-col lg:flex-row gap-5">
        {/* Left — media preview, inset with its own rounded corners like the
            reference rather than bleeding flush into the card edge. */}
        <div className="lg:w-[38%] lg:max-w-[400px] shrink-0">
          <div className="relative aspect-video w-full rounded-input overflow-hidden bg-black border border-rule">
            {/* The source frame stays legible. It used to be dimmed to 30-50%
                opacity, which on a dark page rendered as an almost-black
                rectangle — the preview must actually show the video. */}
            {/* Thumbnail underlay. YouTube's still is a plain image, so it
                paints instantly and — unlike the embed — cannot be refused for
                an age-restricted, region-locked or embedding-disabled video.
                That is what made the preview read as a blank black box: when
                the iframe silently declined to load there was nothing behind
                it. The player still layers on top for motion; this only ever
                shows through when the player has nothing to show. */}
            {isYouTube && videoSrc && !posterFailed && (
              <img
                src={`https://i.ytimg.com/vi/${videoSrc}/hqdefault.jpg`}
                alt=""
                aria-hidden="true"
                onError={() => setPosterFailed(true)}
                className="absolute inset-0 w-full h-full object-cover"
              />
            )}

            <div className={`absolute inset-0 transition-all duration-700 ${isSyncedPlaying ? 'opacity-100' : 'opacity-95'}`}>
              {isYouTube && videoSrc ? (
                <iframe
                  ref={iframeRef}
                  className={`w-full h-full ${isSyncedPlaying ? '' : 'pointer-events-none scale-110'}`}
                  src={`https://www.youtube.com/embed/${videoSrc}?autoplay=1&mute=1&controls=0&loop=1&playlist=${videoSrc}&modestbranding=1&showinfo=0&rel=0&enablejsapi=1`}
                  title="Source preview"
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                />
              ) : videoSrc && !sourceFailed ? (
                <video
                  ref={(el) => {
                    videoRef.current = el;
                    if (el) registerPlayer(el);
                  }}
                  src={videoSrc}
                  className="w-full h-full object-cover"
                  autoPlay
                  muted
                  loop
                  playsInline
                  onLoadedMetadata={(e) => setDuration(e.target.duration)}
                  onError={() => setSourceFailed(true)}
                />
              ) : sourceFailed ? (
                // An uploaded source that's been cleaned up 404s. Say so
                // instead of leaving an unexplained black rectangle.
                <div className="w-full h-full flex flex-col items-center justify-center gap-2 bg-paper2 text-muted">
                  <VideoOff size={20} strokeWidth={1.75} />
                  <span className="text-[10px] lowercase">source no longer on disk</span>
                </div>
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-paper2">
                  <div className="w-10 h-10 border-2 border-paper3 border-t-brass rounded-full animate-spin" />
                </div>
              )}
            </div>

            {/* Real status pill */}
            <div
              className={`absolute top-3 left-3 px-2.5 py-1 rounded-full readout text-[10px] uppercase tracking-wider bg-black/70 ${
                statusText === 'complete' ? 'text-ok' : statusText === 'error' ? 'text-danger' : 'text-brass'
              }`}
            >
              {statusText}
            </div>

            {/* Duration badge (real, from the source's loaded metadata) */}
            {duration != null && (
              <div className="absolute bottom-3 left-3 px-2 py-0.5 rounded-md bg-black/70 readout text-[10px] text-ink2 flex items-center gap-1">
                <Clock size={10} /> {fmtDuration(duration)}
              </div>
            )}

            {/* Center play/pause — participates in the singleton player rule */}
            {!isSyncedPlaying && (
              <button
                onClick={toggleAmbient}
                className="absolute inset-0 m-auto w-12 h-12 rounded-full bg-black/50 hover:bg-black/70 border border-white/20 text-white flex items-center justify-center transition-colors"
                aria-label={ambientPlaying ? 'pause source preview' : 'play source preview'}
              >
                {ambientPlaying ? <Pause size={18} /> : <Play size={18} fill="white" />}
              </button>
            )}

            {isSyncedPlaying && (
              <div className="absolute top-3 right-3 badge-brass bg-black/70">live sync</div>
            )}
          </div>
        </div>

        {/* Right — metadata + metrics + gauge. min-w-0 lets the column shrink
            instead of forcing the row wider than the card (which is what used
            to push the ring past the card's right edge and clip it). */}
        <div className="flex-1 min-w-0 flex flex-col justify-center gap-3">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-center gap-2.5 min-w-0">
              <span className="icon-chip !w-8 !h-8 shrink-0">
                {isYouTube ? <Youtube size={16} /> : <Clapperboard size={16} />}
              </span>
              <h3 className="font-display lowercase text-base text-ink truncate" title={title}>
                {title || 'Untitled video'}
              </h3>
            </div>
            <div className="relative shrink-0">
              <button
                onClick={() => setMenuOpen((v) => !v)}
                className="p-1.5 rounded-full text-muted hover:text-ink hover:bg-paper3 transition-colors"
                aria-label="job actions"
              >
                <MoreVertical size={16} />
              </button>
              {menuOpen && (
                <div className="absolute right-0 top-9 z-20 w-40 rounded-input border border-rule bg-paper2 shadow-lg overflow-hidden">
                  {onCancel && (
                    <button
                      onClick={() => { setMenuOpen(false); onCancel(); }}
                      className="w-full flex items-center gap-2 px-3 py-2 text-xs text-danger hover:bg-paper3 transition-colors"
                    >
                      <X size={13} /> Cancel job
                    </button>
                  )}
                  {!onCancel && (
                    <p className="px-3 py-2 text-[11px] text-muted lowercase">no actions available</p>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Tag row — real data only: format, clip count, stage */}
          <div className="flex flex-wrap gap-1.5">
            {format && (
              <span className="px-2 py-0.5 rounded-full border border-rule text-[10px] readout text-muted">
                {format}
              </span>
            )}
            {clipsTotal > 0 && (
              <span className="px-2 py-0.5 rounded-full border border-rule text-[10px] readout text-muted">
                {clipsDone}/{clipsTotal} clips
              </span>
            )}
            {progress?.stage && !failed && (
              <span className="px-2 py-0.5 rounded-full border border-brass/40 text-[10px] readout text-brass">
                {progress.stage}
              </span>
            )}
          </div>

          {/* Metrics + gauge. The row WRAPS (flex-wrap) rather than overflowing,
              so on a narrow column the ring drops below the grid instead of
              being pushed past the card edge. The ring's own box already
              contains its glow, so no filter wrapper is needed here. */}
          <div className="flex flex-wrap items-center gap-4 mt-1">
            <div className="grid grid-cols-2 gap-2.5 flex-1 min-w-[210px]">
              <div className="rounded-input border border-rule bg-paper2 px-3.5 py-3 min-w-0">
                <p className="readout text-[9px] text-muted uppercase tracking-wider">overall</p>
                <p className="text-xl font-semibold text-ink mt-1 tabular-nums leading-none">{Math.round(pct)}%</p>
              </div>
              <div className="rounded-input border border-rule bg-paper2 px-3.5 py-3 min-w-0">
                <p className="readout text-[9px] text-muted uppercase tracking-wider">clips rendered</p>
                <p className="text-xl font-semibold text-ink mt-1 tabular-nums leading-none">
                  {/* "—/—" not "0/0": zero-found and not-yet-known are
                      different states, and "0/0" reads as the bad one. */}
                  {clipsTotal > 0
                    ? <>{clipsDone}<span className="text-base text-muted">/{clipsTotal}</span></>
                    : <span className="text-muted">—/—</span>}
                </p>
              </div>
              <div className="rounded-input border border-rule bg-paper2 px-3.5 py-3 min-w-0">
                <p className="readout text-[9px] text-muted uppercase tracking-wider">stage</p>
                <p
                  className="text-base font-semibold mt-1 capitalize truncate leading-none"
                  style={{ color: failed ? 'var(--color-danger)' : 'var(--color-ink)' }}
                  title={stageText}
                >
                  {stageText}
                </p>
              </div>
              <div className="rounded-input border border-rule bg-paper2 px-3.5 py-3 min-w-0">
                <p className="readout text-[9px] text-muted uppercase tracking-wider">eta</p>
                <p className="text-base font-semibold text-ink mt-1 truncate leading-none" title={etaText}>
                  {etaText}
                </p>
              </div>
            </div>
            <ProgressRing
              pct={pct}
              size={124}
              stroke={9}
              state={ringState}
              label={isComplete ? 'done' : failed ? 'failed' : 'processing'}
              className="mx-auto lg:mx-0"
            />
          </div>

          {/* LIVE STEP — "what is it doing right now", with a per-stage
              animation, step progress and the raw log tail. This is the
              answer to "I'm staring at 45% and don't know what's happening"
              without reading the whole telemetry grid. */}
          <div className="rounded-input border border-rule bg-paper2 px-3.5 py-3">
            <div className="flex items-center gap-3">
              <span
                className={`icon-chip !w-9 !h-9 shrink-0 ${
                  failed ? '!border-danger/30 text-danger'
                    : isComplete ? '!border-ok/30 text-ok'
                      : '!border-brass/40 text-brass'
                } ${!failed && !isComplete ? 'animate-pulse' : ''}`}
              >
                {isComplete ? <CheckCircle2 size={17} /> : failed ? <X size={17} /> : <StepIcon size={17} />}
              </span>
              <div className="flex-1 min-w-0">
                <p className="readout text-[9px] text-muted uppercase tracking-wider">now doing</p>
                <p className="text-sm font-semibold text-ink truncate leading-snug" title={stepText}>
                  {stepText}
                </p>
              </div>
              {stepPct != null && !isComplete && !failed && (
                <span className="readout text-[11px] text-brass tabular-nums shrink-0">{stepPct}%</span>
              )}
            </div>
            <div className="mt-2 h-1.5 rounded-full bg-paper3 overflow-hidden">
              <div
                className={`h-full rounded-full transition-[width] duration-500 ${
                  failed ? 'bg-danger/70'
                    : isComplete ? 'bg-ok'
                      : stepPct == null ? 'w-full bg-brass/40 animate-pulse'
                        : 'bg-brass'
                }`}
                style={stepPct == null && !failed && !isComplete ? undefined : { width: `${stepPct ?? 100}%` }}
              />
            </div>
            {/* Raw log tail — always visible so the user sees the actual
                pipeline lines, not just a summary word. */}
            {!failed && logTail.length > 0 && (
              <div className="mt-2 space-y-0.5 font-mono text-[9px] text-muted/80 leading-snug max-h-14 overflow-hidden">
                {logTail.map((line, i) => (
                  <p key={i} className="truncate" title={line}>{line}</p>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProcessingAnimation;
