import React, { useEffect, useRef, useState } from 'react';
import {
  MoreVertical, Youtube, Clapperboard, Clock, X, VideoOff, Radio, CheckCircle2,
  AlertTriangle, Maximize2,
} from 'lucide-react';
import { getApiUrl } from '../config';
import { pauseAllOtherPlayers, registerPlayer } from '../lib/playerSync';
import {
  subscribeSourceSync, clearCompareClip, sourceSyncPlay, sourceSyncTime,
  sourceSyncStop, sourceSyncRelease,
} from '../lib/sourceSync';
import ProgressRing from './ProgressRing';

/**
 * The source-preview panel — the "is this shot framed well?" instrument.
 *
 * Three rules it now keeps, all of which it used to break:
 *
 *  1. **It moves.** The source is always live: an ambient muted loop when
 *     nothing else is happening, and a frame-accurate follower of whichever
 *     clip you are playing otherwise. It never registers with playerSync, so
 *     another player claiming audio can no longer freeze it.
 *  2. **No inert chrome.** The old centre play button did nothing at all for
 *     a YouTube source (it drove a <video> ref that is null in that case) and
 *     the "processing" pill just sat in the corner. Both are gone; while the
 *     job runs, a scanning sweep plays OVER the still-visible footage.
 *  3. **It grows when the work is done.** Once the job completes, the metrics
 *     collapse to one compact strip and the preview becomes the hero — with
 *     the clip you picked from the rail sitting right beside it at 9:16, so
 *     the reframe can be judged against the original side by side.
 */
const ProcessingAnimation = ({
  media,
  isComplete,
  status = 'processing',
  progress = null,
  title = '',
  format = '',
  onCancel = null,
  stageDurations = null,
}) => {
  const [videoSrc, setVideoSrc] = useState(null);
  const [isYouTube, setIsYouTube] = useState(false);
  const [duration, setDuration] = useState(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [posterFailed, setPosterFailed] = useState(false);
  const [sourceFailed, setSourceFailed] = useState(false);
  const [sync, setSync] = useState({ playing: false, time: 0, seq: 0, clip: null });
  const videoRef = useRef(null);
  const iframeRef = useRef(null);
  const lastSeq = useRef(0);

  useEffect(() => subscribeSourceSync(setSync), []);

  // A job reaching the finish line starts the preview fresh. Otherwise a clip
  // you played and paused ten minutes into the render still "owns" the source
  // when the finished view mounts, and the preview comes up frozen on a frame
  // nobody is looking at any more.
  useEffect(() => { if (isComplete) clearCompareClip(); }, [isComplete]);

  useEffect(() => {
    setPosterFailed(false);
    setSourceFailed(false);
    if (!media) return undefined;
    if (media.type === 'file') {
      const url = URL.createObjectURL(media.payload);
      setIsYouTube(false);
      setVideoSrc(url);
      return () => URL.revokeObjectURL(url);
    }
    if (media.type === 'server') {
      setIsYouTube(false);
      setVideoSrc(getApiUrl(media.payload));
    } else if (media.type === 'url') {
      setIsYouTube(true);
      setVideoSrc(getYouTubeId(media.payload));
    }
    return undefined;
  }, [media]);

  // --- follow the clip: local <video> --------------------------------------
  // A new `seq` means "play or seek happened" → hard seek. A `time` change on
  // the same seq is the clip reporting its position as it runs → only correct
  // when the two have actually drifted, so the source glides instead of
  // stuttering on every update.
  // Three states, not two — the missing third is why the preview sat dead
  // after a job finished. Once ANY clip had been played, `lastSeq` was set
  // forever, so every later render (including the remount when the layout
  // switches to the finished view) hit the pause branch and stopped the
  // source. It only came back if you played a clip again.
  const ambient = (el) => {
    el.loop = true;
    el.muted = true;
    el.play().catch(() => {});
  };

  useEffect(() => {
    const el = videoRef.current;
    if (isYouTube || !el) return;
    if (sync.playing) {
      // A clip is driving: follow it.
      const seeked = sync.seq !== lastSeq.current;
      lastSeq.current = sync.seq;
      if (seeked || Math.abs(el.currentTime - sync.time) > 0.4) {
        try { el.currentTime = sync.time; } catch (_) { /* not seekable yet */ }
      }
      el.loop = false;
      el.muted = true;
      el.play().catch(() => {});
    } else if (sync.owner) {
      // A clip is loaded but paused: hold this frame so the two can be
      // compared still.
      el.pause();
    } else {
      // Nobody is driving: the source rolls on its own.
      ambient(el);
    }
  }, [sync.playing, sync.owner, sync.time, sync.seq, isYouTube, videoSrc]);

  // --- follow the clip: YouTube iframe -------------------------------------
  useEffect(() => {
    if (!isYouTube || !iframeRef.current || !videoSrc) return;
    const post = (func, args = []) => {
      try {
        iframeRef.current.contentWindow.postMessage(
          JSON.stringify({ event: 'command', func, args }), '*');
      } catch (_) { /* iframe not ready */ }
    };
    if (sync.playing) {
      if (sync.seq !== lastSeq.current) {
        lastSeq.current = sync.seq;
        post('seekTo', [sync.time, true]);
      }
      post('playVideo');
    } else if (sync.owner) {
      post('pauseVideo');
    } else {
      post('playVideo');   // ambient — the embed loops on its own
    }
  }, [sync.playing, sync.owner, sync.time, sync.seq, isYouTube, videoSrc]);

  const getYouTubeId = (url) => {
    const match = (url || '').match(
      /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/);
    return (match && match[2].length === 11) ? match[2] : null;
  };

  const fmtDuration = (s) => {
    if (!Number.isFinite(s) || s < 0) return null;
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  };

  const failed = status === 'error' || status === 'cancelled';
  const scanning = !isComplete && !failed;
  const pct = progress?.overall_pct ?? (isComplete ? 100 : 0);
  const clipsDone = progress?.clips_done ?? 0;
  const clipsTotal = progress?.clips_total ?? 0;
  const ringState = failed ? 'failed' : isComplete ? 'complete' : 'processing';

  const etaText = isComplete
    ? 'Done'
    : failed
      ? '—'
      : typeof progress?.eta_seconds === 'number'
        ? (() => {
          const s = Math.max(0, Math.round(progress.eta_seconds));
          const m = Math.floor(s / 60);
          return m > 0 ? `~${m}m ${s % 60}s` : `~${s % 60}s`;
        })()
        : 'Calculating…';

  const stageText = status === 'error' ? 'Failed'
    : status === 'cancelled' ? 'Cancelled'
      : isComplete ? 'Complete'
        : (progress?.stage || 'Queued');

  const compare = sync.clip || null;

  // ---------------------------------------------------------------- preview
  // A plain element, NOT a nested component: declaring a component inside
  // render gives it a new identity on every state change, which would remount
  // the <video> (and restart the download) roughly once a second while a clip
  // is playing.
  const sourceFrame = (
    <div className="relative w-full h-full bg-black">
      {/* Thumbnail underlay: paints instantly and survives an embed that
          refuses to load (age-restricted / region-locked video). */}
      {isYouTube && videoSrc && !posterFailed && (
        <img
          src={`https://i.ytimg.com/vi/${videoSrc}/hqdefault.jpg`}
          alt=""
          aria-hidden="true"
          onError={() => setPosterFailed(true)}
          className="absolute inset-0 w-full h-full object-cover"
        />
      )}

      {isYouTube && videoSrc ? (
        <iframe
          ref={iframeRef}
          className="absolute inset-0 w-full h-full"
          src={`https://www.youtube.com/embed/${videoSrc}?autoplay=1&mute=1&controls=0&loop=1&playlist=${videoSrc}&modestbranding=1&showinfo=0&rel=0&enablejsapi=1`}
          title="Source preview"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        />
      ) : videoSrc && !sourceFailed ? (
        <video
          // Deliberately NOT registered with playerSync: this is a muted
          // companion view, and letting any other player pause it is what
          // kept it frozen while a clip played.
          ref={videoRef}
          src={videoSrc}
          className="absolute inset-0 w-full h-full object-contain"
          autoPlay
          muted
          loop
          playsInline
          onLoadedMetadata={(e) => setDuration(e.target.duration)}
          onCanPlay={(e) => {
            // The effect above runs before the media is playable on a fresh
            // mount, and play() on an unready element is a no-op — so without
            // this the finished view could come up on a still frame.
            if (!sync.playing && !sync.owner) ambient(e.currentTarget);
          }}
          onError={() => setSourceFailed(true)}
        />
      ) : sourceFailed ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-paper2 text-muted">
          <VideoOff size={20} strokeWidth={1.75} />
          <span className="text-[10px] lowercase">source no longer on disk</span>
        </div>
      ) : (
        <div className="absolute inset-0 flex items-center justify-center bg-paper2">
          <div className="w-10 h-10 border-2 border-paper3 border-t-brass rounded-full animate-spin" />
        </div>
      )}

      {/* Scanning sweep — the footage stays fully visible underneath. */}
      {scanning && (
        <>
          <div className="scan-fx">
            <span className="scan-band" />
            <span className="scan-edge" />
          </div>
          <span className="scan-corner top-2 left-2 border-r-0 border-b-0 rounded-tl-sm" />
          <span className="scan-corner top-2 right-2 border-l-0 border-b-0 rounded-tr-sm" />
          <span className="scan-corner bottom-2 left-2 border-r-0 border-t-0 rounded-bl-sm" />
          <span className="scan-corner bottom-2 right-2 border-l-0 border-t-0 rounded-br-sm" />
          <div className="absolute bottom-2.5 left-1/2 -translate-x-1/2 flex items-center gap-2 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur-sm">
            <Radio size={11} className="text-brass animate-pulse" />
            <span className="readout text-[9px] uppercase tracking-[0.16em] text-ink2">
              {progress?.stage ? `scanning · ${progress.stage}` : 'scanning'}
            </span>
            <span className="readout text-[9px] text-brass tabular-nums">{Math.round(pct)}%</span>
          </div>
        </>
      )}

      {failed && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/45">
          <span className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/70 text-danger text-[11px]">
            <AlertTriangle size={13} /> {status === 'cancelled' ? 'cancelled' : 'failed'}
          </span>
        </div>
      )}

      {duration != null && (
        <div className="absolute bottom-2.5 left-2.5 px-2 py-0.5 rounded-md bg-black/70 readout text-[10px] text-ink2 flex items-center gap-1">
          <Clock size={10} /> {fmtDuration(duration)}
        </div>
      )}

      {sync.playing && (
        <div className="absolute top-2.5 right-2.5 badge-brass bg-black/70">in sync</div>
      )}
    </div>
  );

  const header = (
    <div className="flex items-start justify-between gap-2">
      <div className="flex items-center gap-2.5 min-w-0">
        <span className="icon-chip !w-8 !h-8 shrink-0">
          {isYouTube ? <Youtube size={16} /> : <Clapperboard size={16} />}
        </span>
        <div className="min-w-0">
          <p className="readout text-[9px] uppercase tracking-wider text-muted">source</p>
          <h3 className="font-display lowercase text-base text-ink truncate" title={title}>
            {title || 'Untitled video'}
          </h3>
        </div>
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
            {onCancel ? (
              <button
                onClick={() => { setMenuOpen(false); onCancel(); }}
                className="w-full flex items-center gap-2 px-3 py-2 text-xs text-danger hover:bg-paper3 transition-colors"
              >
                <X size={13} /> Cancel job
              </button>
            ) : (
              <p className="px-3 py-2 text-[11px] text-muted lowercase">no actions available</p>
            )}
          </div>
        )}
      </div>
    </div>
  );

  // The 9:16 companion: whichever clip is being compared right now.
  const compareOwner = compare ? `compare:${compare.id || compare.url}` : null;
  const comparePane = compare && (
    <div className="shrink-0 h-full flex flex-col">
      <div className="flex items-center justify-between mb-1.5 gap-2">
        <p className="readout text-[9px] uppercase tracking-wider text-brass truncate" title={compare.title}>
          {compare.title || 'clip'}
        </p>
        <button
          onClick={clearCompareClip}
          className="text-muted hover:text-ink shrink-0"
          aria-label="close comparison"
        >
          <X size={13} />
        </button>
      </div>
      <div className="relative flex-1 min-h-0 aspect-[9/16] rounded-input overflow-hidden bg-black border border-brass/40">
        <video
          key={compare.url}
          src={compare.url}
          className="w-full h-full object-contain"
          controls
          autoPlay
          playsInline
          // This one HAS audio, so it joins the singleton-player registry:
          // starting a result card must silence it, and vice versa.
          ref={(el) => { if (el) registerPlayer(el); }}
          // This pane drives the source itself, so a clip opened from the
          // rail compares against the original exactly like a result card
          // does. `start` is the clip's offset in the source; without one
          // (a clip from another job) it simply plays alongside.
          onPlay={(e) => {
            pauseAllOtherPlayers(e.currentTarget);
            if (compare.start == null) return;
            sourceSyncPlay(compareOwner, compare.start + e.currentTarget.currentTime, compare);
          }}
          onSeeked={(e) => {
            if (compare.start == null) return;
            sourceSyncPlay(compareOwner, compare.start + e.currentTarget.currentTime, compare);
          }}
          onTimeUpdate={(e) => {
            if (compare.start == null) return;
            sourceSyncTime(compareOwner, compare.start + e.currentTarget.currentTime);
          }}
          onPause={() => sourceSyncStop(compareOwner)}
          onEnded={() => sourceSyncRelease(compareOwner)}
        />
      </div>
    </div>
  );

  const statCell = (label, value, tone) => (
    <div className="rounded-input border border-rule bg-paper2 px-3 py-2 min-w-0">
      <p className="readout text-[9px] text-muted uppercase tracking-wider">{label}</p>
      <p className="text-sm font-semibold mt-1 truncate leading-none tabular-nums"
        style={{ color: tone || 'var(--color-ink)' }}>
        {value}
      </p>
    </div>
  );

  // ------------------------------------------------------------- complete
  // Work is done → the preview is the point of the screen. Everything else
  // shrinks to one strip so the frames get the space.
  if (isComplete) {
    return (
      <div className="card-lit rounded-card bg-paper animate-fade shrink-0 p-3 sm:p-4">
        {/* One row across the full width. Height-driven (a full-width 16:9
            preview is ~800px tall on a 1080p laptop), and the width the 16:9
            frame does not need goes to the readout column instead of sitting
            as a field of black — which is exactly what a centred preview in a
            1750px card looked like. */}
        <div
          className="flex gap-4 items-stretch"
          style={{ height: 'clamp(190px, 34vh, 380px)' }}
        >
          <div className="relative h-full aspect-video shrink-0 max-w-[62%] rounded-input overflow-hidden border border-rule bg-black">
            {sourceFrame}
          </div>

          {comparePane}

          <div className="flex-1 min-w-0 flex flex-col gap-2 py-0.5">
            {header}

            <div className="flex flex-wrap items-center gap-1.5">
              <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-ok/40 bg-ok/10 text-[11px] text-ok">
                <CheckCircle2 size={12} /> complete
              </span>
              {clipsTotal > 0 && (
                <span className="px-2.5 py-1 rounded-full border border-rule text-[10px] readout text-muted">
                  {clipsDone}/{clipsTotal} clips
                </span>
              )}
              {format && (
                <span className="px-2.5 py-1 rounded-full border border-rule text-[10px] readout text-muted">
                  {format}
                </span>
              )}
              {duration != null && (
                <span className="px-2.5 py-1 rounded-full border border-rule text-[10px] readout text-muted">
                  source {fmtDuration(duration)}
                </span>
              )}
            </div>

            <p className="text-[11px] text-muted lowercase flex items-center gap-1.5">
              <Maximize2 size={11} className="shrink-0" />
              {compare
                ? 'playing beside the source, same moment'
                : 'play any clip — it opens here beside the source'}
            </p>

            {/* The measured run, where there is finally room to show it. */}
            <div className="mt-auto grid grid-cols-2 xl:grid-cols-4 gap-1.5">
              {['download', 'transcribe', 'analyze', 'render'].map((k) => {
                const v = Number(stageDurations?.[k]) || 0;
                if (!v) return null;
                return (
                  <div key={k} className="rounded-input border border-rule bg-paper2 px-2.5 py-1.5 min-w-0">
                    <p className="readout text-[8px] uppercase tracking-wider text-muted truncate">{k}</p>
                    <p className="text-xs font-semibold text-ink2 tabular-nums mt-0.5">
                      {v < 60 ? `${Math.round(v)}s` : `${Math.floor(v / 60)}m ${Math.round(v % 60)}s`}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ------------------------------------------------------------ in flight
  return (
    <div className="card-lit rounded-card bg-paper animate-fade shrink-0 p-3 sm:p-4">
      <div className="flex flex-col lg:flex-row gap-4">
        <div className="lg:w-[42%] lg:max-w-[440px] shrink-0 space-y-3">
          <div className="relative aspect-video w-full rounded-input overflow-hidden bg-black border border-rule">
            {sourceFrame}
          </div>
          {comparePane && <div className="flex justify-center">{comparePane}</div>}
        </div>

        <div className="flex-1 min-w-0 flex flex-col justify-center gap-3">
          {header}

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

          <div className="flex flex-wrap items-center gap-4 mt-1">
            <div className="grid grid-cols-2 gap-2.5 flex-1 min-w-[210px]">
              {statCell('overall', `${Math.round(pct)}%`)}
              {statCell(
                'clips rendered',
                clipsTotal > 0 ? `${clipsDone}/${clipsTotal}` : '—/—',
                clipsTotal > 0 ? undefined : 'var(--color-muted)',
              )}
              {statCell('stage', stageText, failed ? 'var(--color-danger)' : undefined)}
              {statCell('eta', etaText)}
            </div>
            <ProgressRing
              pct={pct}
              size={92}
              stroke={7}
              state={ringState}
              label={failed ? 'failed' : 'processing'}
              className="mx-auto lg:mx-0"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProcessingAnimation;
