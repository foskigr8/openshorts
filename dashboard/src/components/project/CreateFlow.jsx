import { useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { getApiUrl } from '../../config';
import SourceStep from './steps/SourceStep';
import RatioStep from './steps/RatioStep';
import CountStep from './steps/CountStep';
import LaunchStep from './steps/LaunchStep';

/**
 * The creation flow: one question at a time, each answered step sliding away
 * in the direction you're travelling.
 *
 * Adding a step later is additive — write a component, put it in STEPS, read
 * and write its field on `draft`. Nothing else in the flow needs to know.
 *
 * The object handed to onSubmit is byte-for-byte the contract the pipeline
 * already accepts (the fields the old MediaInput sent), so the redesign is
 * cosmetic where it matters: at the API boundary.
 */

// Caption placement presets, unchanged from the previous form. MarginV only
// applies to bottom alignment, which is why "raised" exists for bottom only.
// 43 is subtitles.SAFE_MARGIN_V, the server-side default.
const CAPTION_PLACEMENTS = {
  bottom: { position: 'bottom', margin: 43 },
  raised: { position: 'bottom', margin: 120 },
  middle: { position: 'middle', margin: null },
};

const STEPS = ['source', 'ratio', 'count', 'launch'];

const initialDraft = () => ({
  source: null,                  // {type:'url'|'file', payload}
  outputFormat: 'vertical',
  custom: { width: 1080, height: 1920 },
  clipCount: 8,
  captions: true,
  captionPlacement: 'bottom',
  removeBgAudio: false,
  longContextClips: 0,
});

export default function CreateFlow({ onSubmit, onCancel, starting = false, error = '' }) {
  const [draft, setDraft] = useState(initialDraft);
  const [step, setStep] = useState(0);
  const [dir, setDir] = useState('next');
  const [urlEnabled, setUrlEnabled] = useState(true);
  const [launching, setLaunching] = useState(false);

  useEffect(() => {
    fetch(getApiUrl('/api/config'))
      .then((r) => (r.ok ? r.json() : null))
      .then((cfg) => { if (cfg && cfg.youtubeUrlEnabled === false) setUrlEnabled(false); })
      .catch(() => {});
  }, []);

  const patch = (changes) => setDraft((d) => ({ ...d, ...changes }));
  const go = (to) => {
    setDir(to > step ? 'next' : 'prev');
    setStep(Math.max(0, Math.min(STEPS.length - 1, to)));
  };

  // Esc backs out of the flow entirely — it is a modal state over the home.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !starting) onCancel(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCancel, starting]);

  const launch = () => {
    if (!draft.source) { go(0); return; }
    const placement = CAPTION_PLACEMENTS[draft.captionPlacement] || CAPTION_PLACEMENTS.bottom;
    setLaunching(true);
    onSubmit({
      type: draft.source.type,
      payload: draft.source.payload,
      acknowledged: true,          // stated on the launch step, above the button
      outputFormat: draft.outputFormat,
      customWidth: draft.custom.width,
      customHeight: draft.custom.height,
      captions: draft.captions,
      captionPosition: placement.position,
      captionMargin: placement.margin,
      // "isolate" = voice separation, the only mode that removes music.
      removeBackgroundAudio: draft.removeBgAudio ? 'isolate' : '',
      clipCount: draft.clipCount,
      longContextClips: draft.longContextClips,
    });
  };

  // The launch animation ends at opacity 0 (`animation-fill-mode: both`), so
  // anything that leaves this component mounted afterwards leaves it INVISIBLE
  // — a blank panel. That is exactly what happened when the submit came back
  // without a running job: the quality gate, the quota wall, or an error.
  // Belt and braces: clear it when the parent stops starting, when an error
  // arrives, and on a timer that always outlives the 620ms animation.
  useEffect(() => {
    if (!launching) return undefined;
    const t = setTimeout(() => setLaunching(false), 1500);
    return () => clearTimeout(t);
  }, [launching]);
  useEffect(() => { if (error) setLaunching(false); }, [error]);
  const wasStarting = useRef(false);
  useEffect(() => {
    if (wasStarting.current && !starting) setLaunching(false);
    wasStarting.current = starting;
  }, [starting]);

  const body = {
    source: (
      <SourceStep
        value={draft.source}
        urlEnabled={urlEnabled}
        onChange={(v) => patch({ source: v })}
        onNext={() => go(1)}
      />
    ),
    ratio: (
      <RatioStep
        value={draft.outputFormat}
        onChange={(v) => patch({ outputFormat: v })}
        custom={draft.custom}
        onCustomChange={(v) => patch({ custom: v })}
        onNext={() => go(2)}
        onBack={() => go(0)}
      />
    ),
    count: (
      <CountStep
        value={draft.clipCount}
        onChange={(v) => patch({ clipCount: v })}
        longContext={draft.longContextClips}
        onLongContextChange={(v) => patch({ longContextClips: v })}
        onNext={() => go(3)}
        onBack={() => go(1)}
      />
    ),
    launch: (
      <LaunchStep
        draft={draft}
        onPatch={patch}
        onBack={() => go(2)}
        onJump={go}
        onLaunch={launch}
        starting={starting}
        error={error}
      />
    ),
  }[STEPS[step]];

  return (
    <div className="relative w-full h-full flex flex-col">
      <button
        onClick={onCancel}
        className="absolute top-0 right-0 p-2 rounded-full text-muted hover:text-ink hover:bg-paper3 transition-colors z-10"
        aria-label="close"
      >
        <X size={18} />
      </button>

      <div className={`flex-1 flex items-center justify-center px-4 ${launching ? 'launching' : ''}`}>
        {/* Keyed on the step so React remounts it and the entry animation
            actually replays instead of only running on first mount. */}
        <div key={STEPS[step]} className={dir === 'next' ? 'step-enter-next w-full' : 'step-enter-prev w-full'}>
          {body}
        </div>
      </div>

      {/* Progress: four marks, no numbering and no checkmarks. */}
      <div className="flex items-center justify-center gap-1.5 pb-2 shrink-0">
        {STEPS.map((s, i) => (
          <button
            key={s}
            type="button"
            onClick={() => i < step && go(i)}
            aria-label={`step ${s}`}
            className="h-1 rounded-full transition-all duration-300"
            style={{
              width: i === step ? 26 : 10,
              background: i <= step ? 'var(--color-accent)' : 'var(--color-rule-2)',
              boxShadow: i === step ? '0 0 8px var(--color-glow)' : 'none',
              cursor: i < step ? 'pointer' : 'default',
            }}
          />
        ))}
      </div>
    </div>
  );
}
