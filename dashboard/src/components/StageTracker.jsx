import { useEffect, useRef, useState } from 'react';
import { Check, X, Search, Activity, Sparkles, Play, UploadCloud } from 'lucide-react';

/**
 * The 5-stage pipeline tracker.
 *
 * The connector is ONE continuous path drawn through the centre of all five
 * nodes — not five separate segments that happen to sit near each other. It is
 * painted twice on top of itself:
 *   1. the full path in a muted dashed stroke  → the road ahead
 *   2. the same path clipped to the completed fraction, in the accent gradient
 * so the filled and unfilled portions are geometrically identical and can
 * never look jagged or disconnected at the boundary. While a job is actively
 * running, a short "comet" dash travels along the completed portion so the
 * connector itself reads as alive rather than a static colour change.
 *
 * The path percentage, the nodes' checkmarks and the label underneath are all
 * derived from the same `stage`/`pct`, so they cannot disagree with each other
 * or with the metrics grid below.
 */
const STEPS = [
  { label: 'Analyze', sub: 'understanding content', Icon: Search },
  { label: 'Process', sub: 'transcribing & chunking', Icon: Activity },
  { label: 'Edit', sub: 'AI selects viral moments', Icon: Sparkles },
  { label: 'Render', sub: 'encoding clips', Icon: Play },
  { label: 'Publish', sub: 'finalizing & saving', Icon: UploadCloud },
];

const STAGE_INDEX = {
  download: 0,
  transcribe: 1,
  analyze: 2,
  render: 3,
  finalize: 4,
};

// Backend stage key per node, so a completed node can show its real duration.
const STAGE_KEYS = ['download', 'transcribe', 'analyze', 'render', 'finalize'];

const VB_H = 34;
const NODE_Y = 15; // matches the node circles' vertical centre in the row

// Node centres for 5 equal-width columns across the given width.
const nodeX = (i, w) => ((i + 0.5) / STEPS.length) * w;

// One continuous wave through every node centre: each span is a cubic whose
// control points bow alternately down/up, so the curve passes exactly through
// each node rather than stopping short of it.
function buildPath(w) {
  let d = `M ${nodeX(0, w)} ${NODE_Y}`;
  for (let i = 1; i < STEPS.length; i += 1) {
    const x0 = nodeX(i - 1, w);
    const x1 = nodeX(i, w);
    const dx = (x1 - x0) / 3;
    const bow = i % 2 === 1 ? 12 : -12;
    d += ` C ${x0 + dx} ${NODE_Y + bow}, ${x1 - dx} ${NODE_Y + bow}, ${x1} ${NODE_Y}`;
  }
  return d;
}

// The path is drawn in REAL pixel coordinates against a measured width rather
// than a fixed viewBox stretched with preserveAspectRatio="none". The stretched
// version needed vectorEffect="non-scaling-stroke" to keep the line from going
// lumpy, and that combination silently breaks `pathLength`: dash lengths end up
// measured against the rendered geometry instead of the normalised 100 units,
// so the "38% complete" dash pattern repeated and painted a second, phantom
// filled segment at the far end of the pipeline. Measuring avoids both issues.
function useMeasuredWidth() {
  const ref = useRef(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const apply = (w) => setWidth((prev) => (Math.abs(prev - w) > 0.5 ? w : prev));
    apply(el.getBoundingClientRect().width);
    if (typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver((entries) => apply(entries[0].contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

function PipelinePath({ doneFrac, animate, complete, failed }) {
  const [ref, width] = useMeasuredWidth();
  // The comet only makes sense once there's enough filled path to travel along.
  const showComet = animate && doneFrac > 12;
  // Saturated accent while the job runs (this is the reference's red pipeline);
  // green only once the whole job is done, so "finished" reads at a glance.
  const fill = complete
    ? 'color-mix(in oklab, var(--color-ok) 52%, transparent)'
    : 'var(--color-accent)';
  const d = width > 0 ? buildPath(width) : null;

  return (
    <div
      ref={ref}
      className="absolute inset-x-0 top-0 pointer-events-none"
      style={{ height: VB_H }}
      aria-hidden="true"
    >
      {d && (
        <svg width={width} height={VB_H} viewBox={`0 0 ${width} ${VB_H}`} className="block">
          {/* the road ahead — same geometry, muted and dashed */}
          <path
            d={d}
            fill="none"
            stroke="var(--color-rule-2)"
            strokeWidth="2"
            strokeDasharray="5 6"
            strokeLinecap="round"
          />

          {/* the road travelled — identical path, clipped to the done fraction.
              The gap is 1000 (not 100) so the pattern can never wrap and paint
              a second filled run further along the path. */}
          <path
            d={d}
            fill="none"
            stroke={fill}
            strokeWidth="2.5"
            strokeLinecap="round"
            pathLength="100"
            strokeDasharray={`${doneFrac} 1000`}
            style={{
              // Terminal states stay flat — glow means "alive and working".
              filter: complete || failed ? 'none' : 'drop-shadow(0 0 4px var(--color-glow))',
              transition: 'stroke-dasharray 600ms cubic-bezier(0.16, 1, 0.3, 1)',
            }}
          />

          {/* travelling pulse along the completed portion — "actively working" */}
          {showComet && (
            <path
              d={d}
              fill="none"
              stroke="#fff"
              strokeWidth="2.5"
              strokeLinecap="round"
              pathLength="100"
              opacity="0.75"
              strokeDasharray="9 1000"
              style={{ filter: 'drop-shadow(0 0 5px var(--color-accent))' }}
            >
              <animate
                attributeName="stroke-dashoffset"
                from="0"
                to={-(doneFrac - 9)}
                dur="1.9s"
                repeatCount="indefinite"
              />
              <animate
                attributeName="opacity"
                values="0;0.75;0.75;0"
                keyTimes="0;0.15;0.8;1"
                dur="1.9s"
                repeatCount="indefinite"
              />
            </path>
          )}
        </svg>
      )}
    </div>
  );
}

function Node({ step, state, index, duration }) {
  const { Icon } = step;
  const size = state === 'active' || state === 'failed' ? 30 : 24;
  const base = { width: size, height: size };
  let cls = 'border-rule2 text-muted'; // pending
  base.background = 'var(--color-paper)'; // opaque: the rail must not cross the node
  let inner;
  let tip = `${step.label} — ${step.sub}`;

  if (state === 'pending') {
    // Reference nodes are numbered 1-5; the icon appears once the stage is live.
    inner = <span className="text-[13px] font-semibold leading-none">{index + 1}</span>;
    tip = `${step.label} — not started yet`;
  } else if (state === 'failed') {
    cls = 'border-transparent text-white';
    base.background = 'linear-gradient(160deg, var(--color-danger) 0%, #7a0000 100%)';
    // Failed is terminal: flat, contained, NO glow (glow means alive).
    inner = <X size={18} strokeWidth={2.4} />;
    tip = `Failed at ${step.label} — ${step.sub}`;
  } else if (state === 'active') {
    cls = 'border-transparent text-white';
    base.background = 'var(--grad-accent)';
    base.boxShadow = '0 0 18px var(--color-glow), 0 0 0 5px rgba(239,68,68,0.12)';
    inner = <Icon size={18} strokeWidth={2} />;
    tip = `${step.label} — in progress · ${step.sub}`;
  } else if (state === 'done') {
    cls = 'border-ok/50 text-ok';
    // OPAQUE. A translucent fill let the connector path show straight through
    // the middle of every completed node, so the finished pipeline looked like
    // a line with holes punched in it rather than a row of solid checkpoints.
    base.background = 'color-mix(in oklab, var(--color-ok) 13%, var(--color-paper))';
    inner = <Check size={15} strokeWidth={2.6} />;
    tip = duration != null
      ? `${step.label} — completed in ${duration < 10 ? duration.toFixed(1) : Math.round(duration)}s`
      : `${step.label} — completed`;
  }

  return (
    <div
      className="group flex flex-col items-center gap-1.5 min-w-0 w-full"
      style={{ zIndex: 1 }}
      title={tip}
    >
      <div
        className={`rounded-full border flex items-center justify-center shrink-0 transition-all duration-300 group-hover:scale-110 ${cls}`}
        style={base}
      >
        {inner}
      </div>
      <span className={`text-[10px] leading-none truncate max-w-full px-1 ${state === 'failed' ? 'text-danger font-semibold' : state === 'active' ? 'text-ink font-semibold' : state === 'done' ? 'text-ink2' : 'text-muted'}`}>
        {step.label}
      </span>
      {/* Real detail on hover, per node — duration for finished steps. */}
      {state === 'done' && duration != null && (
        <span className="hidden sm:block text-[9px] leading-none text-ok/80 opacity-0 group-hover:opacity-100 transition-opacity">
          {duration < 10 ? duration.toFixed(1) : Math.round(duration)}s
        </span>
      )}
    </div>
  );
}

export default function StageTracker({
  stage,
  compact = false,
  failed = false,
  complete = false,
  pct = null,
  durations = null,
}) {
  // A job that dies before its first successful stage (e.g. the source
  // download itself fails) never gets a real `stage` from progress.json —
  // that used to make the whole tracker vanish, which looked like nothing
  // had been built at all. Default to the first stage instead, so the
  // pipeline is always visible; a failure just marks wherever it died in red.
  const current = complete ? STEPS.length - 1 : (STAGE_INDEX[stage] ?? 0);

  // The connector fills to the real overall percentage when we have one, so
  // the line, the label under it and the metrics grid all show one number.
  //
  // It is floored at the current node's own position: a checkmarked node with
  // an unfilled line running through it is exactly the contradiction this
  // component exists to prevent, and backend `overall_pct` can legitimately lag
  // the stage marker. Whichever says "further along" wins.
  const stepFrac = (current / (STEPS.length - 1)) * 100;

  // The NUMBER shown to the user is always the backend's real overall percent
  // — the same value the OVERALL cell and the ring show, so the three can
  // never print different numbers for one job.
  const overallPct = complete
    ? 100
    : Math.max(0, Math.min(100, typeof pct === 'number' ? pct : stepFrac));

  // The line's GEOMETRY is additionally floored at the current node, so it can
  // never render unfilled underneath a node that already shows a checkmark.
  const doneFrac = complete ? 100 : Math.max(stepFrac, overallPct);

  const dur = (i) => {
    const v = Number(durations?.[STAGE_KEYS[i]]);
    return Number.isFinite(v) && v > 0 ? v : null;
  };

  const nodeState = (i) => {
    if (complete) return 'done';
    if (i < current) return 'done';
    if (i === current) return failed ? 'failed' : 'active';
    return 'pending';
  };

  // Compact variant for narrow columns: dots + icons only, no SVG path —
  // labels there don't have room (Tailwind breakpoints track the viewport,
  // not the column, which is what caused overlap before).
  if (compact) {
    return (
      <div>
        <div className="flex items-center gap-1">
          {STEPS.map((step, i) => {
            const { Icon } = step;
            const st = nodeState(i);
            return (
              <div key={step.label} className="flex items-center flex-1 last:flex-none">
                <div
                  className={`w-6 h-6 rounded-full border flex items-center justify-center shrink-0 transition-all ${
                    st === 'done'
                      ? 'border-ok/60 bg-ok/10 text-ok'
                      : st === 'pending'
                        ? 'border-rule2 bg-transparent text-muted'
                        : 'border-transparent text-white'
                  }`}
                  style={st === 'active' || st === 'failed' ? {
                    background: st === 'failed' ? 'linear-gradient(160deg, var(--color-danger) 0%, #7a0000 100%)' : 'var(--grad-accent)',
                    boxShadow: st === 'failed'
                      ? 'none'
                      : '0 0 10px var(--color-glow), 0 0 0 3px rgba(239,68,68,0.12)',
                  } : undefined}
                  title={step.label}
                >
                  {st === 'done' ? <Check size={11} /> : st === 'failed' ? <X size={12} strokeWidth={2.4} /> : <Icon size={11} strokeWidth={2} />}
                </div>
                {i < STEPS.length - 1 && (
                  <div className={`flex-1 h-px mx-1 ${i < current ? 'bg-ok/50' : 'bg-rule2'}`} />
                )}
              </div>
            );
          })}
        </div>
        <p className={`mt-2 text-[11px] truncate ${failed ? 'text-danger' : 'text-ink2'}`}>
          {failed
            ? `Failed at ${STEPS[current].label} — ${STEPS[current].sub}`
            : `Step ${current + 1}/${STEPS.length} · ${STEPS[current].label} — ${STEPS[current].sub}`}
        </p>
      </div>
    );
  }

  return (
    <div className="relative pt-2">
      <div className="relative">
        <PipelinePath
          doneFrac={doneFrac}
          animate={!failed && !complete}
          complete={complete}
          failed={failed}
        />
        <div className="flex">
          {STEPS.map((step, i) => (
            <div key={step.label} className="flex-1 min-w-0 flex justify-center">
              <Node step={step} state={nodeState(i)} index={i} duration={dur(i)} />
            </div>
          ))}
        </div>
      </div>

      {/* Slim overall bar + label under the whole row, so progress is readable
          before the eye ever reaches the metrics panel below. Same number as
          the ring and the OVERALL cell — one piece of state, three views. */}
      <div className="mt-2 flex items-center gap-3">
        <div className="flex-1 h-1 rounded-full bg-paper3 overflow-hidden">
          <div
            className="h-full rounded-full transition-[width] duration-[600ms] ease-out"
            style={{
              width: `${Math.max(overallPct, 1)}%`,
              background: failed
                ? 'var(--color-danger)'
                : complete || overallPct >= 100
                  ? 'color-mix(in oklab, var(--color-ok) 52%, transparent)'
                  : 'var(--grad-accent)',
              boxShadow: failed || complete ? 'none' : '0 0 8px var(--color-glow)',
            }}
          />
        </div>
        <span
          className="readout text-[10px] uppercase tracking-wider shrink-0 tabular-nums"
          style={{
            color: failed ? 'var(--color-danger)' : complete ? 'var(--color-ok)' : 'var(--color-muted)',
          }}
        >
          {failed ? `failed at ${Math.round(overallPct)}%` : `${Math.round(overallPct)}% complete`}
        </span>
      </div>
    </div>
  );
}
