import { CheckCircle2, Play, SlidersHorizontal, Columns2 } from 'lucide-react';
import { getApiUrl } from '../../config';

/**
 * A finished clip, compact — the same object whether the panel is laid out as
 * a sliding row or a stack.
 *
 * Two actions, and they mean different things:
 *   compare — puts this clip BESIDE the source preview, both playing the same
 *             moment, which is the whole point of having a source on screen.
 *   edit    — opens the full editor card at full width underneath the strip.
 *             It used to expand inside the strip, squeezed to 300px, where its
 *             action grid collapsed into overlapping labels and the card was
 *             clipped by the panel edge.
 */
export default function ClipRow({ clip, index, jobId, active = false, onCompare, onExpand }) {
  const duration = clip.end && clip.start ? Math.max(0, clip.end - clip.start) : null;
  const mmss = duration != null
    ? `${String(Math.floor(duration / 60)).padStart(2, '0')}:${String(Math.round(duration % 60)).padStart(2, '0')}`
    : null;

  return (
    <div
      className={`group rounded-input border overflow-hidden transition-colors duration-200 h-[104px] ${
        active
          ? 'border-brass/60'
          : 'border-rule hover:border-[color:color-mix(in_oklab,var(--color-accent)_35%,var(--color-rule-2))]'
      }`}
      style={{
        background: 'linear-gradient(180deg, rgba(255,255,255,0.03) 0%, transparent 45%), var(--color-paper)',
      }}
    >
      <div className="flex items-stretch gap-2.5 p-2.5 h-full">
        <button
          onClick={onCompare}
          className="w-[54px] shrink-0 relative overflow-hidden rounded-md bg-black"
          title="play beside the source"
        >
          <img
            src={getApiUrl(`/api/thumbnails/${jobId}/${index}`)}
            alt=""
            className="w-full h-full object-cover"
            onError={(e) => { e.currentTarget.style.visibility = 'hidden'; }}
          />
          <span className="absolute inset-0 flex items-center justify-center bg-black/25 group-hover:bg-black/45 transition-colors">
            <Play size={15} className="text-white drop-shadow" fill="white" />
          </span>
          <span className="absolute top-1 left-1 readout text-[8px] text-white/80">
            {String(index + 1).padStart(2, '0')}
          </span>
        </button>

        <div className="min-w-0 flex-1 flex flex-col">
          <p
            className="text-[13px] text-ink leading-snug line-clamp-2"
            title={clip.video_title_for_youtube_short}
          >
            {clip.video_title_for_youtube_short || `Clip ${index + 1}`}
          </p>

          <div className="mt-auto flex items-center gap-1.5">
            <span className="readout text-[9px] uppercase tracking-wider text-ok flex items-center gap-1">
              <CheckCircle2 size={9} /> ready
            </span>
            {mmss && <span className="readout text-[9px] text-muted">{mmss}</span>}

            <span className="flex-1" />

            <button
              onClick={onCompare}
              title="play beside the source"
              className="p-1.5 rounded-md text-muted hover:text-ink hover:bg-paper3 transition-colors"
            >
              <Columns2 size={13} />
            </button>
            <button
              onClick={onExpand}
              title={active ? 'close the editor' : 'edit this clip'}
              className={`p-1.5 rounded-md transition-colors ${
                active ? 'text-brass bg-brass/10' : 'text-muted hover:text-ink hover:bg-paper3'
              }`}
            >
              <SlidersHorizontal size={13} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
