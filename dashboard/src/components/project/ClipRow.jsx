import { CheckCircle2, ChevronDown, Play } from 'lucide-react';
import { getApiUrl } from '../../config';

/**
 * A finished clip, compact.
 *
 * While a job renders, the finished clips used to appear as full editor cards
 * — 420px tall each, so four clips buried the pipeline you were trying to
 * watch. This is the same object at row height: poster, title, length, and a
 * play affordance. Click it and the full card unfolds in place, so nothing is
 * lost — it is just not all shouting at once.
 *
 * Deliberately the same shape as ClipSlotPlaceholder, so a rendered clip and a
 * pending one read as the same kind of thing in the list.
 */
export default function ClipRow({ clip, index, jobId, onExpand, onPlay }) {
  const duration = clip.end && clip.start ? Math.max(0, clip.end - clip.start) : null;
  const mmss = duration != null
    ? `${String(Math.floor(duration / 60)).padStart(2, '0')}:${String(Math.round(duration % 60)).padStart(2, '0')}`
    : null;

  return (
    <div
      className="group rounded-input border border-rule overflow-hidden transition-all duration-200
                 hover:border-[color:color-mix(in_oklab,var(--color-accent)_35%,var(--color-rule-2))]"
      style={{
        background: 'linear-gradient(180deg, rgba(255,255,255,0.03) 0%, transparent 45%), var(--color-paper)',
      }}
    >
      <div className="flex items-center gap-3 p-3">
        <button
          onClick={() => onPlay?.(index)}
          className="w-16 h-24 rounded-md shrink-0 relative overflow-hidden bg-black"
          title="play this clip"
        >
          <img
            src={getApiUrl(`/api/thumbnails/${jobId}/${index}`)}
            alt=""
            className="w-full h-full object-cover"
            onError={(e) => { e.currentTarget.style.visibility = 'hidden'; }}
          />
          <span className="absolute inset-0 flex items-center justify-center bg-black/25 group-hover:bg-black/40 transition-colors">
            <Play size={18} className="text-white drop-shadow" fill="white" />
          </span>
          <span className="absolute top-1 left-1 readout text-[8px] text-white/80">
            {String(index + 1).padStart(2, '0')}
          </span>
          {mmss && (
            <span className="absolute bottom-1 right-1 px-1 rounded bg-black/70 readout text-[8px] text-ink2">
              {mmss}
            </span>
          )}
        </button>

        <button onClick={() => onExpand(index)} className="min-w-0 flex-1 text-left">
          <p className="text-sm text-ink line-clamp-2 leading-snug" title={clip.video_title_for_youtube_short}>
            {clip.video_title_for_youtube_short || `Clip ${index + 1}`}
          </p>
          <span className="mt-1.5 inline-flex items-center gap-1.5 readout text-[9px] uppercase tracking-wider text-ok">
            <CheckCircle2 size={10} /> ready
          </span>
        </button>

        <button
          onClick={() => onExpand(index)}
          className="p-2 rounded-full text-muted hover:text-ink hover:bg-paper3 transition-colors shrink-0"
          title="open the editor for this clip"
          aria-label="expand clip"
        >
          <ChevronDown size={16} />
        </button>
      </div>
    </div>
  );
}
