import { useEffect, useState } from 'react';
import { Film, Loader2 } from 'lucide-react';
import { apiFetch } from '../lib/api';

/**
 * Source section (round-5): the job's original long-form video + lazily
 * extracted stills. The video plays through the low-bitrate +faststart
 * preview proxy (/api/source/{job}/preview.mp4) so it starts instantly
 * without buffering the full-quality original; the stills load from the
 * static mount, so the whole panel is fast by construction.
 */
export default function SourcePanel({ jobId }) {
  const [images, setImages] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    setImages(null);
    apiFetch(`/api/source/${jobId}/images`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (cancelled) return;
        if (!d) setError('Could not load source images.');
        else setImages(d.frames || []);
      })
      .catch(() => { if (!cancelled) setError('Could not load source images.'); });
    return () => { cancelled = true; };
  }, [jobId]);

  return (
    <div className="space-y-4 pb-8">
      <div className="rounded-input border border-rule overflow-hidden bg-black">
        <video
          src={`/api/source/${jobId}/preview.mp4`}
          controls
          preload="metadata"
          playsInline
          className="w-full aspect-video object-contain"
        />
      </div>

      <div>
        <p className="readout text-[10px] uppercase tracking-wider text-muted mb-2">
          Source Stills
        </p>
        {images === null ? (
          <div className="flex justify-center py-8 text-muted">
            <Loader2 size={20} className="animate-spin text-brass" />
          </div>
        ) : error ? (
          <p className="text-xs text-danger">{error}</p>
        ) : images.length === 0 ? (
          <div className="flex flex-col items-center gap-2 text-muted py-8">
            <Film size={22} />
            <p className="text-xs lowercase">no source stills yet</p>
          </div>
        ) : (
          <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
            {images.map((f) => (
              <img
                key={f.index}
                src={f.url}
                alt={`source still ${f.index + 1}`}
                loading="lazy"
                className="w-full aspect-video object-cover rounded-md border border-rule hover:border-rule2 transition-colors"
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
