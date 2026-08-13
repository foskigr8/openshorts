import { useRef, useState } from 'react';
import { ArrowRight, FileVideo, Link2, Paperclip, X } from 'lucide-react';

/**
 * Step 1 — where the video comes from.
 *
 * One field, not three tabs. The old form asked you to declare "YouTube link"
 * vs "video URL" first, which is a question about the app's plumbing rather
 * than about your video: the backend resolves the platform from the link
 * itself. Upload lives as a visible clip on the bubble, so it reads as the
 * alternative to pasting without occupying a whole tab row.
 */
export default function SourceStep({ value, onChange, onNext, urlEnabled = true }) {
  const fileRef = useRef(null);
  const [touched, setTouched] = useState(false);

  const file = value?.type === 'file' ? value.payload : null;
  const url = value?.type === 'url' ? value.payload : '';
  const ready = !!file || !!url.trim();

  const submit = (e) => {
    e?.preventDefault?.();
    setTouched(true);
    if (ready) onNext();
  };

  const pick = (f) => {
    if (f) onChange({ type: 'file', payload: f });
  };

  return (
    <form onSubmit={submit} className="w-full max-w-xl mx-auto text-center space-y-6">
      <div className="space-y-2">
        <h2 className="font-display lowercase text-3xl text-ink">what are we cutting?</h2>
        <p className="text-sm text-muted">
          Paste a link to any video, or bring your own file.
        </p>
      </div>

      {file ? (
        <div className="flex items-center gap-3 px-4 py-3.5 rounded-full border border-brass/50 bg-brass/5 text-left">
          <span className="icon-chip !w-9 !h-9 shrink-0 text-brass"><FileVideo size={17} /></span>
          <span className="flex-1 min-w-0 truncate text-sm text-ink">{file.name}</span>
          <button
            type="button"
            onClick={() => onChange(null)}
            className="p-1.5 rounded-full text-muted hover:text-ink hover:bg-paper3 transition-colors shrink-0"
            aria-label="remove file"
          >
            <X size={15} />
          </button>
        </div>
      ) : (
        <div
          className="relative"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); pick(e.dataTransfer.files?.[0]); }}
        >
          <Link2
            size={17}
            className="absolute left-4 top-1/2 -translate-y-1/2 text-muted pointer-events-none"
          />
          <input
            type="text"
            autoFocus
            inputMode="url"
            value={url}
            disabled={!urlEnabled}
            onChange={(e) => onChange({ type: 'url', payload: e.target.value })}
            placeholder={urlEnabled ? 'paste a video link' : 'link input is disabled on this server'}
            className="w-full h-14 pl-12 pr-14 rounded-full bg-paper2 border border-rule2 text-ink
                       placeholder:text-muted focus:outline-none focus:border-brass/60
                       transition-colors disabled:opacity-50"
          />
          {/* The upload affordance rides on the bubble: unmistakably an
              alternative to pasting, without a tab row of its own. */}
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            title="upload a file instead"
            className="absolute right-2 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full border border-rule2
                       bg-paper3 text-muted hover:text-brass hover:border-brass/50 transition-colors
                       flex items-center justify-center"
          >
            <Paperclip size={16} />
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => pick(e.target.files?.[0])}
          />
        </div>
      )}

      {touched && !ready && (
        <p className="text-xs text-danger">Add a link or a file to continue.</p>
      )}

      <button type="submit" disabled={!ready} className="btn-primary px-6 py-3 disabled:opacity-40">
        continue <ArrowRight size={16} />
      </button>
      <p className="text-[11px] text-muted lowercase">or drop a file anywhere on this panel</p>
    </form>
  );
}
