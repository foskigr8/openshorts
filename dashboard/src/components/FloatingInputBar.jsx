import { useRef, useState } from 'react';
import { Link2, Upload, Youtube, FileText, Layers, Loader2, Paperclip } from 'lucide-react';

/**
 * <FloatingInputBar/> (round 3, item 4): bottom command bar — visible on Home
 * AND during processing (a second job genuinely queues behind the running one
 * via the backend job queue). Only real modes are interactive: YouTube / Upload
 * submit through the same handleProcess path as the full form; Text / Bulk have
 * no backend yet, so they render visibly disabled "coming soon" — never
 * clickable-but-dead.
 */
const SOURCES = [
  { id: 'youtube', label: 'YouTube', Icon: Youtube },
  { id: 'upload', label: 'Upload', Icon: Upload },
  { id: 'text', label: 'Text', Icon: FileText, comingSoon: true },
  { id: 'bulk', label: 'Bulk', Icon: Layers, comingSoon: true },
];

export default function FloatingInputBar({ onProcess, isProcessing }) {
  const [mode, setMode] = useState('youtube');
  const [url, setUrl] = useState('');
  const [file, setFile] = useState(null);
  const [sending, setSending] = useState(false);
  const fileRef = useRef(null);

  const submit = (e) => {
    e.preventDefault();
    if (sending || !canSubmit) return;
    if (mode === 'upload' && file) {
      setSending(true);
      onProcess({
        type: 'file', payload: file, acknowledged: true,
        outputFormat: 'vertical', clipCount: 8, longContextClips: 0,
      });
      setFile(null);
      setSending(false);
    } else if (url) {
      setSending(true);
      onProcess({
        type: 'url', payload: url, acknowledged: true,
        outputFormat: 'vertical', clipCount: 8, longContextClips: 0,
      });
      setUrl('');
      setSending(false);
    }
  };

  const canSubmit = mode === 'upload' ? !!file : !!url.trim();

  return (
    <div className="px-4 pb-4 shrink-0">
      <form
        onSubmit={submit}
        className="flex items-center gap-2 rounded-full border border-rule2 bg-paper2 pl-3 pr-2 py-2 shadow-[0_8px_30px_-12px_rgba(0,0,0,0.8)] focus-within:border-brass/60 transition-colors"
      >
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          className={`p-1.5 rounded-full transition-colors ${
            mode === 'upload' && file ? 'text-brass bg-brass/10' : 'text-muted hover:text-ink'
          }`}
          title="Attach a file (switches to Upload mode)"
          aria-label="attach file"
        >
          <Paperclip size={16} />
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => { if (e.target.files?.[0]) { setFile(e.target.files[0]); setMode('upload'); } }}
        />
        {mode === 'upload' && file ? (
          <span className="flex-1 min-w-0 truncate text-xs text-ink2">{file.name}</span>
        ) : (
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="paste a link / drop a file — generate while this job queues…"
            className="flex-1 min-w-0 bg-transparent text-xs text-ink outline-none placeholder:text-muted"
          />
        )}
        <button
          type="submit"
          disabled={!canSubmit || sending}
          className="px-4 py-2 rounded-full text-xs lowercase font-medium text-white disabled:opacity-40 transition-all"
          style={{ background: 'linear-gradient(90deg, #ef4444 0%, #7f1d1d 100%)' }}
        >
          {sending ? <Loader2 size={13} className="animate-spin" /> : 'Generate'}
        </button>
      </form>

      <div className="flex items-center justify-center gap-2 mt-2.5">
        {SOURCES.map(({ id, label, Icon, comingSoon }) => {
          const active = mode === id;
          return (
            <button
              key={id}
              type="button"
              disabled={comingSoon}
              onClick={() => { setMode(id); setUrl(''); setFile(null); }}
              className={`relative flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] transition-colors border ${
                active
                  ? 'border-brass/50 bg-brass/10 text-ink'
                  : comingSoon
                    ? 'border-rule text-muted/50 cursor-not-allowed'
                    : 'border-rule text-muted hover:border-rule2'
              }`}
            >
              <Icon size={12} />
              {label}
              {comingSoon && (
                <span className="absolute -top-2 -right-2 text-[7px] readout px-1 py-px rounded bg-paper3 border border-rule text-muted">
                  soon
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
