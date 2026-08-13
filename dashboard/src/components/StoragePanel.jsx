import { useEffect, useState } from 'react';
import { HardDrive, Loader2, Trash2, AlertTriangle, Check } from 'lucide-react';
import { apiFetch, apiJson } from '../lib/api';

/**
 * Storage, and the one button that empties it.
 *
 * Nothing on this server deletes your work on a timer — clips, sources and
 * their backup copies are kept until you say otherwise. That policy only
 * works if there is a real way to say otherwise, including for the copies in
 * remote storage, which used to survive a local delete and reappear in the
 * library on the next read.
 *
 * Typing the phrase is the confirmation; the server demands it too, so a
 * stray click here can't wipe anything.
 */
const PHRASE = 'wipe everything';

const fmtBytes = (b) => {
  const n = Number(b);
  if (!Number.isFinite(n) || n <= 0) return '0 B';
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
};

export default function StoragePanel({ onWiped }) {
  const [system, setSystem] = useState(null);
  const [arming, setArming] = useState(false);
  const [phrase, setPhrase] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const load = () => {
    apiFetch('/api/system')
      .then((r) => (r.ok ? r.json() : null))
      .then(setSystem)
      .catch(() => {});
  };
  useEffect(load, []);

  const storage = system?.storage;
  const pct = storage ? Math.min(100, Math.max(0, storage.pct || 0)) : 0;

  const wipe = async () => {
    setBusy(true);
    setError('');
    try {
      const data = await apiJson('/api/storage/wipe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: PHRASE }),
      });
      setResult(data);
      setArming(false);
      setPhrase('');
      load();
      onWiped?.();
    } catch (e) {
      setError(e.detail || 'Could not complete the wipe.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card p-4 sm:p-6 mt-8">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
            <HardDrive size={16} className="text-brass" />
          </div>
          <h2 className="text-base font-medium text-ink lowercase">Storage</h2>
        </div>
        {storage && (
          <span className="readout text-[11px] text-muted">
            {storage.used_gb} / {storage.cap_gb} GB
          </span>
        )}
      </div>

      {storage && (
        <div className="h-2 rounded-full bg-paper3 overflow-hidden mb-4">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${Math.max(pct, 1)}%`,
              background: pct >= 85 ? 'var(--color-danger)' : 'var(--grad-accent)',
            }}
          />
        </div>
      )}

      <p className="text-xs text-muted mb-5 leading-relaxed">
        Clips, downloaded sources and their backup copies are kept until you delete them —
        nothing here expires on a timer. Wiping removes every project on this server,
        the cached sources, and the copies in remote storage.
      </p>

      {result && (
        <p className="text-xs text-ok flex items-center gap-1.5 mb-4">
          <Check size={13} />
          Removed {result.jobs_deleted} project{result.jobs_deleted === 1 ? '' : 's'} ·
          {' '}{fmtBytes(result.freed_bytes + (result.sources_freed_bytes || 0))} freed
          {result.storage_files_deleted ? ` · ${result.storage_files_deleted} backup files` : ''}
        </p>
      )}

      {!arming ? (
        <button
          onClick={() => { setArming(true); setResult(null); setError(''); }}
          className="btn-ghost px-4 py-2 text-sm text-danger hover:text-danger"
        >
          <Trash2 size={14} /> wipe everything
        </button>
      ) : (
        <div className="rounded-input border border-danger/40 bg-danger/5 p-4 space-y-3">
          <p className="text-xs text-ink2 flex items-start gap-2">
            <AlertTriangle size={14} className="text-danger shrink-0 mt-0.5" />
            <span>
              This deletes every project, every cached source, and their copies in remote
              storage. It cannot be undone. Type <strong className="text-ink">{PHRASE}</strong> to confirm.
            </span>
          </p>
          <input
            autoFocus
            value={phrase}
            onChange={(e) => setPhrase(e.target.value)}
            placeholder={PHRASE}
            className="input-field"
          />
          {error && <p className="text-xs text-danger">{error}</p>}
          <div className="flex gap-2">
            <button
              onClick={wipe}
              disabled={busy || phrase.trim().toLowerCase() !== PHRASE}
              className="btn-primary px-4 py-2 text-sm disabled:opacity-40"
            >
              {busy ? <><Loader2 size={14} className="animate-spin" /> wiping…</> : 'wipe everything'}
            </button>
            <button
              onClick={() => { setArming(false); setPhrase(''); setError(''); }}
              className="btn-ghost px-4 py-2 text-sm"
            >
              cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
