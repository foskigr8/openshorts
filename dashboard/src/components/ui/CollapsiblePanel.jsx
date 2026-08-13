import { useCallback, useEffect, useState } from 'react';
import { ChevronDown } from 'lucide-react';

/**
 * A panel that remembers whether you wanted it.
 *
 * Every large surface in the workspace — the right rail, the log, the history
 * sections — needed the same behaviour: fold away, stay folded across
 * reloads, and take no more room than a title bar when closed. One primitive
 * rather than three implementations, so a new panel later is one import.
 */

export function usePanelState(key, defaultOpen = true) {
  const storageKey = `os_panel:${key}`;
  const [open, setOpen] = useState(() => {
    try {
      const v = localStorage.getItem(storageKey);
      return v === null ? defaultOpen : v === '1';
    } catch (_) {
      return defaultOpen;
    }
  });
  useEffect(() => {
    try { localStorage.setItem(storageKey, open ? '1' : '0'); } catch (_) { /* ignore */ }
  }, [storageKey, open]);
  const toggle = useCallback(() => setOpen((v) => !v), []);
  return [open, toggle, setOpen];
}

export default function CollapsiblePanel({
  id, title, subtitle = null, right = null, defaultOpen = true,
  children, className = '', bodyClassName = '',
}) {
  const [open, toggle] = usePanelState(id, defaultOpen);
  return (
    <div className={`rounded-card border border-rule bg-paper overflow-hidden ${className}`}>
      <div className="flex items-center gap-2 px-3 py-2 border-b border-rule">
        <button
          onClick={toggle}
          className="flex items-center gap-2 min-w-0 flex-1 text-left group"
          aria-expanded={open}
        >
          <ChevronDown
            size={14}
            className={`shrink-0 text-muted transition-transform duration-200 ${open ? '' : '-rotate-90'}`}
          />
          <span className="text-xs text-ink2 lowercase font-medium truncate group-hover:text-ink transition-colors">
            {title}
          </span>
          {subtitle && (
            <span className="readout text-[10px] text-muted truncate hidden sm:inline">{subtitle}</span>
          )}
        </button>
        {right && <div className="shrink-0 flex items-center gap-2">{right}</div>}
      </div>
      {open && <div className={bodyClassName}>{children}</div>}
    </div>
  );
}
