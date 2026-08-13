import { useCallback, useEffect, useState } from 'react';
import { ChevronDown } from 'lucide-react';

/**
 * A panel that folds, smoothly, and remembers whether you wanted it.
 *
 * The animation is the `grid-template-rows: 0fr → 1fr` technique rather than a
 * max-height guess: the row resolves to the content's real height, so it
 * glides to exactly the right size with no jump at the end and no magic
 * number to get wrong when the content grows. `min-h-0` on the inner element
 * is load-bearing — without it the grid child refuses to shrink below its
 * content and nothing moves.
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

/** Same idea as usePanelState, for a remembered choice rather than a flag. */
export function useStoredChoice(key, defaultValue) {
  const storageKey = `os_choice:${key}`;
  const [value, setValue] = useState(() => {
    try {
      return localStorage.getItem(storageKey) || defaultValue;
    } catch (_) {
      return defaultValue;
    }
  });
  useEffect(() => {
    try { localStorage.setItem(storageKey, value); } catch (_) { /* ignore */ }
  }, [storageKey, value]);
  return [value, setValue];
}

export default function CollapsiblePanel({
  id, title, subtitle = null, right = null, defaultOpen = true,
  children, className = '', bodyClassName = '', onOpen = null,
}) {
  const [open, toggle] = usePanelState(id, defaultOpen);

  // Let the panel tell its content it just became visible — the log uses this
  // to jump to the line you actually care about instead of the top.
  useEffect(() => { if (open) onOpen?.(); }, [open]);   // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <section className={`rounded-card border border-rule bg-paper/60 overflow-hidden ${className}`}>
      <header
        className="flex items-center gap-2 px-3 h-11 cursor-pointer select-none group
                   hover:bg-paper2/60 transition-colors"
        onClick={toggle}
      >
        <span
          className="w-6 h-6 rounded-md flex items-center justify-center shrink-0
                     text-muted group-hover:text-ink transition-colors"
        >
          <ChevronDown
            size={14}
            className="transition-transform duration-300 ease-[cubic-bezier(0.16,1,0.3,1)]"
            style={{ transform: open ? 'none' : 'rotate(-90deg)' }}
          />
        </span>
        <span className="text-xs text-ink2 lowercase font-medium truncate group-hover:text-ink transition-colors">
          {title}
        </span>
        {subtitle && (
          <span className="readout text-[10px] text-muted/70 truncate hidden sm:inline">{subtitle}</span>
        )}
        <span className="flex-1" />
        {right && (
          // Controls in the header must not toggle the panel underneath them.
          <span className="shrink-0 flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
            {right}
          </span>
        )}
      </header>

      <div
        className="grid transition-[grid-template-rows] duration-300 ease-[cubic-bezier(0.16,1,0.3,1)]"
        style={{ gridTemplateRows: open ? '1fr' : '0fr' }}
      >
        <div className="min-h-0 overflow-hidden">
          <div
            className={`transition-opacity duration-200 ${open ? 'opacity-100 delay-100' : 'opacity-0'} ${bodyClassName}`}
          >
            {children}
          </div>
        </div>
      </div>
    </section>
  );
}
