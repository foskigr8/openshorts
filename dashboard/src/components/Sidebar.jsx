import {
  Home, Image, HardDrive, History, Settings, Flame, LayoutGrid,
  Loader2, TrendingUp, Sparkles, ChevronLeft, ChevronRight,
} from 'lucide-react';
import { BRAND, LOGO_FALLBACK } from '../brand';

/**
 * The navigation rail.
 *
 * A module-level component, deliberately: while it lived inside App() it was
 * a new component type on every render, so React unmounted and rebuilt the
 * whole menu every couple of seconds as job progress ticked in.
 *
 * Icons only until you ask for words — five destinations you already know did
 * not need 256px of permanent labels, and the workspace wants that width.
 */
export default function Sidebar({
  activeTab, onSelect, billingEnabled, isSignedIn,
  todayStats, liveCount, open, onToggle,
}) {
  const navItems = [
    { id: 'dashboard', icon: Home, label: 'Home' },
    { id: 'thumbnails', icon: Image, label: 'YouTube Studio' },
    // Sources are disk-backed like history — visible on self-host always,
    // behind sign-in in cloud mode.
    ...(!billingEnabled || isSignedIn ? [{ id: 'sources', icon: HardDrive, label: 'Sources' }] : []),
    // History must be reachable on self-host too: /api/history is disk-backed
    // and works without sign-in, so the tab was hidden exactly when the user
    // needed it most ("my projects disappeared" was this, not data loss).
    // Cloud mode still gates it behind sign-in (R2 library is per-account).
    ...(!billingEnabled || isSignedIn ? [{ id: 'history', icon: History, label: 'History' }] : []),
    { id: 'settings', icon: Settings, label: 'Settings' },
  ];

  // Icons only until you ask for words. The nav was 256px of permanent
  // labels for five destinations you already know — the workspace wants
  // that width more than the menu does. The choice is remembered.
  const wide = open;

  return (
    <div
      className={`bg-paper2 border-r border-rule flex flex-col h-full shrink-0 transition-[width] duration-300 ${
        wide ? 'w-60' : 'w-[68px]'
      }`}
    >
      <div className={`flex items-center gap-3 ${wide ? 'px-5 py-5' : 'px-3 py-5 justify-center'}`}>
        <button
          onClick={onToggle}
          title={wide ? 'collapse the menu' : 'expand the menu'}
          className="w-9 h-9 bg-paper3 rounded-input flex items-center justify-center shrink-0
                     overflow-hidden border border-rule hover:border-rule2 transition-colors relative group"
        >
          <img src={BRAND.logo} alt="" className="w-full h-full object-contain p-1 group-hover:opacity-25 transition-opacity" />
          <span className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
            {wide ? <ChevronLeft size={15} className="text-ink" /> : <ChevronRight size={15} className="text-ink" />}
          </span>
        </button>
        {wide && (
          <span className="font-display lowercase text-lg text-ink truncate">{BRAND.name}</span>
        )}
      </div>

      <nav className={`flex-1 py-4 space-y-1 ${wide ? 'px-4' : 'px-3'}`}>
        {navItems.map((item) => {
          const NavIcon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelect(item.id)}
              title={item.label}
              className={`relative w-full flex items-center gap-3 py-2.5 rounded-input transition-all ${
                wide ? 'px-3' : 'px-0 justify-center'
              } ${isActive ? 'text-ink' : 'text-muted hover:text-ink2 hover:bg-paper3/50'}`}
              style={isActive ? {
                background: 'linear-gradient(90deg, rgba(239,68,68,0.18) 0%, rgba(239,68,68,0.04) 65%, transparent 100%)',
                boxShadow: 'inset 0 0 0 1px rgba(239,68,68,0.16)',
              } : undefined}
            >
              {isActive && (
                <span
                  className="absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-full"
                  style={{ background: 'var(--color-accent)', boxShadow: '0 0 8px var(--color-glow)' }}
                  aria-hidden="true"
                />
              )}
              <NavIcon size={18} className={`shrink-0 ${isActive ? 'text-brass glow-accent' : ''}`} />
              {wide && <span className="text-sm flex-1 text-left truncate">{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Today's counts — only when there is room for the words. */}
      {wide && (
      <div className="px-4 pb-4">
        <div className="card p-3.5">
          <div className="flex items-center justify-between mb-2.5">
            <p className="readout text-[10px] uppercase tracking-wider text-muted">today</p>
            <Flame size={13} className="text-brass glow-accent" />
          </div>
          <div className="space-y-2">
            {[
              { icon: LayoutGrid, label: 'shorts generated', value: todayStats.generated, tone: 'text-ink' },
              {
                icon: Loader2,
                label: 'processing',
                // Reconciled with the job this tab is actually running: a job
                // visible in the main panel must never coexist with a "0
                // processing" sidebar, even in the window before the backend
                // history has caught up.
                value: Math.max(todayStats.processing, liveCount),
                tone: 'text-brass',
                spin: liveCount > 0,
              },
              { icon: TrendingUp, label: 'success rate', value: `${todayStats.successRate}%`, tone: 'text-ok' },
            ].map((s) => {
              const SIcon = s.icon;
              return (
                <div key={s.label} className="flex items-center gap-2">
                  <SIcon size={13} className={`shrink-0 ${s.tone === 'text-ink' ? 'text-muted' : s.tone} ${s.spin ? 'animate-spin' : ''}`} />
                  <span className="text-[11px] lowercase text-muted flex-1 truncate">{s.label}</span>
                  {/* Never show a static 0 that could be mistaken for a real
                      count while the first fetch is still in flight. */}
                  {todayStats.loaded ? (
                    <span className={`text-xs font-semibold ${s.tone}`}>{s.value}</span>
                  ) : (
                    <span className="inline-block w-6 h-3 rounded bg-paper3 animate-pulse" />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
      )}

      {/* Footer: no landing page, no upstream repo link, no support address
          for somebody else's product. Only what this workspace owns. */}
      <div className={`border-t border-rule space-y-1 ${open ? 'p-4' : 'p-3'}`}>
        {billingEnabled && (
          <a
            href="#/pricing"
            className="flex items-center gap-2 px-3 py-1.5 text-xs lowercase text-muted hover:text-ink2 transition-colors"
          >
            <span className="icon-chip-muted !w-6 !h-6 shrink-0"><Sparkles size={13} /></span>
            {open && <span className="truncate">plans &amp; pricing</span>}
          </a>
        )}
      </div>
    </div>
  );

}
