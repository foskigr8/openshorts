import { useState } from 'react';
import { Plus, Loader2, CheckCircle2, AlertTriangle, Clock } from 'lucide-react';
import CreateFlow from './CreateFlow';

/**
 * The home screen: your projects, and the way to start another.
 *
 * The app used to open on a marketing hero with a long form under it, so
 * "where is my work" had no answer on the first screen. Now the first screen
 * IS the work — a new-project tile that flies in alongside everything already
 * running or finished, each one a click away from the workspace.
 */

const STATE = {
  processing: { label: 'running', Icon: Loader2, tone: 'var(--color-accent)', spin: true },
  queued: { label: 'queued', Icon: Clock, tone: 'var(--color-muted)' },
  complete: { label: 'done', Icon: CheckCircle2, tone: 'var(--color-ok)' },
  error: { label: 'stopped', Icon: AlertTriangle, tone: 'var(--color-danger)' },
  cancelled: { label: 'cancelled', Icon: AlertTriangle, tone: 'var(--color-danger)' },
};

function ProjectTile({ project, index, onOpen }) {
  const s = STATE[project.status] || STATE.complete;
  const { Icon } = s;
  const pct = project.pct;
  return (
    <button
      type="button"
      onClick={() => onOpen(project)}
      className="tile-in group text-left rounded-card border border-rule bg-paper2 p-4 w-[220px]
                 hover:border-[color:color-mix(in_oklab,var(--color-accent)_45%,var(--color-rule-2))]
                 hover:-translate-y-1 transition-all duration-200"
      style={{ animationDelay: `${60 + index * 70}ms` }}
    >
      <div className="flex items-center gap-2 mb-3">
        <Icon size={14} className={s.spin ? 'animate-spin' : ''} style={{ color: s.tone }} />
        <span className="readout text-[9px] uppercase tracking-[0.14em]" style={{ color: s.tone }}>
          {s.label}
        </span>
        {pct != null && s.spin && (
          <span className="readout text-[10px] text-muted tabular-nums ml-auto">{Math.round(pct)}%</span>
        )}
      </div>
      <p className="text-sm text-ink line-clamp-2 leading-snug min-h-[2.6em]" title={project.title}>
        {project.title || 'Untitled project'}
      </p>
      <div className="mt-3 h-1 rounded-full bg-paper3 overflow-hidden">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{
            width: `${project.status === 'complete' ? 100 : Math.max(pct || 0, 2)}%`,
            background: project.status === 'complete'
              ? 'color-mix(in oklab, var(--color-ok) 55%, transparent)'
              : 'var(--grad-accent)',
          }}
        />
      </div>
    </button>
  );
}

export default function ProjectLauncher({
  projects = [], onOpenProject, onSubmit, starting = false, error = '',
}) {
  const [creating, setCreating] = useState(false);

  if (creating) {
    return (
      <div className="h-full flex flex-col animate-fade px-4 py-6">
        <CreateFlow
          onSubmit={onSubmit}
          onCancel={() => setCreating(false)}
          starting={starting}
          error={error}
        />
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto custom-scrollbar">
      <div className="min-h-full flex flex-col items-center justify-center px-6 py-10 gap-10">
        <div className="text-center space-y-2">
          <h1 className="font-display lowercase text-4xl md:text-5xl text-ink tracking-tight">
            {projects.length ? 'your projects' : 'start something'}
          </h1>
          <p className="text-muted text-sm">
            {projects.length
              ? 'Pick one up, or start another — they run side by side.'
              : 'Every video you cut lives here as a project you can come back to.'}
          </p>
        </div>

        <div className="flex flex-wrap items-stretch justify-center gap-4 max-w-4xl">
          <button
            type="button"
            onClick={() => setCreating(true)}
            className="tile-in group rounded-card border border-dashed border-rule2 bg-paper2/60 w-[220px]
                       flex flex-col items-center justify-center gap-3 py-8
                       hover:border-[color:var(--color-accent)] hover:-translate-y-1 transition-all duration-200"
          >
            <span
              className="w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300
                         group-hover:scale-110"
              style={{
                background: 'var(--grad-accent)',
                boxShadow: '0 0 24px -4px var(--color-glow)',
              }}
            >
              <Plus size={22} className="text-white" />
            </span>
            <span className="text-sm lowercase text-ink">new project</span>
          </button>

          {projects.map((p, i) => (
            <ProjectTile key={p.id} project={p} index={i} onOpen={onOpenProject} />
          ))}
        </div>
      </div>
    </div>
  );
}
