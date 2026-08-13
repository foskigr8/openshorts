import React from 'react';

/**
 * The difference between "the UI disappeared" and "something broke, here it
 * is".
 *
 * A React error anywhere in the tree unmounts the WHOLE tree — the screen
 * goes blank with the explanation only in the console, which nobody has open.
 * This catches it, keeps the app on screen, and gives you the message plus a
 * way back in without losing the projects in storage.
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // Still logged for anyone who does have devtools open.
    console.error('UI crash:', error, info?.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="min-h-screen bg-paper text-ink2 flex items-center justify-center p-6">
        <div className="card p-6 max-w-lg w-full space-y-4">
          <h1 className="font-display lowercase text-xl text-ink">something in the interface broke</h1>
          <p className="text-sm text-muted leading-relaxed">
            Your projects are safe — they live on the server and in this browser's storage.
            Reloading usually clears it.
          </p>
          <pre className="text-[11px] font-mono text-danger bg-paper2 border border-rule rounded-input p-3 overflow-x-auto whitespace-pre-wrap break-words max-h-40">
            {String(error?.message || error)}
          </pre>
          <div className="flex flex-wrap gap-2">
            <button onClick={() => window.location.reload()} className="btn-primary px-4 py-2 text-sm">
              reload
            </button>
            <button
              onClick={() => this.setState({ error: null })}
              className="btn-ghost px-4 py-2 text-sm"
            >
              try again without reloading
            </button>
            <button
              onClick={() => {
                // Last resort: a corrupt persisted project can crash on every
                // render, and then reloading just reproduces it.
                try { localStorage.removeItem('os_projects_v2'); } catch (_) { /* ignore */ }
                window.location.reload();
              }}
              className="btn-ghost px-4 py-2 text-sm text-danger hover:text-danger"
              title="Clears the workspace list in this browser. Nothing on the server is deleted."
            >
              reset the workspace
            </button>
          </div>
        </div>
      </div>
    );
  }
}
