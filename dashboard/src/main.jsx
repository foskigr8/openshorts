import { StrictMode, useState, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import Legal from './Legal.jsx'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import { ProjectProvider } from './contexts/ProjectContext'
import { capture as captureAttribution } from './lib/attribution'
import { BRAND, applyBrand } from './brand'
import PricingPage from './components/PricingPage'
import AccountPage from './components/AccountPage'
import LoginModal from './components/LoginModal'

function PageShell({ title, children }) {
  return (
    <div className="min-h-screen bg-paper text-ink2">
      <header className="h-16 border-b border-rule bg-paper flex items-center justify-between px-6">
        <a href="#app" className="font-display lowercase text-lg text-ink">{BRAND.name}</a>
        <a href="#app" className="text-sm lowercase text-muted hover:text-ink transition-colors">← Back to app</a>
      </header>
      <main className="p-8">
        {title && <h1 className="font-display lowercase text-3xl text-ink text-center mb-10">{title}</h1>}
        {children}
      </main>
    </div>
  );
}

function PricingView() {
  const [showLogin, setShowLogin] = useState(false);
  return (
    <PageShell>
      <PricingPage onRequireLogin={() => setShowLogin(true)} />
      {showLogin && <LoginModal onClose={() => setShowLogin(false)} />}
    </PageShell>
  );
}

function AccountView() {
  const { isSignedIn, loading } = useAuth();
  useEffect(() => {
    if (!loading && !isSignedIn) window.location.hash = '#/pricing';
  }, [loading, isSignedIn]);
  return <PageShell><AccountPage /></PageShell>;
}

function Root() {
  // The app IS the site. There is no marketing landing page any more, so an
  // unrecognised hash resolves to the workspace instead of a splash screen you
  // have to click past on every visit.
  const resolveView = () => {
    const hash = window.location.hash || '';
    if (hash.startsWith('#/auth/')) return 'auth';       // AuthContext consumes then redirects
    if (hash.startsWith('#/account')) return 'account';
    if (hash.startsWith('#/pricing')) return 'pricing';
    if (hash === '#legal') return 'legal';
    return 'app';
  };

  const [view, setView] = useState(resolveView);

  useEffect(() => {
    const handleHashChange = () => setView(resolveView());
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  if (view === 'legal') return <Legal />;
  if (view === 'pricing') return <PricingView />;
  if (view === 'account') return <AccountView />;
  if (view === 'auth') {
    return <div className="min-h-screen flex items-center justify-center bg-background text-zinc-400">Signing you in…</div>;
  }
  return <App />;
}

// Before React mounts: AuthContext rewrites the URL on auth redirects, which
// would destroy the referrer and any UTM params we still need to read.
captureAttribution();
applyBrand();

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AuthProvider>
      <ProjectProvider>
        <Root />
      </ProjectProvider>
    </AuthProvider>
  </StrictMode>,
)
