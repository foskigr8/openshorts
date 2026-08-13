/**
 * Everything that identifies this app, in one place.
 *
 * The mark is resolved from a LIST, tried in order, first one that loads
 * wins — so a missing or misnamed file degrades to the previous icon instead
 * of rendering a broken-image glyph in the nav and the tab.
 *
 * To swap the artwork: drop the file in dashboard/public/ and put its path
 * first in LOGO_SOURCES. Transparent background, roughly square.
 */
export const BRAND = {
  name: 'Ai4Ts',
  tagline: 'turn long video into shorts',
};

export const LOGO_SOURCES = [
  '/logo-ai4ts.png',      // the real mark, trimmed and squared
  '/logo-openshorts.png', // last resort, so something always renders
];

// Resolved once per page load rather than per component.
let resolved = null;
const waiting = [];

export function resolveLogo(cb) {
  if (resolved) { cb(resolved); return; }
  waiting.push(cb);
  if (waiting.length > 1) return;   // a probe is already in flight

  const settle = (href) => {
    resolved = href;
    waiting.splice(0).forEach((fn) => fn(href));
  };

  const tryAt = (i) => {
    if (i >= LOGO_SOURCES.length) {
      settle(LOGO_SOURCES[LOGO_SOURCES.length - 1]);
      return;
    }
    const probe = new Image();
    probe.onload = () => settle(LOGO_SOURCES[i]);
    probe.onerror = () => tryAt(i + 1);
    probe.src = LOGO_SOURCES[i];
  };

  try {
    tryAt(0);
  } catch (_) {
    settle(LOGO_SOURCES[LOGO_SOURCES.length - 1]);
  }
}

/** Apply the brand to the document (title + favicon), at boot. */
export function applyBrand() {
  try {
    document.title = BRAND.name;
    resolveLogo((href) => {
      let link = document.querySelector("link[rel='icon']");
      if (!link) {
        link = document.createElement('link');
        link.rel = 'icon';
        document.head.appendChild(link);
      }
      link.type = href.endsWith('.svg') ? 'image/svg+xml' : 'image/png';
      link.href = href;
    });
  } catch (_) { /* SSR / no DOM — nothing to brand */ }
}

export default BRAND;
