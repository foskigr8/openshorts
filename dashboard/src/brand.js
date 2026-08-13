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

// The wide lockup, trimmed tight — for anywhere with horizontal room.
export const LOGO_SOURCES = [
  '/logo-ai4ts.png',
  '/logo-openshorts.png', // last resort, so something always renders
];

// The same lockup on a square canvas, nothing cropped — for the tab and for
// the collapsed nav, where a 1.6:1 image squeezed into a square badge wastes
// 40% of it on empty space and reads as a speck.
export const ICON_SOURCES = [
  '/favicon-ai4ts.png',
  '/logo-openshorts.png',
];

// Resolved once per page load per list, not once per component.
const cache = {};
const waiting = {};

function resolveFrom(list, key, cb) {
  if (cache[key]) { cb(cache[key]); return; }
  waiting[key] = waiting[key] || [];
  waiting[key].push(cb);
  if (waiting[key].length > 1) return;   // a probe is already in flight

  const settle = (href) => {
    cache[key] = href;
    waiting[key].splice(0).forEach((fn) => fn(href));
  };

  const tryAt = (i) => {
    if (i >= list.length) { settle(list[list.length - 1]); return; }
    const probe = new Image();
    probe.onload = () => settle(list[i]);
    probe.onerror = () => tryAt(i + 1);
    probe.src = list[i];
  };

  try {
    tryAt(0);
  } catch (_) {
    settle(list[list.length - 1]);
  }
}

export const resolveLogo = (cb) => resolveFrom(LOGO_SOURCES, 'logo', cb);
export const resolveIcon = (cb) => resolveFrom(ICON_SOURCES, 'icon', cb);

/** Apply the brand to the document (title + favicon), at boot. */
export function applyBrand() {
  try {
    document.title = BRAND.name;
    resolveIcon((href) => {
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
