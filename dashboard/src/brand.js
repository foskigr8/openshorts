/**
 * Everything that identifies this app, in one place.
 *
 * The rename is coming (new name, new logo, new favicon) and it must not be a
 * hunt through JSX. Every visible occurrence of the product's identity reads
 * from here: the sidebar mark, the document title, the page shells. Change
 * these four values and the whole app is rebranded.
 *
 * `logo` and `favicon` are paths under public/.
 */
export const BRAND = {
  name: 'Ai4Ts',
  tagline: 'turn long video into shorts',
  // Vector, so it is sharp at 16px in a tab and at 36px in the nav, and has
  // no white plate around it on a dark surface. To use your own artwork
  // instead, drop it in dashboard/public/ and point these two at it — a PNG
  // is fine. Anything that fails to load falls back rather than showing a
  // broken-image glyph.
  logo: '/logo-ai4ts.svg',
  favicon: '/logo-ai4ts.svg',
};

export const LOGO_FALLBACK = '/logo-openshorts.png';

/**
 * Apply the brand to the document (title + favicon), at boot.
 *
 * The favicon is only swapped in AFTER the image is confirmed to load. Setting
 * it blind is how a missing file becomes a broken-image glyph in the tab —
 * which is worse than the icon we already had.
 */
export function applyBrand() {
  try {
    document.title = BRAND.name;
    const apply = (href) => {
      let link = document.querySelector("link[rel='icon']");
      if (!link) {
        link = document.createElement('link');
        link.rel = 'icon';
        document.head.appendChild(link);
      }
      link.type = href.endsWith('.svg') ? 'image/svg+xml' : 'image/png';
      link.href = href;
    };
    const probe = new Image();
    probe.onload = () => apply(BRAND.favicon);
    probe.onerror = () => apply(LOGO_FALLBACK);
    probe.src = BRAND.favicon;
  } catch (_) { /* SSR / no DOM — nothing to brand */ }
}

export default BRAND;
