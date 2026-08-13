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
  name: 'Studio',
  tagline: 'turn long video into shorts',
  logo: '/logo-openshorts.png',
  favicon: '/logo-openshorts.png',
};

/** Apply the brand to the document (title + favicon). Called once at boot. */
export function applyBrand() {
  try {
    document.title = BRAND.name;
    let link = document.querySelector("link[rel='icon']");
    if (!link) {
      link = document.createElement('link');
      link.rel = 'icon';
      document.head.appendChild(link);
    }
    link.href = BRAND.favicon;
  } catch (_) { /* SSR / no DOM — nothing to brand */ }
}

export default BRAND;
