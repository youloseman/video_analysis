/* Flapp mobile platform bridge.
 *
 * Injected into the BUNDLED index.html only (by scripts/sync-www.mjs) — the
 * web build never loads this file, so nothing here can regress the website.
 *
 * Responsibilities:
 *   1. Point every server call at the real API origin. The SPA uses
 *      server-relative URLs ('/auth/...', '/analyze', ...) everywhere; inside
 *      the app it runs on a capacitor:// origin, so we transparently prefix
 *      fetch() and XMLHttpRequest URLs with the API base. Media URLs go
 *      through withJobToken() in the SPA, which reads FLAPP_API_BASE itself.
 *   2. Persist the auth token in native storage (Capacitor Preferences) —
 *      WebView localStorage can be evicted by the OS. We mirror writes and
 *      restore on boot; the SPA awaits FLAPP_BRIDGE.ready before checkAuth().
 *   3. Open site pages (/academy, /privacy, ...) in the system browser
 *      instead of navigating the WebView away from the app.
 *   4. Hide every purchase path. Store rules (Apple 3.1.1 / Play Billing)
 *      forbid selling digital subscriptions outside IAP; until the RevenueCat
 *      phase the app must show NO buy buttons and NO "purchase on the web"
 *      steering. Free tier + sign-in for existing subscribers is allowed.
 */
(function () {
  'use strict';

  var cfg = window.FLAPP_MOBILE_CONFIG || {};
  var BASE = (cfg.apiBase || '').replace(/\/+$/, '');
  var WEB = (cfg.webOrigin || BASE).replace(/\/+$/, '');
  var TOKEN_KEY = 'flapp_token';

  // No API base configured -> behave like the plain web app (used in dev).
  if (!BASE) { window.FLAPP_BRIDGE = { ready: Promise.resolve(), native: false }; return; }

  window.FLAPP_API_BASE = BASE;   // read by withJobToken() in the SPA
  document.documentElement.classList.add('flapp-native');

  var cap = window.Capacitor || null;
  var plugins = (cap && cap.Plugins) || {};

  /* -- 1. Route server-relative URLs to the API origin -------------------- */
  // Only strings starting with a single '/' are rewritten; absolute URLs,
  // data:/blob: URIs and protocol-relative '//' are left untouched, so a URL
  // already prefixed (e.g. by withJobToken) is never double-prefixed.
  function absolutize(u) {
    return (typeof u === 'string' && u.charCodeAt(0) === 47 && u.charCodeAt(1) !== 47)
      ? BASE + u : u;
  }
  var origFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    if (typeof input === 'string') input = absolutize(input);
    else if (input && input.url) input = new Request(absolutize(input.url), input);
    return origFetch(input, init);
  };
  var origOpen = XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open = function (method, url) {
    var args = Array.prototype.slice.call(arguments);
    args[1] = absolutize(url);
    return origOpen.apply(this, args);
  };

  /* -- 2. Durable auth token --------------------------------------------- */
  var ready = Promise.resolve();
  var prefs = plugins.Preferences;
  if (prefs) {
    var oSet = Storage.prototype.setItem, oRem = Storage.prototype.removeItem;
    Storage.prototype.setItem = function (k, v) {
      oSet.call(this, k, v);
      if (this === window.localStorage && k === TOKEN_KEY) {
        try { prefs.set({ key: k, value: v }); } catch (e) {}
      }
    };
    Storage.prototype.removeItem = function (k) {
      oRem.call(this, k);
      if (this === window.localStorage && k === TOKEN_KEY) {
        try { prefs.remove({ key: k }); } catch (e) {}
      }
    };
    // Restore: native storage wins only when the WebView lost its copy.
    ready = prefs.get({ key: TOKEN_KEY }).then(function (r) {
      if (r && r.value && !window.localStorage.getItem(TOKEN_KEY)) {
        oSet.call(window.localStorage, TOKEN_KEY, r.value);
      }
    }).catch(function () {});
  }

  /* -- 3. Site links open in the system browser --------------------------- */
  document.addEventListener('click', function (e) {
    var a = e.target && e.target.closest && e.target.closest('a[href]');
    if (!a) return;
    var h = a.getAttribute('href');
    if (!h || h.charCodeAt(0) !== 47 || h.charCodeAt(1) === 47) return; // only '/...'
    e.preventDefault();
    if (h === '/') return;               // wordmark: the SPA's own handler resets the app
    var url = WEB + h;
    if (plugins.Browser) plugins.Browser.open({ url: url });
    else window.open(url, '_blank');
  }, true);

  /* -- 4. No purchase paths in the store build ---------------------------- */
  // Every selector here is a place the web app sells something. The list
  // has to grow with the product: the Coach Notes card (.no-card, $12) was
  // added to the report after this list was written and shipped a buy
  // button into the native bundle for two days. When a new offer lands in
  // the SPA, its container goes here in the same commit.
  // The list is pinned by backend/tests/test_native_purchase_paths.py, in
  // both directions: a surface the SPA sells must be here, and a selector
  // here must still exist in the SPA. It drifted both ways once -- #botPricing
  // and #ctaExpert were gone, and .addon-strip had become the ORDERS box, so
  // the store build hid a subscriber's delivered reviews instead of a price.
  var st = document.createElement('style');
  st.textContent = [
    '.flapp-native #navPricing',
    // Teaser card: the upgrade row and the price tag.
    '.flapp-native .upsell .cta-row', '.flapp-native .upsell .price-tag',
    // Pricing page: the tier buttons and the add-on cards' buy block.
    '.flapp-native #pricing .tier-cta', '.flapp-native .solo-buy',
    // Coach Notes / Expert Review offer on the report: the price tag and the
    // button row (which also carries the "coming soon" and the Expert Review
    // rung once notes are delivered).
    '.flapp-native .no-price', '.flapp-native .no-cta',
    // The per-report unlock, wherever the teaser card renders it.
    '.flapp-native .btn-unlock'
  ].join(',') + '{display:none!important}';
  document.head.appendChild(st);

  /* -- 5. Fit the phone: notch, home bar, native feel ---------------------- */
  // The bundle's viewport is viewport-fit=cover (sync-www.mjs), so the page
  // starts at the very top of the screen and env(safe-area-inset-*) carries
  // the notch / home-bar sizes; on the website both are zero and app.css
  // already handles the bottom bar. Everything sticky or fixed at the top
  // moves down by the inset; the header paints that strip white so scrolled
  // content never shows through the status bar.
  var fit = document.createElement('style');
  fit.textContent = [
    'html.flapp-native{background:#fff}',                     // rubber-band shows white, not grey
    '.flapp-native body{-webkit-tap-highlight-color:transparent;-webkit-touch-callout:none}',
    // 16px is the size below which iOS zooms into a focused field.
    '.flapp-native input:not([type=range]):not([type=checkbox]):not([type=radio]):not([type=file]),' +
      '.flapp-native select,.flapp-native textarea{font-size:16px}',
    '@media(max-width:900px){',
    '  .flapp-native .topbar{height:calc(56px + env(safe-area-inset-top,0px));padding-top:env(safe-area-inset-top,0px)}',
    '  .flapp-native .sidebar{padding-top:calc(22px + env(safe-area-inset-top,0px))}',
    '  .flapp-native .rrail{top:calc(56px + env(safe-area-inset-top,0px))}',
    '  .flapp-native .modal{padding-top:calc(20px + env(safe-area-inset-top,0px));' +
        'padding-bottom:calc(20px + env(safe-area-inset-bottom,0px))}',
    // above the bottom tab bar (56px) instead of underneath it
    '  .flapp-native .toast{bottom:calc(76px + env(safe-area-inset-bottom,0px))}',
    '}'
  ].join('\n');
  document.head.appendChild(fit);

  window.FLAPP_BRIDGE = {
    ready: ready,
    native: !!(cap && cap.isNativePlatform && cap.isNativePlatform()) || !!cfg.forceNative,
    base: BASE
  };
})();
