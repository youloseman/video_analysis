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
  var st = document.createElement('style');
  st.textContent = [
    '.flapp-native #navPricing', '.flapp-native #botPricing',
    '.flapp-native .upsell .cta-row', '.flapp-native .upsell .price-tag',
    '.flapp-native #pricing .tier-cta', '.flapp-native #ctaExpert',
    '.flapp-native .addon-strip'
  ].join(',') + '{display:none!important}';
  document.head.appendChild(st);

  window.FLAPP_BRIDGE = {
    ready: ready,
    native: !!(cap && cap.isNativePlatform && cap.isNativePlatform()) || !!cfg.forceNative,
    base: BASE
  };
})();
