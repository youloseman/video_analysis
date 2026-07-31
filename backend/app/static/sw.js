/* Flapp service worker — minimal, deploy-safe.
 *
 * Design rules (learned from the shell-staleness fights):
 *  - NEVER serve the app shell cache-first. The shell is deliberately
 *    `no-cache, must-revalidate` + ETag so deploys are picked up; a cache-first
 *    SW would resurrect the "old build keeps rendering" bug. So navigations are
 *    NETWORK-FIRST, and only fall back to a cached offline page when truly offline.
 *  - Touch nothing else. API calls (/auth, /analyze, /jobs, /me, /billing),
 *    uploads and overlay videos pass straight through to the network. The fetch
 *    handler only intervenes for top-level navigations.
 *  - Precache just the offline fallback + icons, so an install is cheap.
 *
 * Bump CACHE when the precache list or this file changes; old caches are pruned
 * on activate.
 */
const CACHE = 'flapp-v1';
const PRECACHE = [
  '/offline.html',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/manifest.webmanifest',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  // Only ever handle top-level navigations. Everything else — API, media,
  // fonts, the shell's own subrequests — goes to the network untouched.
  if (req.mode !== 'navigate') return;
  // Don't cache API-ish navigations (there shouldn't be any, but be safe).
  event.respondWith(
    fetch(req).catch(() => caches.match('/offline.html', { ignoreSearch: true }))
  );
});
