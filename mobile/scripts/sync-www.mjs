// Build the Capacitor web bundle (www/) from the production SPA.
//
// Copies backend/app/static assets and injects the mobile config + platform
// bridge into <head>. The web build is never touched — this is a one-way copy.
//
//   node scripts/sync-www.mjs                        -> apiBase https://getflapp.com
//   node scripts/sync-www.mjs --api-base http://127.0.0.1:8099   (local testing)
import { cpSync, mkdirSync, readFileSync, writeFileSync, existsSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const mobileDir = resolve(here, '..');
const staticDir = resolve(mobileDir, '..', 'backend', 'app', 'static');
const wwwDir = join(mobileDir, 'www');

const argIdx = process.argv.indexOf('--api-base');
const apiBase = argIdx > -1 ? process.argv[argIdx + 1] : 'https://getflapp.com';

rmSync(wwwDir, { recursive: true, force: true });
mkdirSync(wwwDir, { recursive: true });

// 1. index.html with the injected config + bridge (before </head>).
let html = readFileSync(join(staticDir, 'index.html'), 'utf8');
const inject = [
  '<script>window.FLAPP_MOBILE_CONFIG=' + JSON.stringify({ apiBase, webOrigin: 'https://getflapp.com' }) + ';</script>',
  '<script src="bridge.js"></script>',
].join('\n');
if (!html.includes('</head>')) throw new Error('index.html has no </head> to inject into');
html = html.replace('</head>', inject + '\n</head>');
writeFileSync(join(wwwDir, 'index.html'), html);

// 2. Bridge + static assets the SPA references by absolute path.
cpSync(join(mobileDir, 'bridge', 'bridge.js'), join(wwwDir, 'bridge.js'));
for (const asset of ['app.css', 'tokens.css', 'favicon.svg', 'apple-touch-icon.png']) {
  const src = join(staticDir, asset);
  if (existsSync(src)) cpSync(src, join(wwwDir, asset));
}
if (existsSync(join(staticDir, 'media'))) {
  cpSync(join(staticDir, 'media'), join(wwwDir, 'media'), { recursive: true });
}
// Deliberately NOT copied: sw.js + manifest (service workers don't run on the
// capacitor:// scheme; registration in the SPA fails silently by design).

console.log(`www/ built (apiBase=${apiBase})`);
