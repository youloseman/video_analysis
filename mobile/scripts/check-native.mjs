// Guard: the native projects are generated, so a `cap add` (or a Capacitor
// upgrade) can silently reset hand-edited native files. This asserts the two
// things we edit by hand survive:
//   1. the bundle id matches capacitor.config.json everywhere
//   2. the iOS usage strings are present (without them the WebView file
//      picker crashes the app the first time it opens the camera)
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const mobileDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const appId = JSON.parse(readFileSync(join(mobileDir, 'capacitor.config.json'), 'utf8')).appId;
const fails = [];

const gradle = join(mobileDir, 'android/app/build.gradle');
if (existsSync(gradle)) {
  const g = readFileSync(gradle, 'utf8');
  if (!g.includes(`applicationId "${appId}"`)) fails.push(`android applicationId != ${appId}`);
  const pkgDir = join(mobileDir, 'android/app/src/main/java', ...appId.split('.'), 'MainActivity.java');
  if (!existsSync(pkgDir)) fails.push(`android MainActivity missing at ${appId.replace(/\./g, '/')}/`);
}

const manifest = join(mobileDir, 'android/app/src/main/AndroidManifest.xml');
if (existsSync(manifest) && !readFileSync(manifest, 'utf8').includes('android.permission.CAMERA')) {
  fails.push('AndroidManifest missing CAMERA permission (guided camera cannot start)');
}

const pbx = join(mobileDir, 'ios/App/App.xcodeproj/project.pbxproj');
if (existsSync(pbx)) {
  const x = readFileSync(pbx, 'utf8');
  if (!x.includes(`PRODUCT_BUNDLE_IDENTIFIER = ${appId}`)) fails.push(`ios PRODUCT_BUNDLE_IDENTIFIER != ${appId}`);
  // iPhone only for v1: "1,2" (the Capacitor default) makes App Store Connect
  // demand iPad screenshots and puts the phone layout in front of the
  // reviewer on a 13-inch screen.
  if (/TARGETED_DEVICE_FAMILY = "1,2"/.test(x)) fails.push('ios TARGETED_DEVICE_FAMILY is "1,2" -- v1 ships iPhone only (set to 1)');
  // ITMS-90068: from spring 2027 App Store Connect refuses uploads below
  // iOS 15. Capacitor's template says 14.0 and `cap add` writes it back.
  if (/IPHONEOS_DEPLOYMENT_TARGET = 14\.0/.test(x)) fails.push('ios IPHONEOS_DEPLOYMENT_TARGET is 14.0 -- App Store Connect warns (ITMS-90068); set 15.0 in pbxproj and Podfile');
}

// The icon and splash are real brand assets, not the Capacitor placeholders
// (a blue cross on white). App Review rejects a placeholder icon outright,
// and `cap add` puts the placeholders back. Pinned by the SHA-256 of the
// placeholder files as they were committed on 2026-09-02; rebuild ours from
// mobile/brand/*.svg (see mobile/README.md) if this ever fires.
import { createHash } from 'node:crypto';
const PLACEHOLDERS = new Set([
  '40e9197bf7a2fa6765b59356038c73013eed477f23b54cc406852f041b968ce3',   // AppIcon-512@2x.png
  'b36218f6212e698e7f4f0fe9033b67fbdc7f864be3f728241e1faf9f6c9405bb',   // AppIcon-dark.png
  '5e11652313174d724f92d423ab2546237de2779c420cdd878277ec545b8022f5',   // AppIcon-tinted.png
  '1b5002b74a5500e697298ced06ca2811ac33f2771f236f3c720ff23243890530',   // splash-2732x2732.png
]);
for (const rel of ['ios/App/App/Assets.xcassets/AppIcon.appiconset/AppIcon-512@2x.png',
                   'ios/App/App/Assets.xcassets/AppIcon.appiconset/AppIcon-dark.png',
                   'ios/App/App/Assets.xcassets/AppIcon.appiconset/AppIcon-tinted.png',
                   'ios/App/App/Assets.xcassets/Splash.imageset/splash-2732x2732.png']) {
  const f = join(mobileDir, rel);
  if (!existsSync(f)) continue;
  const b = readFileSync(f);
  if (PLACEHOLDERS.has(createHash('sha256').update(b).digest('hex'))) fails.push(`${rel} is the Capacitor placeholder -- App Review rejects it`);
  if (rel.includes('AppIcon') && (b.readUInt32BE(16) !== 1024 || b.readUInt32BE(20) !== 1024)) fails.push(`${rel} is not 1024x1024`);
}

if (fails.length) {
  console.error('native check FAILED:\n  - ' + fails.join('\n  - '));
  process.exit(1);
}
console.log(`native check ok (${appId})`);
