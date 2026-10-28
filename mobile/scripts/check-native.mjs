// Guard: the native projects are generated, so a `cap add` (or a Capacitor
// upgrade) can silently reset hand-edited native files. This asserts the
// things we edit by hand survive:
//   1. the bundle id matches capacitor.config.json everywhere
//   2. the iOS usage strings are present (without them the WebView file
//      picker crashes the app the first time it opens the camera)
//   3. the icons and splash are brand assets, not the Capacitor placeholders
//   4. the Android release can actually be uploaded (versionCode from the
//      build, a signing config) and fits the screen (edge-to-edge insets)
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const mobileDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const capConfig = JSON.parse(readFileSync(join(mobileDir, 'capacitor.config.json'), 'utf8'));
const appId = capConfig.appId;
const fails = [];

// Android 15 (targetSdk 35) forces the window edge-to-edge: with Capacitor's
// default ("disable") nothing insets the WebView, so the page paints under the
// status bar and the bottom tab bar sits under the navigation bar. "auto" lets
// Capacitor apply the insets as margins -- the Android half of what
// contentInset/env(safe-area-inset-*) do on iOS.
if ((capConfig.android?.adjustMarginsForEdgeToEdge ?? 'disable') !== 'auto') {
  fails.push('android.adjustMarginsForEdgeToEdge != "auto" -- content draws under the status/navigation bars on Android 15');
}

const gradle = join(mobileDir, 'android/app/build.gradle');
if (existsSync(gradle)) {
  const g = readFileSync(gradle, 'utf8');
  if (!g.includes(`applicationId "${appId}"`)) fails.push(`android applicationId != ${appId}`);
  const pkgDir = join(mobileDir, 'android/app/src/main/java', ...appId.split('.'), 'MainActivity.java');
  if (!existsSync(pkgDir)) fails.push(`android MainActivity missing at ${appId.replace(/\./g, '/')}/`);
  // Play refuses a versionCode it has seen before, so the hard-coded 1 the
  // template writes makes every release after the first unuploadable.
  if (/versionCode\s+\d/.test(g)) fails.push('android versionCode is hard-coded 1 -- Play rejects the second upload; take it from the build');
  // Without this the release build type is signed with the debug key, which
  // Play rejects at upload ("debuggable" / wrong certificate).
  if (!g.includes('signingConfig signingConfigs.release')) fails.push('android release build has no signing config');
}

const styles = join(mobileDir, 'android/app/src/main/res/values/styles.xml');
if (existsSync(styles)) {
  const x = readFileSync(styles, 'utf8');
  // The inset strips (above/below the WebView) show the window background;
  // the SPA is light-only, so they are white and the system icons in them
  // must be dark or they vanish.
  if (!x.includes('windowLightStatusBar')) fails.push('android theme has no windowLightStatusBar -- white status bar strip with white icons');
}

// `cap add android` restores the template launcher icon, whose adaptive
// background is white. Ours is the brand navy, which is the cheapest way to
// tell our mark from the placeholder (a blue cross) without pinning the hash
// of an asset we are still free to redraw.
const iconBg = join(mobileDir, 'android/app/src/main/res/values/ic_launcher_background.xml');
if (existsSync(iconBg) && !/#14294B/i.test(readFileSync(iconBg, 'utf8'))) {
  fails.push('android ic_launcher_background is not the brand navy -- the Capacitor placeholder icon is back');
}
for (const rel of ['android/app/src/main/res/mipmap-xxxhdpi/ic_launcher.png',
                   'android/app/src/main/res/mipmap-xxxhdpi/ic_launcher_foreground.png',
                   'android/app/src/main/res/mipmap-xxxhdpi/ic_launcher_round.png']) {
  if (existsSync(join(mobileDir, 'android')) && !existsSync(join(mobileDir, rel))) fails.push(`${rel} missing`);
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
