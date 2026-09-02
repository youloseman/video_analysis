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

const pbx = join(mobileDir, 'ios/App/App.xcodeproj/project.pbxproj');
if (existsSync(pbx) && !readFileSync(pbx, 'utf8').includes(`PRODUCT_BUNDLE_IDENTIFIER = ${appId}`)) {
  fails.push(`ios PRODUCT_BUNDLE_IDENTIFIER != ${appId}`);
}

const plist = join(mobileDir, 'ios/App/App/Info.plist');
if (existsSync(plist)) {
  const p = readFileSync(plist, 'utf8');
  for (const k of ['NSCameraUsageDescription', 'NSMicrophoneUsageDescription',
                   'NSPhotoLibraryUsageDescription', 'NSPhotoLibraryAddUsageDescription']) {
    if (!p.includes(k)) fails.push(`Info.plist missing ${k} (camera picker will crash)`);
  }
}

if (fails.length) {
  console.error('native check FAILED:\n  - ' + fails.join('\n  - '));
  process.exit(1);
}
console.log(`native check ok (${appId})`);
