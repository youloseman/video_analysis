# Flapp mobile (Capacitor shell)

iOS + Android apps for Flapp. The analysis stays on the server; this shell
bundles the production SPA (`backend/app/static/index.html`) and adds native
behavior through `bridge/bridge.js` — the web build never loads the bridge.

## Layout

- `bridge/bridge.js` — all mobile-only behavior (API routing, durable auth
  token, external links, hidden purchase paths). Injected into the bundle by
  the sync script; **never** referenced from the web build.
- `scripts/sync-www.mjs` — builds `www/` from the production SPA.
  `--api-base http://127.0.0.1:8099` points the bundle at a local backend.
- `android/`, `ios/` — generated native projects (committed, Capacitor
  convention). `www/` and `node_modules/` are generated → gitignored.

## Commands

```bash
npm install               # once
npm run sync              # rebuild www/ + cap sync android
npm run android           # sync + open in Android Studio (build/run from there)
npm run ios:ci            # what Codemagic runs before xcodebuild
```

## Video capture (MVP)

The SPA's `<input type="file" accept="video/*">` opens the **native** camera /
gallery sheet inside the WebView on both platforms — no plugin code needed.
iOS requires the usage strings in `ios/App/App/Info.plist` (already added);
removing them makes the picker crash the app. The guided capture screen
(framing overlay, 30fps, 15s cap) replaces this in Phase 2.

## Store rules (do not undo)

No purchase UI may be visible in the app until IAP ships (RevenueCat phase):
no pricing page, no upgrade buttons, no "buy on the web" links. The bridge
hides these with injected CSS. Free tier + login for existing subscribers is
allowed. See the plan artifact + `flapp-mobile-plan` memory.

## iOS builds (no Mac needed)

`codemagic.yaml` at the repo root builds and ships to TestFlight. Setup steps
are in that file's header comment (App Store Connect API key → Codemagic
integration named `appstore`, a `CERTIFICATE_PRIVATE_KEY` secure variable in
the `ios_signing_creds` group, create the app with bundle `ca.clarivagroups.flapp`).

## Android builds

`codemagic.yaml` has two Android workflows:

- **`android-debug-apk`** — start it by hand, open the artifact link *on the
  phone* to sideload. No keystore, no Play account. This is how a change gets
  looked at on a real device.
- **`android-release`** — a signed `.aab`, built on the same `mobile/*` tag as
  the iOS workflow so the two stores never drift apart. The AAB comes back as
  an artifact and goes to Play Console → Internal testing by hand; the
  service-account upload is written out, commented, in the workflow.

The upload keystore is `secrets/android/flapp-upload.jks` (gitignored);
`secrets/android/KEYSTORE.md` has the password and both ways of using it.
It is only the *upload* key — Play App Signing holds the real one, and Google
can reset an upload key that is lost.

Three Android-only things the shell has to get right, all pinned by
`backend/tests/test_native_android_shell.py`:

- **Back.** Capacitor leaves `onBackPressed` alone, so the default is
  `finish()` — the app closes from any screen. `bridge.js` section 6 makes it
  close a dialog, else go back a view, else background the app.
- **The forced edge-to-edge window.** On Android 15 (targetSdk 35) the page
  would paint under the status bar and the tab bar under the navigation bar.
  `adjustMarginsForEdgeToEdge: "auto"` insets the WebView instead, and the
  theme paints the two strips white with dark system icons.
- **`versionCode`.** Play refuses a number it has seen, so it comes from
  `$PROJECT_BUILD_NUMBER`; the template's hard-coded `1` would make the second
  release unuploadable.

A local release build (needs a JDK — Android Studio's `jbr` is one):

```bash
export JAVA_HOME="/c/Program Files/Android/Android Studio/jbr"
export ANDROID_HOME="$LOCALAPPDATA/Android/Sdk"
cd mobile && npm run sync && cd android
./gradlew assembleDebug      # or bundleRelease with the FLAPP_KEYSTORE_* vars
```

## Brand assets

`brand/*.svg` are the source of truth for the app icon (light / dark / tinted)
and the launch splash. `python scripts/render-brand.py` rasterises them into
`ios/App/App/Assets.xcassets`; `check-native.mjs` refuses the Capacitor
placeholders, which a `cap add` would put back. The Android launcher icon is
not rendered by that script — its mipmaps were drawn from the same mark and
are committed; `check-native.mjs` watches the adaptive background colour
(brand navy, vs the template's white) to notice a `cap add` restoring the
placeholder. The design files they come from
are the "Flapp Logo" canvases in the design export.
