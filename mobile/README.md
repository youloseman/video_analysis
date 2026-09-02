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
integration named `appstore`, create the app with bundle `com.clariva.flapp`).
