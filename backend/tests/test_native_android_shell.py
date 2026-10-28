"""The Android shell: Back, the forced edge-to-edge window, the signed release.

Three Android-only ways the app breaks that no iOS check would notice, each
of them silent -- the build succeeds and the damage only shows on a phone or
at the Play upload:

* Capacitor's ``BridgeActivity`` does not override ``onBackPressed``, so
  without a ``backButton`` listener Back is the plain Activity default:
  ``finish()``. The app closes from whatever screen, with a dialog open,
  mid-analysis. Android users read that as a crash.
* On Android 15 (targetSdk 35) the window is forced edge-to-edge. Capacitor
  defaults ``adjustMarginsForEdgeToEdge`` to ``"disable"``, which leaves the
  page painting under the status bar and the tab bar under the navigation bar.
* A release built without a signing config is rejected at upload, and a
  hard-coded ``versionCode 1`` makes every release after the first
  unuploadable.

``mobile/scripts/check-native.mjs`` asserts most of this too, but it only runs
inside the mobile CI workflows; this suite gates every push.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MOBILE = ROOT / "mobile"
BRIDGE = MOBILE / "bridge" / "bridge.js"
SPA = ROOT / "backend" / "app" / "static" / "index.html"
GRADLE = MOBILE / "android" / "app" / "build.gradle"
STYLES = MOBILE / "android" / "app" / "src" / "main" / "res" / "values" / "styles.xml"


@pytest.fixture(scope="module")
def bridge() -> str:
    return BRIDGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def spa() -> str:
    return SPA.read_text(encoding="utf-8")


def test_webview_is_inset_from_the_system_bars() -> None:
    cfg = json.loads((MOBILE / "capacitor.config.json").read_text(encoding="utf-8"))
    assert cfg["android"].get("adjustMarginsForEdgeToEdge") == "auto"


def test_app_plugin_is_installed() -> None:
    """Without it ``plugins.App`` is undefined and Back silently kills the app."""
    deps = json.loads((MOBILE / "package.json").read_text(encoding="utf-8"))[
        "dependencies"
    ]
    assert "@capacitor/app" in deps


def test_back_is_handled(bridge: str) -> None:
    assert "App.addListener('backButton'" in bridge


def test_back_at_the_first_screen_backgrounds_rather_than_exits(bridge: str) -> None:
    """Coming back should land on the same report, not a cold start."""
    assert "minimizeApp()" in bridge
    assert "exitApp" not in bridge


def test_back_closes_a_dialog_the_way_the_spa_does(bridge: str, spa: str) -> None:
    """The handler synthesises Escape; the SPA must still be listening for it.

    Escape is what reaches each modal's own close routine (stopping the camera
    recorder, releasing the stream, restoring focus) instead of just hiding the
    element -- so the bridge dispatching it is only correct while the SPA's
    shared dialog containment is there to receive it.
    """
    assert "key: 'Escape'" in bridge
    assert "modalFocus.onClose" in spa
    assert "e.key==='Escape'" in spa
    # The fallback for a dialog that opened without registering.
    assert ".modal-close" in bridge
    assert 'class="modal-close' in spa


def test_back_leaves_the_menu_by_closing_it(bridge: str, spa: str) -> None:
    assert "nav-open" in bridge
    assert "nav-open" in spa


def test_back_depth_counts_the_spa_pushes(bridge: str, spa: str) -> None:
    """history.back() at the root entry is a no-op, so the pushes are counted.

    The SPA takes over the launch entry with replaceState, which means
    history.length includes WebView entries from before the app and cannot be
    used to tell whether there is a view to go back to.
    """
    assert "history.pushState = function" in bridge
    assert "history.replaceState(entry" in spa


def test_release_version_code_comes_from_the_build() -> None:
    g = GRADLE.read_text(encoding="utf-8")
    assert "versionCode flappVersionCode" in g
    assert "PROJECT_BUILD_NUMBER" in g


def test_release_is_signed_when_a_keystore_is_supplied() -> None:
    g = GRADLE.read_text(encoding="utf-8")
    assert "signingConfig signingConfigs.release" in g
    # Codemagic's own names, so attaching the keystore to the workflow is the
    # whole configuration.
    for var in (
        "CM_KEYSTORE_PATH",
        "CM_KEYSTORE_PASSWORD",
        "CM_KEY_ALIAS",
        "CM_KEY_PASSWORD",
    ):
        assert var in g


def test_the_system_bar_strips_stay_legible() -> None:
    """The inset strips show the window background: white page, dark icons."""
    x = STYLES.read_text(encoding="utf-8")
    assert "android:windowLightStatusBar" in x
    assert "android:windowBackground" in x
