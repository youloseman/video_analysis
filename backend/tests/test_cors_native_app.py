"""CORS for the mobile shell.

The website is same-origin and needs no CORS at all, which is why production
used to install no middleware. The installed app is the same SPA on a
capacitor://localhost (iOS) or http://localhost (Android) origin calling the
API by absolute URL -- and with no allow-list every one of its requests died
on the preflight. Safari on the same phone logged in fine, the app said
"Network error" (first seen on the first TestFlight build, 2026-09-11).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import NATIVE_APP_ORIGINS, app, cors_origins


def test_production_allows_exactly_the_native_app_origins():
    assert cors_origins({"RAILWAY_ENVIRONMENT": "production"}) == [
        "capacitor://localhost", "http://localhost",
    ]


def test_an_explicit_list_is_added_on_top_of_the_native_origins():
    got = cors_origins({
        "RAILWAY_ENVIRONMENT": "production",
        "VA_CORS_ORIGINS": "https://studio.example, http://localhost",
    })
    assert got == NATIVE_APP_ORIGINS + ["https://studio.example"]


def test_wildcard_still_opts_into_the_old_permissive_behaviour():
    assert cors_origins({"RAILWAY_ENVIRONMENT": "production", "VA_CORS_ORIGINS": "*"}) == ["*"]


def test_a_local_run_stays_wide_open():
    assert cors_origins({}) == ["*"]


def test_the_login_preflight_from_the_ios_app_is_answered():
    """End to end on the running app: the exact request the WebView sends
    before POST /auth/login must come back with the origin allowed and the
    JSON content-type permitted."""
    with TestClient(app) as c:
        r = c.options(
            "/auth/login",
            headers={
                "Origin": "capacitor://localhost",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") in ("*", "capacitor://localhost")
    assert "content-type" in r.headers.get("access-control-allow-headers", "").lower()
