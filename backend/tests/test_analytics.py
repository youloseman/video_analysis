"""Product analytics: the switch, the snippet, and the promises around it.

Three things are worth a test here, and they are not the ones that look
obvious.

1. **Off means off.** No key -> no snippet in any page, no request off the box.
   The suite itself is the first customer of that: nothing in CI should be
   phoning an analytics vendor.
2. **The privacy policy must not lie in either direction.** Section 7 ships in
   two branches -- "we run product analytics" and "we run none" -- and the
   server keeps whichever is true for this deploy. A page that describes
   PostHog on an install that never set a key is as wrong as one that hides it.
3. **Failure is not an outage.** The capture path swallows everything: a dead
   analytics host may not turn a purchase webhook into a 500 that Stripe then
   retries forever.
"""

from __future__ import annotations

import dataclasses
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from app.core.config import settings
from app.services import analytics

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"

KEY = "phc_test_key_not_a_real_one"


@pytest.fixture
def analytics_on(monkeypatch):
    """Turn analytics on for one test, without a real key or a real host."""
    monkeypatch.setattr(
        analytics, "settings",
        dataclasses.replace(settings, posthog_key=KEY),
    )
    analytics.snippet_html.cache_clear()
    yield
    analytics.snippet_html.cache_clear()


@pytest.fixture(autouse=True)
def _clear_snippet_cache():
    """The snippet is cached for the life of the process; a test that changes
    settings must not leak its version into the next one."""
    analytics.snippet_html.cache_clear()
    yield
    analytics.snippet_html.cache_clear()


# --------------------------------------------------------------------------
# Off by default
# --------------------------------------------------------------------------
def test_disabled_without_a_key():
    """The suite runs with no POSTHOG_KEY, and that has to be the quiet path."""
    assert not analytics.enabled()
    assert analytics.snippet_html() == ""


def test_inject_is_a_no_op_when_disabled():
    doc = "<html><head><title>x</title></head><body></body></html>"
    assert analytics.inject(doc) == doc


def test_capture_sends_nothing_when_disabled(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("analytics must not open a socket without a key")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert analytics.capture("u1", "purchase_completed", {}) is False
    analytics.capture_bg("u1", "purchase_completed", {})


# --------------------------------------------------------------------------
# The browser snippet
# --------------------------------------------------------------------------
def test_snippet_carries_the_key_and_no_leftover_tokens(analytics_on):
    html = analytics.snippet_html()
    assert html.startswith("<script>") and html.endswith("</script>")
    assert json.dumps(KEY) in html
    # An unreplaced placeholder is a JS syntax error on every page of the site.
    assert "__PH_" not in html
    assert "posthog.init" in html


def test_snippet_defaults_protect_the_athletes_footage(analytics_on):
    """Replay is off unless someone opts in, because a results page has the
    clip on it. Anonymous visitors get no person profile."""
    html = analytics.snippet_html()
    assert "disable_session_recording:!REC" in html
    assert "person_profiles:'identified_only'" in html
    assert "respect_dnt:true" in html


def test_session_recording_is_opt_in(monkeypatch):
    monkeypatch.setattr(
        analytics, "settings",
        dataclasses.replace(settings, posthog_key=KEY, posthog_session_recording=True),
    )
    analytics.snippet_html.cache_clear()
    assert "REC=true" in analytics.snippet_html()


def test_cloud_library_is_loaded_from_the_assets_host():
    """PostHog Cloud serves array.js from a sibling host; getting this wrong is
    a 404 on the library and silent, total data loss."""
    assert analytics._assets_host("https://us.i.posthog.com") == (
        "https://us-assets.i.posthog.com"
    )
    assert analytics._assets_host("https://eu.i.posthog.com") == (
        "https://eu-assets.i.posthog.com"
    )


def test_a_reverse_proxy_serves_its_own_library():
    """A proxy on our own domain (the ad-blocker workaround) has no sibling
    assets host -- it must serve array.js from itself."""
    assert analytics._assets_host("https://getflapp.com/ph") == "https://getflapp.com/ph"


def test_inject_puts_the_snippet_in_the_head(analytics_on):
    doc = "<html><head><title>x</title></head><body>hi</body></html>"
    out = analytics.inject(doc)
    assert out.index("posthog.init") < out.index("</head>")


def test_inject_leaves_a_fragment_alone(analytics_on):
    assert analytics.inject("<div>no head here</div>") == "<div>no head here</div>"


# --------------------------------------------------------------------------
# The privacy policy branches
# --------------------------------------------------------------------------
DISCLOSURE_DOC = (
    "<p>always</p>"
    "<!--ANALYTICS:ON--><p>we use posthog</p><!--/ANALYTICS:ON-->"
    "<!--ANALYTICS:OFF--><p>we run no analytics</p><!--/ANALYTICS:OFF-->"
)


def test_disclosure_denies_analytics_when_it_is_off():
    out = analytics.apply_disclosure(DISCLOSURE_DOC)
    assert "we run no analytics" in out
    assert "posthog" not in out
    assert "ANALYTICS:" not in out          # markers never reach the reader


def test_disclosure_admits_analytics_when_it_is_on(analytics_on):
    out = analytics.apply_disclosure(DISCLOSURE_DOC)
    assert "we use posthog" in out
    assert "we run no analytics" not in out
    assert "ANALYTICS:" not in out


def test_disclosure_survives_an_unbalanced_marker():
    """A half-written edit must degrade to "the marker is ignored", not to a
    truncated privacy policy."""
    out = analytics.apply_disclosure("<p>keep</p><!--ANALYTICS:ON--><p>tail</p>")
    assert "keep" in out and "tail" in out and "ANALYTICS:" not in out


def test_the_real_policy_has_both_branches_and_they_balance():
    doc = (STATIC / "privacy.html").read_text(encoding="utf-8")
    assert doc.count("<!--ANALYTICS:ON-->") == doc.count("<!--/ANALYTICS:ON-->") > 0
    assert doc.count("<!--ANALYTICS:OFF-->") == doc.count("<!--/ANALYTICS:OFF-->") > 0
    off = analytics.apply_disclosure(doc)
    assert "PostHog" not in off, "the policy claims analytics that is switched off"
    assert "no analytics or advertising trackers" in off


def test_the_real_policy_names_posthog_when_it_is_on(analytics_on):
    doc = (STATIC / "privacy.html").read_text(encoding="utf-8")
    on = analytics.apply_disclosure(doc)
    assert "PostHog" in on
    # The claim that has to stay true in code: replay off, media never sent.
    assert "Session replay is off" in on
    assert "no analytics or advertising trackers" not in on


# --------------------------------------------------------------------------
# Server-side capture
# --------------------------------------------------------------------------
class _Resp:
    def __init__(self, status: int = 200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_capture_posts_the_documented_payload(analytics_on, monkeypatch):
    sent: dict = {}

    def fake_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data.decode("utf-8"))
        sent["method"] = req.get_method()
        return _Resp(200)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert analytics.capture("u7", "purchase_completed", {"plan": "full_yearly"})
    assert sent["method"] == "POST"
    assert sent["url"].endswith("/i/v0/e/")
    body = sent["body"]
    assert body["api_key"] == KEY
    assert body["event"] == "purchase_completed"
    assert body["distinct_id"] == "u7"
    assert body["properties"]["plan"] == "full_yearly"
    # Server-side events are always about a known account.
    assert body["properties"]["$process_person_profile"] is True


def test_a_dead_analytics_host_never_raises(analytics_on, monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert analytics.capture("u7", "purchase_completed", {}) is False


def test_a_rejected_event_is_reported_not_raised(analytics_on, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(400))
    assert analytics.capture("u7", "purchase_completed", {}) is False


async def test_person_id_matches_what_the_browser_is_told(make_user):
    """The browser identifies with ``analytics_id`` from the auth payload and
    the webhook files purchases under :func:`person_id`. If these two ever
    diverge, every paying customer becomes two people in the funnel."""
    from app.api.auth import UserOut

    user = await make_user()
    out = UserOut(
        email=user.email, tier=user.tier, is_pro=user.is_paid,
        expert_credits=0, analytics_id=analytics.person_id(user),
    )
    assert out.analytics_id == f"u{user.id}"


# --------------------------------------------------------------------------
# The webhook: the only trustworthy account of a payment
# --------------------------------------------------------------------------
async def test_a_paid_checkout_is_recorded_with_its_revenue(
    db, make_user, analytics_on, monkeypatch,
):
    """The browser's own `checkout_returned` needs the tab to survive the
    redirect back from Stripe. The charge does not. So the sale is filed from
    here, under the same person id the browser identifies with."""
    from app.api import billing
    from tests.test_billing_webhook import checkout_completed

    sent: list = []
    monkeypatch.setattr(
        analytics, "capture_bg",
        lambda did, event, props=None: sent.append((did, event, props)),
    )
    user = await make_user()
    await billing._apply_event(
        db, checkout_completed(user.id, mode="payment", plan="expert",
                               tier=None, amount=3900, currency="usd"),
    )
    assert sent, "a completed checkout recorded no purchase"
    distinct_id, event, props = sent[0]
    assert distinct_id == analytics.person_id(user)
    assert event == "purchase_completed"
    assert props["plan"] == "expert"
    assert props["revenue"] == 39.0            # cents -> dollars, once
    assert props["currency"] == "USD"


async def test_a_cancelled_subscription_is_recorded(db, make_user, analytics_on, monkeypatch):
    from app.api import billing
    from tests.test_billing_webhook import subscription_event

    sent: list = []
    monkeypatch.setattr(
        analytics, "capture_bg",
        lambda did, event, props=None: sent.append(event),
    )
    user = await make_user()
    user.stripe_customer_id = "cus_gone"
    await db.commit()
    await billing._apply_event(
        db, subscription_event("customer.subscription.deleted", customer="cus_gone"),
    )
    assert "subscription_canceled" in sent


def test_an_unparseable_amount_is_not_a_free_sale():
    """`None` beats `0.0`: a missing amount must not read as a zero-dollar
    purchase in the revenue chart."""
    from app.api.billing import _amount_major

    assert _amount_major(3900) == 39.0
    assert _amount_major(None) is None
    assert _amount_major("oops") is None


# --------------------------------------------------------------------------
# The client wiring
# --------------------------------------------------------------------------
@pytest.mark.parametrize("page", ["index.html", "landing.html", "privacy.html"])
def test_pages_only_touch_analytics_through_the_facade(page):
    """`window.flappAnalytics` does not exist when analytics is off, and every
    call site is written to survive that. A direct `posthog.` reference is a
    ReferenceError on a deploy with no key -- i.e. on every local run."""
    doc = (STATIC / page).read_text(encoding="utf-8")
    assert "flappAnalytics" in doc, f"{page} does no analytics at all"
    for reach in ("window.posthog", "posthog.capture", "posthog.identify",
                  "posthog.init", "posthog.reset"):
        assert reach not in doc, f"{page} reaches past the facade ({reach})"
