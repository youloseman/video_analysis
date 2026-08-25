"""Minimal settings for the standalone video-analysis MVP (Milestone 1).

IMPORTANT: the copied biomechanics/detector core reads **no** ``settings.*``
fields at all -- a repo-wide search for ``settings.`` across
``app/services/video_analysis`` returns nothing. So this object is
deliberately tiny; it exists only to give the driver (and future
milestones) a single place to grow configuration.

No ``pydantic`` / ``pydantic-settings`` dependency on purpose: Milestone 1
requirements are limited to packages the core actually imports
(mediapipe, opencv, numpy, scipy, structlog). R2 / LLM keys are out of
scope for this milestone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# backend/ directory (this file is backend/app/core/config.py -> parents[2]).
BACKEND_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Overridable via environment variables."""

    # Pose model. The detector searches several locations
    # (see detectors/mediapipe_detector.py::_MODEL_SEARCH_PATHS); this
    # is the primary one the driver checks for a friendly error message.
    models_dir: Path = BACKEND_DIR / "models"
    model_filename: str = "pose_landmarker_heavy.task"

    # Cap on frames analyzed for long clips (mirrors the Motus default
    # baked into the frame extractor). Kept here so it is tunable later.
    max_analysis_frames: int = 450

    # Where the API stores uploaded clips + generated overlays (M3).
    uploads_dir: Path = BACKEND_DIR / "uploads"

    # LLM coaching (M5). Read from env; never hard-code the key. When absent,
    # recommendations are skipped gracefully and analysis still works.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    # Hard ceiling on a single Gemini request. Every LLM call is made from a
    # threadpool thread (the photo endpoint blocks the user on it outright), so
    # without this a hung request holds a worker slot until the socket gives up
    # on its own -- which can be minutes. Coaching is optional by design: a
    # timeout degrades to "no coaching", never to a failed analysis.
    gemini_timeout_s: float = 20.0

    # Abuse guard: max analyses per client (IP) per rolling 24h. 0 disables.
    rate_limit_per_day: int = 3

    # --- Job lifecycle. The job store is in-memory and uploads land on the
    # container's ephemeral disk, so both need an explicit reaper: nothing else
    # deletes them, and a long-lived instance would otherwise grow until the
    # disk (or the heap) fills. ---
    # How long a finished job stays fetchable (result JSON + overlay download)
    # before it and its upload directory are deleted. 0 disables the sweeper.
    job_ttl_hours: float = 6.0
    # How often the sweeper runs.
    job_sweep_interval_s: int = 600

    # --- Analysis concurrency. MediaPipe "heavy" is CPU- and RAM-bound; the
    # work runs in Starlette's threadpool, which would happily start ~40 of them
    # at once and OOM the container. Bound it, and reject new uploads once the
    # backlog is longer than a caller would sensibly wait through. ---
    max_concurrent_analyses: int = 2
    max_queued_analyses: int = 8

    # Accounts (auth). DATABASE_URL defaults to a local SQLite file; set it to a
    # Postgres URL (Railway) in prod. jwt_secret MUST be overridden in prod.
    database_url: str = f"sqlite+aiosqlite:///{BACKEND_DIR / 'flapp.db'}"
    jwt_secret: str = "dev-insecure-change-me"
    jwt_expire_days: int = 30

    # Account promoted to the ``admin`` tier on startup (15 analyses/day). Set to
    # your own email via ADMIN_EMAIL in prod; case-insensitive match.
    admin_email: str | None = None

    # Free-tier teaser: how many annotated phase photos the starter plan sees
    # (1 = one photo with the angle NUMBERS hidden; the "soft" default. 2 =
    # both phase photos). Env-overridable so we can A/B without a redeploy.
    starter_teaser_photos: int = 1

    # --- Billing (Stripe, Stage 4). All from env. Secret keys MUST stay in the
    # environment (never in git). Price IDs are NOT secret; they differ between
    # test and live mode, so they live in env too (swap when going live). When
    # stripe_secret_key is unset, the /billing endpoints return 503 and the
    # frontend degrades to a "coming soon" message. ---
    # Amounts live in services/pricing.py (the display catalogue); these are the
    # Stripe ids that actually charge. The two move together -- the comments
    # below are the only place the pairing is written down.
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None
    stripe_price_enthusiast_m: str | None = None   # $9 / month
    stripe_price_enthusiast_y: str | None = None   # $69 / year
    stripe_price_full_y: str | None = None         # $99 / year
    stripe_price_expert: str | None = None         # $39 one-time
    stripe_price_unlock: str | None = None         # $4 one-time

    # Expert Reviews accepted per rolling week. The deliverable is ~40 minutes
    # of one person's attention, so this is the only product here whose supply
    # is finite -- and the failure mode of ignoring that is a queue of people
    # who paid and are waiting, which is worse than a sold-out sign. Set while
    # the queue is empty on purpose: a limit added after the rush is an apology.
    # 0 disables the check.
    expert_review_slots_per_week: int = 5
    # Absolute base URL for Checkout success/cancel redirects. Falls back to the
    # request origin when unset (works on any host).
    public_base_url: str | None = None

    # --- Outbound email (see services/notify.py). Optional: with none of this
    # set, a delivered Expert Review is only discoverable inside the app. ---
    # "resend" | "smtp" | "" (disabled)
    email_provider: str = ""
    email_api_key: str | None = None          # resend
    # The From: address. Must be on a domain verified with the provider, or
    # everything lands in spam.
    email_from: str | None = None
    # Where "reply" goes when the athlete answers the notification. Falls back
    # to email_from.
    email_reply_to: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587                      # STARTTLS; 25 is blocked on most hosts
    smtp_user: str | None = None
    smtp_password: str | None = None

    # --- Product analytics (PostHog, see services/analytics.py). Optional: with
    # no key the snippet is never injected and the server never phones home, so
    # local development and a self-hosted copy stay silent by default. ---
    # Project API key (`phc_…`). Public by design -- it ends up in the page.
    posthog_key: str | None = None
    # Ingestion host. `us` or `eu` cloud, or a reverse proxy on our own domain
    # (which is what gets past ad-blockers; see docs/POSTHOG_RU.md).
    posthog_host: str = "https://us.i.posthog.com"
    # Session replay is OFF unless this is set. It records the DOM -- which on a
    # results page includes the athlete's own overlay video -- so it is a
    # deliberate decision, not a default.
    posthog_session_recording: bool = False

    @property
    def stripe_enabled(self) -> bool:
        return bool(self.stripe_secret_key)

    @property
    def plan_price_map(self) -> dict[str, str | None]:
        """Frontend plan key -> Stripe price ID. Keys match startCheckout()."""
        return {
            "enthusiast_monthly": self.stripe_price_enthusiast_m,
            "enthusiast_yearly": self.stripe_price_enthusiast_y,
            "full_yearly": self.stripe_price_full_y,
            "expert": self.stripe_price_expert,
            "unlock": self.stripe_price_unlock,
        }

    @property
    def price_tier_map(self) -> dict[str, str]:
        """Stripe price ID -> subscription tier (for webhook lifecycle). Only
        recurring plans map to a tier; the one-time Expert Review does not."""
        out: dict[str, str] = {}
        if self.stripe_price_enthusiast_m:
            out[self.stripe_price_enthusiast_m] = "enthusiast"
        if self.stripe_price_enthusiast_y:
            out[self.stripe_price_enthusiast_y] = "enthusiast"
        if self.stripe_price_full_y:
            out[self.stripe_price_full_y] = "full"
        return out

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename

    @property
    def async_database_url(self) -> str:
        """Normalize to an async driver URL (asyncpg for Postgres)."""
        url = self.database_url
        if url.startswith("postgres://"):
            return "postgresql+asyncpg://" + url[len("postgres://"):]
        if url.startswith("postgresql://"):
            return "postgresql+asyncpg://" + url[len("postgresql://"):]
        return url

    @property
    def auth_secure(self) -> bool:
        return self.jwt_secret != "dev-insecure-change-me"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def analytics_enabled(self) -> bool:
        return bool(self.posthog_key)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_settings() -> Settings:
    models_dir = os.environ.get("VA_MODELS_DIR")
    uploads_dir = os.environ.get("VA_UPLOADS_DIR")
    return Settings(
        models_dir=Path(models_dir).resolve() if models_dir else Settings.models_dir,
        uploads_dir=Path(uploads_dir).resolve() if uploads_dir else Settings.uploads_dir,
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
        gemini_model=os.environ.get("GEMINI_MODEL") or Settings.gemini_model,
        gemini_timeout_s=_float_env("GEMINI_TIMEOUT_S", Settings.gemini_timeout_s),
        rate_limit_per_day=_int_env("VA_RATE_LIMIT_PER_DAY", Settings.rate_limit_per_day),
        job_ttl_hours=_float_env("VA_JOB_TTL_HOURS", Settings.job_ttl_hours),
        job_sweep_interval_s=_int_env(
            "VA_JOB_SWEEP_INTERVAL_S", Settings.job_sweep_interval_s,
        ),
        max_concurrent_analyses=max(1, _int_env(
            "VA_MAX_CONCURRENT_ANALYSES", Settings.max_concurrent_analyses,
        )),
        max_queued_analyses=max(1, _int_env(
            "VA_MAX_QUEUED_ANALYSES", Settings.max_queued_analyses,
        )),
        database_url=os.environ.get("DATABASE_URL") or Settings.database_url,
        jwt_secret=os.environ.get("JWT_SECRET") or Settings.jwt_secret,
        jwt_expire_days=_int_env("JWT_EXPIRE_DAYS", Settings.jwt_expire_days),
        admin_email=(os.environ.get("ADMIN_EMAIL") or "").strip().lower() or None,
        starter_teaser_photos=_int_env(
            "STARTER_TEASER_PHOTOS", Settings.starter_teaser_photos,
        ),
        stripe_secret_key=os.environ.get("STRIPE_SECRET_KEY") or None,
        stripe_webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET") or None,
        stripe_price_enthusiast_m=os.environ.get("STRIPE_PRICE_ENTHUSIAST_M") or None,
        stripe_price_enthusiast_y=os.environ.get("STRIPE_PRICE_ENTHUSIAST_Y") or None,
        stripe_price_full_y=os.environ.get("STRIPE_PRICE_FULL_Y") or None,
        stripe_price_expert=os.environ.get("STRIPE_PRICE_EXPERT") or None,
        stripe_price_unlock=os.environ.get("STRIPE_PRICE_UNLOCK") or None,
        expert_review_slots_per_week=_int_env(
            "EXPERT_REVIEW_SLOTS_PER_WEEK", Settings.expert_review_slots_per_week,
        ),
        public_base_url=(os.environ.get("PUBLIC_BASE_URL") or "").rstrip("/") or None,
        email_provider=(os.environ.get("EMAIL_PROVIDER") or "").strip().lower(),
        email_api_key=os.environ.get("EMAIL_API_KEY") or None,
        email_from=os.environ.get("EMAIL_FROM") or None,
        email_reply_to=os.environ.get("EMAIL_REPLY_TO") or None,
        smtp_host=os.environ.get("SMTP_HOST") or None,
        smtp_port=_int_env("SMTP_PORT", Settings.smtp_port),
        smtp_user=os.environ.get("SMTP_USER") or None,
        smtp_password=os.environ.get("SMTP_PASSWORD") or None,
        posthog_key=(os.environ.get("POSTHOG_KEY") or "").strip() or None,
        posthog_host=(
            (os.environ.get("POSTHOG_HOST") or "").strip().rstrip("/")
            or Settings.posthog_host
        ),
        posthog_session_recording=_bool_env(
            "POSTHOG_SESSION_RECORDING", Settings.posthog_session_recording,
        ),
    )


settings = _load_settings()
