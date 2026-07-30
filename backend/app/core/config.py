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

    # Abuse guard: max analyses per client (IP) per rolling 24h. 0 disables.
    rate_limit_per_day: int = 3

    # Accounts (auth). DATABASE_URL defaults to a local SQLite file; set it to a
    # Postgres URL (Railway) in prod. jwt_secret MUST be overridden in prod.
    database_url: str = f"sqlite+aiosqlite:///{BACKEND_DIR / 'flapp.db'}"
    jwt_secret: str = "dev-insecure-change-me"
    jwt_expire_days: int = 30

    # Account promoted to the ``admin`` tier on startup (5 analyses/day). Set to
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
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None
    stripe_price_enthusiast_m: str | None = None   # $9 / month
    stripe_price_enthusiast_y: str | None = None   # $59 / year
    stripe_price_full_y: str | None = None         # $80 / year
    stripe_price_expert: str | None = None         # $29 one-time
    # Absolute base URL for Checkout success/cancel redirects. Falls back to the
    # request origin when unset (works on any host).
    public_base_url: str | None = None

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


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _load_settings() -> Settings:
    models_dir = os.environ.get("VA_MODELS_DIR")
    uploads_dir = os.environ.get("VA_UPLOADS_DIR")
    return Settings(
        models_dir=Path(models_dir).resolve() if models_dir else Settings.models_dir,
        uploads_dir=Path(uploads_dir).resolve() if uploads_dir else Settings.uploads_dir,
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
        gemini_model=os.environ.get("GEMINI_MODEL") or Settings.gemini_model,
        rate_limit_per_day=_int_env("VA_RATE_LIMIT_PER_DAY", Settings.rate_limit_per_day),
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
        public_base_url=(os.environ.get("PUBLIC_BASE_URL") or "").rstrip("/") or None,
    )


settings = _load_settings()
