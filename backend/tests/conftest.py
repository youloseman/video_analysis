"""Test fixtures.

Two constraints shape this file.

1. ``app.core.db`` builds its engine at import time from ``settings``, and
   ``settings`` is read from the environment at import time too. So the
   environment has to be set up **before** anything under ``app`` is imported --
   hence the ``os.environ`` block at the top, above the app imports.

2. The analysis core (mediapipe/opencv) is a heavy native dependency and is not
   needed to test billing, gating or quotas. Nothing here imports ``app.main``
   or ``app.services.video_analysis``, so the suite runs on a plain Python
   install with no ML stack.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

# --- must precede any `app.*` import (see note 1 above) --------------------
_TMP_DB_DIR = Path(tempfile.mkdtemp(prefix="flapp-tests-"))
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP_DB_DIR / 'test.db'}"
os.environ["JWT_SECRET"] = "test-secret-not-the-prod-one"
os.environ.pop("ADMIN_EMAIL", None)
# ``Settings`` is a frozen dataclass loaded once from the environment, so price
# IDs have to be planted here rather than monkeypatched onto it later. These are
# the ids the webhook tests map to tiers.
os.environ["STRIPE_PRICE_ENTHUSIAST_M"] = "price_test_enthusiast_monthly"
os.environ["STRIPE_PRICE_ENTHUSIAST_Y"] = "price_test_enthusiast_yearly"
os.environ["STRIPE_PRICE_FULL_Y"] = "price_test_full_yearly"
os.environ["STRIPE_PRICE_EXPERT"] = "price_test_expert"
os.environ["STRIPE_PRICE_UNLOCK"] = "price_test_unlock"
# --------------------------------------------------------------------------

from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.db import Base  # noqa: E402
from app.models import analysis, feedback, order, usage, user  # noqa: E402,F401
from app.models.user import TIER_STARTER, User  # noqa: E402


@pytest.fixture
async def db() -> AsyncSession:
    """A session on a fresh, empty database — one file per test.

    Per-test isolation (rather than rollback-per-test on a shared DB) keeps the
    quota tests honest: they count rows in a window, so leftovers from another
    test would quietly change their answers.
    """
    path = _TMP_DB_DIR / f"{uuid.uuid4().hex}.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        # Windows will not unlink a file another handle still holds, and
        # aiosqlite's connection thread can outlive engine.dispose() by a beat
        # -- reliably so after a test that rolled back a failed INSERT. The
        # directory is a mkdtemp() the OS reclaims anyway, so failing teardown
        # over it would turn a passing test red for no product reason.
        pass


@pytest.fixture
async def make_user(db: AsyncSession):
    """Factory for persisted users: ``await make_user(tier=...)``."""
    counter = {"n": 0}

    async def _make(tier: str = TIER_STARTER, **kwargs) -> User:
        counter["n"] += 1
        u = User(
            email=kwargs.pop("email", f"u{counter['n']}@example.com"),
            password_hash="x",
            tier=tier,
            **kwargs,
        )
        db.add(u)
        await db.commit()
        await db.refresh(u)
        return u

    return _make
