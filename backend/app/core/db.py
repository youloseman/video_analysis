"""Async SQLAlchemy engine + session + Base for accounts.

Local dev uses SQLite (default DATABASE_URL); production uses Postgres on
Railway. Tables are created on startup (``init_db``) -- fine for the current
simple schema; switch to Alembic migrations when the schema starts evolving.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

logger = structlog.get_logger()


class Base(DeclarativeBase):
    pass


engine = create_async_engine(settings.async_database_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a scoped async session."""
    async with SessionLocal() as session:
        yield session


def _existing_columns(conn, table: str) -> set[str]:
    """Column names on ``table`` via SQLAlchemy's dialect inspector (works for
    both SQLite and Postgres)."""
    from sqlalchemy import inspect

    insp = inspect(conn)
    if table not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def _migrate_users(conn) -> None:
    """Lightweight, idempotent schema evolution for the ``users`` table.

    We don't use Alembic yet and ``create_all`` never ALTERs existing tables,
    so newly added columns must be back-filled here on startup. Safe to run on
    every boot: each step checks the current schema first.
    """
    from app.models.user import PAID_TIERS, TIER_STARTER

    cols = _existing_columns(conn, "users")
    if not cols:  # fresh DB -- create_all already built the current schema.
        return

    if "tier" not in cols:
        # Add the column with a safe default, then backfill from is_pro:
        # legacy pro accounts -> enthusiast, everyone else -> starter.
        conn.execute(text(
            f"ALTER TABLE users ADD COLUMN tier VARCHAR(20) "
            f"NOT NULL DEFAULT '{TIER_STARTER}'"
        ))
        if "is_pro" in cols:
            conn.execute(text(
                "UPDATE users SET tier = 'enthusiast' WHERE is_pro = true"
            ))
        logger.info("MIGRATED", change="users.tier added + backfilled")

    # Promote the configured admin account (idempotent).
    admin = (settings.admin_email or "").strip().lower()
    if admin:
        conn.execute(
            text("UPDATE users SET tier = 'admin' WHERE lower(email) = :e"),
            {"e": admin},
        )


async def init_db() -> None:
    # Import models so they register on Base.metadata before create_all.
    from app.models import analysis, usage, user  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_users)
    logger.info("DB_READY", backend=settings.async_database_url.split("://", 1)[0])
