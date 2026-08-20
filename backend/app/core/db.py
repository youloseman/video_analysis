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
    from app.models.user import TIER_STARTER

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

    # Billing columns (Stage 4). Nullable -> no backfill needed.
    if "stripe_customer_id" not in cols:
        conn.execute(text("ALTER TABLE users ADD COLUMN stripe_customer_id VARCHAR(64)"))
        logger.info("MIGRATED", change="users.stripe_customer_id added")
    if "subscription_status" not in cols:
        conn.execute(text("ALTER TABLE users ADD COLUMN subscription_status VARCHAR(32)"))
        logger.info("MIGRATED", change="users.subscription_status added")
    if "expert_credits" not in cols:
        # NOT NULL with a default, unlike the nullable billing columns above:
        # every read of this is arithmetic ("do they have one left?"), and a
        # NULL balance would make that a three-way answer.
        conn.execute(text(
            "ALTER TABLE users ADD COLUMN expert_credits INTEGER NOT NULL DEFAULT 0"
        ))
        # Anyone already on Full paid for a term that included a review they
        # had no way to claim, because the mechanism did not exist. Grant it.
        conn.execute(text(
            "UPDATE users SET expert_credits = 1 WHERE tier = 'full'"
        ))
        logger.info("MIGRATED", change="users.expert_credits added + granted to full")
    if "free_preview_job_id" not in cols:
        conn.execute(text(
            "ALTER TABLE users ADD COLUMN free_preview_job_id VARCHAR(64)"
        ))
        # Left NULL for everyone, including existing accounts: they signed up
        # when the free tier showed a score and nothing else, so giving them
        # the preview is a reason to come back rather than a giveaway.
        logger.info("MIGRATED", change="users.free_preview_job_id added")
    if "expert_credit_grant_ref" not in cols:
        conn.execute(text(
            "ALTER TABLE users ADD COLUMN expert_credit_grant_ref VARCHAR(255)"
        ))
        logger.info("MIGRATED", change="users.expert_credit_grant_ref added")

    # Promote the configured admin account (idempotent).
    admin = (settings.admin_email or "").strip().lower()
    if admin:
        conn.execute(
            text("UPDATE users SET tier = 'admin' WHERE lower(email) = :e"),
            {"e": admin},
        )


def _migrate_orders(conn) -> None:
    """Same idempotent, Alembic-less evolution as ``_migrate_users``.

    The Expert Review deliverable (which analysis was bought, the report itself,
    when it shipped) landed after the first orders were already in the table.
    """
    cols = _existing_columns(conn, "orders")
    if not cols:  # fresh DB -- create_all already built the current schema.
        return
    # Match what the model declares per dialect (JSONB on Postgres), or the
    # column ends up a plain JSON that the ORM then treats as JSONB.
    json_type = "JSONB" if conn.dialect.name == "postgresql" else "JSON"
    # All nullable, so no backfill: an order placed before this existed simply
    # has no linked analysis, which is the truth about it.
    for name, ddl in (
        ("analysis_client_id", "ALTER TABLE orders ADD COLUMN analysis_client_id VARCHAR(64)"),
        ("report", f"ALTER TABLE orders ADD COLUMN report {json_type}"),
        ("delivered_at_ms", "ALTER TABLE orders ADD COLUMN delivered_at_ms BIGINT"),
        ("read_at_ms", "ALTER TABLE orders ADD COLUMN read_at_ms BIGINT"),
    ):
        if name not in cols:
            conn.execute(text(ddl))
            logger.info("MIGRATED", change=f"orders.{name} added")


def _migrate_analyses(conn) -> None:
    """Same idempotent, Alembic-less evolution as the two above.

    ``job_id`` links a stored analysis to the footage it came from. Rows written
    before it existed keep NULL, which is the truth about them: those clips were
    swept hours after the analysis and nothing recorded where they had been.
    """
    cols = _existing_columns(conn, "analyses")
    if not cols:  # fresh DB -- create_all already built the current schema.
        return
    if "job_id" not in cols:
        conn.execute(text("ALTER TABLE analyses ADD COLUMN job_id VARCHAR(64)"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_analyses_job_id ON analyses (job_id)"
        ))
        logger.info("MIGRATED", change="analyses.job_id added")
    json_type = "JSONB" if conn.dialect.name == "postgresql" else "JSON"
    if "result" not in cols:
        # NULL for every row written before this: those analyses were trimmed
        # before anything could store them, so there is genuinely nothing to
        # back-fill. They stay unsellable, and the reader says so rather than
        # offering an unlock that would reveal an empty report.
        conn.execute(text(f"ALTER TABLE analyses ADD COLUMN result {json_type}"))
        logger.info("MIGRATED", change="analyses.result added")
    if "preview" not in cols:
        conn.execute(text(
            "ALTER TABLE analyses ADD COLUMN preview BOOLEAN NOT NULL DEFAULT false"
        ))
        logger.info("MIGRATED", change="analyses.preview added")
    if "unlocked_at_ms" not in cols:
        conn.execute(text("ALTER TABLE analyses ADD COLUMN unlocked_at_ms BIGINT"))
        logger.info("MIGRATED", change="analyses.unlocked_at_ms added")


async def init_db() -> None:
    # Import models so they register on Base.metadata before create_all.
    from app.models import analysis, feedback, order, usage, user  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_users)
        await conn.run_sync(_migrate_orders)
        await conn.run_sync(_migrate_analyses)
    logger.info("DB_READY", backend=settings.async_database_url.split("://", 1)[0])
