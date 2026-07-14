"""Async SQLAlchemy engine + session + Base for accounts.

Local dev uses SQLite (default DATABASE_URL); production uses Postgres on
Railway. Tables are created on startup (``init_db``) -- fine for the current
simple schema; switch to Alembic migrations when the schema starts evolving.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import structlog
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


async def init_db() -> None:
    # Import models so they register on Base.metadata before create_all.
    from app.models import analysis, user  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("DB_READY", backend=settings.async_database_url.split("://", 1)[0])
