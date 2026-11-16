"""The report chat: questions an athlete asked about one saved report, and the
answers they got.

One thread per (reader, report). The reader is whoever asked -- the athlete,
or a coach reading a linked athlete's report on their own plan -- and the
report is named the way every other ``/me/analyses`` route names it: the
owner's account and the client id. Two coaches asking about the same athlete
keep two threads; neither sees the other's.

A row is written only once an answer came back. A question the models could
not answer is not stored and not counted, so a provider outage never spends
the athlete's monthly allowance. That is also why the quota counts rows here
and not ``usage_events``: the analysis quota counts every event in that table
regardless of kind, and a chat question must never eat an analysis.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

MAX_QUESTION_CHARS = 600


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Who asked. The quota is theirs.
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    # Whose report it is (the same account for an athlete's own report).
    owner_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False,
    )
    analysis_client_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Which model wrote an answer; NULL on the athlete's own question.
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at_ms: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
