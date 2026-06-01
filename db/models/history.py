import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as pgUUID
from sqlmodel import Field, SQLModel


class LLMHistoryV2(SQLModel, table=True):
    """Dedicated persisted history for the PydanticAI connector."""

    __tablename__ = "chat_history"

    session_id: uuid.UUID = Field(
        sa_column=Column(pgUUID(as_uuid=True), primary_key=True, nullable=False)
    )
    schema_version: int = Field(
        default=1,
        sa_column=Column(Integer, nullable=False, server_default="1"),
    )
    system_prompt: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    messages: list[dict] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default="'[]'::jsonb"),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
        ),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
        ),
    )
