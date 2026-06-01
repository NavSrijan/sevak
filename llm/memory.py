import os
import uuid
import logging
import asyncio
from datetime import datetime, timezone
from typing import Awaitable, Callable, TypeVar

import psycopg
from pydantic import ValidationError
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from utils.helpers import to_uuid

logger = logging.getLogger(__name__)

HISTORY_TABLE = "chat_history"
HISTORY_SCHEMA_VERSION = 1
T = TypeVar("T")

class MemoryManager:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.pg_conn: psycopg.AsyncConnection | None = None
        self._conn_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize async resources."""
        await self._ensure_pg_conn()
        await self._create_history_table()

    async def aclose(self) -> None:
        """Close the dedicated history connection."""
        async with self._conn_lock:
            if self.pg_conn is not None and not self.pg_conn.closed:
                await self.pg_conn.close()
            self.pg_conn = None

    async def _connect_pg(self) -> psycopg.AsyncConnection:
        conn = await psycopg.AsyncConnection.connect(os.getenv("PG_DIRECT_URL"))
        await conn.set_autocommit(False)
        return conn

    async def _ensure_pg_conn(self, *, force_reconnect: bool = False) -> psycopg.AsyncConnection:
        async with self._conn_lock:
            if force_reconnect and self.pg_conn is not None and not self.pg_conn.closed:
                await self.pg_conn.close()
                self.pg_conn = None

            if (
                self.pg_conn is None
                or self.pg_conn.closed
                or getattr(self.pg_conn, "broken", False)
            ):
                self.pg_conn = await self._connect_pg()
                logger.info("History database connection established.")

            return self.pg_conn

    async def _rollback_pg_conn(self) -> None:
        if self.pg_conn is None or self.pg_conn.closed:
            return

        try:
            await self.pg_conn.rollback()
        except psycopg.Error:
            logger.warning("Rollback on history connection failed; reconnecting on next use.")

    async def _run_db_op(self, op: Callable[[psycopg.AsyncConnection], Awaitable[T]]) -> T:
        last_error: Exception | None = None

        for attempt in range(2):
            conn = await self._ensure_pg_conn(force_reconnect=attempt == 1)
            try:
                result = await op(conn)
                await conn.commit()
                return result
            except psycopg.Error as exc:
                last_error = exc
                await self._rollback_pg_conn()
                logger.warning("History DB operation failed on attempt %s: %s", attempt + 1, exc)

        assert last_error is not None
        raise last_error

    async def _create_history_table(self) -> None:
        """Create the history table if it doesn't exist."""

        async def op(conn: psycopg.AsyncConnection) -> None:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
                        session_id UUID PRIMARY KEY,
                        schema_version INTEGER NOT NULL DEFAULT {HISTORY_SCHEMA_VERSION},
                        system_prompt TEXT NULL,
                        messages JSONB NOT NULL DEFAULT '[]'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )

        await self._run_db_op(op)

    def _serialize_history_messages(self, messages: list) -> list[dict]:
        """Convert messages to a minimal JSON-serializable list of dicts to save space."""
        serialized: list[dict] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                parts = [
                    {"content": part.content, "part_kind": "user-prompt"}
                    for part in message.parts
                    if isinstance(part, UserPromptPart)
                    and isinstance(part.content, str)
                    and part.content.strip()
                ]
                if parts:
                    serialized.append(
                        {
                            "kind": "request",
                            "parts": parts,
                        }
                    )
            elif isinstance(message, ModelResponse):
                parts = [
                    {"content": part.content, "part_kind": "text"}
                    for part in message.parts
                    if isinstance(part, TextPart) and part.content.strip()
                ]
                if parts:
                    serialized.append(
                        {
                            "kind": "response",
                            "parts": parts,
                        }
                    )

        return serialized

    async def load_history(self, session_id: str) -> tuple[list, str | None]:
        """Load message history and stored system prompt from the dedicated v2 table."""
        async def op(conn: psycopg.AsyncConnection) -> tuple[list, str | None]:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT messages, system_prompt
                    FROM {HISTORY_TABLE}
                    WHERE session_id = %s
                    """,
                    (to_uuid(session_id),),
                )
                row = await cur.fetchone()
                return ([], None) if row is None else (row[0], row[1])

        raw_messages, stored_system_prompt = await self._run_db_op(op)
        if not raw_messages:
            return [], stored_system_prompt

        try:
            return ModelMessagesTypeAdapter.validate_python(raw_messages), stored_system_prompt
        except ValidationError:
            logger.warning("Stored history for session %s is invalid; ignoring it.", session_id)
            return [], stored_system_prompt

    async def save_history(self, session_id: str, messages: list, system_prompt: str | None = None) -> None:
        """Persist the session history in one row."""
        payload = self._serialize_history_messages(messages)
        session_uuid = to_uuid(session_id)
        prompt_value = system_prompt or self.system_prompt

        async def op(conn: psycopg.AsyncConnection) -> None:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    INSERT INTO {HISTORY_TABLE} (
                        session_id,
                        schema_version,
                        system_prompt,
                        messages,
                        created_at,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (session_id)
                    DO UPDATE SET
                        schema_version = EXCLUDED.schema_version,
                        system_prompt = EXCLUDED.system_prompt,
                        messages = EXCLUDED.messages,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        session_uuid,
                        HISTORY_SCHEMA_VERSION,
                        prompt_value,
                        psycopg.types.json.Jsonb(payload),
                        datetime.now(timezone.utc),
                        datetime.now(timezone.utc),
                    ),
                )

        await self._run_db_op(op)
