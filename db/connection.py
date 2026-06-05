import os
import asyncio
import logging
import psycopg
from typing import Awaitable, Callable, TypeVar
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

T = TypeVar("T")

class DatabaseManager:
    def __init__(self):
        self._engine = None
        self._session_factory = None
        self._raw_conn: psycopg.AsyncConnection | None = None
        self._raw_conn_lock = asyncio.Lock()

    async def connect(self):
        """Initializes the async engine and connection pool."""
        raw_url = os.getenv("PG_DIRECT_URL")
        if not raw_url:
            raise ValueError("PG_DIRECT_URL environment variable is missing.")
        
        # Enforce async psycopg3 driver for the application
        async_url = raw_url.replace("postgresql://", "postgresql+psycopg_async://")

        self._engine = create_async_engine(
            async_url,
            pool_size=5,         # Base number of connections kept open
            max_overflow=10,     # Allow up to 10 extra connections during spikes
            pool_pre_ping=True,  # Verify connection is alive before checking out
            echo=False           # Set to True to debug generated SQL queries
        )
        
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        logger.info("Database pool initialized (SQLModel + Async psycopg3).")

    async def disconnect(self):
        """Drains the connection pool and raw connection. Call during app shutdown."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database pool disposed.")

        async with self._raw_conn_lock:
            if self._raw_conn is not None and not self._raw_conn.closed:
                await self._raw_conn.close()
            self._raw_conn = None
            logger.info("Raw database connection closed.")

    @asynccontextmanager
    async def get_session(self):
        """Yields an async session. Usage: async with db.get_session() as session:"""
        if not self._session_factory:
            raise RuntimeError("Database pool not initialized. Call db.connect() first.")
        
        async with self._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    # ------------------------------------------------------------------
    # Raw psycopg helper methods for custom queries and libraries
    # ------------------------------------------------------------------

    async def _connect_raw_pg(self) -> psycopg.AsyncConnection:
        raw_url = os.getenv("PG_DIRECT_URL")
        if not raw_url:
            raise ValueError("PG_DIRECT_URL environment variable is missing.")
        conn = await psycopg.AsyncConnection.connect(raw_url)
        await conn.set_autocommit(False)
        return conn

    async def _ensure_raw_conn(self, *, force_reconnect: bool = False) -> psycopg.AsyncConnection:
        async with self._raw_conn_lock:
            if force_reconnect and self._raw_conn is not None and not self._raw_conn.closed:
                await self._raw_conn.close()
                self._raw_conn = None

            if (
                self._raw_conn is None
                or self._raw_conn.closed
                or getattr(self._raw_conn, "broken", False)
            ):
                self._raw_conn = await self._connect_raw_pg()
                logger.info("Raw psycopg connection established.")

            return self._raw_conn

    async def _rollback_raw_conn(self) -> None:
        if self._raw_conn is None or self._raw_conn.closed:
            return

        try:
            await self._raw_conn.rollback()
        except psycopg.Error:
            logger.warning("Rollback on raw connection failed; reconnecting on next use.")

    async def run_raw_op(self, op: Callable[[psycopg.AsyncConnection], Awaitable[T]]) -> T:
        """Runs a database operation on a raw psycopg AsyncConnection with auto-commit/rollback and retry."""
        last_error: Exception | None = None

        for attempt in range(2):
            conn = await self._ensure_raw_conn(force_reconnect=attempt == 1)
            try:
                result = await op(conn)
                await conn.commit()
                return result
            except psycopg.Error as exc:
                last_error = exc
                await self._rollback_raw_conn()
                logger.warning("Raw DB operation failed on attempt %s: %s", attempt + 1, exc)

        assert last_error is not None
        raise last_error

# Export singleton instance
db = DatabaseManager()
