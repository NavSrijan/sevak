import os
import logging
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self._engine = None
        self._session_factory = None

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
        """Drains the connection pool. Call during app shutdown."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database pool disposed.")

    @asynccontextmanager
    async def get_session(self):
        """Yields an async session. Usage: async with db.get_session() as session:"""
        if not self._session_factory:
            raise RuntimeError("Database pool not initialized. Call db.connect() first.")
        
        async with self._session_factory() as session:
            try:
                yield session
                # Note: No auto-commit here. If your agent mutates state, 
                # you must explicitly call `await session.commit()` in the skill.
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

# Export singleton instance
db = DatabaseManager()
