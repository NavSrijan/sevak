import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

from sqlmodel import SQLModel
import db.models  # Crucial: Must be imported so metadata is populated

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata

def get_sync_url():
    """Fetches the DB URL and enforces the synchronous psycopg3 driver."""
    raw_url = os.environ.get("PG_DIRECT_URL")
    if not raw_url:
        raise ValueError("PG_DIRECT_URL is not set in the environment.")
    
    # Converts postgresql:// to postgresql+psycopg:// for the sync engine
    return raw_url.replace("postgresql://", "postgresql+psycopg://")

# Dynamically inject the URL into the config
config.set_main_option("sqlalchemy.url", get_sync_url())


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
