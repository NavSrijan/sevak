"""change chat_history to request_response rows

Revision ID: c1e9dfe789a3
Revises: b2a7d9579673
Create Date: 2026-06-02 18:24:21.704661

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c1e9dfe789a3'
down_revision: Union[str, Sequence[str], None] = 'b2a7d9579673'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Drop foreign key constraint on episodes pointing to chat_history
    op.drop_constraint('episodes_session_id_fkey', 'episodes', type_='foreignkey')
    
    # 2. Drop the old chat_history table
    op.drop_table('chat_history')
    
    # 3. Create the new chat_history table
    op.create_table(
        'chat_history',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('system_prompt', sa.Text(), nullable=False),
        sa.Column('request', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('response', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_chat_history_session_id'), 'chat_history', ['session_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # 1. Drop new chat_history index and table
    op.drop_index(op.f('ix_chat_history_session_id'), table_name='chat_history')
    op.drop_table('chat_history')
    
    # 2. Recreate old chat_history table
    op.create_table(
        'chat_history',
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('system_prompt', sa.Text(), nullable=False),
        sa.Column('messages', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('session_id')
    )
    
    # 3. Re-add foreign key constraint to episodes table
    op.create_foreign_key('episodes_session_id_fkey', 'episodes', 'chat_history', ['session_id'], ['session_id'])
