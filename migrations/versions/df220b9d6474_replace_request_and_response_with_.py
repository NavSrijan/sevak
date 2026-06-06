"""replace request and response with payload in chat_history

Revision ID: df220b9d6474
Revises: f419011736ce
Create Date: 2026-06-06 15:35:43.780180

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'df220b9d6474'
down_revision: Union[str, Sequence[str], None] = 'f419011736ce'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Add payload column as nullable
    op.add_column('chat_history', sa.Column('payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    
    # 2. Populate payload from existing request/response data
    op.execute("""
        UPDATE chat_history 
        SET payload = CASE 
            WHEN jsonb_typeof(response) = 'array' THEN jsonb_build_array(request) || response
            ELSE jsonb_build_array(request, response)
        END
    """)
    
    # 3. Alter payload column to be NOT NULL
    op.alter_column('chat_history', 'payload', nullable=False)
    
    # 4. Drop old columns
    op.drop_column('chat_history', 'request')
    op.drop_column('chat_history', 'response')


def downgrade() -> None:
    """Downgrade schema."""
    # 1. Add request/response columns as nullable
    op.add_column('chat_history', sa.Column('request', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('chat_history', sa.Column('response', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    
    # 2. Populate request/response from payload data
    op.execute("""
        UPDATE chat_history 
        SET request = payload->0,
            response = CASE 
                WHEN jsonb_array_length(payload) > 2 THEN payload - 0 
                ELSE payload->1 
            END
    """)
    
    # 3. Alter request/response to NOT NULL
    op.alter_column('chat_history', 'request', nullable=False)
    op.alter_column('chat_history', 'response', nullable=False)
    
    # 4. Drop payload column
    op.drop_column('chat_history', 'payload')
