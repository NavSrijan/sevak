import uuid
import enum
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Integer, Text, Float, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID as pgUUID
from sqlalchemy import Enum as PgEnum
from sqlmodel import Field, SQLModel
from pgvector.sqlalchemy import Vector



# ─────────────────────────────────────────────────────────────
# STATE MACHINES (ENUMS)
# ─────────────────────────────────────────────────────────────

class EpisodeStatus(str, enum.Enum):
    """State Transitions"""
    active = "active"
    closed = "closed"
    summarized = "summarized"
    dreamed = "dreamed"


class DreamType(str, enum.Enum):
    """Categorization of dream types based on their origin and purpose:"""
    pattern = "pattern"
    contradiction = "contradiction"
    insight = "insight"
    hypothesis = "hypothesis"
    emotional = "emotional"


class ClarificationStatus(str, enum.Enum):
    """Status of a clarification question."""
    pending = "pending"
    asked = "asked"
    resolved = "resolved"
    dismissed = "dismissed"


# ─────────────────────────────────────────────────────────────
# LAYER 1: Chat History
# ─────────────────────────────────────────────────────────────

class ChatHistory(SQLModel, table=True):
    """Chat History stores individual request-response pairs for a session."""
    __tablename__ = "chat_history"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(pgUUID(as_uuid=True), primary_key=True),  # (PK)
    )
    session_id: uuid.UUID = Field(
        sa_column=Column(pgUUID(as_uuid=True), nullable=False, index=True)
    )
    payload: list = Field(
        sa_column=Column(JSONB, nullable=False)
    )
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


# ─────────────────────────────────────────────────────────────
# LAYER 2: Episodic Memory
# ─────────────────────────────────────────────────────────────

class Episode(SQLModel, table=True):
    """
    Episode History

    Episode is a collection of interactions, which are grouped together based on temporal proximity or thematic similarity, or due to technical constraints (i.e. token limits).
    It is also stored as an embedding vector.
    """
    __tablename__ = "episodes"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(pgUUID(as_uuid=True), primary_key=True),  # (PK)
    )
    
    # Logical reference to Chat History session
    session_id: uuid.UUID = Field(
        sa_column=Column(pgUUID(as_uuid=True), nullable=False, index=True),
    )
    
    status: EpisodeStatus = Field(
        default=EpisodeStatus.active,
        sa_column=Column(PgEnum(EpisodeStatus), nullable=False, server_default="active"),
    )
    title: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    summary: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    
    # 768 vector for nomic-embed-text-v2-moe
    embedding: list[float] | None = Field(
        default=None,
        sa_column=Column(Vector(768), nullable=True),
    )
    
    # Metadata dictionary for entities mentioned, user sentiment, and temporal anchors
    meta_data: dict = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="'{}'::jsonb"),
    )
    importance_score: float = Field(
        default=0.5, sa_column=Column(Float, nullable=False, server_default="0.5")
    )
    
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    ended_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    last_accessed: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )


# ─────────────────────────────────────────────────────────────
# LAYER 3: Graph Memory (Entities & Relations)
# ─────────────────────────────────────────────────────────────

class EntityTaxonomy(SQLModel, table=True):
    """Mapped from Entity-taxonomy block in image_998ce0.png"""
    __tablename__ = "entity_taxonomy"

    id: str = Field(primary_key=True)  # e.g., 'car', 'vehicle', 'person' (PK)
    
    # Self-referential hierarchy link
    parent_id: str | None = Field(
        default=None,
        sa_column=Column(Text, ForeignKey("entity_taxonomy.id"), nullable=True),  # (FK)
    )
    description: str = Field(sa_column=Column(Text, nullable=False))

class MemoryEntity(SQLModel, table=True):
    """Mapped from Graph Memory-entity block in image_998ce0.png"""
    __tablename__ = "memory_entities"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(pgUUID(as_uuid=True), primary_key=True),  # (PK)
    )
    name: str = Field(sa_column=Column(Text, nullable=False, index=True))
    
    # Direct relationship to dynamic Taxonomy Tree
    entity_type: str = Field(
        sa_column=Column(Text, ForeignKey("entity_taxonomy.id"), nullable=False)  # (FK)
    )
    description: str = Field(sa_column=Column(Text, nullable=False))
    
    # Profile serialization embedding
    embedding: list[float] | None = Field(
        default=None,
        sa_column=Column(Vector(768), nullable=True),
    )
    
    confidence: float = Field(
        default=1.0, sa_column=Column(Float, nullable=False, server_default="1.0")  # Float [0->1]
    )
    first_seen_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    last_seen_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

class Predicates(SQLModel, table=True):
    """Predicates for entities"""
    __tablename__ = "predicates"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(pgUUID(as_uuid=True), primary_key=True),
    )

    predicate: str = Field(sa_column=Column(Text, nullable=False))

    usage_count: int = Field(
        default=1, sa_column=Column(Integer, nullable=False, server_default="1")
    )
    description: str = Field(sa_column=Column(Text, nullable=True))
    embedding: list[float] | None = Field(
        default=None,
        sa_column=Column(Vector(768), nullable=True),
    )

    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

class EntityFacts(SQLModel, table=True):
    """Facets for entities"""
    __tablename__ = "entity_facts"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(pgUUID(as_uuid=True), primary_key=True),
    )

    entity_id: uuid.UUID = Field(
        sa_column=Column(pgUUID(as_uuid=True), ForeignKey("memory_entities.id"), nullable=False)
    )

    predicate_id: uuid.UUID = Field(
        sa_column=Column(pgUUID(as_uuid=True), ForeignKey("predicates.id"), nullable=False)
    )

    value_json: dict = Field(
        sa_column=Column(JSONB, nullable=False, server_default="'{}'::jsonb"),
    )

    target_entity_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(pgUUID(as_uuid=True), ForeignKey("memory_entities.id"), nullable=True)
    )

    confidence: float = Field(
        default=0.5, sa_column=Column(Float, nullable=False, server_default="0.5")  # Float [0->1]
    )
    source_episode_id: uuid.UUID = Field(
        sa_column=Column(pgUUID(as_uuid=True), ForeignKey("episodes.id"), nullable=False)  # (FK)
    )
    valid_from: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    valid_until: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    last_updated: datetime = Field(
            default_factory=lambda: datetime.now(timezone.utc),
            sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    created_at: datetime = Field(
            default_factory=lambda: datetime.now(timezone.utc),
            sa_column=Column(DateTime(timezone=True), nullable=False),
    )


# ─────────────────────────────────────────────────────────────
# LAYER 4: Dreams & Operational Feedback Loop
# ─────────────────────────────────────────────────────────────

class Dream(SQLModel, table=True):
    """Mapped from Dreams block in image_998ce0.png"""
    __tablename__ = "dreams"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(pgUUID(as_uuid=True), primary_key=True),  # (PK)
    )
    dream_type: DreamType = Field(
        sa_column=Column(PgEnum(DreamType), nullable=False)
    )
    content: str = Field(sa_column=Column(Text, nullable=False))
    
    embedding: list[float] | None = Field(
        default=None,
        sa_column=Column(Vector(768), nullable=True),
    )
    
    # Tracking linear paths to original data points
    source_episode_ids: list[uuid.UUID] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default="'[]'::jsonb"),  # list[UUID]
    )
    related_entity_ids: list[uuid.UUID] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default="'[]'::jsonb"),  # list[UUID]
    )
    
    confidence: float = Field(
        default=0.5, sa_column=Column(Float, nullable=False, server_default="0.5")  # Float [0->1]
    )
    reinforcement_count: int = Field(
        default=1, sa_column=Column(Integer, nullable=False, server_default="1")
    )
    dreamed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    reinforced_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class Clarification(SQLModel, table=True):
    """Mapped from Clarifications block in image_998ce0.png"""
    __tablename__ = "clarifications"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(pgUUID(as_uuid=True), primary_key=True),  # (PK)
    )
    question: str = Field(sa_column=Column(Text, nullable=False))
    context_rationale: str = Field(sa_column=Column(Text, nullable=False))
    status: ClarificationStatus = Field(
        default=ClarificationStatus.pending,
        sa_column=Column(PgEnum(ClarificationStatus), nullable=False, server_default="'pending'"),
    )
    priority_score: int = Field(
        default=1, sa_column=Column(Integer, nullable=False, server_default="1")
    )
    
    # Linked lineage fields tracing back to either background Insight or Live Interaction
    source_dream_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(pgUUID(as_uuid=True), ForeignKey("dreams.id"), nullable=True),  # (FK)
    )
    source_episode_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(pgUUID(as_uuid=True), ForeignKey("episodes.id"), nullable=True),  # (FK)
    )
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
