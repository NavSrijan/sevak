import uuid
import logging
import httpx
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError, BaseModel, Field
from pydantic_ai import Agent, ModelMessagesTypeAdapter, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from sqlmodel import select
from sqlalchemy import func, text

from utils.helpers import to_uuid
from db.connection import db
from db.models import ChatHistory, EntityTaxonomy, Episode, MemoryEntity, MemoryRelation, EpisodeStatus
import config

logger = logging.getLogger(__name__)

# Shared HTTP client for background memory model requests
http_client = httpx.AsyncClient()

model = OllamaModel(
    config.LLM_MODEL,
    provider=OllamaProvider(
        base_url=config.OLLAMA_BASE_URL,
        http_client=http_client
    ),
)

# Core background helper agent to process semantic structure and memory
memory_agent = Agent(
    model=model,
    system_prompt=(
        "You are a core background memory helper agent. Your task is to process episodic memory and extract entities/relationships.\n"
        "Crucial Instructions:\n"
        "1. Pronoun Resolution: Never output pronouns (like 'he', 'she', 'they', 'it') as entity names. Use the search/profile tools to resolve pronouns to the actual entity names (e.g. resolve 'he' to 'Nav' or 'John Doe').\n"
        "2. Duplication & Aliasing: Use search tools to find if an entity already exists under a slightly different name (e.g. 'Nav' vs 'Navsrijan'). If you find a duplicate/alias, use merge_duplicate_entities.\n"
        "3. Taxonomy: Query allowed types using get_entity_taxonomy and keep entities within valid types.\n"
        "4. Relations: Check if a relationship link already exists using check_existing_relation before registering to prevent duplicate records."
    ),
    retries=3,
)

# Structured output schemas for memory helper agent
class ExtractedEntity(BaseModel):
    name: str = Field(description="Name of the entity (e.g. 'John Doe', 'Google', 'Paris')")
    entity_type: str = Field(description="Type of entity (must be one of: 'person', 'organization', 'place', 'concept', 'event')")
    description: str = Field(description="Brief description context for this entity based on the discussion.")

class ExtractedRelation(BaseModel):
    source_entity_name: str = Field(description="Name of the source entity")
    target_entity_name: str = Field(description="Name of the target entity")
    relation_type: str = Field(description="Type of relation (e.g. 'works_at', 'located_in', 'participant_of')")
    description: str = Field(description="Brief description explaining their relationship.")

class MemoryExtraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list, description="Entities extracted from the interaction.")
    relations: list[ExtractedRelation] = Field(default_factory=list, description="Relations extracted from the interaction.")
    episode_title_update: str | None = Field(default=None, description="Proposed title or title update for this interaction episode.")
    episode_summary_update: str | None = Field(default=None, description="Updated running summary of the conversation episode so far.")


@memory_agent.tool
async def search_entities(ctx: RunContext[None], query: str) -> list[dict]:
    """Search existing entities in the graph by name or description to resolve pronouns or identify duplicates."""
    async with db.get_session() as session:
        res = await session.execute(
            select(MemoryEntity).where(
                (MemoryEntity.name.ilike(f"%{query}%")) | (MemoryEntity.description.ilike(f"%{query}%"))
            )
        )
        entities = res.scalars().all()
        return [{"id": str(e.id), "name": e.name, "type": e.entity_type, "description": e.description} for e in entities]


@memory_agent.tool
async def get_entity_profile(ctx: RunContext[None], entity_name: str) -> dict:
    """Fetch details and all active relationships for a specific entity name."""
    async with db.get_session() as session:
        res = await session.execute(
            select(MemoryEntity).where(MemoryEntity.name == entity_name)
        )
        entity = res.scalar_one_or_none()
        if not entity:
            return {"error": f"Entity '{entity_name}' not found."}
        
        rel_out_res = await session.execute(
            select(MemoryRelation).where(MemoryRelation.source_id == entity.id)
        )
        rels_out = rel_out_res.scalars().all()
        
        rel_in_res = await session.execute(
            select(MemoryRelation).where(MemoryRelation.target_id == entity.id)
        )
        rels_in = rel_in_res.scalars().all()
        
        relationships = []
        for r in rels_out:
            target = await session.get(MemoryEntity, r.target_id)
            relationships.append({
                "type": "outgoing",
                "relation": r.relation_type,
                "target": target.name if target else "Unknown",
                "description": r.description
            })
        for r in rels_in:
            source = await session.get(MemoryEntity, r.source_id)
            relationships.append({
                "type": "incoming",
                "relation": r.relation_type,
                "source": source.name if source else "Unknown",
                "description": r.description
            })
            
        return {
            "id": str(entity.id),
            "name": entity.name,
            "type": entity.entity_type,
            "description": entity.description,
            "attributes": entity.attributes,
            "relationships": relationships
        }


@memory_agent.tool
async def check_existing_relation(ctx: RunContext[None], source_name: str, target_name: str, relation_type: str) -> dict:
    """Check if a relationship already exists between two entities to prevent duplicate links."""
    async with db.get_session() as session:
        src_res = await session.execute(select(MemoryEntity).where(MemoryEntity.name == source_name))
        src = src_res.scalar_one_or_none()
        tgt_res = await session.execute(select(MemoryEntity).where(MemoryEntity.name == target_name))
        tgt = tgt_res.scalar_one_or_none()
        
        if not src or not tgt:
            return {"exists": False, "reason": "One or both entities do not exist."}
            
        res = await session.execute(
            select(MemoryRelation).where(
                MemoryRelation.source_id == src.id,
                MemoryRelation.target_id == tgt.id,
                MemoryRelation.relation_type == relation_type
            )
        )
        relation = res.scalar_one_or_none()
        if relation:
            return {"exists": True, "relation_id": str(relation.id), "description": relation.description}
        return {"exists": False}


@memory_agent.tool
async def get_entity_taxonomy(ctx: RunContext[None]) -> list[dict]:
    """Retrieve the list of valid entity classifications and types."""
    async with db.get_session() as session:
        res = await session.execute(select(EntityTaxonomy))
        taxonomy = res.scalars().all()
        return [{"id": t.id, "description": t.description} for t in taxonomy]


@memory_agent.tool
async def merge_duplicate_entities(ctx: RunContext[None], primary_name: str, alias_name: str) -> dict:
    """Merge an alias/duplicate entity into a primary entity, updating all relationships and deleting the alias."""
    async with db.get_session() as session:
        pri_res = await session.execute(select(MemoryEntity).where(MemoryEntity.name == primary_name))
        primary = pri_res.scalar_one_or_none()
        ali_res = await session.execute(select(MemoryEntity).where(MemoryEntity.name == alias_name))
        alias = ali_res.scalar_one_or_none()
        
        if not primary or not alias:
            return {"success": False, "error": "One or both entities not found."}
            
        out_res = await session.execute(select(MemoryRelation).where(MemoryRelation.source_id == alias.id))
        for r in out_res.scalars().all():
            r.source_id = primary.id
            session.add(r)
            
        in_res = await session.execute(select(MemoryRelation).where(MemoryRelation.target_id == alias.id))
        for r in in_res.scalars().all():
            r.target_id = primary.id
            session.add(r)
            
        await session.delete(alias)
        await session.commit()
        return {"success": True, "message": f"Merged '{alias_name}' successfully into '{primary_name}'."}


@memory_agent.tool
async def get_recent_episodes(ctx: RunContext[None], limit: int = 5) -> list[dict]:
    """Get titles and summaries of recent conversation episodes to understand history context."""
    async with db.get_session() as session:
        res = await session.execute(
            select(Episode).order_by(Episode.started_at.desc()).limit(limit)
        )
        episodes = res.scalars().all()
        return [{"id": str(ep.id), "title": ep.title, "summary": ep.summary, "status": ep.status} for ep in episodes]


class MemoryManager:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    async def get_total_tokens(self, session_id: str) -> tuple[int, int]:
        """Query and return (total_input_tokens, total_output_tokens) from the database."""
        session_uuid = uuid.UUID(to_uuid(session_id))
        async with db.get_session() as session:
            result = await session.execute(
                select(
                    func.sum(text("(response->'usage'->>'input_tokens')::integer")),
                    func.sum(text("(response->'usage'->>'output_tokens')::integer"))
                )
                .where(ChatHistory.session_id == session_uuid)
            )
            row = result.first()
            if row and row[0] is not None and row[1] is not None:
                return int(row[0]), int(row[1])
            return 0, 0

    async def initialize(self) -> None:
        """Initialize database history table and seed EntityTaxonomy."""
        try:
            async with db.get_session() as session:
                for ent_id, desc in [
                    ("person", "A human being"),
                    ("organization", "A company, institution, or group"),
                    ("place", "A location, city, country, or building"),
                    ("concept", "An abstract idea, topic, or field of study"),
                    ("event", "An occurrence, meeting, or milestone"),
                ]:
                    res = await session.get(EntityTaxonomy, ent_id)
                    if not res:
                        session.add(EntityTaxonomy(id=ent_id, description=desc))
                await session.commit()
        except Exception as e:
            logger.warning("Failed to seed EntityTaxonomy: %s", e)

    async def aclose(self) -> None:
        """No-op as the shared database pool handles raw connection shutdown."""
        pass

    async def load_history(self, session_id: str) -> tuple[list, str | None]:
        """Load message history and stored system prompt from individual request-response rows."""
        session_uuid = uuid.UUID(to_uuid(session_id))
        
        async with db.get_session() as session:
            result = await session.execute(
                select(ChatHistory.request, ChatHistory.response)
                .where(ChatHistory.session_id == session_uuid)
                .order_by(ChatHistory.created_at.asc())
            )
            rows = result.all()

        if not rows:
            return [], None

        flat_messages = []
        for req, resp in rows:
            flat_messages.append(req)
            flat_messages.append(resp)

        return flat_messages, None

    async def save_history(self, session_id: str, messages: list, system_prompt: str | None = None) -> None:
        """Persist the latest request-response pair as a new row."""
        if len(messages) < 2:
            logger.warning("Attempted to save message history with less than 2 messages; skipping.")
            return

        last_request = messages[-2]
        last_response = messages[-1]

        # Dump using TypeAdapter to format correctly as JSON-serializable types
        payload = ModelMessagesTypeAdapter.dump_python([last_request, last_response], mode='json')
        req_dict = payload[0]
        resp_dict = payload[1]

        # Clean up provider-specific details that don't add value
        resp_dict.pop("provider_name", None)
        resp_dict.pop("provider_url", None)
        resp_dict.pop("provider_details", None)
        resp_dict.pop("provider_response_id", None)
                
        session_uuid = uuid.UUID(to_uuid(session_id))
        now = datetime.now(timezone.utc)

        async with db.get_session() as session:
            history_record = ChatHistory(
                session_id=session_uuid,
                request=req_dict,
                response=resp_dict,
                created_at=now
            )
            session.add(history_record)
            await session.commit()

    async def retrieve_memory_context(self, session_id: str) -> str:
        """Fetch the active episode summary, entities, and relations to construct a prompt context."""
        session_uuid = uuid.UUID(to_uuid(session_id))
        try:
            async with db.get_session() as session:
                # 1. Fetch active episode
                res = await session.execute(
                    select(Episode).where(Episode.session_id == session_uuid, Episode.status == EpisodeStatus.active)
                )
                episode = res.scalar_one_or_none()
                if not episode:
                    return ""

                # 2. Fetch relations for this episode
                rel_res = await session.execute(
                    select(MemoryRelation).where(MemoryRelation.source_episode_id == episode.id)
                )
                relations = rel_res.scalars().all()

                # 3. Format as context
                context_parts = []
                if episode.summary:
                    context_parts.append(f"Episode Summary: {episode.summary}")
                
                if relations:
                    context_parts.append("Known Relations:")
                    for rel in relations:
                        context_parts.append(f"- Relation: {rel.relation_type} ({rel.description})")

                return "\n".join(context_parts)
        except Exception as e:
            logger.warning("Failed to retrieve memory context: %s", e)
            return ""

    async def ensure_active_episode(self, session_id: str) -> None:
        """Verify an active Episode exists for this session, creating one if not exists."""
        session_uuid = uuid.UUID(to_uuid(session_id))
        async with db.get_session() as session:
            ep_res = await session.execute(
                select(Episode).where(Episode.session_id == session_uuid, Episode.status == EpisodeStatus.active)
            )
            episode = ep_res.scalar_one_or_none()
            if not episode:
                episode = Episode(
                    session_id=session_uuid,
                    status=EpisodeStatus.active,
                    title="Active Conversation Episode",
                    summary=""
                )
                session.add(episode)
                await session.commit()
                logger.info("Created new active episode for session %s.", session_id)

    async def check_semantic_drift(self, session_id: str, memory_agent: Agent) -> bool:
        """Placeholder for semantic drift check. User will implement this later."""
        return False

    async def check_and_process_episode(self, session_id: str, memory_agent: Agent, current_time: datetime = None) -> None:
        """Check if active episode has ended due to 2 min time gap or semantic drift, extracting memory if so."""
        session_uuid = uuid.UUID(to_uuid(session_id))
        now = current_time or datetime.now(timezone.utc)
        
        async with db.get_session() as session:
            # 1. Fetch active episode
            ep_res = await session.execute(
                select(Episode).where(Episode.session_id == session_uuid, Episode.status == EpisodeStatus.active)
            )
            episode = ep_res.scalar_one_or_none()
            if not episode:
                return

            hist_res = await session.execute(
                select(ChatHistory)
                .where(ChatHistory.session_id == session_uuid)
                .order_by(ChatHistory.created_at.desc())
                .limit(1)
            )
            last_chat = hist_res.scalar_one_or_none()
            if not last_chat:
                return

            gap = (now - last_chat.created_at).total_seconds()
            
            # Check semantic drift
            drift = await self.check_semantic_drift(session_id, memory_agent)

            # Query the sum of total tokens in this active episode
            tok_res = await session.execute(
                select(
                    func.sum(text("(response->'usage'->>'input_tokens')::integer")),
                    func.sum(text("(response->'usage'->>'output_tokens')::integer"))
                )
                .where(ChatHistory.session_id == session_uuid, ChatHistory.created_at >= episode.started_at)
            )
            row = tok_res.first()
            total_tokens = 0
            if row and row[0] is not None and row[1] is not None:
                total_tokens = int(row[0]) + int(row[1])
            
            if gap > 120 or drift or total_tokens > 4000:
                logger.info("Episode ended (gap: %ss, drift: %s, tokens: %s). Extracting and closing episode.", gap, drift, total_tokens)
                
                # Fetch all ChatHistory turns since active episode started
                turns_res = await session.execute(
                    select(ChatHistory)
                    .where(ChatHistory.session_id == session_uuid, ChatHistory.created_at >= episode.started_at)
                    .order_by(ChatHistory.created_at.asc())
                )
                turns = turns_res.scalars().all()
                
                if not turns:
                    episode.status = EpisodeStatus.closed
                    episode.ended_at = now
                    session.add(episode)
                    await session.commit()
                    return

                # Format the conversation transcript
                transcript_parts = []
                for turn in turns:
                    # Resolve requests and responses from dict payloads
                    req_text = ""
                    if "parts" in turn.request:
                        req_text = turn.request["parts"][0].get("content", "")
                    resp_text = ""
                    if "parts" in turn.response:
                        resp_text = turn.response["parts"][0].get("content", "")
                    transcript_parts.append(f"User: {req_text}\nAgent: {resp_text}")
                
                transcript = "\n".join(transcript_parts)
                
                # 3. Extract memory using memory_agent
                result = await memory_agent.run(
                    f"Analyze this conversation transcript and extract entities/relationships:\n{transcript}",
                    output_type=MemoryExtraction
                )
                extracted: MemoryExtraction = result.output

                # 4. Save extracted entities
                entity_ids = {}
                for ent in extracted.entities:
                    ent_type = ent.entity_type if ent.entity_type in ['person', 'organization', 'place', 'concept', 'event'] else 'concept'
                    
                    db_ent_res = await session.execute(
                        select(MemoryEntity).where(MemoryEntity.name == ent.name, MemoryEntity.entity_type == ent_type)
                    )
                    db_ent = db_ent_res.scalar_one_or_none()
                    if not db_ent:
                        db_ent = MemoryEntity(
                            name=ent.name,
                            entity_type=ent_type,
                            description=ent.description,
                            confidence=1.0
                        )
                        session.add(db_ent)
                        await session.flush()
                    else:
                        db_ent.description = ent.description
                        db_ent.last_seen_at = datetime.now(timezone.utc)
                        session.add(db_ent)
                    
                    entity_ids[ent.name] = db_ent.id

                # 5. Save relations
                for rel in extracted.relations:
                    source_id = entity_ids.get(rel.source_entity_name)
                    target_id = entity_ids.get(rel.target_entity_name)
                    
                    if source_id and target_id:
                        db_rel = MemoryRelation(
                            source_id=source_id,
                            target_id=target_id,
                            relation_type=rel.relation_type,
                            description=rel.description,
                            source_episode_id=episode.id,
                            confidence=1.0
                        )
                        session.add(db_rel)

                # 6. Close the active episode
                episode.title = extracted.episode_title_update or episode.title
                episode.summary = extracted.episode_summary_update or episode.summary
                episode.status = EpisodeStatus.closed
                episode.ended_at = now
                session.add(episode)
                await session.commit()
                logger.info("Successfully closed episode and committed episodic/graph memory.")

