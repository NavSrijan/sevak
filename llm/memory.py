import uuid
import logging
import httpx
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError, BaseModel
from pydantic_ai import Agent, ModelMessagesTypeAdapter
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from sqlmodel import select
from sqlalchemy import func, text

from utils.helpers import to_uuid
from db.connection import db
from db.models import ChatHistory, EntityTaxonomy, Episode, MemoryEntity, Predicates, EntityFacts, EpisodeStatus
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
    instructions="You are a core background memory helper agent.",
)


class MemoryManager:
    def __init__(self, instruction: str):
        self.instruction = instruction

    async def get_total_tokens(self, session_id: str) -> tuple[int, int]:
        """Query and return (total_input_tokens, total_output_tokens) from the database."""
        session_uuid = uuid.UUID(to_uuid(session_id))
        total_input = 0
        total_output = 0
        async with db.get_session() as session:
            result = await session.execute(
                select(ChatHistory.payload)
                .where(ChatHistory.session_id == session_uuid)
            )
            rows = result.scalars().all()
            for payload in rows:
                if isinstance(payload, list):
                    for msg in payload:
                        if isinstance(msg, dict) and "usage" in msg and msg["usage"]:
                            usage = msg["usage"]
                            total_input += usage.get("input_tokens", 0) or 0
                            total_output += usage.get("output_tokens", 0) or 0
            return total_input, total_output

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
                select(ChatHistory.payload)
                .where(ChatHistory.session_id == session_uuid)
                .order_by(ChatHistory.created_at.asc())
            )
            rows = result.scalars().all()

        if not rows:
            return [], None

        flat_messages = []
        for payload in rows:
            if isinstance(payload, list):
                flat_messages.extend(payload)
            else:
                flat_messages.append(payload)

        return flat_messages, None

    async def save_history(
        self,
        session_id: str,
        messages: list,
        prev_history_len: int = 0,
        instruction: str | None = None
    ) -> None:
        """Persist the latest conversation turn (all new messages) as a new row."""
        new_messages = messages[prev_history_len:]
        if not new_messages:
            logger.warning("No new messages to save in history.")
            return

        # Dump using TypeAdapter to format correctly as JSON-serializable types
        payload = ModelMessagesTypeAdapter.dump_python(new_messages, mode='json')
                
        session_uuid = uuid.UUID(to_uuid(session_id))
        now = datetime.now(timezone.utc)

        # Store it in a file in logs exactly as Pydantic does without modification
        import json
        import os
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/llm_payloads.log", "a") as f:
                log_entry = {
                    "timestamp": now.isoformat(),
                    "session_id": str(session_id),
                    "payload": payload
                }
                f.write(json.dumps(log_entry, indent=2) + "\n" + "="*80 + "\n")
        except Exception as e:
            logger.warning("Failed to write to logs/llm_payloads.log: %s", e)

        async with db.get_session() as session:
            history_record = ChatHistory(
                session_id=session_uuid,
                payload=payload,
                created_at=now
            )
            session.add(history_record)
            await session.commit()

    async def retrieve_memory_context(self, session_id: str) -> str:
        """Fetch the active episode summary, entities, and facts to construct a prompt context."""
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

                # 2. Fetch facts for this episode
                from sqlalchemy.orm import aliased
                TargetEntity = aliased(MemoryEntity)
                
                stmt = (
                    select(
                        EntityFacts,
                        MemoryEntity.name.label("subject_name"),
                        Predicates.predicate.label("predicate_name"),
                        TargetEntity.name.label("target_name")
                    )
                    .join(MemoryEntity, EntityFacts.entity_id == MemoryEntity.id)
                    .join(Predicates, EntityFacts.predicate_id == Predicates.id)
                    .outerjoin(TargetEntity, EntityFacts.target_entity_id == TargetEntity.id)
                    .where(EntityFacts.source_episode_id == episode.id)
                )
                
                fact_res = await session.execute(stmt)
                facts_data = fact_res.all()

                # 3. Format as context
                context_parts = []
                if episode.summary:
                    context_parts.append(f"Episode Summary: {episode.summary}")
                
                if facts_data:
                    context_parts.append("Known Facts:")
                    for row in facts_data:
                        fact = row.EntityFacts
                        subject = row.subject_name
                        predicate = row.predicate_name
                        if row.target_name:
                            object_val = row.target_name
                        else:
                            import json
                            val = fact.value_json
                            if isinstance(val, dict) and len(val) == 1 and "value" in val:
                                object_val = str(val["value"])
                            elif isinstance(val, dict) and len(val) == 0:
                                object_val = ""
                            else:
                                object_val = json.dumps(val)
                        
                        context_parts.append(f"- {subject} {predicate} {object_val}".strip())

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
        """Check if active episode has ended due to 2 min time gap or semantic drift, and close it."""
        session_uuid = uuid.UUID(to_uuid(session_id))
        now = current_time or datetime.now(timezone.utc)
        
        async with db.get_session() as session:
            # 1. Fetch active episode
            import ipdb; ipdb.set_trace()
            ep_res = await session.execute(
                select(Episode).where(Episode.session_id == session_uuid, Episode.status == EpisodeStatus.active)
            )
            episode = ep_res.scalar_one_or_none()
            if not episode:
                return

            # 2. Get most recent ChatHistory record to check time gap
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
                logger.info("Episode ended (gap: %ss, drift: %s, tokens: %s). Closing episode.", gap, drift, total_tokens)
                episode.status = EpisodeStatus.closed
                episode.ended_at = now
                session.add(episode)
                await session.commit()
                logger.info("Successfully closed episode.")
