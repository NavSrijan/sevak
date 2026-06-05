import asyncio
import logging
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple

from llm.commands import CommandHandler
from llm.connect_v2 import LLM, LLMResponse
from llm.memory import MemoryManager, memory_agent
import config

from pydantic import BaseModel
from pydantic_ai import Agent

logger = logging.getLogger(__name__)


####
# Pydantic Graph config
####

@dataclass
class PipelineState:
    """Shared state object passed through the graph execution."""
    session_id: str
    instruction: str
    message_history: List[Any]
    background_context: str | None = None

@dataclass
class PipelineDeps:
    """Dependencies injected into graph steps."""
    memory_agent: Agent
    diti: Agent
    memory: MemoryManager
    history_enabled: bool

####
# Nodes 
####

async def start_node(state, deps) -> 'step_2':
    """Initial node to decide if we need to fetch history based on user's prompt."""
    class Response(BaseModel):
        is_history_required: bool

    res = await deps.memory_agent.run(
        f"Given the user instruction: '{state.instruction}', determine if we need to fetch historical context. Or, if this instruction can we safely run without history.",
        output_type=Response
    )

    return res

async def memory_agent_node(state, deps):
    pass


class Pipeline:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.memory_agent = memory_agent
        self.memory = MemoryManager(llm.system_prompt)

    async def initialize(self):
        await self.memory.initialize()

    async def aclose(self):
        await self.memory.aclose()

    async def process_input(self, instruction: str, session_id: str, other_context: dict = None) -> LLMResponse:
        """
        Orchestrates pipeline execution: resolves prompt, pulls context, runs LLM, saves.
        """
        message_history = []
        stored_system_prompt = None

        # 1. Fetch historical data and ensure active episode if active
        if self.llm.history:
            await self.memory.ensure_active_episode(session_id)
            message_history, stored_system_prompt = await self.memory.load_history(session_id)

        # 2. Resolve systemic context hierarchy: runtime context -> database -> base default
        system_prompt = (other_context or {}).get("system_prompt") or stored_system_prompt or self.llm.system_prompt

        # 3. Retrieve episodic/graph memory context if active
        if self.llm.history:
            memory_context = await self.memory.retrieve_memory_context(session_id)
            if memory_context:
                system_prompt = f"{system_prompt}\n\nRetrieved Episodic/Graph Context:\n{memory_context}"

        # 4. Hand off clean data payloads to the optimized wrapper
        response = await self.llm.invoke_with_memory(
            instruction,
            session_id=session_id,
            message_history=message_history,
            resolved_prompt=system_prompt,
        )

        # 5. Extract total updated state and write behind asynchronously
        if self.llm.history:
            await self.memory.save_history(
                session_id, 
                response.raw.all_messages(), 
                system_prompt=system_prompt
            )
            
            # Extract token usage directly from the run result's usage property
            total_input = response.raw.usage().input_tokens
            total_output = response.raw.usage().output_tokens
            GOLD = "\033[38;5;220m"
            RESET = "\033[0m"
            logger.info(f"{GOLD}Total Tokens Used - Input: {total_input}, Output: {total_output}{RESET}")

            # Conditionally check if episode has ended, running extraction in background
            asyncio.create_task(
                self.memory.check_and_process_episode(
                    session_id,
                    memory_agent=self.memory_agent
                )
            )

        return response


async def llm_worker(input_queue: asyncio.Queue, pipeline: Pipeline):
    """
    The 'Brain' asynchronous background consumer loop.
    """
    logger.info("LLM Worker started.")

    while True:
        task = await input_queue.get()

        # Route system-level commands instantly out of band
        if task['text'].startswith("/"):
            try:
                logger.info(f"Command received: {task['text']}")
                res = await CommandHandler.handle_command(task['text'])
                await task['reply_func'](res)
            except Exception as e:
                logger.error(f"Command handling error: {e}")
                await task['reply_func']("Command execution failed.")
            finally:
                input_queue.task_done()
                continue

        try:
            logger.info(f"Processing task from source: {task.get('source', 'unknown')}")

            # Dynamically resolve session_id after picking up from queue
            current_session = config.get_current_session_id()

            response = await pipeline.process_input(
                task['text'],
                session_id=current_session,
                other_context=task.get('context') # Support pipeline prompt-injections
            )

            await task['reply_func'](response.content)

        except Exception as e:
            logger.error(f"Error processing task within worker loop: {e}", exc_info=True)
            await task['reply_func']("An internal processing error occurred.")
        finally:
            input_queue.task_done()
