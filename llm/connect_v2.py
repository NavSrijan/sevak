import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

from prompts.system_prompt import SYSTEM_PROMPT
from utils.helpers import to_uuid
from llm.memory import MemoryManager

load_dotenv()

logger = logging.getLogger(__name__)

settings: ModelSettings = {
    "think": False,
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
}


def _make_agent(system_prompt: str) -> Agent[str]:
    """Create the base PydanticAI agent."""
    model = OllamaModel(
        "granite4.1:3b",
        provider=OllamaProvider(base_url="http://localhost:11434/v1"),
    )
    return Agent(
        model=model,
        system_prompt=system_prompt,
        capabilities=[ReinjectSystemPrompt()],
    )


@dataclass
class LLMResponse:
    content: str
    raw: object


class LLM:
    def __init__(self, name: str = "Diti", history: bool = True, system_prompt: str | None = None):
        self.name = name
        self.history = history
        self.system_prompt = system_prompt or SYSTEM_PROMPT.format(name=name)
        self.agent = _make_agent(self.system_prompt)
        self.memory: Optional[MemoryManager] = None

    @classmethod
    async def create(
        cls,
        name: str = "Diti",
        history: bool = True,
        system_prompt: str | None = None,
    ) -> "LLM":
        """Asynchronous factory method to create an instance of LLM."""
        instance = cls(name=name, history=history, system_prompt=system_prompt)
        await instance._async_init()
        return instance

    async def _async_init(self) -> None:
        """Initialize async resources."""
        if self.history:
            self.memory = MemoryManager(self.system_prompt)
            await self.memory.initialize()

    async def aclose(self) -> None:
        """Close the dedicated history connection."""
        if self.memory:
            await self.memory.aclose()

    def get_chain(self) -> Agent[str]:
        """Return the underlying agent, useful for testing."""
        return self.agent

    async def invoke(self, input: str, session_id: str, system_prompt: str | None = None) -> LLMResponse:
        message_history = []
        stored_system_prompt = None

        if self.memory:
            message_history, stored_system_prompt = await self.memory.load_history(session_id)

        # Resolve active system prompt: explicitly passed, stored in DB, or default
        active_system_prompt = system_prompt or stored_system_prompt or self.system_prompt

        run_agent = self.agent if active_system_prompt == self.system_prompt else _make_agent(active_system_prompt)
        result = await run_agent.run(
            input,
            message_history=message_history,
            conversation_id=to_uuid(session_id),
            model_settings=settings,
        )

        if self.memory:
            await self.memory.save_history(session_id, result.all_messages(), system_prompt=active_system_prompt)

        return LLMResponse(content=result.output, raw=result)


if __name__ == "__main__":
    pass
