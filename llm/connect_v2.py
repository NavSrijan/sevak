import logging
import httpx
import os
from datetime import datetime
from typing import Any
from pydantic import BaseModel, PrivateAttr
from pydantic_ai import Agent, UsageLimits, ModelMessagesTypeAdapter
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

from prompts.instruction import INSTRUCTION
from utils.helpers import to_uuid
import config

logger = logging.getLogger(__name__)

SETTINGS = ModelSettings(
    response_format={"type": "json_object"},
    **config.LLM_SETTINGS
)

# Share client
http_client = httpx.AsyncClient()

class LLMResponse(BaseModel):
    content: str
    
    _raw: Any = PrivateAttr(default=None)

    @property
    def raw(self) -> Any:
        return self._raw

    @raw.setter
    def raw(self, value: Any):
        self._raw = value

class LLM:
    def __init__(self, name: str = config.DEFAULT_BOT_NAME, history: bool = True, instruction: str | None = None):
        self.name = name
        self.history = history
        self.instruction = instruction or INSTRUCTION.format(name=name)
        
        self.model = OllamaModel(
            config.LLM_MODEL,
            provider=OllamaProvider(
                base_url=config.OLLAMA_BASE_URL,
                http_client=http_client
            ),
        )
        
        # Define a single, static agent using structured output and settings
        self.agent = Agent(
            model=self.model,
            output_type=LLMResponse,
            model_settings=SETTINGS,
            retries=3,
        )

    def get_agent(self) -> Agent[LLMResponse]:
        """Return the underlying agent, useful for testing."""
        return self.agent

    async def invoke(self, input_text: str, instruction: str | None = None) -> LLMResponse:
        """Invoke the agent without history."""
        resolved_instruction = instruction or self.instruction
        json_prompt = f"{resolved_instruction}\n\nYou MUST return a JSON object with a single string field 'content' containing your response text. Do not output anything other than raw valid JSON."
        result = await self.agent.run(
            input_text,
            instructions=json_prompt,
            usage_limits=UsageLimits(request_limit=10),
        )
        response = result.output
        response.raw = result
        return response

    async def invoke_with_memory(
        self,
        input_text: str,
        session_id: str,
        message_history: list,
        resolved_prompt: str
    ) -> LLMResponse:
        """Runs the agent with explicit pre-loaded history and a resolved prompt."""
        try:
            validated_history = ModelMessagesTypeAdapter.validate_python(message_history)
        except Exception as e:
            logger.warning("Failed to validate message history: %s. Starting with empty history.", e)
            validated_history = []
            
        json_prompt = f"{resolved_prompt}\n\nYou MUST return a JSON object with a single string field 'content' containing your response text. Do not output anything other than raw valid JSON."
        result = await self.agent.run(
            input_text,
            message_history=validated_history,
            instructions=json_prompt,
            conversation_id=to_uuid(session_id),
            usage_limits=UsageLimits(request_limit=10),
        )
        response = result.output
        response.raw = result
        return response
