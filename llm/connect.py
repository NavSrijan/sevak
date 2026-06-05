import os
import psycopg
from utils.helpers import to_uuid
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_postgres import PostgresChatMessageHistory, PGEngine
from langchain_core.globals import set_debug

from prompts.system_prompt import SYSTEM_PROMPT
import config

# set_debug(True)

class LLM():
    def __init__(self, name=config.DEFAULT_BOT_NAME, history=True, system_prompt=None):
        self.name = name
        self.history_messages_key = "history"
        self.llm = ChatOllama(
                model=config.LLM_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=config.LLM_SETTINGS.get("temperature", 0.7),
                top_p=config.LLM_SETTINGS.get("top_p", 0.9),
                top_k=64,
                reasoning=False,
                # seed=1
                )

        self.history = history

        if system_prompt is None:
            self.system_prompt = SYSTEM_PROMPT.format(name=self.name)
        else:
            self.system_prompt = system_prompt

    @classmethod
    async def create(cls, name="Diti", history=True, system_prompt=None):
        """Asynchronous factory method to create an instance of LLM."""
        instance = cls(name, history, system_prompt)
        await instance._async_init()
        return instance

    async def _async_init(self):
        """Async init, self-explanatory."""
        if self.history: 
            self.pg_conn = await psycopg.AsyncConnection.connect(os.getenv("PG_DIRECT_URL"))
            await PostgresChatMessageHistory.acreate_tables(self.pg_conn, "chat_history")

        self._setup_chain()


    def _setup_chain(self):
        if self.history:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                MessagesPlaceholder(variable_name=self.history_messages_key),
                ("human", "{input}"),
            ])

            self.chain = prompt | self.llm

            def return_history(session_id: str):
                return PostgresChatMessageHistory(
                        "chat_history",
                        to_uuid(session_id),
                        async_connection=self.pg_conn,
                    )

            self.wrapped_chain = RunnableWithMessageHistory(
                    self.chain,
                    return_history,
                    history_messages_key=self.history_messages_key,
                    )
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}"),
                ("human", "{input}"),
            ])

            self.chain = prompt | self.llm
            self.wrapped_chain = self.chain

    def get_chain(self):
        """Returns the chain, useful for testing."""
        return self.wrapped_chain

    async def invoke_with_history(self, input: str, session_id: str, system_prompt: str = None):
        config = {"configurable": {"session_id": to_uuid(session_id)}}

        return await self.wrapped_chain.ainvoke(
                {
                    "input": input,
                    "system_prompt": system_prompt or self.system_prompt,
                },
                config=config
            )



if __name__ == "__main__":
    pass

