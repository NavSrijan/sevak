from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

class LLM():
    def __init__(self, name="Ram Prakash", history=True, system_prompt=None):
        self.name = name
        self.history_messages_key = "history"
        self.store = {}
        self.llm = ChatOllama(
                model="gemma4:e2b",
                temperature=1,
                top_p=0.95,
                top_k=64,
                reasoning=False,
                seed=1
                )

        self.history = history
        self.system_prompt = system_prompt

        self.setup_chain()

    def setup_chain(self):
        system_prompt = self.system_prompt
        history = self.history

        if system_prompt is None:
            system_prompt = f"You are my assistant named '{self.name}'."

        if history:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name=self.history_messages_key),
                ("human", "{input}"),
            ])

            self.chain = prompt | self.llm

            def return_history(session_id: str):
                if session_id not in self.store:
                    self.store[session_id] = ChatMessageHistory()
                return self.store[session_id]

            self.wrapped_chain = RunnableWithMessageHistory(
                    self.chain,
                    return_history,
                    history_messages_key=self.history_messages_key,
                    )
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            self.chain = prompt | self.llm
            self.wrapped_chain = self.chain

    def get_chain(self):
        return self.wrapped_chain



if __name__ == "__main__":
    config = {"configurable": {"session_id": "nav"}}

    x = ""
    while x != "exit":
        x = input("You: ")
        if x != "exit":
            response = wrapped_chain.invoke({"input": x}, config=config)
            print(f"Ram Prakash: {response.content}")




