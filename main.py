from llm.connect import LLM

def main():
    print("Hello from sevak!")
    config = {"configurable": {"session_id": "nav"}}
    llm = LLM()
    chain = llm.get_chain()

    x = ""
    while x != "exit":
        x = input("You: ")
        if x != "exit":
            response = chain.invoke({"input": x}, config=config)
            print(f"Ram Prakash: {response.content}")


if __name__ == "__main__":
    main()
