import os
import asyncio
from dotenv import load_dotenv

from llm.connect import LLM
from comm.telegram_bot import TelegramBot, run_bot

load_dotenv()  # Load environment variables from .env file


bot = TelegramBot(token=os.getenv("TELEGRAM_BOT_TOKEN"))

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


@bot.on_message()
async def greet(update, context):
    await update.message.reply_text(f"Hello, {update.effective_user.first_name}!")


if __name__ == "__main__":
    # main()
    asyncio.run(run_bot(bot))

