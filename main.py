import os
import asyncio
import uvicorn
import logging
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from llm.connect_v2 import LLM
from llm.worker import llm_worker

from comm.telegram_bot import TelegramBot, run_bot
from comm.server import app

from db.connection import db

from utils.setup_logging import setup_logging


logger = logging.getLogger(__name__)

async def main():

    load_dotenv()  # Load environment variables from .env file
    setup_logging()  # Set up logging configuration
    input_queue = asyncio.Queue()

    await db.connect() # Database
    
    engine = create_async_engine(os.getenv("PG_DATABASE_URL"))
    AsyncSessionLocal = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

    llm = await LLM.create()

    bot = TelegramBot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
    bot_data = {}
    bot_data['input_queue'] = input_queue

    app.state.input_queue = input_queue
    config = uvicorn.Config(app, host="localhost", port=6969, log_level="warning")
    server = uvicorn.Server(config)


    try:
        await asyncio.gather(
                server.serve(),
                run_bot(bot, bot_data),
                llm_worker(input_queue, llm)
            )
    finally:
        await llm.aclose()
        await db.disconnect()



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting gracefully...")
