import asyncio
import logging
from llm.commands import CommandHandler

logger = logging.getLogger(__name__)

async def llm_worker(input_queue: asyncio.Queue, llm):
    """
    The 'Brain'
    """
    logger.info("LLM Worker started.")

    while True:
        task = await input_queue.get()

        if task['text'][0] == "/":
            logger.info(f"Command received: {task['text']}")
            res = await CommandHandler.handle_command(task['text'])
            await task['reply_func'](res)
            input_queue.task_done()
            continue

        try:
            logger.info(f"Input from : {task['source']}")

            response = await llm.invoke(
                    task['text'],
                    session_id=task['session_id'],
                    )

            await task['reply_func'](response.content)

        except Exception as e:
            logger.error(f"Error processing task: {e}")
            await task['reply_func']("Check logs.")
        finally:
            input_queue.task_done()


