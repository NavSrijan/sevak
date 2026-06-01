import os
import asyncio
import uvicorn
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI server started.")

@app.post("/message")
async def message(req: Request):
    future: asyncio.Future = asyncio.get_event_loop().create_future()

    async def reply_func(response: str):
        future.set_result(response)

    user_input = (await req.json())["text"]
    logger.info(f"Terminal: {user_input}")
    task = {
            "source": "terminal",
            "session_id": os.getenv("SESSION_ID", "nav"),
            "text": user_input,
            "reply_func": reply_func,
            }

    await req.app.state.input_queue.put(task)
    reply = await future
    return {"response": reply}

