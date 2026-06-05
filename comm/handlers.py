import os
import logging
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler, MessageHandler, filters

from llm.connect_v2 import LLM

logger = logging.getLogger(__name__)

###
# THIS IS FOR TELEGRAM BOT
###

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for the /start command."""

    await update.message.reply_text("If you are seeing this message, you're in the wrong place.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for incoming messages."""

    user_input = update.message.text
    chat_id = update.effective_chat.id

    logger.info(f"Received message from {update.effective_user.first_name}: {user_input}")

    async def reply_func(response):
        await context.bot.send_message(chat_id=chat_id, text=response)

    task = {
            "source": "telegram",
            "text": user_input,
            "reply_func": reply_func,
            }

    await context.bot_data['input_queue'].put(task) 

def register_handlers(app):
    """Registers all handlers with the given Application."""

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

