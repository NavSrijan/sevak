"""
telegram_bot.py
---------------
An async Telegram bot built on python-telegram-bot v20+.
Designed to be imported and composed into a larger async system.

Install:
    pip install "python-telegram-bot>=20.0"

Quick usage:
    from telegram_bot import TelegramBot

    bot = TelegramBot(token="YOUR_TOKEN")

    @bot.on_message()
    async def greet(update, context):
        await update.message.reply_text(f"Hello, {update.effective_user.first_name}!")

    # In your main async entrypoint:
    await bot.run()

    # Or if you need the Application object directly (e.g. to share it):
    app = bot.build()
"""

import logging
from typing import Callable, Optional
import asyncio

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)


class TelegramBot:
    """
    A thin, composable wrapper around python-telegram-bot v20+.

    Supports:
    - Registering command handlers via @bot.on_command("start")
    - Registering text/message handlers via @bot.on_message()
    - Optional error handler
    - Both blocking (run) and non-blocking (start/stop) lifecycle modes
      for integration into a larger async system.
    """

    def __init__(self, token: str, *, concurrency: int = 4):
        """
        Args:
            token:       Your Telegram Bot API token.
            concurrency: Max concurrent handler invocations (default: 4).
        """
        self.token = token
        self.concurrency = concurrency
        self._app: Optional[Application] = None

    # ------------------------------------------------------------------
    # Decorator helpers
    # ------------------------------------------------------------------

    def on_command(self, command: str):
        """
        Decorator — registers an async function as a /<command> handler.

        Example:
            @bot.on_command("start")
            async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await update.message.reply_text("Welcome!")
        """
        def decorator(func: Callable):
            self.build().add_handler(CommandHandler(command, func))
            return func
        return decorator

    def on_message(self, filter=filters.TEXT & ~filters.COMMAND):
        """
        Decorator — registers an async function for incoming messages.

        Args:
            filter: A python-telegram-bot Filter (default: plain text, no commands).

        Example:
            @bot.on_message()
            async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await update.message.reply_text(update.message.text)
        """
        def decorator(func: Callable):
            self.build().add_handler(MessageHandler(filter, func))
            return func
        return decorator

    def on_error(self, func: Callable):
        """
        Decorator — registers an async error handler.

        Example:
            @bot.on_error
            async def handle_error(update, context):
                logger.error("Update %s caused error: %s", update, context.error)
        """
        self.build().add_error_handler(func)
        return func

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def build(self) -> Application:
        """
        Returns the underlying Application, building it once on first call.
        Use this to access the full python-telegram-bot API directly.
        """
        if self._app is None:
            self._app = (
                ApplicationBuilder()
                .token(self.token)
                .concurrent_updates(self.concurrency)
                .build()
            )
        return self._app

    def run(self):
        """
        Blocking entry point — starts polling and blocks until Ctrl+C.
        Suitable for scripts where the bot IS the main program.
        """
        logger.info("Starting bot (blocking)...")
        self.build().run_polling(allowed_updates=Update.ALL_TYPES)

    async def start(self):
        """
        Non-blocking start — initialises the bot and begins polling.
        Use this when embedding into an existing asyncio event loop
        (e.g. alongside a FastAPI/aiohttp server or other async services).

        Must be paired with a matching call to `await bot.stop()`.
        """
        app = self.build()
        await app.initialize()
        await app.start()
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("Bot polling started.")

    async def stop(self):
        """
        Graceful shutdown — mirrors `start()`.
        """
        app = self.build()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        logger.info("Bot stopped.")


async def run_bot(bot):
    app = bot.build()
    await app.initialize()
    await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    await app.start()

    stop_event = asyncio.Event()

    try:
        await stop_event.wait()  # Run until externally stopped
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        await app.stop()
        await app.shutdown()

# ------------------------------------------------------------------
# Example — run directly: python telegram_bot.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TOKEN_HERE")
    bot = TelegramBot(token=TOKEN)

    @bot.on_command("start")
    async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            f"Hello, {update.effective_user.first_name}! I'm alive."
        )

    @bot.on_command("help")
    async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Commands: /start, /help")

    @bot.on_message()
    async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(f"You said: {update.message.text}")

    @bot.on_error
    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
        logger.error("Unhandled error: %s", context.error, exc_info=context.error)

    bot.run()

