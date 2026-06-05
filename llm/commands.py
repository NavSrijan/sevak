
import config

class CommandHandler:

    @classmethod
    async def handle_command(cls, command: str) -> str:
        if command == "/help":
            return "This is help"
        elif command == "/new":
            new_session = config.increment_session_id()
            return f"Session incremented. Current active session is now: {new_session}"


if __name__ == "__main__":
    import asyncio

    async def test_command():
        res = await CommandHandler.handle_command("/help")
        print(res)

    asyncio.run(test_command())

