
class CommandHandler:

    @classmethod
    async def handle_command(cls, command: str) -> str:
        if command == "/help":
            return "This is help"


if __name__ == "__main__":
    import asyncio

    async def test_command():
        res = await CommandHandler.handle_command("/help")
        print(res)

    asyncio.run(test_command())

