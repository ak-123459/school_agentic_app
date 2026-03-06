import asyncio
from reasoning_engine.websocket_server import start_server

if __name__ == "__main__":
    asyncio.run(start_server())
