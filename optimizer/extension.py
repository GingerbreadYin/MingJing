"""Official extension entrypoint; no upstream core patches required."""
import asyncio
from .service import get_service


class Lifecycle:
    def __init__(self, server):
        self.server = server

    async def _mcp_lifespan(self):
        async with self.server.session_manager.run():
            self.ready.set()
            await self.shutdown.wait()

    async def start(self,deps):
        self.ready = asyncio.Event()
        self.shutdown = asyncio.Event()
        self.runner = asyncio.create_task(self._mcp_lifespan())
        ready = asyncio.create_task(self.ready.wait())
        try:
            done, _ = await asyncio.wait({self.runner, ready}, return_when=asyncio.FIRST_COMPLETED)
            if self.runner in done:
                await self.runner
                raise RuntimeError('MCP session manager exited during startup')
        finally:
            ready.cancel()

    async def stop(self):
        self.shutdown.set()
        await self.runner
        await asyncio.to_thread(get_service().close)


def install(registry,config):
    from .routes import build_router
    from .mcp_server import create_mcp, mcp_router
    server = create_mcp()
    registry.service(Lifecycle(server))
    registry.routers([build_router(get_service()), mcp_router(server)])
