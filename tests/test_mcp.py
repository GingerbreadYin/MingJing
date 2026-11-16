import unittest
from datetime import timedelta
from unittest.mock import Mock

import httpx
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from optimizer.mcp_server import create_mcp, mcp_router
from optimizer.permissions import CAPABILITY_HEADER, run_scope


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_and_reads_require_task_scoped_capability(self):
        server = create_mcp()
        app = FastAPI()
        app.include_router(mcp_router(server))
        service = Mock()
        service.agent_view.side_effect = lambda task_id: {'id':task_id, 'development_cases':[]}
        url = 'http://127.0.0.1:8030/api/rag-optimizer/mcp'

        async def call(token):
            headers = {CAPABILITY_HEADER:token} if token else {}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), headers=headers) as client, \
                       streamable_http_client(url, http_client=client) as (read, write, _), \
                       ClientSession(read, write, read_timeout_seconds=timedelta(seconds=8)) as session:
                await session.initialize()
                tools = await session.list_tools()
                self.assertEqual({tool.name for tool in tools.tools}, {'mcp_inspect_development','mcp_read_policies'})
                self.assertTrue(all(tool.annotations.readOnlyHint for tool in tools.tools))
                self.assertFalse(any('ctx' in tool.inputSchema.get('properties', {}) for tool in tools.tools))
                valid = await session.call_tool('mcp_inspect_development', {'task_id':'A'})
                wrong = await session.call_tool('mcp_inspect_development', {'task_id':'B'})
                self.assertTrue(wrong.isError)
                return valid

        async with server.session_manager.run():
            self.assertTrue((await call(None)).isError)
            with run_scope(service, 'A') as binding:
                token = binding.tokens['retrieval-diagnostician']
                result = await call(token)
                self.assertFalse(result.isError)
                self.assertEqual(result.structuredContent['id'], 'A')
            self.assertTrue((await call(token)).isError)
        service.agent_view.assert_called_once_with('A')
        self.assertTrue(any(call.args[1] == 'tool_denied' for call in service.store.event.call_args_list))
