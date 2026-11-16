"""Inject a server-issued capability into the native DeerFlow MCP request."""
import json
from urllib.parse import urlsplit

from anyio import fail_after
from langchain_core.tools import ToolException

from .permissions import CAPABILITY_HEADER, MCP_SERVER, MCP_TOOLS, PermissionDenied, authorize, runtime_identity


def check_local_configuration(root):
    path = root/'extensions_config.json'
    if not path.exists():
        return
    config = json.loads(path.read_text(encoding='utf-8'))
    for name, server in config.get('mcpServers', {}).items():
        if not server.get('enabled', True):
            continue
        url = urlsplit(server.get('url', ''))
        if (name != MCP_SERVER or server.get('type') != 'http' or url.scheme != 'http'
                or url.hostname != '127.0.0.1' or url.port is None or url.username is not None
                or url.path != '/api/rag-optimizer/mcp' or url.query or url.fragment):
            raise PermissionDenied('RAG 优化任务仅允许配置指定的本机只读 MCP 地址')


def build():
    async def authorize_request(request, handler):
        if request.server_name != MCP_SERVER:
            return await handler(request)
        try:
            binding, role = runtime_identity(request.runtime)
            if request.name not in MCP_TOOLS:
                raise PermissionDenied('未知的 RAG MCP 工具')
            authorize(binding, role, request.name, request.args.get('task_id'), source='mcp-interceptor')
        except PermissionDenied as exc:
            raise ToolException(str(exc)) from None
        headers = {key:value for key,value in (request.headers or {}).items()
                   if key.lower() != CAPABILITY_HEADER.lower()}
        headers[CAPABILITY_HEADER] = binding.tokens[role]
        with fail_after(15):
            return await handler(request.override(headers=headers))
    return authorize_request
