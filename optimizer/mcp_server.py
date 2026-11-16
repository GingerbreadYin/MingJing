"""Local MCP Streamable HTTP facade over the bounded RAG read tools."""
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from .permissions import CAPABILITY_HEADER, authorize, capability_identity


def compact_development(view):
    fields = ('id','status','metrics','config','attempted_configs','allowed_configs','development_case_count')
    result = {key:view[key] for key in fields if key in view}
    if view.get('previous_round'):
        previous = view['previous_round']
        result['previous_round'] = {key:previous[key] for key in ('round','config','metrics','checks')}
        result['previous_round']['case_deltas'] = previous['case_deltas'][:4]
    rows = view.get('development_cases', [])
    result['development_cases'] = [
        {**{key:row[key] for key in ('id','question','required','scores')},
         'results':[{**{key:hit[key] for key in ('id','document_id','evidence_ids','vector_score','bm25_score')},
                     'text_preview':hit['text'][:160]} for hit in row['results'][:2]]}
        for row in rows[:4]
    ]
    for compact, original in zip(result['development_cases'], rows[:4], strict=True):
        if 'rrf_trace' in original:
            compact['rrf_trace'] = original['rrf_trace']
    result['note'] = '优先返回开发集漏召回题与短证据片段，可按 document_id 调用原文工具。'
    return result


def create_mcp():
    server = FastMCP('RAG development evidence', stateless_http=True, json_response=True)
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    def guarded(ctx, name, task_id):
        request = ctx.request_context.request
        token = request.headers.get(CAPABILITY_HEADER) if request is not None else None
        binding, role = capability_identity(token)
        authorize(binding, role, name, task_id, source='mcp')
        binding.service.store.event(task_id, 'mcp_tool_called', {'tool':name,'role':role})
        return binding.service

    @server.tool(annotations=read_only)
    def mcp_inspect_development(task_id: str, ctx: Context) -> dict[str, Any]:
        """Read bounded development evidence for the current experiment; no held-out data."""
        return compact_development(guarded(ctx, 'mcp_inspect_development', task_id).agent_view(task_id))

    @server.tool(annotations=read_only)
    def mcp_read_policies(task_id: str, document_ids: list[str], ctx: Context) -> list[dict]:
        """Read up to eight policy documents by ID for the current experiment."""
        return guarded(ctx, 'mcp_read_policies', task_id).sources(task_id, document_ids)

    server.streamable_http_app()  # Initializes the SDK session manager.
    return server


def mcp_router(server):
    router = APIRouter()

    class MCPResponse(Response):
        async def __call__(self, scope, receive, send):
            await server.session_manager.handle_request(scope, receive, send)

    @router.api_route('/api/rag-optimizer/mcp', methods=['POST', 'GET', 'DELETE'], include_in_schema=False)
    async def endpoint(request: Request):
        if request.url.hostname not in {'127.0.0.1','localhost','::1'}:
            return Response(status_code=403)
        origin = request.headers.get('origin')
        if origin and origin != f'{request.url.scheme}://{request.url.netloc}':
            return Response(status_code=403)
        return MCPResponse()

    return router
