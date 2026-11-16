"""Independent MCP SDK client for a local RAG evidence server."""
import argparse
import asyncio
import json
import os
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def probe(url, task_id=None, expect_denied=False):
    target = urlsplit(url)
    if target.scheme != 'http' or target.hostname not in {'127.0.0.1','localhost','::1'} or target.username:
        raise ValueError('MCP 验收客户端仅连接本机 HTTP')
    token = os.environ.get('RAGOPT_MCP_CAPABILITY')
    headers = {'X-Ragopt-Capability':token} if token else {}
    async with httpx.AsyncClient(headers=headers, trust_env=False, timeout=15) as client, \
               streamable_http_client(url, http_client=client) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            result = {'tools':sorted(tool.name for tool in listing.tools)}
            if task_id is not None:
                response = await session.call_tool('mcp_inspect_development', {'task_id':task_id})
                result['call_denied'] = bool(response.isError)
                if response.isError != expect_denied:
                    raise RuntimeError('MCP 调用的权限判定与预期不符')
                if not response.isError:
                    content = response.structuredContent or {}
                    if content.get('id') != task_id:
                        raise RuntimeError('MCP 返回了非当前任务数据')
                    result['task_id'] = content['id']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('--task-id')
    parser.add_argument('--expect-denied', action='store_true')
    args = parser.parse_args()
    print(json.dumps(asyncio.run(probe(args.url, args.task_id, args.expect_denied)), ensure_ascii=False))
