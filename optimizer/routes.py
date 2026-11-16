"""Local experiment API; approvals are intentionally absent from model tools."""
from pathlib import Path
import asyncio
import json
from time import monotonic
from urllib.parse import urlsplit
from fastapi import APIRouter, HTTPException, Request, Depends, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from starlette.concurrency import run_in_threadpool
from .store import Conflict, MAX_ROUNDS
from .isolation import confined_path


class NewTask(BaseModel):
    model_config=ConfigDict(extra='forbid')
    goal: str=Field(min_length=1,max_length=2000)
    mode: str='deerflow'
    iterative: bool=True
    overlap_experiment: bool=False
    rrf_experiment: bool=False


class Approval(BaseModel):
    plan_hash: str=Field(min_length=1,max_length=128)


class Retention(BaseModel):
    keep: bool


class MemoryNote(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=1, max_length=600)
    include_in_model: bool = False


def build_router(service):
    def local_request(request: Request):
        origin=request.headers.get('origin')
        if origin and urlsplit(origin).netloc!=request.url.netloc:
            raise HTTPException(403,'只接受当前本地页面发出的操作')

    router=APIRouter(prefix='/api/rag-optimizer',tags=['rag-optimizer'],dependencies=[Depends(local_request)])

    def perform(function,*args,**kwargs):
        try:
            return function(*args,**kwargs)
        except KeyError:
            raise HTTPException(404,'任务不存在') from None
        except Conflict as exc:
            raise HTTPException(409,str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from None

    @router.get('/ui',response_class=HTMLResponse)
    def ui():
        return (Path(__file__).parent/'ui.html').read_text(encoding='utf-8')

    @router.get('/status')
    def status():
        import importlib.util, os
        return {'upstream_pin':'1dd6d056278536af1cb4fc60347f4ca40fcb629b','deerflow_installed':importlib.util.find_spec('deerflow') is not None,
                'credential_configured':bool(os.environ.get('DEEPSEEK_API_KEY')) and os.environ.get('DEEPSEEK_API_KEY')!='not-configured',
                'budget':service.store.budget(),'root':str(service.root)}

    @router.get('/tasks')
    def tasks():
        detail_fields = {'baseline','candidate','diagnosis','round_summary','previous_round','attempted_configs'}
        return [{k:v for k,v in t.items() if k not in detail_fields} for t in service.store.list()]

    @router.get('/query-split')
    def query_split_report():
        report = perform(confined_path, service.root, 'runtime/query-split.json')
        if not report.is_file():
            raise HTTPException(404, '请先运行 scripts/compare_query_split.py 生成报告')
        return FileResponse(report, media_type='application/json')

    @router.get('/answer-evaluation')
    def answer_evaluation():
        for filename in ('answer-evaluation.json', 'answer-evaluation-offline.json'):
            report = perform(confined_path, service.root, 'runtime/' + filename)
            if report.is_file():
                return FileResponse(report, media_type='application/json')
        raise HTTPException(404, '请先运行 scripts/evaluate_answers.py 生成回答评测报告')

    @router.get('/onnx-benchmark')
    def onnx_benchmark():
        report = perform(confined_path, service.root, 'runtime/onnx-benchmark.json')
        if not report.is_file():
            raise HTTPException(404, '请先运行 scripts/compare_onnx.py 生成 ONNX 对照报告')
        return FileResponse(report, media_type='application/json')

    @router.post('/tasks',status_code=201)
    def create(body: NewTask):
        return perform(service.create,body.goal,mode=body.mode,iterative=body.iterative,
                       overlap_experiment=body.overlap_experiment, rrf_experiment=body.rrf_experiment)

    @router.get('/tasks/{task_id}')
    def detail(task_id: str):
        return perform(service.store.get,task_id) | {'max_rounds': MAX_ROUNDS}

    @router.get('/tasks/{task_id}/events')
    def events(task_id: str):
        return perform(service.store.events,task_id)

    @router.get('/tasks/{task_id}/memory')
    def memory(task_id: str):
        return perform(service.memory.view, task_id)

    @router.post('/tasks/{task_id}/memory', status_code=201)
    def memory_note(task_id: str, body: MemoryNote):
        return perform(service.add_memory_note, task_id, body.text, include_in_model=body.include_in_model)

    @router.get('/tasks/{task_id}/stream')
    def stream(task_id: str, request: Request, after: int = Query(0, ge=0, le=2**63-1), once: bool = False):
        perform(service.store.get, task_id)
        header = request.headers.get('last-event-id')
        if header is not None:
            if len(header) > 19 or not header.isascii() or not header.isdecimal() or int(header) > 2**63-1:
                raise HTTPException(422, '无效事件序号')
            after = int(header)

        async def updates():
            cursor = after
            last_write = monotonic()
            yield ': connected\n\n'
            while not await request.is_disconnected():
                events = await run_in_threadpool(service.store.events_after, task_id, cursor)
                for event in events:
                    cursor = event['seq']
                    yield f'id: {cursor}\nevent: update\ndata: {json.dumps(event)}\n\n'
                if once:
                    return
                if events:
                    last_write = monotonic()
                    continue
                if monotonic() - last_write >= 15:
                    yield ': heartbeat\n\n'
                    last_write = monotonic()
                await asyncio.sleep(0.5)

        return StreamingResponse(updates(), media_type='text/event-stream',
                                 headers={'Cache-Control':'no-cache', 'X-Accel-Buffering':'no'})

    @router.get('/tasks/{task_id}/replay')
    def replay(task_id: str):
        return perform(service.replay, task_id)

    @router.post('/tasks/{task_id}/resume')
    def resume(task_id: str, body: Approval):
        return perform(service.resume, task_id, body.plan_hash)

    @router.post('/tasks/{task_id}/replan', status_code=201)
    def replan(task_id: str, body: Approval):
        return perform(service.replan, task_id, body.plan_hash)

    @router.post('/tasks/{task_id}/finalize')
    def finalize(task_id: str, body: Approval):
        return perform(service.finalize, task_id, body.plan_hash)

    @router.post('/tasks/{task_id}/approve')
    def approve(task_id: str,body: Approval):
        return perform(service.approve,task_id,body.plan_hash)

    @router.post('/tasks/{task_id}/reject')
    def reject(task_id: str):
        return perform(service.reject,task_id)

    @router.post('/tasks/{task_id}/cancel')
    def cancel(task_id: str):
        return perform(service.store.cancel,task_id)

    @router.post('/tasks/{task_id}/retain')
    def retain(task_id: str,body: Retention):
        return perform(service.retain,task_id,body.keep)

    return router
