"""Explicit source-span chunks and bounded retrieval configurations."""
import json, os, subprocess, sys, time
from pathlib import Path
from .evaluation import score_case, aggregate

DEFAULT_CONFIG={'strategy':'vector','top_k':2,'chunking':'paragraph','chunk_size':300,'context_chars':1800,'rerank':False,'candidate_k':8,'query_split':False,'chunk_overlap':0,'rrf_vector_pct':50}


def validate_config(config: dict) -> dict:
    if not isinstance(config,dict) or set(config)-set(DEFAULT_CONFIG):
        raise ValueError('包含不支持的检索配置')
    value=DEFAULT_CONFIG|config
    if value['strategy'] not in {'vector','bm25','hybrid'} or value['chunking'] not in {'paragraph','section','fixed'}:
        raise ValueError('未知检索或切分方式')
    for key, low, high in [('top_k',1,10),('chunk_size',16,1000),('context_chars',100,6000),('candidate_k',2,20)]:
        if type(value[key]) is not int or not low<=value[key]<=high:
            raise ValueError(f'{key} 超出允许范围')
    if type(value['rerank']) is not bool:
        raise ValueError('rerank 必须为布尔值')
    if type(value['query_split']) is not bool:
        raise ValueError('query_split 必须为布尔值')
    if value['query_split'] and (value['strategy'] != 'vector' or value['rerank']):
        raise ValueError('本版 Query 拆分仅支持纯向量检索')
    overlap = value['chunk_overlap']
    if type(overlap) is not int or not 0 <= overlap < value['chunk_size']:
        raise ValueError('切分重叠必须为非负整数且小于窗口长度')
    if overlap and value['chunking'] != 'fixed':
        raise ValueError('切分重叠只用于定长窗口')
    weight = value['rrf_vector_pct']
    if type(weight) is not int or not 10 <= weight <= 90:
        raise ValueError('RRF 向量权重须为 10–90 的整数百分比')
    if weight != 50 and value['strategy'] != 'hybrid':
        raise ValueError('自定义 RRF 权重只用于混合检索')
    return value


def build_chunks(documents: list[dict], config: dict) -> list[dict]:
    config=validate_config(config)
    chunks=[]
    for doc in documents:
        text=doc['title']+'\n\n'
        spans=[]
        for fact in doc['facts']:
            start=len(text)
            text+=fact['text']+'\n\n'
            spans.append((start,start+len(fact['text']),fact['id']))
        if config['chunking']=='section':
            ranges=[(0,len(text))]
        elif config['chunking']=='paragraph':
            ranges=[(start,end) for start,end,_ in spans]
        else:
            size = config['chunk_size']
            stride = size - config['chunk_overlap']
            ranges = []
            for start in range(0, len(text), stride):
                end = min(start + size, len(text))
                ranges.append((start, end))
                if end == len(text):
                    break
        for number,(start,end) in enumerate(ranges):
            body=text[start:end].strip()
            if not body:
                continue
            chunks.append({'id':f"{doc['id']}-{number:03d}",'document_id':doc['id'],'text':body,
                           'title':doc['title'],'start':start,'end':end,
                           'evidence_ids':[eid for a,b,eid in spans if start<=a and b<=end],
                           'evidence_spans':[{'id':eid,'start':a,'end':b,'covered_start':max(start,a),'covered_end':min(end,b)}
                                             for a,b,eid in spans if max(start,a)<min(end,b)]})
    return chunks


def worker_python(root: Path) -> Path:
    relative = 'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
    local=root/'.venv-retrieval'/relative
    if local.exists() and (root/'.venv-retrieval/READY').exists():
        return local
    raise RuntimeError('检索环境缺失，请运行 scripts/setup-retrieval.ps1')


def run_retrieval(root: Path, documents: list[dict], questions: list[str], config: dict, work: Path, candidate_path=None, *, compare_query_split=False):
    from .isolation import confined_path
    config=validate_config(config)
    work=confined_path(root,work.relative_to(root).as_posix())
    # Fresh per-run directories prevent following pre-existing report leaf links.
    work.mkdir(parents=True,exist_ok=False)
    request={'chunks':build_chunks(documents,config),'queries':questions,'config':config,'cache':str(root/'.cache/models'),'index_dir':str(work/'index')}
    if compare_query_split:
        if candidate_path is not None or config['query_split'] or config['strategy'] != 'vector' or config['rerank'] or not questions:
            raise ValueError('Query 拆分对照需要纯向量基线与非空问题集')
        request['compare_query_split'] = True
    if config['query_split'] and candidate_path is None:
        raise ValueError('Query 拆分评测需要获批后生成的候选目录')
    if candidate_path is not None:
        request['candidate_path']=str(confined_path(root,candidate_path.relative_to(root).as_posix()))
    request_path=work/'retrieval-request.json'
    response_path=work/'retrieval-response.json'
    request_path.write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8')
    env=os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8',PYTHONPATH=str(root),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    result=subprocess.run([str(worker_python(root)),str(root/'optimizer/retrieval_worker.py'),str(request_path),str(response_path)],
                          cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
    if result.returncode:
        # Preserve a bounded dependency error, not environment variables.
        (work/'retrieval-error.log').write_text(result.stderr[-16000:],encoding='utf-8')
        raise RuntimeError('本地检索失败，详见实验目录 retrieval-error.log')
    return json.loads(response_path.read_text(encoding='utf-8'))


def evaluate(root: Path, documents: list[dict], cases: list[dict], config: dict, work: Path, split: str, candidate_path=None):
    started=time.perf_counter()
    output=run_retrieval(root,documents,[case['question'] for case in cases],config,work,candidate_path=candidate_path)
    rows=[]
    for case, hits in zip(cases,output['results'],strict=True):
        rows.append({'id':case['id'],'question':case['question'],'required':case['required'],
                     'scores':score_case(case,hits,validate_config(config)['top_k']),'results':hits})
    report={'split':split,'config':validate_config(config),'metrics':aggregate([row['scores'] for row in rows]),
            'cases':rows,'elapsed_seconds':time.perf_counter()-started,'worker':output['metadata'],
            'generation_quality':'未执行；此报告仅验证检索','evaluation':'真实本地检索；模拟政策数据；无生成 API 费用'}
    report['stage_metrics']={}
    for stage in ['initial_candidates','post_ranking','budgeted_context','potential_expanded_context']:
        stage_k=min(validate_config(config)['top_k']+2,10) if stage=='potential_expanded_context' else validate_config(config)['top_k']
        report['stage_metrics'][stage]=aggregate([score_case(case,result[stage],stage_k) for case,result in zip(cases,output.get('stages',[]),strict=True)]) if output.get('stages') else None
    report['expanded_context_note']='潜在补检窗口，不是全部问题实际触发补检的证明；真实分支见问答轨迹。'
    if 'query_split_traces' in output:
        for row, trace in zip(rows, output['query_split_traces'], strict=True):
            row['query_split_trace'] = trace
    for row, stage in zip(rows, output.get('stages', [])):
        if 'rrf_trace' in stage:
            row['rrf_trace'] = stage['rrf_trace']
    if report['config']['chunking'] == 'fixed':
        chunks = build_chunks(documents, config)
        ends = {}
        overlap_chars = 0
        for chunk in chunks:
            doc = chunk['document_id']
            overlap_chars += max(0, min(ends.get(doc, 0), chunk['end']) - chunk['start'])
            ends[doc] = chunk['end']
        repeated_hits = 0
        for row in rows:
            seen = set()
            for hit in row['results']:
                facts = set(hit['evidence_ids'])
                repeated_hits += bool(seen & facts)
                seen.update(facts)
        index = candidate_path/'vector_index.pt' if candidate_path else work/'index/vectors.pt'
        report['chunking_stats'] = {'chunks':len(chunks), 'overlap_source_chars':overlap_chars,
                                    'retrieved_chunks_repeating_complete_evidence':repeated_hits,
                                    'index_bytes':index.stat().st_size}
    (work/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
