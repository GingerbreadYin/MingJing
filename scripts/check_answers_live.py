"""Authorized small live RAG answer check on a baseline-equivalent fixture.

Does not approve or execute the real Agent's pending proposal. Credentials are
consumed by the application; only simulated policy evidence is transmitted.
"""
from pathlib import Path
import hashlib, json, os, sys, uuid

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from optimizer.launch import prepare_environment
prepare_environment()
from optimizer.store import Store
from optimizer.artifacts import candidate_hash

fixture=json.loads((root/'reports/offline-acceptance.json').read_text(encoding='utf-8'))
candidate=Path(fixture['retained_path'])
before=candidate_hash(candidate)
sys.path.insert(0,str(candidate))
import model_client

store=Store(root/'runtime/optimizer.db')
budget_task='answer-smoke-'+uuid.uuid4().hex
records=[]
original_call=model_client.call_deepseek
original_open=model_client.urlopen
attempts=[0]


def counted_open(*args,**kwargs):
    attempts[0]+=1
    return original_open(*args,**kwargs)


def accounted_call(payload):
    call_id=store.reserve(budget_task,.3,cap=3.,task_cap=2.)
    attempts[0]=0
    try:
        result=original_call(payload)
        usage=result.get('usage') or {}
        input_tokens=usage.get('prompt_tokens',usage.get('input_tokens',0))
        output_tokens=usage.get('completion_tokens',usage.get('output_tokens',0))
        estimate=(input_tokens*.3+output_tokens*1.2)/1_000_000*8 if usage and attempts[0]==1 else None
        store.settle(call_id,estimate)
        records.append({'call_id':call_id,'stage':'evidence' if payload.get('response_format') else 'generation',
                        'usage':usage,'physical_attempts':attempts[0],'estimated_cny':estimate,
                        'coverage':'successful single attempt' if estimate is not None else 'uncertain; reservation retained'})
        return result
    except BaseException:
        store.settle(call_id,None)
        raise


model_client.urlopen=counted_open
model_client.call_deepseek=accounted_call
import retrieval, rag_graph
resources=retrieval.load_retriever()
graph=rag_graph.build_rag_graph(resources)
cases=[
    ('return','普通商品签收第五天，未使用且包装完整，可以申请无理由退货吗？','ANSWERED'),
    ('invoice','发票已经开具，能立即修改抬头吗？','ANSWERED'),
    ('unknown','店铺是否支持比特币付款？','INSUFFICIENT_EVIDENCE'),
    ('clarify','商品损坏需要退货，运费由谁承担？','NEED_CLARIFICATION'),
]
rows=[]
for case_id,question,expected in cases:
    state=graph.invoke({'question':question})
    row={'id':case_id,'question':question,'expected_status':expected,'status':state['status'],
         'answer':state['answer'],'rounds':state['rounds'],'sources':[c['id'] for c in state['selected_chunks']],
         'status_pass':state['status']==expected}
    rows.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
report={'fixture':str(candidate),'fixture_source':'baseline-equivalent acceptance fixture, not the pending hybrid proposal',
        'candidate_hash_before':before,'candidate_hash_after':candidate_hash(candidate),'cases':rows,'calls':records,
        'status_checks_pass':all(r['status_pass'] for r in rows),'content_review':'pending explicit inspection of the actual answers',
        'budget_task':budget_task,'data':'simulated policies/questions/evidence only','modified_original_rag':False}
(root/'reports/live-rag-answer-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
if not report['status_checks_pass'] or report['candidate_hash_before']!=report['candidate_hash_after']:
    raise SystemExit(1)
