"""Real candidate graph/retriever, stubbed generation. This is not a live LLM test."""
from pathlib import Path
import json, sys

candidate=Path(sys.argv[1]).resolve()
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from optimizer.isolation import confined_path
output_input=Path(sys.argv[2])
output=confined_path(root,output_input.relative_to(root).as_posix())
if output.exists():
    raise RuntimeError('不覆盖已有图检查结果')
sys.path.insert(0,str(candidate))
import retrieval, rag_graph

resources=retrieval.load_retriever()
question='普通商品签收五天，未使用且包装完整，能申请无理由退货吗？'
split_enabled = resources['config'].get('query_split', False)
if split_enabled:
    question='物流停滞何时核查，核查后能否立即按丢件退款？'
ranked=retrieval.rank_chunks(question,resources)
if not ranked:
    raise RuntimeError('候选没有返回检索结果')
check_calls=[0]
def check_evidence(question,evidence):
    check_calls[0]+=1
    return {'status':'NEED_RETRIEVAL' if check_calls[0]==1 else 'ANSWERABLE','reason':'验收用固定分支检查，不评价语义正确性','question':''}
rag_graph.check_evidence=check_evidence
rag_graph.generate_answer=lambda question,evidence:(f"这是接口验收用模拟回答。[{ranked[0]['id']}]",None)
graph=rag_graph.build_rag_graph(resources)
state=graph.invoke({'question':question})
if state['status']!='ANSWERED':
    raise RuntimeError('实际候选图未到达回答节点')
trace = {}
if split_enabled:
    trace = resources['last_query_split']
    if len(trace['branches']) < 2 or [c['id'] for c in state['ranked_chunks']] != [c['id'] for c in ranked]:
        raise RuntimeError('候选问答图未使用同一拆分排序')
output.write_text(json.dumps({'status':state['status'],'retrieved_ids':[c['id'] for c in state['selected_chunks']],
    'actual_retrieval':True,'actual_graph':True,'model_generation':'stubbed','semantic_quality':'not evaluated',
    'rounds':state['rounds'],'actual_final_context_ids':[c['id'] for c in state['selected_chunks']],
    'index_exists':(candidate/'vector_index.pt').exists(), 'query_split_trace':trace},ensure_ascii=False,indent=2),encoding='utf-8')
print('实际候选 RAG 图和检索接口检查通过；生成与证据判断为 mock。')
