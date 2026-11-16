"""A real single-Agent control on the same development evidence.

Separate task database; shared bounded billing ledger. No candidate execution.
"""
from pathlib import Path
import json, os, sys, time

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from optimizer.launch import prepare_environment
prepare_environment()
from optimizer.store import Store
import optimizer.service as service_module
from optimizer.service import get_service
from optimizer.permissions import run_scope
from deerflow.config.app_config import get_app_config
from deerflow.extensions import load_extensions,set_loaded_extensions
from deerflow.client import DeerFlowClient
from langgraph.checkpoint.memory import InMemorySaver

billing=Store(root/'runtime/optimizer.db')


class BenchmarkStore(Store):
    def __init__(self,path):
        super().__init__(root/'runtime/live-single-benchmark.db')
    def reserve(self,task_id,amount,cap=70.,task_cap=None):
        return billing.reserve(task_id,amount,cap=min(cap,3.),task_cap=1.)
    def settle(self,call_id,actual):
        return billing.settle(call_id,actual)
    def budget(self):
        return billing.budget()


service_module.Store=BenchmarkStore
service=get_service()
loaded,diagnostics=load_extensions(get_app_config().plugins)
set_loaded_extensions(loaded)
task=service.create('真实单 Agent 对照：同一开发集证据，只诊断和提案，不修改',run=False)
service._baseline(task['id'])
service.store.transition(task['id'],{'diagnosing'},'diagnosing',agent_running=True)
text=[]
started=time.monotonic()
try:
    client=DeerFlowClient(config_path=str(root/'config.yaml'),checkpointer=InMemorySaver(),agent_name='rag-optimizer',
                          model_name='deepseek-flash',thinking_enabled=False,subagent_enabled=False,available_skills=set())
    prompt=f'''这是单 Agent 对照验收，任务 ID {task['id']}。不要委派子任务。
先 inspect_retrieval_task 获取开发集失败案例和对照。必要时 read_policy_sources 核查证据。
诊断关键证据漏召回和相关性问题，区分事实与假设。保持 top_k=2 和 context_chars=1800，选择一个最小改法，
调用 propose_retrieval_plan 提交一项配置改动与 experiment（待验证假设、实际开发集漏召回题 ID、最小平均召回绝对增量），然后总结并停止。不要批准或执行修改，不读取源码、个人文件或留出题。'''
    with run_scope(service, task['id'], single_agent_control=True) as binding:
        for event in client.stream(prompt,thread_id=binding.thread_id,authz_attributes={'ragopt_run':binding.nonce},recursion_limit=80):
            service._guard(task['id'])
            if time.monotonic()-started>180:
                raise TimeoutError('single control timeout')
            if event.type=='messages-tuple' and event.data.get('type')=='ai' and event.data.get('content'):
                text.append(str(event.data['content']))
    task=service.store.get(task['id'])
    if task['status']!='proposal_ready':
        raise RuntimeError('single control did not propose a plan')
    task=service.store.transition(task['id'],{'proposal_ready'},'awaiting_approval',agent_running=False)
    events=service.store.events(task['id'])
    calls=[e['payload'] for e in events if e['kind']=='model_call_completed']
    bill=[c for c in billing.budget()['calls'] if c['task_id']==task['id']]
    report={'task_id':task['id'],'model':'deepseek-flash','architecture':'real DeerFlow, subagents disabled',
            'same_development_evidence':True,'model_calls':calls,'model_steps':len(calls),'elapsed_seconds':time.monotonic()-started,
            'estimated_cny':sum(c['actual'] if c['actual'] is not None else c['reserved'] for c in bill),
            'plan':task['plan'],'reason':task['reason'],'summary':''.join(text)[-10000:],
            'approved':False,'comparison_limit':'单次对照，不证明多 Agent 的普遍质量优势'}
    (root/'reports/live-single-control.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'model_calls','reason','summary'}},ensure_ascii=False))
    service.reject(task['id'])
finally:
    service.close()
