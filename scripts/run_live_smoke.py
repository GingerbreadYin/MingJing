"""User-authorized live diagnostic only; never approves a proposed change."""
from pathlib import Path
import collections, json, time, urllib.request

root=Path(__file__).resolve().parents[1]
base='http://127.0.0.1:8026/api/rag-optimizer'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(path,body=None):
    data=json.dumps(body,ensure_ascii=False).encode('utf-8') if body is not None else None
    req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
    with opener.open(req,timeout=15) as response:
        return json.load(response)


task=request('/tasks',{'goal':'验收：根据模拟电商政策开发集失败案例诊断漏召回和无关结果。真实委派诊断与优化子 Agent，保持 top_k=2，提交最小修改方案；不要执行修改。只使用获准的模拟政策、测试问题和检索结果，不读取源码或其他个人文件。','mode':'deerflow'})
task_id=task['id']
(root/'runtime/live-smoke-task.txt').write_text(task_id,encoding='utf-8')
print('live_task='+task_id,flush=True)
deadline=time.monotonic()+300
last=None
while time.monotonic()<deadline:
    task=request('/tasks/'+task_id)
    if task['status']!=last:
        print('status='+task['status'],flush=True)
        last=task['status']
    if task['status'] in {'awaiting_approval','failed','cancelled','interrupted'}:
        break
    time.sleep(2)
events=request('/tasks/'+task_id+'/events')
budget=request('/status')['budget']
report={'task_id':task_id,'status':task['status'],'error':task.get('error'),'plan':task.get('plan'),'reason':task.get('reason'),
        'diagnosis':task.get('diagnosis'),'event_kinds':dict(collections.Counter(e['kind'] for e in events)),
        'model_calls':[e['payload'] for e in events if e['kind'] in {'model_call_completed','deerflow_error'}],
        'budget_calls':[c for c in budget['calls'] if c['task_id']==task_id],
        'approved':False,'modified_original_rag':False,
        'data_authorization':'用户明确允许模拟政策、测试问题、检索结果发送至 DeepSeek，本次预计<=3元，总预算100元内；不发送源码或个人文件'}
(root/'reports/live-deerflow-smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'task_id':task_id,'status':task['status'],'error':task.get('error'),'plan':task.get('plan'),
                  'event_kinds':report['event_kinds'],'recorded_model_steps':len([e for e in events if e['kind']=='model_call_completed']),
                  'estimated_cny':sum(c['actual'] if c['actual'] is not None else c['reserved'] for c in report['budget_calls'])},ensure_ascii=False),flush=True)
if task['status']!='awaiting_approval':
    raise SystemExit(1)
