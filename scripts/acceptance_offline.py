"""No-provider-call acceptance runner. Uses real local retrieval and harness."""
from pathlib import Path
import json, os, subprocess, sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
os.environ.update(PYTHONIOENCODING='utf-8',PYTHONPATH=str(root),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')


def run(name,args,python=None):
    log=root/'reports'/f'{name}.log'
    with log.open('w',encoding='utf-8') as output:
        result=subprocess.run([str(python or sys.executable),*args],cwd=root,env=os.environ.copy(),stdout=output,stderr=subprocess.STDOUT,timeout=480)
    if result.returncode:
        raise RuntimeError(f'{name} failed; see {log}')
    print(name+': PASS',flush=True)


run('final-unit-tests',['-m','unittest','discover','-s','tests','-v'])
run('baseline-equivalence',['scripts/check_baseline_equivalence.py'],root/'.venv-retrieval/Scripts/python.exe')
run('deerflow-offline-multi',['scripts/check_deerflow_offline.py'])
run('deerflow-offline-single',['scripts/check_deerflow_offline.py','--single'])

from optimizer.service import OptimizerService
import optimizer.service as service_module
from optimizer.store import Store
service_module.Store=lambda path: Store(root/'runtime/offline-acceptance.db')
service=OptimizerService(root)
try:
    task=service.create('验收：基线等价候选，模拟审批和保留，仅验证机制，无性能提升声明',run=False,mode='offline')
    service._baseline(task['id'])
    proposal=service.propose(task['id'],{},'验收用基线等价方案：不宣称优化收益，验证冻结代码、回归和保留机制。')
    service.approve(task['id'],proposal['plan_hash'],run=False)
    comparison=service.execute(task['id'])
    retained=service.retain(task['id'],True)
    summary={'task_id':task['id'],'status':retained['status'],'provider_api_calls':0,
             'development':comparison['development']['metrics'],'heldout':comparison['heldout'],
             'regression':comparison['original_regression'],'checks':comparison['retention_checks'],
             'graph_check':comparison['graph_check'],'retained_path':retained['retained_path'],
             'semantic_quality':'not evaluated; graph generation/evidence decision stubbed',
             'performance_claim':'none; baseline-equivalent acceptance fixture'}
    (root/'reports/offline-acceptance.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
finally:
    service.close()
