"""Exercise the REAL DeerFlow lead/task executor using a no-network test model."""
from pathlib import Path
import json, os, shutil, sys
from unittest.mock import patch
from functools import partial
import yaml

root=Path(__file__).resolve().parents[1]
single='--single' in sys.argv
sys.path.insert(0,str(root))
home=root/'runtime/deerflow-offline'
agent=home/'users/default/agents/rag-optimizer'
agent.mkdir(parents=True,exist_ok=True)
for name in ['config.yaml','SOUL.md']:
    shutil.copy2(root/'runtime/deerflow/users/default/agents/rag-optimizer'/name,agent/name)
env_file=root/'runtime/offline-test.env'
env_file.write_text('OFFLINE_TEST=1\n',encoding='utf-8')
offline_extensions = root/'runtime/offline-extensions.json'
offline_extensions.write_text('{"mcpServers":{},"skills":{}}', encoding='utf-8')
config=yaml.safe_load((root/'config.yaml').read_text(encoding='utf-8'))
config['models'][0]['use']='tests.support.scripted_model:ScriptedChatModel'
config['models'][0]['api_key']='offline-no-secret'
config['models'][0]['model']='offline-scripted'
config['database']['sqlite_dir']=str(home/'data')
config_path=root/'runtime/offline-config.yaml'
config_path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
os.environ.update(DEER_FLOW_PROJECT_ROOT=str(root),DEER_FLOW_HOME=str(home),DEER_FLOW_CONFIG_PATH=str(config_path),
                  DEER_FLOW_EXTENSIONS_CONFIG_PATH=str(offline_extensions),DEER_FLOW_ENV_FILE=str(env_file),
                  DEEPSEEK_API_KEY='offline-no-secret',PYTHONIOENCODING='utf-8')
from optimizer.service import get_service
import optimizer.service as service_module
from optimizer.store import Store
from optimizer.agent import run_diagnosis
service_module.Store=lambda path: Store(root/'runtime/offline-harness.db')
from deerflow.client import DeerFlowClient
from deerflow.config.app_config import get_app_config
from deerflow.extensions import load_extensions,set_loaded_extensions
real_client=DeerFlowClient
service=get_service()
loaded,diagnostics=load_extensions(get_app_config().plugins)
set_loaded_extensions(loaded)
task=service.create(('单 Agent 对照' if single else '主子 Agent')+'离线调度验收：模型为脚本模拟，不是 DeepSeek 调用',run=False)
service.store.transition(task['id'],{'created'},'created',mode='deerflow-offline-scripted')

class FixtureClient(real_client):
    def __init__(self,**kwargs):
        kwargs['config_path']=str(config_path)
        if single:
            kwargs['subagent_enabled']=False
        super().__init__(**kwargs)

try:
    with patch('deerflow.client.DeerFlowClient',FixtureClient), \
         patch('optimizer.agent.run_diagnosis', partial(run_diagnosis, single_agent_control=single)):
        service.diagnose_deerflow(task['id'])
    task=service.store.get(task['id'])
    events=service.store.events(task['id'])
    calls=[e for e in events if e['kind']=='model_call_completed']
    assert task['status']=='awaiting_approval',task['status']
    assert task['experiment_plan']['target_case_ids']==['demo-13-combined']
    assert calls,'budget middleware not exercised'
    scopes={e['payload'].get('scope') for e in calls}
    assert scopes==({'lead'} if single else {'lead','subagent'}),scopes
    report={'task_id':task['id'],'real_deerflow_client':True,'real_task_executor':not single,'model':'test-only scripted/no network',
            'provider_api_calls':0,'audited_model_steps':len(calls),'audited_scopes':sorted(scopes),'status':task['status'],'events':events,
            'comparison_limit':'只比较调度与调用开销，不证明多 Agent 的诊断质量收益'}
    (root/('reports/deerflow-offline-single.json' if single else 'reports/deerflow-offline-integration.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='events'},ensure_ascii=False))
    service.reject(task['id'])
finally:
    service.close()
