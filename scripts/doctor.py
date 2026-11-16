"""Readiness check that never prints credential values or resolved configs."""
from pathlib import Path
import json, sys, os

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from optimizer.launch import prepare_environment

root=prepare_environment()
result={'root':str(root),'credential_configured':os.environ.get('DEEPSEEK_API_KEY') not in {None,'','not-configured'}}
try:
    from deerflow.config.app_config import get_app_config
    config=get_app_config()
    result.update(models=[m.name for m in config.models],tools=[t.name for t in config.tools],subagents=list(config.subagents.custom_agents))
    from deerflow.client import DeerFlowClient
    client=DeerFlowClient(config_path=str(root/'config.yaml'),agent_name='rag-optimizer',subagent_enabled=True,available_skills=set(),thinking_enabled=False)
    result['client_initialized']=True
except Exception as exc:
    result['error_type']=type(exc).__name__
    if hasattr(exc,'errors'):
        result['validation_errors']=[{'location':str(e['loc']),'type':e['type']} for e in exc.errors()]
    else:
        result['error_message']=str(exc).replace(os.environ.get('DEEPSEEK_API_KEY','not-configured'),'[REDACTED]')[:1000]
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(1)
print(json.dumps(result,ensure_ascii=False,indent=2))
