"""Only these explicit, scoped functions are model-visible."""
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from .experiment_plan import ExperimentSpec
from .mcp_server import compact_development
from .permissions import authorize, runtime_identity


def bound(task_id, runtime, name):
    binding, role = runtime_identity(runtime)
    authorize(binding, role, name, task_id)
    return binding.service


@tool
def inspect_retrieval_task(task_id: str, runtime: ToolRuntime) -> dict:
    """Read current task, actual development retrieval results and bounded config options; no held-out labels."""
    service=bound(task_id,runtime,'inspect_retrieval_task')
    service.store.event(task_id,'tool',{'name':'inspect_retrieval_task'})
    return compact_development(service.agent_view(task_id))


@tool
def read_policy_sources(task_id: str, document_ids: list[str], runtime: ToolRuntime) -> list[dict]:
    """Read at most eight original policy documents by document ID to inspect evidence and conditions."""
    service=bound(task_id,runtime,'read_policy_sources')
    service.store.event(task_id,'tool',{'name':'read_policy_sources','document_ids':document_ids})
    return service.sources(task_id,document_ids)


@tool
def propose_retrieval_plan(task_id: str, retrieval_config: dict, reason: str, experiment: ExperimentSpec, runtime: ToolRuntime) -> dict:
    """Submit one retrieval change and a testable experiment for human approval. For multi-condition misses, query_split=true is available with plain vector retrieval and no reranking."""
    service=bound(task_id,runtime,'propose_retrieval_plan')
    task=service.propose(task_id,retrieval_config,reason,experiment)
    service.store.event(task_id,'tool',{'name':'propose_retrieval_plan','plan_hash':task['plan_hash']})
    return {'status':task['status'],'plan_hash':task['plan_hash'],'plan':task['plan'],'reason':task['reason'],
            'experiment_plan':task['experiment_plan'],'next':'必须等待网页上的用户批准；此工具没有执行权限'}
