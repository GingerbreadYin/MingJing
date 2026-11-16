"""Run-scoped authority sourced from DeerFlow runtime, never model arguments."""
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass, field
from secrets import token_urlsafe
from threading import Lock


MCP_SERVER = 'ragopt_read'
MCP_TOOLS = frozenset({'mcp_inspect_development', 'mcp_read_policies'})
READ = frozenset({'inspect_retrieval_task', 'read_policy_sources'}) | MCP_TOOLS
ROLE_TOOLS = {
    'lead': READ | {'task'},
    'retrieval-diagnostician': READ,
    'retrieval-optimizer': READ | {'propose_retrieval_plan'},
}
PROJECT_TOOLS = frozenset().union(*ROLE_TOOLS.values())
CAPABILITY_HEADER = 'X-Ragopt-Capability'
_lock = Lock()
_runs = {}
_capabilities = {}


class PermissionDenied(ValueError):
    pass


@dataclass
class RunBinding:
    service: object
    task_id: str
    thread_id: str
    nonce: str = field(repr=False)
    skill: str | None
    tokens: dict[str, str] = field(repr=False)
    single_agent_control: bool = False
    skill_recorded: bool = False


@contextmanager
def run_scope(service, task_id, skill=None, *, single_agent_control=False):
    nonce = token_urlsafe(32)
    binding = RunBinding(service, task_id, 'ragopt-'+task_id, nonce, skill,
                         {role:token_urlsafe(32) for role in ROLE_TOOLS}, single_agent_control)
    with _lock:
        _runs[nonce] = binding
        for role, token in binding.tokens.items():
            _capabilities[token] = (binding, role)
    try:
        yield binding
    finally:
        with _lock:
            _runs.pop(nonce, None)
            for token in binding.tokens.values():
                _capabilities.pop(token, None)


def runtime_identity(runtime):
    context = getattr(runtime, 'context', None) or {}
    attributes = context.get('authz_attributes') if isinstance(context, Mapping) else None
    nonce = attributes.get('ragopt_run') if isinstance(attributes, Mapping) else None
    if not isinstance(nonce, str):
        raise PermissionDenied('缺少有效的任务运行身份')
    with _lock:
        binding = _runs.get(nonce)
    role = context.get('agent_id') if context.get('is_subagent') else 'lead'
    if binding is None or context.get('thread_id') != binding.thread_id or not isinstance(role, str) or role not in ROLE_TOOLS:
        raise PermissionDenied('运行身份已失效或不匹配')
    return binding, role


def capability_identity(token):
    with _lock:
        identity = _capabilities.get(token)
    if identity is None:
        raise PermissionDenied('需要有效的本机任务凭据')
    return identity


def authorize(binding, role, tool, task_id, *, delegate=None, source='native'):
    reason = None
    with _lock:
        active = _runs.get(binding.nonce) is binding
    if not active or task_id != binding.task_id:
        reason = '任务范围不匹配'
    elif tool not in ROLE_TOOLS.get(role, ()) and not (
            binding.single_agent_control and role == 'lead' and tool == 'propose_retrieval_plan'):
        reason = '角色不允许调用该工具'
    elif tool == 'task' and delegate not in {'retrieval-diagnostician', 'retrieval-optimizer'}:
        reason = '不允许委派给该角色'
    if reason:
        binding.service.store.event(binding.task_id, 'tool_denied', {'role':role,'tool':tool,'source':source,'reason':reason})
        raise PermissionDenied(reason)
    binding.service._guard(binding.task_id)


def mark_skill_injected(binding):
    with _lock:
        if binding.skill_recorded:
            return False
        binding.skill_recorded = True
        return True


def redact_runtime_secrets(text):
    with _lock:
        secrets = [*_runs, *_capabilities]
    for secret in secrets:
        text = text.replace(secret, '[REDACTED]')
    return text
