"""Fail-closed policy through DeerFlow's native tool authorization hook."""
from deerflow.guardrails.provider import GuardrailDecision, GuardrailReason
from langgraph.runtime import get_runtime

from .permissions import PermissionDenied, authorize, runtime_identity


class ScopedToolGuardrail:
    name = 'ragopt-scoped-tools'
    policy_id = 'ragopt.scoped-tools'
    policy_version = '1.4'

    def __init__(self, *, framework='deerflow'):
        if framework != 'deerflow':
            raise ValueError('该工具策略只支持 DeerFlow')

    def evaluate(self, request):
        try:
            binding, role = runtime_identity(get_runtime())
            if request.thread_id != binding.thread_id:
                raise PermissionDenied('授权请求与运行线程不一致')
            args = request.tool_input
            authorize(binding, role, request.tool_name, args.get('task_id', binding.task_id),
                      delegate=args.get('subagent_type'), source='deerflow-guardrail')
        except PermissionDenied as exc:
            return GuardrailDecision(allow=False, reasons=[GuardrailReason(code='ragopt.scope_denied', message=str(exc))])
        return GuardrailDecision(allow=True, reasons=[GuardrailReason(code='ragopt.scope_allowed')])

    async def aevaluate(self, request):
        return self.evaluate(request)
