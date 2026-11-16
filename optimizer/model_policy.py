"""Apply task binding, diagnostic input and accounting at the actual model call."""
from langchain_core.messages import SystemMessage
from langgraph.runtime import get_runtime

from .diagnostic_skills import load_guide
from .permissions import mark_skill_injected, runtime_identity


def add_diagnostic_guide(messages, binding, role):
    if role != 'retrieval-diagnostician' or not binding.skill:
        return messages
    guide = load_guide(binding.skill)
    note = f'\n\n<reviewed_diagnostic_skill name="{binding.skill}">\n{guide}\n</reviewed_diagnostic_skill>'
    if messages and isinstance(messages[0], SystemMessage):
        first = messages[0]
        content = first.content + note if isinstance(first.content, str) else [*first.content, {'type':'text','text':note}]
        result = [first.model_copy(update={'content':content}), *messages[1:]]
    else:
        result = [SystemMessage(content=note), *messages]
    if mark_skill_injected(binding):
        binding.service.store.event(binding.task_id, 'skill_injected', {'role':role,'skill':binding.skill})
    return result


class ScopedModelMixin:
    """The provider entry point rejects unbound or over-budget requests before IO."""

    def _prepare_ragopt_call(self, model_input):
        binding, role = runtime_identity(get_runtime())
        service, task_id = binding.service, binding.task_id
        service._guard(task_id)
        messages = add_diagnostic_guide(self._convert_input(model_input).to_messages(), binding, role)
        chars = sum(len(str(message.content).encode('utf-8')) for message in messages)
        if chars > 300000:
            raise RuntimeError('输入上下文超过本项目预算上限')
        reserve = max(.15, ((chars+4096)*.3+4096*1.2)/1_000_000*8)
        call_id = service.store.reserve(task_id, reserve, task_cap=5.)
        scope = 'lead' if role == 'lead' else 'subagent'
        agent = 'rag-optimizer' if role == 'lead' else role
        service.store.event(task_id, 'model_call_started', {'call_id':call_id,'reserved_cny':reserve,
                            'input_utf8_bytes':chars,'scope':scope,'agent_name':agent})
        return messages, (service, task_id, call_id, scope, agent)

    @staticmethod
    def _settle_ragopt_call(binding, message):
        service, task_id, call_id, scope, agent = binding
        usage = getattr(message, 'usage_metadata', None)
        actual = ((usage.get('input_tokens',0)*.3+usage.get('output_tokens',0)*1.2)/1_000_000*8) if usage is not None else None
        service.store.settle(call_id, actual)
        if message is not None:
            service.store.event(task_id, 'model_call_completed', {'call_id':call_id,'usage':usage or {},
                                'estimated_cny':actual,'coverage':'provider token metadata; peak/no-cache estimate',
                                'scope':scope,'agent_name':agent})

    def invoke(self, input, *args, **kwargs):
        messages, binding = self._prepare_ragopt_call(input)
        try:
            result = super().invoke(messages, *args, **kwargs)
        except BaseException:
            self._settle_ragopt_call(binding, None)
            raise
        self._settle_ragopt_call(binding, result)
        return result

    async def ainvoke(self, input, *args, **kwargs):
        messages, binding = self._prepare_ragopt_call(input)
        try:
            result = await super().ainvoke(messages, *args, **kwargs)
        except BaseException:
            self._settle_ragopt_call(binding, None)
            raise
        self._settle_ragopt_call(binding, result)
        return result


def require_scoped_model(model_class):
    if not isinstance(model_class, type) or not issubclass(model_class, ScopedModelMixin):
        raise RuntimeError('模型未启用任务绑定与预算控制；拒绝启动模型')
