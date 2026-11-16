"""Test-only deterministic chat model; it has no network client or API calls."""
import re
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from optimizer.model_policy import ScopedModelMixin


class ScriptedChatModel(ScopedModelMixin, BaseChatModel):
    model: str='offline-scripted'

    @property
    def _llm_type(self):
        return 'offline-scripted-no-network'

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        names=[t.name if hasattr(t,'name') else t['function']['name'] for t in tools]
        return self.bind(tool_names=names)

    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        text='\n'.join(str(m.content) for m in messages)
        task_ids=re.findall(r'\b[0-9a-f]{32}\b',text)
        if not task_ids:
            raise RuntimeError('scripted fixture task id missing')
        task_id=task_ids[0]
        names=set(kwargs.get('tool_names',[]))
        past=[tc['name'] for m in messages for tc in getattr(m,'tool_calls',[])]
        read_tool = 'mcp_inspect_development' if 'mcp_inspect_development' in names else 'inspect_retrieval_task'
        if read_tool == 'mcp_inspect_development' and read_tool in past:
            evidence = [str(m.content) for m in messages if m.type == 'tool' and m.name == read_tool]
            if not any('development_cases' in item and 'demo-13-combined' in item for item in evidence):
                raise RuntimeError('MCP development evidence did not reach the model')
        calls=[]
        if 'task' in names and read_tool == 'mcp_inspect_development' and read_tool not in past:
            calls=[{'name':read_tool,'args':{'task_id':task_id},'id':'offline-lead-mcp','type':'tool_call'}]
        elif 'task' in names:
            count=past.count('task')
            if count<2:
                role=['retrieval-diagnostician','retrieval-optimizer'][count]
                calls=[{'name':'task','args':{'prompt':f'离线调度验收：任务 {task_id}，读取真实开发集。'+('请提出方案并调用 propose_retrieval_plan。' if count else '只读检查后返回证据。'),
                    'subagent_type':role,'description':'离线调度验收'},'id':f'offline-task-{count+1}','type':'tool_call'}]
        elif read_tool not in past:
            if read_tool == 'mcp_inspect_development' and 'propose_retrieval_plan' not in names:
                if '<reviewed_diagnostic_skill name="missing-evidence">' not in text:
                    raise RuntimeError('diagnostic skill did not reach the model')
            calls=[{'name':read_tool,'args':{'task_id':task_id},'id':'offline-inspect','type':'tool_call'}]
        elif 'propose_retrieval_plan' in names and 'propose_retrieval_plan' not in past:
            calls=[{'name':'propose_retrieval_plan','args':{'task_id':task_id,'retrieval_config':{'strategy':'hybrid'},
                    'reason':'离线脚本模型仅验证真实主子工具调度，不证明模型诊断质量。',
                    'experiment':{'hypothesis':'政策关键词可能补足向量漏召回','target_case_ids':['demo-13-combined'],
                                  'min_recall_gain':0.1}},'id':'offline-plan','type':'tool_call'}]
        message=AIMessage(content='' if calls else '离线脚本模型完成本阶段，仅作为框架调度证据。',tool_calls=calls,
                          usage_metadata={'input_tokens':0,'output_tokens':0,'total_tokens':0})
        return ChatResult(generations=[ChatGeneration(message=message)])
