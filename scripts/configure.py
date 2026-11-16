"""Generate operator config without reading or copying credential values."""
from pathlib import Path
import argparse
import json
from uuid import NAMESPACE_URL, uuid5

ROOT=Path(__file__).resolve().parents[1]


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8026)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('invalid port')
    tools=['inspect_retrieval_task','read_policy_sources','propose_retrieval_plan']
    mcp_tools=['mcp_inspect_development','mcp_read_policies']
    config={
        'models':[{'name':'deepseek-flash','display_name':'DeepSeek Flash','use':'optimizer.models:ScopedDeepSeek',
                   'model':'deepseek-flash','api_base':'https://api.deepseek.com','api_key':'$DEEPSEEK_API_KEY','max_tokens':3072,
                   'timeout':60,'max_retries':0,'supports_thinking':True,'supports_vision':False,
                   'when_thinking_enabled':{'extra_body':{'thinking':{'type':'enabled'}}},
                   'when_thinking_disabled':{'extra_body':{'thinking':{'type':'disabled'}}}}],
        'tool_groups':[{'name':'rag-read'},{'name':'rag-proposal'}],
        'tools':[{'name':name,'group':'rag-proposal' if name=='propose_retrieval_plan' else 'rag-read','use':'optimizer.tools:'+name} for name in tools],
        'sandbox':{'use':'deerflow.sandbox.local:LocalSandboxProvider','allow_host_bash':False},
        'title':{'enabled':False},'memory':{'enabled':False},'summarization':{'enabled':False},
        'suggestions':{'enabled':False},'input_polish':{'enabled':False},
        'tool_search':{'enabled':False},
        'skills':{'path':str(ROOT/'runtime/empty-skills')},
        'token_usage':{'enabled':True},'token_budget':{'enabled':True,'max_tokens':250000},
        'subagent_runtime':{'max_running':2},
        'subagents':{'max_turns':8,'timeout_seconds':180,'max_total_per_run':2,
                     'token_budget':{'enabled':True,'max_tokens':60000},
                     'custom_agents':{
                         'retrieval-diagnostician':{'description':'Read-only diagnosis of missing evidence, ranking and chunking issues in current RAG.',
                             'system_prompt':'你是只读检索诊断专家。优先使用 MCP 查看开发集和政策，缺少 MCP 工具时使用本地只读工具。按收到的诊断指南工作，区分事实、假设和待验证项，不访问留出题。',
                             'tools':tools[:2]+mcp_tools,'skills':[],'max_turns':8,'timeout_seconds':180,'thinking_enabled':False},
                         'retrieval-optimizer':{'description':'Design one bounded retrieval change and submit it for human approval; cannot execute changes.',
                             'system_prompt':'你是检索改进方案专家。依据诊断和真实开发集结果，必要时读取原文。只改一个检索配置项。调用 propose_retrieval_plan，提交配置、理由和 experiment：假设、实际漏召回的开发集题 ID、目标题平均证据召回最小绝对增量。说明风险和不确定性。不能批准、执行修改或读取留出题。优先保持 K 和上下文预算。',
                             'tools':tools+mcp_tools,'skills':[],'max_turns':8,'timeout_seconds':180,'thinking_enabled':False}}},
        'guardrails':{'enabled':True,'fail_closed':True,'provider':{'use':'optimizer.tool_guardrail:ScopedToolGuardrail','config':{}}},
        'plugins':[{'name':'rag-optimizer','use':'optimizer.extension:install','required':True,'config':{}}],
        'database':{'backend':'sqlite','sqlite_dir':str(ROOT/'runtime/deerflow/data')},
        'run_events':{'backend':'db'},
    }
    (ROOT/'config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    home=ROOT/'runtime/deerflow'
    agent=home/'users/default/agents/rag-optimizer'
    agent.mkdir(parents=True,exist_ok=True)
    (ROOT/'runtime/empty-skills').mkdir(parents=True,exist_ok=True)
    (agent/'config.yaml').write_text(yaml.safe_dump({'name':'rag-optimizer','display_name':'RAG 检索诊断与优化','description':'只读诊断、受审批的隔离整改与可重复评测',
        'model':'deepseek-flash','tool_groups':['rag-read','rag-proposal'],'skills':[],
        'mcp_plugins':[str(uuid5(NAMESPACE_URL, 'deerflow:mcp:ragopt_read'))],
        'allowed_subagents':['retrieval-diagnostician','retrieval-optimizer'],'memory_enabled':False,'thinking_enabled':False},allow_unicode=True,sort_keys=False),encoding='utf-8')
    (agent/'SOUL.md').write_text('你协调现有电商政策 RAG 的检索诊断与优化。所有结论必须关联实际工具证据。只将必要工作委派给获准子 Agent。生成方案后等待用户在网页批准，不自行执行或绕过审批。不要宣称未运行的评测已经通过。',encoding='utf-8')
    extensions = {'mcpServers':{'ragopt_read':{'enabled':True,'type':'http',
        'url':f'http://127.0.0.1:{args.port}/api/rag-optimizer/mcp','tool_name_prefix':False,
        'session_init_timeout':10}},
        'mcpInterceptors':['optimizer.mcp_interceptor:build'],'skills':{}}
    (ROOT/'extensions_config.json').write_text(json.dumps(extensions,indent=2),encoding='utf-8')
    print('已生成项目配置；只引用环境变量，未读取或复制密钥。')


if __name__=='__main__':
    main()
