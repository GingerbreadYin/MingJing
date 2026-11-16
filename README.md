# 电商政策 RAG 检索诊断与优化 Agent

v1.11 基于固定版本 DeerFlow，诊断现有电商政策 RAG 的检索失败，在用户审批后生成隔离候选并比较实际检索结果。

## Windows 本地启动

需要 Git、uv 和 Python 3.12。从本仓库根目录执行以下 PowerShell 命令：

```powershell
git clone https://github.com/bytedance/deer-flow.git upstream
git -C upstream checkout --detach 1dd6d056278536af1cb4fc60347f4ca40fcb629b
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv'
uv sync --project upstream/backend --python 3.12 --locked --no-dev --extra redis
Remove-Item Env:UV_PROJECT_ENVIRONMENT
uv pip install --python .venv/Scripts/python.exe -e .
& ./.venv/Scripts/python.exe scripts/configure.py --port 8026
& ./scripts/setup-retrieval.ps1
& ./.venv-retrieval/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='.cache/models/hub')"
& ./scripts/start.ps1
```

打开 http://127.0.0.1:8026/api/rag-optimizer/ui。可先选择「离线统计检查」，真实 DeerFlow 诊断需要通过环境变量或本仓库的本地 `.env` 提供 `DEEPSEEK_API_KEY`。实际检索读取已下载的 BGE 缓存。用 `./scripts/stop.ps1` 停止服务。此入口绑定本机回环地址。

## 使用

v1.1 的新诊断任务会生成单变量实验计划。审批前可查看假设、开发集漏召回题 ID、配置前后差异、检索预算和目标题平均证据召回的最小增量。执行前校验获批版本的完整内容。对照结果分别报告实验目标与原有回归门槛，任一未通过就不能保留。旧任务按原格式显示。

v1.2 为已批准的本地执行保存三个检查点：候选构建、开发集评测、最终验证。服务进程中断后，新版本任务进入「等待续跑」，在工作台选择「从检查点继续」。已完成产物按审批、源快照和文件指纹校验，未完成阶段在新目录重做。诊断、在途付费模型请求与保留操作仍需人工检查。工作台的「离线回放」是只读 JSON，可查看事件序列、检查点状态，并用已保存的开发集证据重新计算指标。

v1.3 的工作台新任务按实验系列运行，v1.11 最多四轮，跨轮共享 5 元预留预算和 60 次物理模型调用上限。批准只执行本轮开发集评测。你可以根据结果请求新计划，新轮次需独立批准。开发集门槛通过后，可冻结该候选再运行留出集和原题回归。同一系列开启最终验收后进入保留流程。进度通过 SSE 推送，断线按事件序号补发，审批和续跑仍用普通请求。

v1.4 通过本机 `/api/rag-optimizer/mcp` 提供两项只读证据工具。DeerFlow 原生 MCP 调用从服务端运行身份取得短期任务凭据，工具服务检查角色与任务范围。独立客户端 `scripts/check_mcp_client.py` 可列出工具；读取数据需要由本机验收进程注入任务凭据。诊断子 Agent 根据开发集证据接收固定目录中的诊断指南。权限和预算使用同一运行身份。生成配置时用 `scripts/configure.py --port 8026` 指向工作台监听端口。

1. 输入目标，选择 DeerFlow 主子 Agent 诊断，或明确标注的离线统计模式。
2. 查看开发集基线、实际轨迹、原因假设与方案。离线模式没有模型主子任务，不可作为该能力的证据。
3. 决定是否批准方案。拒绝时不产生候选代码。
4. 批准后，在独立目录生成检索适配代码、配置和切分结果，运行索引、开发集、留出集及实际问答图接口检查。
5. 查看收益、退化及未验证项，决定保留或不保留。保留只保存到新项目 retained/，不合入原 RAG、不推送代码。

第一版修改是受限配置与固定适配器生成，不开放任意代码/Shell 执行。支持向量、BM25、RRF 混合检索、段落/章节/固定长度切分、候选数及上下文预算。重排序只有本地模型缓存就绪时才能启用，不静默下载。

## 目录与来源

- upstream/：真实 DeerFlow 源码，提交 1dd6d056278536af1cb4fc60347f4ca40fcb629b，独立 Git 分支；保留上游许可证，不作为个人原创。
- baseline/rag/：原 RAG 提交 90d871cacdc8acacf5999707286a72594d9f0cf3 的 39 个跟踪文件快照，不含真实 .env 或模型缓存。
- optimizer/：个人定制的证据评测、审批、隔离整改、模型工具、预算审计和工作台扩展。
- data/：26 篇演示/模拟政策、50 道开发题、26 道留出题、10 道原回归题及文件指纹。
- experiments/：每个任务独立的基线、候选、差异和机器可读报告。
- runtime/：SQLite 任务库、DeerFlow 状态和日志；不提交 Git。
- docs/ISSUES.md：实际遇到的问题与修复过程。

扩展通过 config.yaml 中 plugins 和官方扩展注册接口接入 Gateway；主子 Agent 使用原生 DeerFlowClient/task，不自行仿造角色循环。工具不包含审批或任意文件执行入口，审批来自网页 API。主子模型均通过物理调用预算中间件。

## 配置与费用

优先使用 `DEEPSEEK_API_KEY` 环境变量。应用也可读取本项目的 `.env`，或由 `DEER_FLOW_ENV_FILE` 指定本机配置文件。

Flash 是第一版启用的模型；Pro 仅为后续待实测升级候选，没有默认启用。模型验证前需授权将模拟政策、测试问题和检索结果发送给 DeepSeek。不会发送原项目源码或其他个人文件。

人民币总预算 100 元，制作预留上限 70 元，学习预留 30 元。按高峰、无缓存折扣及保守换算估算；失败且用量不明确的调用保留预留额。客户端记录不是厂商账单，外部网络超时不等于未计费，不自动充值或无限重试。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\doctor.py
```

doctor 只输出密钥存在性、模型和工具名，不输出密钥。测试不发付费请求。实际检索需要本地 BGE 模型缓存，默认离线加载。

### Query 拆分对照

运行 `python scripts/compare_query_split.py`，在现有开发集上比较整句查询和分句查询。两组使用同一个本地 BGE 索引、Top-2 和 1800 字上下文上限。报告记录实际分支、片段去重结果、证据指标、查询次数和延迟。工作台中的“查看 Query 拆分对照报告”可读取最近一次完整报告。

当开发集存在可拆分的多条件漏召回题时，诊断可提出单项配置 `query_split=true`。批准后，候选问答与候选评测调用同一个冻结的检索适配器。默认关闭，目前限纯向量且关闭重排。多条件问题按分支召回并合并，单条件问题进入整句检索。

### 实验记忆与 Compose

多轮任务的“实验记忆”展示同系列前序轮次的开发集结果与备注。每条摘要标出来源事件。备注默认只在本机查看，可在保存时允许后续模型诊断读取。模型只接收最多 1000 字符的结构化摘要。

### 回答质量评测

使用独立 Python 环境安装 `scripts/requirements-answer-eval.txt`。运行 `python scripts/evaluate_answers.py` 会用离线回放验证真实 Ragas 计分。工作台的“查看回答质量评测”只读展示报告，有真实裁判报告时优先显示该报告。检索指标继续从原有实验报告读取。

固定样本包含历史回答及构造的边界案例。自定义裁判分别判断证据、完整性和答复方式，Ragas Faithfulness 衡量已有陈述的证据支持比例。参考标签只在本地对照。授权的模型实测需要显式指定 `--live-ledger` 使用现有预算账本，凭据从环境变量或指定的本地文件读取。

### ONNX CPU 检索对照

准备原有检索解释器及本地 BGE 缓存后，使用下列 PowerShell 命令导出并比较 FP32 ONNX。

```powershell
uv pip install --python .venv-retrieval/Scripts/python.exe --target .onnx-deps --no-deps -r scripts/requirements-onnx.txt
& ./.venv-retrieval/Scripts/python.exe scripts/compare_onnx.py
```

命令在独立进程执行 PyTorch 和 ONNX Runtime，保存模型、向量和逐题检索对照。工作台的“查看 ONNX CPU 对照”只读展示最近的完整报告。

在 Linux 或 WSL 的项目目录运行 `docker compose up --build -d --wait`，然后访问本机端口 8033。默认从项目下的 `upstream/` 和 `.cache/models/` 读取固定版本 DeerFlow 源码和 BGE 缓存，可用 `DEERFLOW_SOURCE` 和 `RAGOPT_MODEL_CACHE` 指定路径。Compose 预览使用离线任务模式。Redis 只在 Compose 网络内供 Gateway 缓存摘要，SQLite 和实验产物使用持久卷。重启可用 `docker compose restart`。保留历史时使用 `docker compose down`，持久卷随 `down -v` 删除。

检索指标以原始条款证据组计算；替代证据属于同一组，重复片段不重复增加召回。无答案题不进入 Recall 分母。主要报告 Recall、Precision、MRR 和完整证据覆盖，当前不报告缺少分级标注依据的标准 NDCG。

## 重要限制

模拟题集不代表真实业务分布；留出题主要测试同主题不同表述，不代表未见主题泛化。留出标签不提供给优化 Agent；人工持续依据留出指标调参也会造成污染，应冻结最终候选并使用新的测试集。

取消会阻止后续步骤和新模型调用，已提交给外部服务的请求不保证撤销，已运行的本地子进程可能需要完成或超时后退出。不得把取消理解为立即终止厂商计费。

已有反例：混合检索+章节切分在开发集提高召回，却在留出集退化；记录为不保留，不以开发集数字宣传提升。已知问题见 [问题记录](docs/ISSUES.md)。


公开仓库包含模拟电商政策、测试数据和评测脚本。工作台的历史报告入口在本轮完成最终验收后显示，报告由上述独立评测脚本生成。
