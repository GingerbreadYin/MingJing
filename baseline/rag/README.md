# 电商政策 RAG 学习项目

基于 FastAPI、LangGraph、本地中文向量模型和 DeepSeek API 的政策问答后端。根据检索证据回答、追问用户或停止，并提供文档块引用。资料为练习店铺的虚构规则。

## 独立目录迁移状态

本目录已从原混合学习项目复制为独立 Git 仓库。四份离线检查已在本目录运行通过，使用的是原项目已有的 Python 环境；检查未请求真实模型。真实端到端报告是迁移前运行结果。

本目录没有复制真实 .env、虚拟环境、模型缓存或 vector_index.pt。首次独立运行前需恢复环境、根据 .env.example 配置自己的密钥，再切分资料、构建索引。下方“现有环境启动”命令以这些准备已完成为前提。

## 当前范围

- 正式知识库：退货与发票两份 Markdown，共七个文本块。
- 正式检索：纯向量检索；首次取前两块，证据不足时扩大到前四块，最多两轮检查。
- 纯问候采用完整匹配规则直接回复。
- 回答前检查证据，回答后检查引用编号合法性。
- 请求具有有限重试、指数退避与抖动；接口区分模型故障和引用校验失败。
- BM25、RRF、重排是独立对比实验，尚未接入正式问答图。

## 架构与模块

```text
POST /chat
  → route
    → direct → 结束
    → retrieve → check
                   → generate → 引用校验 → 结束
                   → clarify → 结束
                   → expand → check
                                → 仍不足：insufficient → 结束
```

| 文件 | 职责 |
|---|---|
| rag_api.py | 请求校验、服务启动时准备图、HTTP 错误转换 |
| rag_graph.py | 状态、节点、条件边与两轮上限 |
| router.py | 完整问候语的规则路由 |
| retrieval.py | 加载索引、排列候选、整理带来源的证据 |
| evidence.py | 调用模型判断可回答、需澄清或需补检 |
| generation.py | 根据资料和问题生成答案 |
| citations.py | 检查回答有引用且编号来自本次证据 |
| model_client.py | 密钥读取、HTTP 请求、有限重试 |
| chunk_practice.py | 按示例格式切分多份文档 |
| build_index.py | 保存文档向量、模型名称、源文件指纹 |
| rag_service.py | 保留旧顺序流程及响应格式化函数；API 目前复用 make_result |

main.py 属于原学习记录 API，不是本 RAG 服务入口。

## 现有环境启动（Windows PowerShell）

在项目文件夹中执行命令。已验证环境为 Python 3.12，依赖版本记录在 requirements-rag.txt。运行脚本时直接指定虚拟环境中的 Python，无需先激活。

在项目根目录创建 .env：

```dotenv
DEEPSEEK_API_KEY=替换为自己的密钥
```

.env 已列入 .gitignore，不应提交、分享。若进程中已有同名环境变量，当前 load_dotenv 默认不会覆盖它；排查密钥问题时需同时注意终端环境。

准备资料与索引：

```powershell
.\.venv-rag\Scripts\python.exe chunk_practice.py
.\.venv-rag\Scripts\python.exe build_index.py
```

预期分别生成七块及七块向量。文档或切分结果更新后，重新执行这两步，再重启服务。只改变模型配置时也要保证文档向量和问题向量使用相同模型。

启动：

```powershell
.\.venv-rag\Scripts\python.exe -m uvicorn rag_api:app --host 127.0.0.1 --port 8010
```

等待“RAG 图准备完成”和“Application startup complete.”。保持服务终端打开；Ctrl+C 停止。当前命令未启用自动重载，代码修改后需要重启。

- 健康检查：http://127.0.0.1:8010/health
- 接口文档：http://127.0.0.1:8010/docs

POST /chat 示例：

```json
{"question":"退货商品验收通过后，退款多久退回？"}
```

## 新环境恢复参考

需要事先安装 uv，并确保能下载依赖和公开模型。以下恢复命令尚未在一套全新环境中完整验收：

```powershell
uv venv --python 3.12 .venv-rag
uv pip install --python .\.venv-rag\Scripts\python.exe "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .\.venv-rag\Scripts\python.exe -r requirements-rag.txt
```

先安装记录的 CPU 版 PyTorch，再恢复其余依赖。公开模型首次运行时会下载并缓存至 .model-cache；下载失败应保留完整输出定位，不反复重装。

## 接口含义

HTTP 200 表示流程正常完成，业务结果看 status：DIRECT、ANSWERED、NEED_CLARIFICATION 或 INSUFFICIENT_EVIDENCE。

HTTP 422 为输入校验失败；503 的 MODEL_REQUEST_FAILED 为模型请求失败；502 的 INVALID_MODEL_OUTPUT 为引用校验失败。503 不是政策资料不足。

retrieved_sources 是最后一轮提供的候选资料，不等于答案实际引用。final_answer_usage 只记录最终生成调用，不包括证据检查或可能的重试。

## 检查与评测

在另一个终端、同一项目目录运行。以下模拟检查不发起真实模型请求：

```powershell
.\.venv-rag\Scripts\python.exe check_citations.py
.\.venv-rag\Scripts\python.exe check_model_client.py
.\.venv-rag\Scripts\python.exe check_rag_graph.py
.\.venv-rag\Scripts\python.exe check_rag_api.py
```

端到端评测需要服务运行，会产生真实 API 费用：

```powershell
.\.venv-rag\Scripts\python.exe evaluate_api.py
.\.venv-rag\Scripts\python.exe evaluate_api.py e2e_invoice_cases.json
```

报告按题集命名保存。若本地 HTTP 请求受终端代理影响，评测中的本地请求使用 trust_env=False；这不等于应对 DeepSeek 请求禁用代理。

## 已观察到的结果与限制

七块知识库上的六题退货、四题发票开发检查，本轮自动检查 10/10，内容复核符合预期。I03 曾重复确认已知条件、跨场景套用申请材料；修改生成要求后，本轮未再出现。

这些问题已参与开发调试，不是独立盲测。结果不能表述为整体准确率 100% 或幻觉率为零。独立的十二条政策检索实验也不能替代这里的端到端评测。

当前切分依赖示例格式的空行；新增或调整章节后需维护块编号。索引使用内容指纹检查文件是否变化，但尚未固定模型权重的精确 revision。无跨请求会话记忆、真实订单工具、自动子问题分解、流式输出、上线负载测试；引用编号校验不证明结论被证据支持。检索两轮仍不足只表示当前检索证据不足，不证明整库绝无答案。

学习者已在引导下完成实现和运行，独立解释、定位与修改能力需另行验证。
