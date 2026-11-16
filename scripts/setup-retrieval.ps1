$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR=Join-Path $projectRoot '.cache/uv'
$env:UV_PYTHON_INSTALL_DIR=Join-Path $projectRoot '.cache/python'
$targetEnv=Join-Path $projectRoot '.venv-retrieval'
if (-not (Test-Path -LiteralPath (Join-Path $targetEnv 'Scripts/python.exe'))) {
    uv venv --python 3.12 $targetEnv
    if ($LASTEXITCODE -ne 0) { throw '无法创建检索环境' }
}
$retrievalPython=Join-Path $targetEnv 'Scripts/python.exe'
uv pip install --python $retrievalPython 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch 安装失败' }
uv pip install --python $retrievalPython -r (Join-Path $projectRoot 'baseline/rag/requirements-rag.txt') --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw '检索依赖安装失败' }
& $retrievalPython -c 'import torch,sentence_transformers,jieba,rank_bm25; from langgraph.types import Send; print("检索环境导入检查通过")'
if ($LASTEXITCODE -ne 0) { throw '检索环境导入检查失败' }
'verified imports' | Set-Content -LiteralPath (Join-Path $targetEnv 'READY') -Encoding utf8
