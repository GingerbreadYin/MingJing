param([switch]$NoBrowser)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$runtimeDir=Join-Path $projectRoot 'runtime'
$pythonExe=Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw '项目环境缺失，请先按 README 安装。' }
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$pidFile=Join-Path $runtimeDir 'server.json'
$url='http://127.0.0.1:8026/api/rag-optimizer/ui'
try {
    $existing=Invoke-RestMethod -Uri 'http://127.0.0.1:8026/api/rag-optimizer/status' -TimeoutSec 2
    if ($existing.root -eq $projectRoot) {
        Write-Output $url
        if (-not $NoBrowser) { Start-Process $url }
        exit 0
    }
    throw '8026 端口已由另一个项目占用。'
} catch [System.Net.Http.HttpRequestException] { }
catch [System.Threading.Tasks.TaskCanceledException] { }
$env:PYTHONPATH=$projectRoot
$env:PYTHONIOENCODING='utf-8'
$process=Start-Process -FilePath $pythonExe -ArgumentList @('-m','optimizer.launch') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDir 'server.stdout.log') -RedirectStandardError (Join-Path $runtimeDir 'server.stderr.log')
@{pid=$process.Id;started_ticks=$process.StartTime.Ticks;executable=$pythonExe;root=$projectRoot} | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding utf8
for ($attempt=0;$attempt -lt 45;$attempt++) {
    Start-Sleep -Seconds 1
    if ($process.HasExited) { throw '服务启动失败，请查看 runtime/server.stderr.log。' }
    try {
        $status=Invoke-RestMethod -Uri 'http://127.0.0.1:8026/api/rag-optimizer/status' -TimeoutSec 1
        if ($status.root -eq $projectRoot) {
            Write-Output $url
            if (-not $NoBrowser) { Start-Process $url }
            exit 0
        }
    } catch { }
}
throw '启动超过等待时间，请查看日志；未自动重复启动。'
