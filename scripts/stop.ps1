$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$pidFile=Join-Path $projectRoot 'runtime/server.json'
if (-not (Test-Path -LiteralPath $pidFile)) { Write-Output '没有本项目启动记录。'; exit 0 }
$record=Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
$process=Get-Process -Id $record.pid -ErrorAction SilentlyContinue
if ($process) {
    if ($process.StartTime.Ticks -ne $record.started_ticks -or $process.Path -ne $record.executable) { throw '进程身份已变化，不停止不相关进程。' }
    Stop-Process -Id $process.Id
}
Remove-Item -LiteralPath $pidFile
Write-Output '已停止本项目服务；下次启动会标记中断状态，可续跑的任务由用户手动继续。'
