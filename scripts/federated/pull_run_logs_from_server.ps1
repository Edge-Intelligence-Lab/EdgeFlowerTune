param(
  [string]$RunId = "",
  [switch]$Latest
)

$ErrorActionPreference = "Stop"

$localRoot = Split-Path -Parent $PSScriptRoot
$remoteRunsRoot = "/datapool/BESTTOOLBOX/L-shaped/outputs/runs"
$localRunsRoot = Join-Path $localRoot "outputs\\runs"
New-Item -ItemType Directory -Force -Path $localRunsRoot | Out-Null

if ($Latest) {
  $RunId = ssh GPU_3 "python3 - <<'PY'
from pathlib import Path
root = Path('$remoteRunsRoot')
runs = sorted([p.name for p in root.iterdir() if p.is_dir()], reverse=True)
print(runs[0] if runs else '')
PY"
  $RunId = $RunId.Trim()
}

if (-not $RunId) {
  throw "Provide -RunId or use -Latest"
}

$remotePath = "GPU_3:${remoteRunsRoot}/${RunId}"
$localPath = Join-Path $localRunsRoot $RunId
if (Test-Path $localPath) {
  Remove-Item -Recurse -Force $localPath
}

scp -r $remotePath $localRunsRoot
Write-Host "Pulled run logs to $localPath"
