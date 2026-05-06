$ErrorActionPreference = "Stop"

$localRoot = Split-Path -Parent $PSScriptRoot
$remoteRoot = "/datapool/BESTTOOLBOX/L-shaped"

ssh GPU_3 "mkdir -p $remoteRoot"
scp -r `
  "$localRoot\\README.md" `
  "$localRoot\\pyproject.toml" `
  "$localRoot\\requirements-server.txt" `
  "$localRoot\\clients" `
  "$localRoot\\configs" `
  "$localRoot\\docs" `
  "$localRoot\\scripts" `
  "$localRoot\\src" `
  "$localRoot\\third_party" `
  "GPU_3:${remoteRoot}/"

Write-Host "Synced to GPU_3:${remoteRoot}"
