param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
$AgniScopeRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $AgniScopeRoot
try {
    & (Join-Path $PSScriptRoot 'run.ps1') (Join-Path $PSScriptRoot 'dashboard_server.py') --port $Port
    exit $LASTEXITCODE
}
finally { Pop-Location }
