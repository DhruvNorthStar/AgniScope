$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CacheParent = Join-Path $ProjectRoot '.cache\pytest'
New-Item -ItemType Directory -Path $CacheParent -Force | Out-Null
$RunId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
$BaseTemp = Join-Path $CacheParent $RunId
# Always use a new base directory: pytest must never clean up a previous run.
if (Test-Path -LiteralPath $BaseTemp) { throw 'Test directory collision; no files removed.' }
& (Join-Path $PSScriptRoot 'run.ps1') -m pytest -q "--basetemp=$BaseTemp" '--junitxml=outputs/tables/pytest_final.xml'
exit $LASTEXITCODE
