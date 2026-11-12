param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArguments)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Prefix = Join-Path $ProjectRoot '.conda\agniscope'
$PythonExe = Join-Path $Prefix 'python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) { throw 'Create the AgniScope environment first.' }
$env:PATH = "$Prefix;$Prefix\Library\bin;$Prefix\Scripts;" + $env:PATH
$env:CONDA_PREFIX = $Prefix
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = Join-Path $ProjectRoot '.cache\tmp'
$env:TMP = $env:TEMP
$env:GDAL_PAM_ENABLED = 'NO'
$env:GDAL_DATA = Join-Path $Prefix 'Library\share\gdal'
$env:PROJ_DATA = Join-Path $Prefix 'Library\share\proj'
$env:PROJ_NETWORK = 'OFF'
& $PythonExe -B @PythonArguments
exit $LASTEXITCODE
