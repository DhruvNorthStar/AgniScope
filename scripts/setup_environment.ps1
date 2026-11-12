$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CondaCommand = Get-Command conda -ErrorAction SilentlyContinue
$CondaExe = if ($CondaCommand) { $CondaCommand.Source } else { Join-Path $env:USERPROFILE 'miniforge3\Scripts\conda.exe' }
if (-not (Test-Path -LiteralPath $CondaExe)) { throw 'Conda executable not found. Install Miniforge/Conda or add it to PATH.' }
$EnvironmentName = 'agniscope-public-source'
$Existing = & $CondaExe env list --json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect Conda environments.' }
if (@($Existing.envs | ForEach-Object { Split-Path $_ -Leaf }) -contains $EnvironmentName) { throw 'The setup environment already exists; inspect it before proceeding.' }
& $CondaExe create --name $EnvironmentName --file (Join-Path $ProjectRoot 'environment.lock.win-64.txt') --yes
if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
& $CondaExe run --name $EnvironmentName python -m pip install -e $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw 'Editable package installation failed.' }
& $CondaExe run --name $EnvironmentName python -m pip install --no-deps --require-hashes --only-binary=:all: -r (Join-Path $ProjectRoot 'dashboard\requirements.lock.txt') -c (Join-Path $ProjectRoot 'dashboard\constraints-scientific.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dashboard dependency installation failed.' }
& $CondaExe run --name $EnvironmentName python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
