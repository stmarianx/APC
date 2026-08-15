[CmdletBinding()]
param(
    [ValidateSet('core', 'vision', 'transcription', 'all')]
    [string]$Profile = 'vision',
    [string]$VenvPath = '.venv',
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$resolvedVenv = Join-Path $workspaceRoot $VenvPath
$pythonExe = Join-Path $resolvedVenv 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonExe)) {
    py -3.11 -m venv $resolvedVenv
}

if ($SkipInstall) {
    Write-Host "APC environment exists at $resolvedVenv"
    exit 0
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install --editable (Join-Path $workspaceRoot 'coach')

if ($Profile -in @('vision', 'all')) {
    & $pythonExe -m pip install --requirement (
        Join-Path $workspaceRoot 'apc\requirements-dev.txt'
    )
}

if ($Profile -in @('transcription', 'all')) {
    & $pythonExe -m pip install --requirement (
        Join-Path $workspaceRoot 'analysis\requirements-transcription.txt'
    )
}

Write-Host "APC $Profile environment is ready at $resolvedVenv"
Write-Host "Run with: & '$pythonExe' -m pytest apc/tests coach/tests"
