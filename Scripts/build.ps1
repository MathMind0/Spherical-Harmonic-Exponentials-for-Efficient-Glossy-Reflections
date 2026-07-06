param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Debug",

    [ValidateSet("x64")]
    [string]$Platform = "x64",

    [switch]$Run,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$SolutionPath = Join-Path $RepoRoot "Build\ImageBasedPBR.sln"

function Find-MSBuild {
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $VsWhere) {
        $Found = & $VsWhere -latest -products * -requires Microsoft.Component.MSBuild -find "MSBuild\Current\Bin\MSBuild.exe" | Select-Object -First 1
        if ($Found) {
            return $Found
        }
    }

    $Command = Get-Command MSBuild.exe -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }

    throw "MSBuild.exe was not found. Install Visual Studio 2022 with the Desktop development with C++ workload, or run this from a Developer PowerShell."
}

$MSBuild = Find-MSBuild
$Target = if ($Clean) { "Clean;Build" } else { "Build" }

Write-Host "MSBuild: $MSBuild"
Write-Host "Solution: $SolutionPath"
Write-Host "Configuration: $Configuration|$Platform"

& $MSBuild $SolutionPath `
    /m `
    /t:$Target `
    /p:Configuration=$Configuration `
    /p:Platform=$Platform

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Run) {
    $ExeName = if ($Configuration -eq "Debug") { "ImageBasedPBRDebug.exe" } else { "ImageBasedPBR.exe" }
    $ExePath = Join-Path $RepoRoot $ExeName

    if (-not (Test-Path $ExePath)) {
        throw "Build succeeded, but $ExePath was not found."
    }

    Write-Host "Starting: $ExePath"
    Start-Process -FilePath $ExePath -WorkingDirectory $RepoRoot
}
