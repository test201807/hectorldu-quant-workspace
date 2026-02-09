<#
.SYNOPSIS
    Creates virtual environments and installs dependencies for all projects.
.DESCRIPTION
    For each project under projects/, creates a .venv (or project-specific venv)
    and runs pip install -r requirements.txt.
    Run from repo root: .\scripts\bootstrap.ps1
.PARAMETER Project
    Optional: bootstrap only a specific project (e.g. TWF, BTC_ANALIST)
#>
param(
    [string]$Project = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "$RepoRoot\projects")) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}

$Projects = @(
    @{ Name = "MT5_Data_Extraction"; Venv = "venv1" },
    @{ Name = "TWF";                 Venv = ".venv" },
    @{ Name = "BTC_ANALIST";         Venv = ".venv" },
    @{ Name = "GESTOR DE IA";        Venv = ".venv" }
)

function Bootstrap-Project {
    param($Proj)
    $projDir = Join-Path $RepoRoot "projects" $Proj.Name
    $venvDir = Join-Path $projDir $Proj.Venv
    $reqFile = Join-Path $projDir "requirements.txt"

    Write-Host "`n=== $($Proj.Name) ===" -ForegroundColor Cyan

    if (-not (Test-Path $reqFile)) {
        Write-Host "  SKIP: no requirements.txt found" -ForegroundColor Yellow
        return
    }

    # Create venv if it doesn't exist
    if (-not (Test-Path (Join-Path $venvDir "Scripts\python.exe"))) {
        Write-Host "  Creating venv at $($Proj.Venv)..."
        python -m venv $venvDir
    } else {
        Write-Host "  Venv already exists at $($Proj.Venv)"
    }

    # Install deps
    $pip = Join-Path $venvDir "Scripts\pip.exe"
    Write-Host "  Installing dependencies..."
    & $pip install -r $reqFile --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK" -ForegroundColor Green
    } else {
        Write-Host "  FAILED (exit code $LASTEXITCODE)" -ForegroundColor Red
    }
}

Write-Host "Bootstrap — Quant Workspace" -ForegroundColor White
Write-Host "Repo root: $RepoRoot"

if ($Project -ne "") {
    $match = $Projects | Where-Object { $_.Name -eq $Project }
    if ($null -eq $match) {
        Write-Host "Unknown project: $Project" -ForegroundColor Red
        Write-Host "Available: $($Projects.Name -join ', ')"
        exit 1
    }
    Bootstrap-Project $match
} else {
    foreach ($p in $Projects) {
        Bootstrap-Project $p
    }
}

Write-Host "`nDone." -ForegroundColor Green
