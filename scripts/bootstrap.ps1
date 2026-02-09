<#
.SYNOPSIS
    Bootstrap dev environment: detect Python, create per-project venvs, install deps.
.DESCRIPTION
    Each project gets its own .venv because dependency stacks conflict:
      - MT5 needs MetaTrader5 (Windows-only binary)
      - GESTOR needs openai + simfin
      - TWF needs ruptures
      - BTC needs scikit-learn + yfinance
    A single root venv would risk version conflicts across ~40 packages.

    Run from repo root:
      .\scripts\bootstrap.ps1              # all projects
      .\scripts\bootstrap.ps1 -Project TWF # single project
.PARAMETER Project
    Optional. Bootstrap only this project. Valid: MT5_Data_Extraction, TWF, BTC_ANALIST, "GESTOR DE IA"
#>
param(
    [string]$Project = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:Failures = 0

# ── Resolve repo root ──────────────────────────────────────────────
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot   = Split-Path -Parent $ScriptDir
if (-not (Test-Path (Join-Path $RepoRoot "projects"))) {
    Write-Host "ERROR: Cannot find projects/ under $RepoRoot" -ForegroundColor Red
    exit 1
}

# ── Detect Python ──────────────────────────────────────────────────
function Find-Python {
    # Try py launcher first (standard on Windows)
    try {
        $pyInfo = & py -0p 2>&1 | Select-String -Pattern "3\.\d+" | Select-Object -First 1
        if ($pyInfo) {
            $pyPath = ($pyInfo -split "\s+")[-1]
            if (Test-Path $pyPath) {
                Write-Host "  Found Python via py launcher: $pyPath" -ForegroundColor DarkGray
                return "py -3"
            }
        }
    } catch {}

    # Fallback: python on PATH
    try {
        $ver = & python --version 2>&1
        if ($ver -match "Python 3\.") {
            $loc = (Get-Command python).Source
            Write-Host "  Found Python on PATH: $loc ($ver)" -ForegroundColor DarkGray
            return "python"
        }
    } catch {}

    # Fallback: python3 on PATH
    try {
        $ver = & python3 --version 2>&1
        if ($ver -match "Python 3\.") {
            return "python3"
        }
    } catch {}

    return $null
}

# ── Project definitions ────────────────────────────────────────────
$Projects = @(
    @{ Name = "MT5_Data_Extraction"; Venv = ".venv" }
    @{ Name = "TWF";                 Venv = ".venv" }
    @{ Name = "BTC_ANALIST";         Venv = ".venv" }
    @{ Name = "GESTOR DE IA";        Venv = ".venv" }
)

# ── Bootstrap one project ─────────────────────────────────────────
function Bootstrap-Project {
    param(
        [hashtable]$Proj,
        [string]$PythonCmd
    )
    $projDir  = Join-Path $RepoRoot "projects" $Proj.Name
    $venvDir  = Join-Path $projDir $Proj.Venv
    $reqFile  = Join-Path $projDir "requirements.txt"
    $pipExe   = Join-Path $venvDir "Scripts\pip.exe"
    $pythonExe = Join-Path $venvDir "Scripts\python.exe"

    Write-Host "`n=== $($Proj.Name) ===" -ForegroundColor Cyan

    # Check project dir exists
    if (-not (Test-Path $projDir)) {
        Write-Host "  SKIP: directory not found ($projDir)" -ForegroundColor Yellow
        return
    }

    # Check requirements.txt
    if (-not (Test-Path $reqFile)) {
        Write-Host "  SKIP: no requirements.txt" -ForegroundColor Yellow
        return
    }

    # Create venv if needed
    if (-not (Test-Path $pythonExe)) {
        Write-Host "  Creating venv..."
        try {
            if ($PythonCmd -eq "py -3") {
                & py -3 -m venv $venvDir
            } else {
                & $PythonCmd -m venv $venvDir
            }
        } catch {
            Write-Host "  FAILED to create venv: $_" -ForegroundColor Red
            $script:Failures++
            return
        }
        if (-not (Test-Path $pythonExe)) {
            Write-Host "  FAILED: venv created but python.exe not found" -ForegroundColor Red
            $script:Failures++
            return
        }
        Write-Host "  Venv created at $($Proj.Venv)\" -ForegroundColor Green
    } else {
        $ver = & $pythonExe --version 2>&1
        Write-Host "  Venv exists ($ver)" -ForegroundColor DarkGray
    }

    # Upgrade pip quietly
    & $pythonExe -m pip install --upgrade pip --quiet 2>$null

    # Install requirements
    Write-Host "  Installing dependencies from requirements.txt..."
    & $pipExe install -r $reqFile --quiet 2>&1 | ForEach-Object {
        if ($_ -match "ERROR|Could not") { Write-Host "    $_" -ForegroundColor Red }
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED: pip install exited with code $LASTEXITCODE" -ForegroundColor Red
        $script:Failures++
        return
    }

    # Verify: count installed packages
    $pkgCount = (& $pipExe list --format=columns 2>$null | Measure-Object -Line).Lines - 2
    Write-Host "  OK ($pkgCount packages installed)" -ForegroundColor Green
}

# ── Main ───────────────────────────────────────────────────────────
Write-Host "======================================" -ForegroundColor White
Write-Host "  Quant Workspace — Bootstrap" -ForegroundColor White
Write-Host "======================================" -ForegroundColor White
Write-Host "Repo root : $RepoRoot"

# Detect Python
Write-Host "`nDetecting Python..." -ForegroundColor White
$PythonCmd = Find-Python
if ($null -eq $PythonCmd) {
    Write-Host "ERROR: Python 3 not found. Install from https://python.org" -ForegroundColor Red
    exit 1
}

# Run
if ($Project -ne "") {
    $match = $Projects | Where-Object { $_.Name -eq $Project }
    if ($null -eq $match) {
        Write-Host "ERROR: Unknown project '$Project'" -ForegroundColor Red
        Write-Host "Valid projects: $($Projects | ForEach-Object { $_.Name } | Join-String -Separator ', ')"
        exit 1
    }
    Bootstrap-Project -Proj $match -PythonCmd $PythonCmd
} else {
    foreach ($p in $Projects) {
        Bootstrap-Project -Proj $p -PythonCmd $PythonCmd
    }
}

# Summary
Write-Host "`n======================================" -ForegroundColor White
if ($script:Failures -gt 0) {
    Write-Host "  DONE with $($script:Failures) failure(s)" -ForegroundColor Red
    exit 1
} else {
    Write-Host "  ALL OK" -ForegroundColor Green
    exit 0
}
