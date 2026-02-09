<#
.SYNOPSIS
    Creates runtime directories that notebooks expect but are NOT in Git.
.DESCRIPTION
    Ensures all data/, outputs/, logs/, artifacts/, bulk_data/, restore/
    directories exist on disk so notebooks don't fail on first run.
    These dirs are in .gitignore — this script recreates them after a fresh clone.
    Run from repo root: .\scripts\workspace_init.ps1
#>

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "$RepoRoot\projects")) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}

Write-Host "Workspace Init — Quant" -ForegroundColor White
Write-Host "Repo root: $RepoRoot`n"

# --- MT5_Data_Extraction ---
$mt5 = Join-Path $RepoRoot "projects\MT5_Data_Extraction"
$mt5Dirs = @(
    "data\bulk_data",
    "data\historical_data",
    "data\processed_data",
    "data\metadata",
    "data\backups",
    "data\restore",
    "data\logs",
    "outputs\er_filter_5m",
    "diagnostics_global",
    "artifacts"
)

# Strategy Lab runtime dirs
$slDirs = @(
    "ER_STRATEGY_LAB\artifacts\features",
    "ER_STRATEGY_LAB\artifacts\deploy",
    "ER_STRATEGY_LAB\artifacts\backtests",
    "ER_STRATEGY_LAB\artifacts\wfo",
    "ER_STRATEGY_LAB\research_logs",
    "ER_STRATEGY_LAB\notebooks\outputs"
)

Write-Host "=== MT5_Data_Extraction ===" -ForegroundColor Cyan
foreach ($d in ($mt5Dirs + $slDirs)) {
    $full = Join-Path $mt5 $d
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
        Write-Host "  CREATED: $d" -ForegroundColor Green
    } else {
        Write-Host "  EXISTS:  $d" -ForegroundColor DarkGray
    }
}

# --- TWF ---
$twf = Join-Path $RepoRoot "projects\TWF"
$twfDirs = @(
    "data",
    "outputs",
    "logs"
)

Write-Host "`n=== TWF ===" -ForegroundColor Cyan
foreach ($d in $twfDirs) {
    $full = Join-Path $twf $d
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
        Write-Host "  CREATED: $d" -ForegroundColor Green
    } else {
        Write-Host "  EXISTS:  $d" -ForegroundColor DarkGray
    }
}

# --- BTC_ANALIST ---
$btc = Join-Path $RepoRoot "projects\BTC_ANALIST"
$btcDirs = @(
    "data\raw",
    "data\processed"
)

Write-Host "`n=== BTC_ANALIST ===" -ForegroundColor Cyan
foreach ($d in $btcDirs) {
    $full = Join-Path $btc $d
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
        Write-Host "  CREATED: $d" -ForegroundColor Green
    } else {
        Write-Host "  EXISTS:  $d" -ForegroundColor DarkGray
    }
}

# --- GESTOR DE IA ---
$gestor = Join-Path $RepoRoot "projects\GESTOR DE IA"
$gestorDirs = @(
    "data",
    "logs",
    "reports"
)

Write-Host "`n=== GESTOR DE IA ===" -ForegroundColor Cyan
foreach ($d in $gestorDirs) {
    $full = Join-Path $gestor $d
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
        Write-Host "  CREATED: $d" -ForegroundColor Green
    } else {
        Write-Host "  EXISTS:  $d" -ForegroundColor DarkGray
    }
}

# --- Root-level dirs ---
$rootDirs = @("_archive", "_inbox")
Write-Host "`n=== Root ===" -ForegroundColor Cyan
foreach ($d in $rootDirs) {
    $full = Join-Path $RepoRoot $d
    if (-not (Test-Path $full)) {
        New-Item -ItemType Directory -Path $full -Force | Out-Null
        Write-Host "  CREATED: $d" -ForegroundColor Green
    } else {
        Write-Host "  EXISTS:  $d" -ForegroundColor DarkGray
    }
}

Write-Host "`nWorkspace init complete." -ForegroundColor Green
Write-Host "Next: run .\scripts\bootstrap.ps1 to create venvs and install dependencies."
