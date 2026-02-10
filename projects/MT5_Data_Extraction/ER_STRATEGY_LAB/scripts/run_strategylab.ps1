# =============================================================================
# run_strategylab.ps1 — StrategyLab Runner (TREND v2 + RANGE v1)
# =============================================================================
# Usage:
#   cd C:\Quant
#   .\projects\MT5_Data_Extraction\ER_STRATEGY_LAB\scripts\run_strategylab.ps1
#
# Options (env vars):
#   $env:TREND_M5_RUN_ID = "some_id"    # Force a specific run ID
#   $env:RANGE_M5_RUN_ID = "some_id"
#   $env:STRATEGY = "TREND"              # Run only TREND (default: both)
#   $env:STRATEGY = "RANGE"              # Run only RANGE
#   $env:STRATEGY = "ALL"                # Run both (default)
# =============================================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$NbDir = Join-Path $PSScriptRoot ".." "notebooks"
$NbDir = (Resolve-Path $NbDir).Path

$Strategy = if ($env:STRATEGY) { $env:STRATEGY.ToUpper() } else { "ALL" }

Write-Host "=============================================="
Write-Host "  StrategyLab Runner"
Write-Host "  Project Root: $ProjectRoot"
Write-Host "  Notebooks:    $NbDir"
Write-Host "  Strategy:     $Strategy"
Write-Host "=============================================="

# Check jupyter/papermill
$hasPapermill = $false
try {
    python -c "import papermill" 2>$null
    $hasPapermill = $true
} catch {}

if (-not $hasPapermill) {
    Write-Host ""
    Write-Host "papermill not installed. Install with: pip install papermill"
    Write-Host ""
    Write-Host "Alternative: Open notebooks in Jupyter and run cells manually:"
    Write-Host "  1. cd $NbDir"
    Write-Host "  2. jupyter notebook"
    Write-Host "  3. Open 03_TREND_M5_Strategy_v2.ipynb -> Run All"
    Write-Host "  4. Open 04_RANGE_M5_Strategy_v1.ipynb -> Run All"
    exit 1
}

Push-Location $NbDir

if ($Strategy -eq "ALL" -or $Strategy -eq "TREND") {
    Write-Host ""
    Write-Host ">>> Running TREND v2 ..."
    $TrendNb = Join-Path $NbDir "03_TREND_M5_Strategy_v2.ipynb"
    $TrendOut = Join-Path $NbDir "outputs" "03_TREND_M5_Strategy_v2_output.ipynb"
    New-Item -ItemType Directory -Force -Path (Split-Path $TrendOut) | Out-Null
    python -m papermill $TrendNb $TrendOut --no-progress-bar
    if ($LASTEXITCODE -ne 0) {
        Write-Host "TREND v2 FAILED (exit code $LASTEXITCODE)"
    } else {
        Write-Host "TREND v2 DONE"
    }
}

if ($Strategy -eq "ALL" -or $Strategy -eq "RANGE") {
    Write-Host ""
    Write-Host ">>> Running RANGE v1 ..."
    $RangeNb = Join-Path $NbDir "04_RANGE_M5_Strategy_v1.ipynb"
    $RangeOut = Join-Path $NbDir "outputs" "04_RANGE_M5_Strategy_v1_output.ipynb"
    New-Item -ItemType Directory -Force -Path (Split-Path $RangeOut) | Out-Null
    python -m papermill $RangeNb $RangeOut --no-progress-bar
    if ($LASTEXITCODE -ne 0) {
        Write-Host "RANGE v1 FAILED (exit code $LASTEXITCODE)"
    } else {
        Write-Host "RANGE v1 DONE"
    }
}

Pop-Location

Write-Host ""
Write-Host "=============================================="
Write-Host "  StrategyLab Runner COMPLETE"
Write-Host "=============================================="
