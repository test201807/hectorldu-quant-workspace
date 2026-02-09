<#
.SYNOPSIS
    Smoke-test imports for each project's venv.
.DESCRIPTION
    Activates each project's .venv and verifies that:
      1) Core third-party dependencies import correctly
      2) Project src/ modules import correctly (if they have code)
      3) All .py files compile without syntax errors

    Run from repo root: .\scripts\smoke_imports.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:Pass = 0
$script:Fail = 0

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = Split-Path -Parent $ScriptDir

Write-Host "==========================================" -ForegroundColor White
Write-Host "  Quant Workspace — Import Smoke Tests" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor White

function Test-Import {
    param(
        [string]$PythonExe,
        [string]$Label,
        [string]$Code
    )
    $result = & $PythonExe -c $Code 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    PASS  $Label" -ForegroundColor Green
        $script:Pass++
    } else {
        Write-Host "    FAIL  $Label" -ForegroundColor Red
        # Extract the actual error line
        $errLine = ($result | Select-String "ModuleNotFoundError|ImportError" | Select-Object -First 1)
        if ($errLine) {
            Write-Host "          $errLine" -ForegroundColor DarkRed
            $mod = if ($errLine -match "'([^']+)'") { $Matches[1] } else { "unknown" }
            Write-Host "          Fix: pip install $mod" -ForegroundColor Yellow
        } else {
            $result | Select-Object -Last 3 | ForEach-Object {
                Write-Host "          $_" -ForegroundColor DarkRed
            }
        }
        $script:Fail++
    }
}

# ── MT5_Data_Extraction ────────────────────────────────────────────
$mt5Python = Join-Path $RepoRoot "projects\MT5_Data_Extraction\.venv\Scripts\python.exe"
Write-Host "`n=== MT5_Data_Extraction ===" -ForegroundColor Cyan
if (Test-Path $mt5Python) {
    Test-Import $mt5Python "polars"      "import polars; print(f'polars {polars.__version__}')"
    Test-Import $mt5Python "pyarrow"     "import pyarrow; print(f'pyarrow {pyarrow.__version__}')"
    Test-Import $mt5Python "numpy"       "import numpy; print(f'numpy {numpy.__version__}')"
    Test-Import $mt5Python "pandas"      "import pandas; print(f'pandas {pandas.__version__}')"
    Test-Import $mt5Python "scipy"       "import scipy; print(f'scipy {scipy.__version__}')"
    Test-Import $mt5Python "matplotlib"  "import matplotlib; print(f'matplotlib {matplotlib.__version__}')"
    Test-Import $mt5Python "httpx"       "import httpx; print(f'httpx {httpx.__version__}')"
    Test-Import $mt5Python "yaml"        "import yaml; print(f'PyYAML {yaml.__version__}')"
} else {
    Write-Host "  SKIP: .venv not found. Run bootstrap.ps1 first." -ForegroundColor Yellow
}

# ── TWF ────────────────────────────────────────────────────────────
$twfPython = Join-Path $RepoRoot "projects\TWF\.venv\Scripts\python.exe"
Write-Host "`n=== TWF ===" -ForegroundColor Cyan
if (Test-Path $twfPython) {
    Test-Import $twfPython "numpy"       "import numpy; print(f'numpy {numpy.__version__}')"
    Test-Import $twfPython "scipy"       "import scipy; print(f'scipy {scipy.__version__}')"
    Test-Import $twfPython "polars"      "import polars; print(f'polars {polars.__version__}')"
    Test-Import $twfPython "statsmodels" "import statsmodels; print(f'statsmodels {statsmodels.__version__}')"
    Test-Import $twfPython "ruptures"    "import ruptures; print(f'ruptures {ruptures.__version__}')"
    Test-Import $twfPython "httpx"       "import httpx; print(f'httpx {httpx.__version__}')"
    Test-Import $twfPython "matplotlib"  "import matplotlib; print(f'matplotlib {matplotlib.__version__}')"
    # src/ modules (real code)
    $twfDir = Join-Path $RepoRoot "projects\TWF"
    Test-Import $twfPython "twf.utils.config"   "import sys; sys.path.insert(0, r'$twfDir\src'); from twf.utils.config import *; print('twf.utils.config OK')"
    Test-Import $twfPython "twf.utils.logging"  "import sys; sys.path.insert(0, r'$twfDir\src'); from twf.utils.logging import *; print('twf.utils.logging OK')"
    Test-Import $twfPython "twf.io.binance"     "import sys; sys.path.insert(0, r'$twfDir\src'); from twf.io.binance import get_klines; print('twf.io.binance OK')"
} else {
    Write-Host "  SKIP: .venv not found. Run bootstrap.ps1 first." -ForegroundColor Yellow
}

# ── BTC_ANALIST ────────────────────────────────────────────────────
$btcPython = Join-Path $RepoRoot "projects\BTC_ANALIST\.venv\Scripts\python.exe"
Write-Host "`n=== BTC_ANALIST ===" -ForegroundColor Cyan
if (Test-Path $btcPython) {
    Test-Import $btcPython "pandas"       "import pandas; print(f'pandas {pandas.__version__}')"
    Test-Import $btcPython "numpy"        "import numpy; print(f'numpy {numpy.__version__}')"
    Test-Import $btcPython "sklearn"      "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
    Test-Import $btcPython "scipy"        "import scipy; print(f'scipy {scipy.__version__}')"
    Test-Import $btcPython "matplotlib"   "import matplotlib; print(f'matplotlib {matplotlib.__version__}')"
    Test-Import $btcPython "seaborn"      "import seaborn; print(f'seaborn {seaborn.__version__}')"
    Test-Import $btcPython "yfinance"     "import yfinance; print(f'yfinance {yfinance.__version__}')"
} else {
    Write-Host "  SKIP: .venv not found. Run bootstrap.ps1 first." -ForegroundColor Yellow
}

# ── GESTOR DE IA ───────────────────────────────────────────────────
$gestorPython = Join-Path $RepoRoot "projects\GESTOR DE IA\.venv\Scripts\python.exe"
Write-Host "`n=== GESTOR DE IA ===" -ForegroundColor Cyan
if (Test-Path $gestorPython) {
    Test-Import $gestorPython "openai"       "import openai; print(f'openai {openai.__version__}')"
    Test-Import $gestorPython "dotenv"       "import dotenv; print(f'python-dotenv {dotenv.__version__}')"
    Test-Import $gestorPython "pandas"       "import pandas; print(f'pandas {pandas.__version__}')"
    Test-Import $gestorPython "polars"       "import polars; print(f'polars {polars.__version__}')"
    Test-Import $gestorPython "numpy"        "import numpy; print(f'numpy {numpy.__version__}')"
    Test-Import $gestorPython "sklearn"      "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
    Test-Import $gestorPython "matplotlib"   "import matplotlib; print(f'matplotlib {matplotlib.__version__}')"
} else {
    Write-Host "  SKIP: .venv not found. Run bootstrap.ps1 first." -ForegroundColor Yellow
}

# ── Syntax check: compile all .py ──────────────────────────────────
Write-Host "`n=== Syntax Check (py_compile) ===" -ForegroundColor Cyan
$pyFiles = Get-ChildItem -Path (Join-Path $RepoRoot "projects"), (Join-Path $RepoRoot "shared") `
    -Filter "*.py" -Recurse -File |
    Where-Object { $_.FullName -notmatch '[\\/](\.venv|venv1|btc_env|__pycache__)[\\/]' }

$syntaxFail = 0
foreach ($f in $pyFiles) {
    $result = & python -m py_compile $f.FullName 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    FAIL  $($f.FullName -replace [regex]::Escape($RepoRoot), '.')" -ForegroundColor Red
        Write-Host "          $result" -ForegroundColor DarkRed
        $syntaxFail++
        $script:Fail++
    }
}
if ($syntaxFail -eq 0) {
    Write-Host "    PASS  All $($pyFiles.Count) .py files compile OK" -ForegroundColor Green
    $script:Pass++
}

# ── Summary ────────────────────────────────────────────────────────
Write-Host "`n==========================================" -ForegroundColor White
Write-Host "  Results: $($script:Pass) PASS / $($script:Fail) FAIL" -ForegroundColor $(if ($script:Fail -gt 0) { "Red" } else { "Green" })
Write-Host "==========================================" -ForegroundColor White

if ($script:Fail -gt 0) {
    exit 1
} else {
    exit 0
}
