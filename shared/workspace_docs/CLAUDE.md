# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Quant is a multi-project quantitative analysis workspace (Python, notebooks). It contains four independent sub-projects under `projects/`, each with its own virtual environment and data pipeline. All interactive work happens in Jupyter notebooks with supporting library code in `src/` directories.

**Remote**: `origin` → `https://github.com/test201807/hectorldu-quant-workspace.git` (private)
**Branch**: `main`

## Workspace Structure

```
C:\Quant\
├── .gitignore              # Monorepo gitignore (data, outputs, venvs, binaries excluded)
├── WORKSPACE_INDEX.md       # Project index, entry points, execution order
├── projects/                # All projects live here (real paths)
│   ├── MT5_Data_Extraction/ # Primary pipeline: data engine + ER filter + strategy lab
│   ├── TWF/                 # Statistical microstructure analysis
│   ├── BTC_ANALIST/         # Bitcoin cycle analysis
│   └── GESTOR DE IA/        # AI portfolio management
├── shared/
│   ├── audit/               # 00_AUDIT_REPORT.md + _audit_tools/ (3 scripts)
│   └── workspace_docs/      # This file (CLAUDE.md)
├── _archive/                # Backups, legacy artifacts, migration logs
│   ├── backup_20260207_1435/ # Pre-migration snapshot
│   └── legacy/              # Old 99_archive contents
└── Junctions (backward compat):
    MT5_Data_Extraction → projects/MT5_Data_Extraction
    TWF                 → projects/TWF
    BTC_ANALIST         → projects/BTC_ANALIST
    GESTOR DE IA        → projects/GESTOR DE IA
```

### What's in Git (62 files, ~1.4 MiB)
Notebooks, Python source, configs (YAML/JSON), path contracts, docs, requirements.txt.

### What's NOT in Git (on disk only, ignored)
All `data/`, `outputs/`, `artifacts/`, `bulk_data/`, `logs/`, `restore/`, `backups/`, `diagnostics_global/`, virtual environments, `.env`, binary docs (PDF/DOCX), `_archive/`, `_inbox/`.

## Projects

### MT5_Data_Extraction (primary, largest)
Multi-stage pipeline for MetaTrader 5 market data processing and strategy research:
- **01_MT5_DE_5M_V1.ipynb** — Extracts 5M candles from MT5 broker, cleans data, builds metadata. Outputs to `data/bulk_data/m5_raw/` and `data/historical_data/m5_clean/` as Hive-partitioned parquet (`symbol=<SYM>/year=<YYYY>/month=<MM>/`).
- **02_ER_FILTER_5M_V4.ipynb** — Economic regime (ER) filter: computes ER/PD metrics, classifies TREND/RANGE events, scores and ranks symbols, produces basket exports. Each run writes to `outputs/er_filter_5m/<RUN_ID>/` (RUN_ID = `YYYYMMDD_HHMMSS` UTC).
- **ER_STRATEGY_LAB/notebooks/** — Strategy design notebooks (03_TREND_M5, 04_RANGE_M5) that consume ER filter outputs.

Notebooks 01 and 02 have **formal path contracts** (CLOSED documents): `NOTEBOOK1_PATH_CONTRACT.md` and `NOTEBOOK2_PATH_CONTRACT_CLOSED.md`. These are the single source of truth for all path conventions and must not be contradicted.

### TWF
Statistical pipeline for market microstructure analysis (NO trading). Pulls real OHLCV data from Binance public KLINES endpoint (closed candles only). Organized as an 18-cell sequential pipeline (`scripts/cell_00_setup.py` …). Output tree: `outputs/{symbol}/{timeframe}/{stats,figures,deliverables,logs}`.

### BTC_ANALIST
Bitcoin market cycle analysis using on-chain indicators (MVRV, profit addresses). Main logic lives in `notebooks/BTC_ANALIST_v1.ipynb`; `src/` modules are scaffolding stubs.

### GESTOR DE IA
AI-driven portfolio management. Single notebook `notebooks/GESTOR.ipynb`. Uses GPT model via OpenAI API + SIMFIN for financial data. Config in `configs/runtime.yaml` (monthly rebalance, 15% target vol, 25% max weight, 5 bps tcost). API keys in `.env` (never commit).

## Environment Setup

Each project has its own `.venv`. Activate before running:

```powershell
# Example for TWF
cd C:\Quant\projects\TWF
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

All four projects have `requirements.txt`. Core stacks:
- **MT5_Data_Extraction**: polars, pyarrow, numpy, scipy, statsmodels, matplotlib
- **TWF**: numpy, scipy, polars, pyarrow, statsmodels, ruptures, httpx, matplotlib, joblib, tqdm
- **BTC_ANALIST**: pandas, pyarrow, numpy, scikit-learn, scipy, statsmodels, matplotlib, seaborn, yfinance
- **GESTOR DE IA**: openai, python-dotenv, simfin, yfinance, pandas, polars, pyarrow, numpy, scikit-learn

## Execution Order (Core Pipeline)

```
1) 01_MT5_DE_5M_V1.ipynb       ← run from projects/MT5_Data_Extraction/
2) 02_ER_FILTER_5M_V4.ipynb    ← run from projects/MT5_Data_Extraction/
3) Strategy Lab notebooks       ← run from projects/MT5_Data_Extraction/ER_STRATEGY_LAB/notebooks/
     03_TREND_M5_Strategy_v1/v2
     04_RANGE_M5_Strategy_v1
```

Rule: **always `cd` to the project root before running notebooks**.

## Key Architectural Patterns

### Path Resolution
- **MT5 Notebook 1:** `DATA_ROOT = <PROJECT_ROOT>/data`. Override via `MT5_DE_DATA_ROOT` env var.
- **MT5 Notebook 2:** Auto-detects `PROJECT_ROOT` by walking up from `cwd()` looking for marker directories (`bulk_data/`, `processed_data/`, `outputs/`). Reads `data/metadata/config_snapshot.json` for path overrides if it exists.
- **TWF:** Paths managed via `src/twf/utils/config.py`.

### Junctions (Backward Compatibility)
Root-level Windows junctions (`mklink /J`) point from `C:\Quant\<project>` to `C:\Quant\projects\<project>`. This ensures hardcoded paths in notebooks (e.g., `Path(r"C:\Quant\MT5_Data_Extraction\data")`) continue working. **Do not remove junctions** without first updating all hardcoded references.

Known hardcoded paths:
- `01_MT5_DE_5M_V1.ipynb`: 7 instances of `C:\Quant\MT5_Data_Extraction\data` (fallback in `globals().get()`)
- `GESTOR.ipynb`: 16 instances of `C:\Quant\GESTOR DE IA` (PROJECT_ROOT assignments)

### Data Flow (MT5_Data_Extraction)
```
MT5 broker → 01_MT5_DE → m5_raw/ → m5_clean/ → metadata/
                                                    ↓
                              02_ER_FILTER ← config_snapshot.json
                                    ↓
                         outputs/er_filter_5m/<RUN_ID>/
                              {metrics, events, scores, baskets, exports}/
                                    ↓
                         ER_STRATEGY_LAB notebooks
```

Downstream notebooks resolve `DATA_ROOT` once and read `metadata/config_snapshot.json` as the canonical path source. `metadata/m5_manifest.parquet` (per-file detail) and `metadata/manifest.json` (run-level manifest) are distinct artifacts.

### M5 Input Resolution (Notebook 02)
The ER filter selects its M5 data source in this order:
1. `M5_CLEAN_DIR` (preferred)
2. `M5_RAW_DIR` (fallback)
3. Restore: `data/restore/<restore_id>/historical_data/m5_clean/*.parquet` (last resort)

### Fallback Mode (Notebook 02)
If `GLOBAL_STATE["paths"]` can't be resolved (partial/isolated execution), outputs may land in `PROJECT_ROOT/artifacts/` instead of the standard `OUT_ROOT`. This mode is not recommended for official runs.

### Run Isolation
Each ER filter run is fully isolated under `outputs/er_filter_5m/<RUN_ID>/` with subdirectories: `logs/`, `diagnostics/`, `metrics/`, `events/`, `stability/`, `scores/`, `baskets/`, `exports/`, `reports/`. Files listed in the path contracts are immutable — do not rename or restructure.

### Config Precedence (ER Filter)
1. `config/er_filter_5m.json` — master config (persistent, auto-generated with conservative defaults if missing)
2. `data/metadata/config_snapshot.json` — optional overrides from data engine
3. Autogenerated watchlist/params in `diagnostics/` if `ea_watchlist` / `ea_params` files are absent

### Parallelism
TWF uses `runtime_defaults.json` (`n_jobs: 8`) for joblib parallelization.

## Data Formats
- All tabular data uses **Parquet** (via polars/pyarrow), Hive-partitioned where applicable
- Exports also written as CSV for EA (Expert Advisor) consumption
- Metadata/config as JSON; run logs as JSONL
- Correlation matrices as CSV (`processed_data/corr_matrix_5m.csv`); built atomically via temp file rename

## Audit Tools
`shared/audit/_audit_tools/` contains `quant_audit_local.py` (inventory), `quant_audit_openai.py` (AI analysis), and `quant_audit_full.ps1` (PowerShell). Outputs go to `_audit_out/`.

## Important Conventions
- All data is real market data (never simulated)
- Binance endpoint is public REST (no API key needed); OpenAI and SIMFIN keys go in `.env`
- Notebooks use explicit cell-level print/log statements for traceability
- TWF logging goes through `src/twf/utils/logging.py` (CSV-based event logger)
- The workspace is written in Spanish (variable names, comments, documentation)
- **Git rule**: only code, configs, and docs are tracked. Data is always regenerable from the pipeline.
- **Git workflow**: `git add <files> && git commit -m "msg" && git push` — upstream is `origin/main`.
- **Backups**: bundle backups in `C:\Backups\` (recoverable via `git clone <bundle>`)
