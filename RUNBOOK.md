# Runbook — Quant Workspace

How to run each project without dirtying Git.

## First-time setup (after clone)

```powershell
cd C:\Quant

# 1. Create runtime directories (data/, outputs/, logs/, etc.)
.\scripts\workspace_init.ps1

# 2. Create virtual environments and install dependencies
.\scripts\bootstrap.ps1

# 3. (Optional) Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## MT5_Data_Extraction

### Prerequisites
- MetaTrader 5 terminal running and logged in
- venv1 activated

### Run sequence
```powershell
cd C:\Quant\projects\MT5_Data_Extraction
.\venv1\Scripts\Activate.ps1

# Step 1: Data extraction (requires MT5 connection)
jupyter notebook 01_MT5_DE_5M_V1.ipynb

# Step 2: ER Filter (uses output from Step 1)
jupyter notebook 02_ER_FILTER_5M_V4.ipynb

# Step 3: Strategy Lab (uses output from Step 2)
cd ER_STRATEGY_LAB\notebooks
jupyter notebook 03_TREND_M5_Strategy_v2.ipynb
```

### What gets created (all ignored by Git)
- `data/bulk_data/m5_raw/` — Hive-partitioned M5 candles
- `data/historical_data/m5_clean/` — Cleaned M5 data
- `data/metadata/` — Manifests, config snapshots
- `outputs/er_filter_5m/<RUN_ID>/` — ER filter results per run
- `ER_STRATEGY_LAB/artifacts/` — Features, signals, deploy configs

### Gotchas
- NB2 auto-detects PROJECT_ROOT by looking for `bulk_data/` directory. If missing, run `workspace_init.ps1`.
- Each ER filter run creates an isolated `<RUN_ID>/` folder. Old runs are safe to delete.

## TWF

```powershell
cd C:\Quant\projects\TWF
.\.venv\Scripts\Activate.ps1
jupyter notebook A_E_P_v1.ipynb
```

### What gets created (all ignored by Git)
- `outputs/{symbol}/{timeframe}/` — Stats, figures, deliverables
- `logs/` — Run logs

### Gotchas
- Uses `runtime_defaults.json` for `n_jobs: 8`. Lower this on machines with fewer cores.
- Data comes from Binance public API (no key needed). Requires internet.

## BTC_ANALIST

```powershell
cd C:\Quant\projects\BTC_ANALIST
.\.venv\Scripts\Activate.ps1
jupyter notebook notebooks\BTC_ANALIST_v1.ipynb
```

### What gets created (all ignored by Git)
- `data/raw/` — Downloaded BTC price + on-chain data
- `data/processed/` — Feature-engineered datasets

### Gotchas
- Downloads data from yfinance + blockchain APIs. Requires internet.

## GESTOR DE IA

### Prerequisites
- `.env` file with `OPENAI_API_KEY=sk-...` (and optionally `SIMFIN_API_KEY`)
- Create `.env` from scratch — it's in `.gitignore` and never committed

```powershell
cd "C:\Quant\projects\GESTOR DE IA"
.\.venv\Scripts\Activate.ps1
jupyter notebook notebooks\GESTOR.ipynb
```

### What gets created (all ignored by Git)
- `data/` — SIMFIN + yfinance downloads
- `logs/` — LLM call logs
- `reports/` — Portfolio recommendations

### Gotchas
- Hardcoded `PROJECT_ROOT = Path(r"C:\Quant\GESTOR DE IA")`. Works via junction.
- OpenAI API calls cost money. Check `configs/runtime.yaml` for model/parameters.

## Git hygiene

After running notebooks:
```powershell
cd C:\Quant
git status   # Should show nothing — all outputs are ignored
git diff     # Should be empty unless you edited code
```

If `git status` shows unexpected untracked files, add them to `.gitignore` before committing.

**Never commit**: `.parquet`, `.csv`, `.zip`, `.pkl`, `.env`, `data/`, `outputs/`, `venv*/`.
