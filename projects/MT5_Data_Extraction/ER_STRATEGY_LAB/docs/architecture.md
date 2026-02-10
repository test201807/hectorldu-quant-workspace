# StrategyLab Architecture

## Overview

StrategyLab contains two strategy notebooks that consume the output of the
ER_FILTER_5M pipeline (NB1 + NB2):

| Notebook | Strategy | Version | Cells |
|----------|----------|---------|-------|
| `03_TREND_M5_Strategy_v2.ipynb` | Trend-following (momentum breakout) | v2.0.1 | 21 |
| `04_RANGE_M5_Strategy_v1.ipynb` | Mean-reversion (Bollinger bands) | v1.0.0 | 21 |

## Cell Pipeline (both notebooks follow the same structure)

```
Cell 00: Run Manifest + Paths + Artifacts
Cell 01: Universe + Instrument Specs
Cell 02: Load M5 OHLCV + QA
Cell 03: Cost Model (base/stress)
Cell 04: WFO Builder (IS=18m, OOS=3m, >=6 folds)
Cell 05: Feature Engineering
Cell 06: Regime Gate (IS-only calibration)
Cell 07: Signal Generation + t+1 Execution + Costs
Cell 08: QA Timing Trades
Cell 09: Alpha Multi-Horizon Report
Cell 10: Backtest Engine (bar-by-bar simulation)
Cell 11: QA Weekend Entries
Cell 12: Engine Report (Equity + KPIs + Exit Reasons)
Cell 13: Diagnostics (Edge Alignment)
Cell 14: Engine Tuning (IS-only grid search)
Cell 15: Alpha Design (IS-only side+horizon selection)
Cell 16: Execution & Risk Overlay
Cell 17: Institutional Selection (OOS gates + score)
Cell 18: Deploy Pack (per-symbol JSON configs)
Cell 19: QA Alpha-Motor Alignment
Cell 20: Run Summary + Manifest Final
```

## Key Differences: TREND vs RANGE

| Aspect | TREND v2 | RANGE v1 |
|--------|----------|----------|
| Universe | 4 hardcoded symbols | basket_range_core.parquet (NB2) |
| Regime gate | ER >= threshold (trending) | ER <= threshold (ranging) |
| Entry signal | Momentum breakout | Mean-reversion (dist_mean_atr bands) |
| SL | 2.0x ATR | 1.5x ATR |
| TP | 5.0x ATR | 2.0x ATR |
| Trailing stop | 3.0x ATR | None |
| Time stop | 288 bars (1 day) | 144 bars (12h) |
| Entry confirm | 12 bars | 6 bars |
| Cooldown | 24 bars | 12 bars |
| Extra features | EMA cross, trend_slope | Bollinger %B, dist_mean_atr, range_width_atr |

## Data Flow

```
NB1 (01_MT5_DE_5M_V1.ipynb)
  -> bulk_data/ (raw M5 parquets)

NB2 (02_ER_FILTER_5M_V2.ipynb)
  -> processed_data/ (m5_clean, features, baskets)
  -> outputs/er_filter_5m/ (basket_range_core.parquet, etc.)

StrategyLab TREND v2
  <- processed_data/m5_clean/ OR ohlcv_clean_m5.parquet
  -> outputs/trend_m5_strategy/v2/run_<ID>/

StrategyLab RANGE v1
  <- processed_data/m5_clean/
  <- outputs/er_filter_5m/basket_range_core.parquet
  -> outputs/range_m5_strategy/v1/run_<ID>/
```

## Bug Fixes (v2.0.1)

1. **Trail > SL**: TRAIL_ATR=3.0 > SL_ATR=2.0 (was 2.0 < 2.5, making SL unreachable)
2. **SHORT gate independent**: thr_mom_short calibrated separately using negative percentiles
3. **Dedup keep="last"**: Uniform dedup strategy (was mixed first/last)
4. **Overlay double-exec guard**: Prevents Cell 16 from running twice in same session

## How to Run

### Option A: Jupyter (interactive)
```bash
cd C:\Quant\projects\MT5_Data_Extraction\ER_STRATEGY_LAB\notebooks
jupyter notebook
# Open and run all cells in order
```

### Option B: PowerShell script
```powershell
cd C:\Quant
.\projects\MT5_Data_Extraction\ER_STRATEGY_LAB\scripts\run_strategylab.ps1
```

### Option C: Papermill (headless)
```bash
pip install papermill
cd C:\Quant\projects\MT5_Data_Extraction\ER_STRATEGY_LAB\notebooks
python -m papermill 03_TREND_M5_Strategy_v2.ipynb outputs/trend_output.ipynb
python -m papermill 04_RANGE_M5_Strategy_v1.ipynb outputs/range_output.ipynb
```

## Outputs

All outputs go to `notebooks/outputs/<strategy>/v<N>/run_<ID>/`:
- `run_manifest_*.json` — Run metadata and artifact paths
- `*_v2.parquet` / `*_range_v1.parquet` — Data artifacts
- `*_snapshot_*.json` — Human-readable snapshots
- `deploy/` — Per-symbol deployment configs
