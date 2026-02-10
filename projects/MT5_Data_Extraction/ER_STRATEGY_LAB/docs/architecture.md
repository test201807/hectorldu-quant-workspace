# StrategyLab Architecture

## Overview

StrategyLab is an institutional-grade backtest framework implemented as a Python
package under `src/strategylab/` with CLI, YAML config, and full test suite.

It also includes two strategy notebooks that consume the output of the
ER_FILTER_5M pipeline (NB1 + NB2).

## Python Module (`src/strategylab/`)

### Module Map

| Module | Purpose |
|--------|---------|
| `config.py` | Dataclasses (CostsConfig, RiskConfig, EngineConfig, WFOConfig, MCConfig) + YAML/JSON loading |
| `schema.py` | Canonical column schemas + `validate_columns()` runtime check |
| `data_loader.py` | Parquet loading with UTC cast, dedup (keep=last), sort + synthetic bar generator |
| `costs.py` | Cost model: spread + commission + slippage + borrow, stressed config |
| `risk.py` | Position sizing (ATR-based, clamped), DailyRiskTracker (daily caps, DD kill-switch) |
| `execution.py` | Fill model with slippage |
| `backtest_engine.py` | Bar-by-bar engine: SL/TP/trail, gate confirmation, time stop, regime-off, weekend flatten |
| `signals_trend_v2.py` | TREND signals: EMA features, regime gate (independent LONG/SHORT thresholds) |
| `signals_range_v1.py` | RANGE signals: Bollinger %B, mean-reversion bands, ranging regime gate |
| `metrics.py` | CAGR, Sharpe, Sortino, Calmar, MDD, hit rate, expectancy, profit factor, turnover |
| `monte_carlo.py` | 3 MC methods (IID bootstrap, block bootstrap, stress) + VaR/CVaR |
| `wfo.py` | Walk-forward: leakage-free splits with embargo, grid search, min_folds guard |
| `report.py` | Export: parquet, CSV, JSON snapshot, markdown report |
| `cli.py` | CLI entry point: `run`, `wfo`, `mc` subcommands |
| `paths.py` | Path resolution via env vars |

### How to Run (3 commands)

```bash
cd C:\Quant\projects\MT5_Data_Extraction\ER_STRATEGY_LAB

# 1. Single backtest (synthetic demo)
PYTHONPATH=src python -m strategylab.cli run --output outputs/demo_run

# 2. Walk-forward optimization (synthetic demo)
PYTHONPATH=src python -m strategylab.cli wfo --output outputs/demo_wfo

# 3. Monte Carlo simulation (demo returns)
PYTHONPATH=src python -m strategylab.cli mc --n-sims 1000 --output outputs/demo_mc
```

With real data:
```bash
PYTHONPATH=src python -m strategylab.cli run \
    --config configs/strategylab.yaml \
    --data ../../processed_data/m5_clean/ohlcv_clean_m5.parquet \
    --symbol EURUSD \
    --output outputs/real_run
```

### Where Outputs Go

All CLI outputs go to `outputs/<run_name>/`:
- `run_manifest.json` — Full config snapshot + hash + timestamp
- `report_trades.parquet` / `.csv` — Trade-level data
- `report_snapshot.json` — KPI summary
- `report_summary.md` — Human-readable report
- `wfo_snapshot.json` — WFO best params per fold (wfo command)
- `mc_snapshot.json` — MC percentiles (mc command)

### What is NOT versioned (in .gitignore)

- `outputs/` — All CLI and notebook outputs
- `artifacts/` — Intermediate artifacts
- `logs/` — Logs
- `data/`, `bulk_data/`, `processed_data/` — All data files
- `*.parquet`, `*.csv`, `*.pkl` — Data files anywhere
- `__pycache__/`, `.pytest_cache/` — Python cache
- `venv*/`, `.venv/` — Virtual environments

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MT5_PROJECT_ROOT` | Override project root detection | Auto-detect |
| `STRATEGY_LAB_ROOT` | Override StrategyLab root | `<project_root>/ER_STRATEGY_LAB` |
| `STRATEGYLAB_OUTPUTS_ROOT` | Override outputs directory | `<lab_root>/outputs` |

### Config File

Default config: `configs/strategylab.yaml`

Key parameters:
- **Engine**: sl_atr=2.0, tp_atr=5.0, trail_atr=3.0 (trail > SL enforced)
- **Risk**: risk_per_trade=1%, daily_loss_cap=-2%, max_dd_cap=-15%, max_trades/day=3
- **WFO**: IS=18m, OOS=3m, embargo=5d, min_folds=6, max_combos=100
- **MC**: n_sims=1000, block_size=20, stress_cost_factor=2x

## Notebooks

| Notebook | Strategy | Cells |
|----------|----------|-------|
| `03_TREND_M5_Strategy_v2.ipynb` | Trend-following (momentum breakout) | 21 |
| `04_RANGE_M5_Strategy_v1.ipynb` | Mean-reversion (Bollinger bands) | 21 |

### Key Differences: TREND vs RANGE

| Aspect | TREND v2 | RANGE v1 |
|--------|----------|----------|
| Regime gate | ER >= threshold (trending) | ER <= threshold (ranging) |
| Entry signal | Momentum breakout | Mean-reversion (dist_mean_atr bands) |
| SL / TP / Trail | 2.0 / 5.0 / 3.0 ATR | 1.5 / 2.0 / None ATR |
| Time stop | 288 bars (1 day) | 144 bars (12h) |
| Confirm / Cooldown | 12 / 24 bars | 6 / 12 bars |

## Bug Fixes (v2.0.1)

1. **Trail > SL**: TRAIL_ATR=3.0 > SL_ATR=2.0 (was 2.0 < 2.5, SL unreachable)
2. **SHORT gate independent**: thr_mom_short calibrated separately
3. **Dedup keep="last"**: Uniform in data_loader + engine
4. **Overlay double-exec guard**: Prevents re-execution in notebooks

## Tests

```bash
# Run full suite (59 tests, ~1s)
python -m pytest tests/ -v

# Lint
python -m ruff check src/ tests/
```

| File | Tests | Covers |
|------|-------|--------|
| `test_strategylab_engine.py` | 8 | Engine basic, fields, no-lookahead, exit reasons |
| `test_strategylab_risk.py` | 14 | Sizing, daily caps, DD kill, reset |
| `test_strategylab_mc.py` | 11 | IID, block, stress, determinism, percentiles |
| `test_strategylab_wfo.py` | 10 | Folds, grid search, trail>SL skip, full WFO |
| `test_strategylab_guards.py` | 16 | Costs, NaN, dedup, schema, VaR/CVaR, embargo |
