# StrategyLab v2 — Institutional Ready Report

## DoD Checklist

### A) Motor de Backtest

| # | Requisito | Status | Evidencia |
|---|-----------|--------|-----------|
| A.1 | Cost model: spread + commission + slippage + borrow | PASS | `costs.py:CostsConfig` (4 campos), `roundtrip_cost_dec()`, `compute_fill_cost()`, `stressed_config()` |
| A.2a | Daily loss stop | PASS | `risk.py:DailyRiskTracker.can_trade()` L46: `daily_pnl <= daily_loss_cap` |
| A.2b | DD kill-switch | PASS | `risk.py:DailyRiskTracker.is_killed()` L65: `drawdown() <= max_drawdown_cap` |
| A.2c | Max trades/day | PASS | `risk.py:DailyRiskTracker.can_trade()` L43: `daily_trades >= max_trades_per_day` |
| A.2d | Max positions | PASS | `config.py:RiskConfig.max_positions=1`; engine is single-position by design (pos=0/1/-1) |
| A.2e | Sizing ATR-based + clamp | PASS | `risk.py:compute_position_size()` L8-20: `risk_amount/sl_distance`, clamped `[min, max]` |
| A.3a | Dedup keep=last | PASS | `data_loader.py:load_bars()` L18: `unique(subset=["symbol","time_utc"], keep="last")` |
| A.3b | NaN/None handling | PASS | `backtest_engine.py:_is_finite()` L39-45, used for H/L/O/C/ATR |
| A.3c | UTC timezone | PASS | `data_loader.py` L14: `cast(pl.Datetime("us","UTC"))` |
| A.3d | Temporal sort | PASS | `data_loader.py` L17: `sort(["symbol","time_utc"])` |
| A.4a | SL/TP/Trail consistent | PASS | `backtest_engine.py` L157-178: separate LONG/SHORT logic |
| A.4b | Trail > SL default | PASS | `config.py:EngineConfig` trail_atr=3.0 > sl_atr=2.0 |
| A.4c | Weekend flatten | PASS | `backtest_engine.py` L186: `_is_weekend(dow_list[idx])` |
| A.5 | LONG/SHORT independent thresholds | PASS | `signals_trend_v2.py:compute_regime_gate()` L74-78: `thr_mom_long` vs `thr_mom_short` |

### B) Módulos Institucionales

| # | Requisito | Status | Evidencia |
|---|-----------|--------|-----------|
| B.1a | Señales TREND v2 (pure functions) | PASS | `signals_trend_v2.py`: `add_trend_features()`, `compute_regime_gate()`, `confirm_signals()` |
| B.1b | Señales RANGE v1 (pure functions) | PASS | `signals_range_v1.py`: `add_range_features()`, `compute_range_gate()`, `mean_reversion_signals()` |
| B.1c | Schema validation runtime | PASS | `schema.py:validate_columns()` raises ValueError on missing cols |
| B.2 | Métricas completas | PASS | `metrics.py`: `cagr`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `hit_rate`, `expectancy`, `profit_factor`, `turnover`, `exposure`, `compute_kpis()` |
| B.3a | MC IID bootstrap | PASS | `monte_carlo.py:iid_bootstrap()` L68-99 |
| B.3b | MC block bootstrap | PASS | `monte_carlo.py:block_bootstrap()` L106-144 |
| B.3c | MC stress scenarios | PASS | `monte_carlo.py:stress_test()` L151-209 |
| B.3d | Percentiles + VaR/CVaR | PASS | `monte_carlo.py:_percentiles()` + `var_cvar()` |
| B.3e | Seed controlado | PASS | All MC functions accept `seed` param, use `random.Random(seed)` |
| B.4a | WFO leakage-free splits | PASS | `wfo.py:grid_search()` scores IS only, OOS for eval |
| B.4b | WFO embargo | PASS | `wfo.py:build_wfo_folds()` `embargo_days` param shifts oos_start |
| B.4c | WFO min_folds guard | PASS | `wfo.py:run_wfo()` returns empty if `len(folds) < min_folds` |
| B.4d | Grid search max_combos | PASS | `wfo.py:grid_search()` L113: `combos[:max_combos]` |
| B.4e | IS/OOS KPIs + best params exportable | PASS | `wfo.py:GridResult` has `kpis_is`, `kpis_oos`, `params` |
| B.5a | Parquet/CSV/JSON/MD export | PASS | `report.py`: `export_trades()`, `export_csv()`, `snapshot_json()`, `markdown_report()` |
| B.5b | Run manifest with config hash | PASS | `cli.py:_write_run_manifest()` writes config + SHA256 hash |

### C) CLI

| # | Requisito | Status | Evidencia |
|---|-----------|--------|-----------|
| C.1 | `cli run --config --data --output` | PASS | `cli.py:cmd_run()` + `main()` argparse |
| C.2 | `cli wfo --config --data` | PASS | `cli.py:cmd_wfo()` uses config param_grid |
| C.3 | `cli mc --trades --seed` | PASS | `cli.py:cmd_mc()` |
| C.4 | Demo mode (synthetic) with clear logs | PASS | `[cli] No data file provided, using synthetic bars.` |

### D) Tests + CI

| # | Requisito | Status | Evidencia |
|---|-----------|--------|-----------|
| D.1 | Engine tests | PASS | `test_strategylab_engine.py`: 8 tests |
| D.2 | Risk tests | PASS | `test_strategylab_risk.py`: 14 tests |
| D.3 | MC tests | PASS | `test_strategylab_mc.py`: 11 tests |
| D.4 | WFO tests | PASS | `test_strategylab_wfo.py`: 10 tests |
| D.5 | Guards tests (costs/NaN/dedup/schema/VaR/embargo) | PASS | `test_strategylab_guards.py`: 16 tests |
| D.6 | Ruff clean | PASS | `All checks passed!` |
| D.7 | Pytest clean | PASS | `59 passed in 1.16s` |

### E) Documentación

| # | Requisito | Status | Evidencia |
|---|-----------|--------|-----------|
| E.1 | architecture.md with module map + 3 commands | PASS | `docs/architecture.md` |
| E.2 | Output locations documented | PASS | `docs/architecture.md` "Where Outputs Go" |
| E.3 | Path contracts + env vars | PASS | `docs/architecture.md` "Environment Variables" |
| E.4 | What's NOT versioned | PASS | `docs/architecture.md` ".gitignore" section |
| E.5 | Ready report (this file) | PASS | `docs/READY_REPORT.md` |

## Validation Results

```
ruff check: All checks passed!
pytest:     59 passed in 1.16s
CLI run:    76 trades, outputs/cli_demo/ with run_manifest.json
CLI mc:     3 methods x 100 sims, mc_snapshot.json
```

## Files Created/Modified

### Source (src/strategylab/)
- `__init__.py`, `__main__.py`, `paths.py`, `config.py`, `schema.py`
- `data_loader.py`, `costs.py`, `risk.py`, `execution.py`
- `backtest_engine.py`, `signals_trend_v2.py`, `signals_range_v1.py`
- `metrics.py`, `monte_carlo.py`, `wfo.py`, `report.py`, `cli.py`

### Config
- `configs/strategylab.yaml`

### Tests
- `tests/test_strategylab_engine.py` (8 tests)
- `tests/test_strategylab_risk.py` (14 tests)
- `tests/test_strategylab_mc.py` (11 tests)
- `tests/test_strategylab_wfo.py` (10 tests)
- `tests/test_strategylab_guards.py` (16 tests)

### Docs
- `docs/architecture.md` (rewritten for Python module)
- `docs/READY_REPORT.md` (this file)
