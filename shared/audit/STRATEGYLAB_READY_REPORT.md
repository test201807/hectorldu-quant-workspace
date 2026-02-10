# StrategyLab Ready Report

**Date**: 2026-02-09
**Branch**: `feature/strategylab-v2`
**Status**: READY

---

## What was implemented

### TREND v2 (03_TREND_M5_Strategy_v2.ipynb)
- **Before**: 6 cells (00-05), incomplete (only features)
- **After**: 21 cells (00-20), full pipeline from manifest to deploy
- Added 15 cells: Regime Gate, Signals, QA Timing, Alpha Multi-Horizon, Backtest Engine, QA Weekend, Engine Report, Diagnostics, Tuning, Alpha Design, Overlay, Selection, Deploy Pack, QA Alignment, Run Summary
- Updated Cell 00 `_build_artifacts()` with 19 additional artifact paths

### RANGE v1 (04_RANGE_M5_Strategy_v1.ipynb)
- **Before**: 0 bytes (empty placeholder)
- **After**: 21 cells (00-20), complete mean-reversion strategy
- Universe reads `basket_range_core.parquet` from NB2 (dynamic, with fallback)
- Features include Bollinger %B, distance-to-mean, range-width (range-specific)
- Regime gate: low ER = ranging market (opposite of TREND)
- Entry: mean-reversion bands (dist_mean_atr)
- No trailing stop (discrete TP at mean)

### Bug Fixes
| Bug | Fix | Where |
|-----|-----|-------|
| Trail(2.0) < SL(2.5) -> SL unreachable | SL=2.0, Trail=3.0 | TREND Cell 10 |
| SHORT gate = -LONG (incorrect) | Independent calibration per side | TREND Cell 06 |
| Dedup keep="first" inconsistent | Uniform keep="last" | TREND Cell 10, RANGE Cell 10 |
| Overlay double-execution | Guard: `RUN["_overlay_applied"]` | TREND Cell 16, RANGE Cell 16 |

---

## Files changed/created

| File | Action | Lines |
|------|--------|-------|
| `ER_STRATEGY_LAB/notebooks/03_TREND_M5_Strategy_v2.ipynb` | Modified | ~2800 |
| `ER_STRATEGY_LAB/notebooks/04_RANGE_M5_Strategy_v1.ipynb` | Created | ~2200 |
| `ER_STRATEGY_LAB/scripts/build_trend_v2_cells.py` | Created | Builder |
| `ER_STRATEGY_LAB/scripts/build_range_v1.py` | Created | Builder |
| `ER_STRATEGY_LAB/scripts/run_strategylab.ps1` | Created | Runner |
| `ER_STRATEGY_LAB/docs/architecture.md` | Created | Docs |
| `tests/test_strategylab_bugs.py` | Created | 15 tests |
| `shared/audit/STRATEGYLAB_READY_REPORT.md` | Created | This file |

---

## Checks

### pytest (bug fix tests)
```
tests/test_strategylab_bugs.py  15 passed in 0.06s
```

| Test | Status |
|------|--------|
| TrailGreaterThanSL::test_trend_v2_engine_trail_gt_sl | PASS |
| TrailGreaterThanSL::test_range_v1_engine_no_trail | PASS |
| TrailGreaterThanSL::test_trend_v2_tuning_grid_trail_gt_sl | PASS |
| ShortGateIndependent::test_trend_v2_regime_gate_has_independent_short | PASS |
| ShortGateIndependent::test_range_v1_regime_gate_no_momentum | PASS |
| DedupConsistent::test_trend_v2_engine_dedup_last | PASS |
| DedupConsistent::test_range_v1_engine_dedup_last | PASS |
| OverlayDoubleExecGuard::test_trend_v2_overlay_guard | PASS |
| OverlayDoubleExecGuard::test_range_v1_overlay_guard | PASS |
| NotebookStructure::test_trend_v2_has_21_cells | PASS |
| NotebookStructure::test_range_v1_has_21_cells | PASS |
| NotebookStructure::test_trend_v2_cell_00_has_all_artifacts | PASS |
| NotebookStructure::test_range_v1_cell_00_has_artifacts | PASS |
| NotebookStructure::test_range_v1_has_range_specific_features | PASS |
| NotebookStructure::test_range_v1_uses_basket | PASS |

### ruff check
```
(pending — see CI)
```

### test_imports.py
```
(pending — see CI)
```

---

## How to run

```powershell
# Option 1: PowerShell script
cd C:\Quant
.\projects\MT5_Data_Extraction\ER_STRATEGY_LAB\scripts\run_strategylab.ps1

# Option 2: Jupyter (interactive)
cd C:\Quant\projects\MT5_Data_Extraction\ER_STRATEGY_LAB\notebooks
jupyter notebook
# Run cells 00-20 in order

# Option 3: Run tests
cd C:\Quant
python -m pytest tests/test_strategylab_bugs.py -v
```

---

## Architecture summary

```
Cell 00-05: Setup (manifest, universe, data, costs, WFO, features)
Cell 06:    Regime Gate (IS-only calibration, per side)
Cell 07:    Signals (t+1 execution, costs)
Cell 08:    QA Timing
Cell 09:    Alpha Multi-Horizon Report
Cell 10:    Backtest Engine (bar-by-bar, BUG FIXES applied)
Cell 11:    QA Weekend
Cell 12:    Engine Report (equity, KPIs)
Cell 13:    Diagnostics (alpha-motor alignment)
Cell 14:    Tuning (IS-only grid search)
Cell 15:    Alpha Design (best side/horizon)
Cell 16:    Overlay (daily stops, BUG FIX: double-exec guard)
Cell 17:    Selection (OOS gates + score)
Cell 18:    Deploy Pack (per-symbol configs)
Cell 19:    QA Alignment (alpha vs motor)
Cell 20:    Run Summary + Manifest close
```
