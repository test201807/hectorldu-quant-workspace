# Edge Diagnosis Tool

## Overview

`_edge_diagnosis.py` is a comprehensive diagnostic script that analyzes whether the TREND (NB3) and RANGE (NB4) strategies have exploitable edge by computing forward returns at signal generation.

## Purpose

Instead of looking at backtest results (which can be optimized), this script analyzes raw forward returns from the moment signals fire, providing an unbiased view of whether the strategy signals predict future price movement.

## Usage

```bash
cd C:\Quant\projects\MT5_Data_Extraction
venv1\Scripts\python.exe tools\_edge_diagnosis.py
```

## What It Does

1. **Finds Latest Runs**: Automatically locates the most recent run for each strategy
2. **Loads Data**: Reads signals and features from the latest runs
3. **Computes Forward Returns**: Calculates returns at multiple horizons (1h, 2h, 4h, 8h, 12h, 24h)
4. **Segments Analysis**:
   - By signal side (LONG vs SHORT)
   - By symbol
   - By regime (ER, volatility, momentum)
   - By sample (IS vs OOS)
5. **Statistical Testing**: Bootstrap confidence intervals to confirm edge significance

## Key Metrics

- **Mean Return**: Average forward return at each horizon
- **Median Return**: Median forward return (robust to outliers)
- **% Positive**: Win rate (percentage of signals with positive returns)
- **Sharpe Ratio**: Risk-adjusted return (approximate, assuming independence)
- **Bootstrap CI**: 95% confidence interval for mean return

## Output Structure

The script generates:

1. **Overall Edge Analysis**: Aggregate statistics across all signals
2. **Edge by Side**: LONG vs SHORT performance comparison
3. **Edge by Symbol**: Per-symbol breakdown
4. **Edge by Regime**: Performance in different market conditions
5. **IS vs OOS**: In-sample vs out-of-sample comparison
6. **Bootstrap CI**: Statistical significance testing
7. **Executive Summary**: High-level findings and recommendations

## Interpreting Results

### Positive Edge Indicators
- Mean return > 0 (consistently positive)
- 95% CI excludes zero (statistically significant)
- OOS performance ≥ IS performance (no overfitting)
- Win rate > 50% (better than random)

### Negative Edge Indicators
- Mean return < 0 (losing on average)
- 95% CI excludes zero on negative side (significant losses)
- Win rate < 50% (worse than random)

### No Edge
- Mean return ≈ 0
- 95% CI includes zero (not statistically significant)
- Win rate ≈ 50% (random)

## Forward Return Horizons

The script analyzes returns at 6 different horizons:

| Bars | Time  | Use Case |
|------|-------|----------|
| 12   | 1h    | Intraday scalping |
| 24   | 2h    | Short-term swing |
| 48   | 4h    | Medium-term swing |
| 96   | 8h    | Day trading |
| 144  | 12h   | Overnight holds |
| 288  | 24h   | Daily swing |

## Regime Analysis

The script segments signals by market regime:

1. **ER (Efficiency Ratio)**:
   - Low = ranging market
   - High = trending market

2. **Volatility** (vol_bps_288):
   - Low = calm market
   - High = volatile market

3. **Momentum** (mom_bps_288):
   - Positive = uptrend
   - Negative = downtrend

## Current Results Summary

### TREND Strategy (NB3)
- **Edge**: POSITIVE ✓
- **Best horizon**: 24h (+0.12% mean)
- **Best segment**: LONG on BTCUSD (+0.36% / 24h)
- **Best regime**: High ER (trending) + Mid volatility
- **OOS performance**: Better than IS (robust)

### RANGE Strategy (NB4)
- **Edge**: NEGATIVE ✗
- **Best horizon**: 1h (-0.009% mean)
- **Issue**: Fat left tail (losses larger than wins)
- **OOS performance**: Still negative
- **Status**: Requires redesign

## Customization

You can modify the following parameters at the top of the script:

```python
HORIZONS = [12, 24, 48, 96, 144, 288]  # Forward return horizons
BOOTSTRAP_ITERS = 1000                  # Bootstrap iterations
QUANTILE_TERCILES = [0.0, 0.333, 0.667, 1.0]  # Regime boundaries
```

## Dependencies

- polars >= 1.35
- numpy
- pathlib (standard library)
- datetime (standard library)

## Technical Notes

1. **Forward Returns**: Computed as `(close[t+N] - close[t]) / close[t]`
2. **Signed Returns**: Negated for SHORT signals
3. **Null Handling**: Signals at the end of data with no forward data are excluded
4. **Bootstrap**: 1000 iterations with replacement for CI estimation
5. **Regime Quantiles**: Terciles (33rd, 67th percentiles)

## Files Analyzed

### TREND (NB3)
- `outputs/trend_v2/run_*/signals_all_v2.parquet`
- `outputs/trend_v2/run_*/features_m5_v2.parquet`
- `outputs/trend_v2/run_*/wfo_folds_v2.parquet`

### RANGE (NB4)
- `outputs/range_v1/run_*/signals_all_range_v1.parquet`
- `outputs/range_v1/run_*/features_m5_range_v1.parquet`
- `outputs/range_v1/run_*/wfo_folds_range_v1.parquet`

## Error Handling

The script handles:
- Missing directories gracefully
- Missing files gracefully
- Empty dataframes
- Null values in forward returns
- Edge cases at data boundaries

## Performance

- Runtime: ~30-60 seconds for both strategies
- Memory: ~2GB peak (loads full feature sets)
- Output: Console only (pipe to file if needed)

## Future Enhancements

Potential additions:
1. Export results to parquet/CSV
2. Generate plots (equity curves by regime)
3. Add more regime definitions
4. Include transaction cost impact
5. Multi-strategy comparison matrix
6. Time-series decomposition of edge

## Author

Created for the MT5_Data_Extraction project
Last updated: 2026-02-18
