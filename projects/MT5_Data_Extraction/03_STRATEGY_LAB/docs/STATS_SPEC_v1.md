# Institutional Statistical Suite -- Specification v1.0

**Target**: BTCUSD LONG Trend-Following M5 | Prop-Firm Challenge ($25k)
**Date**: 2026-02-19
**Author**: @stats-institutional

---

## Table of Contents

- [0. Data Contract and Input Schema](#0-data-contract-and-input-schema)
- [A. Investable Core Metrics](#a-investable-core-metrics)
- [B. Significance Tests](#b-significance-tests)
- [C. Monte Carlo Simulation](#c-monte-carlo-simulation)
- [D. Alpha Decay Analysis](#d-alpha-decay-analysis)
- [E. Overfitting / WFO Analysis](#e-overfitting--wfo-analysis)
- [F. Cost Sensitivity](#f-cost-sensitivity)
- [G. Thresholds Summary Table](#g-thresholds-summary-table)
- [H. Output Schema](#h-output-schema)
- [I. Library Dependencies](#i-library-dependencies)
- [J. Implementation Notes and Gotchas](#j-implementation-notes-and-gotchas)

---

## 0. Data Contract and Input Schema

### 0.1 Primary Input: `trades_engine_v2.parquet`

Produced by NB3 Celda 10. Each row is one completed trade.

| Column            | Type     | Description                                             |
|-------------------|----------|---------------------------------------------------------|
| `symbol`          | Utf8     | e.g. "BTCUSD"                                          |
| `fold_id`         | Utf8     | e.g. "F00" .. "F09"                                    |
| `segment`         | Utf8     | "IS" or "OOS"                                          |
| `side`            | Utf8     | "LONG" or "SHORT"                                      |
| `signal_time_utc` | Datetime | Bar where signal was confirmed                         |
| `entry_time_utc`  | Datetime | Execution bar = signal + 1                             |
| `exit_time_utc`   | Datetime | Exit bar                                               |
| `entry_price`     | Float64  | Entry fill price                                       |
| `exit_price`      | Float64  | Exit fill price                                        |
| `gross_pnl`       | Float64  | Fractional: `sign * (exit_price / entry_price - 1)`    |
| `net_pnl_base`    | Float64  | `gross_pnl - cost_base_dec` (base cost scenario)       |
| `net_pnl_stress`  | Float64  | `gross_pnl - cost_stress_dec` (stressed cost scenario) |
| `hold_bars`       | Int64    | Number of M5 bars held                                 |
| `exit_reason`     | Utf8     | SL, TP, TRAIL, TIME, REGIME_OFF, WEEKEND, DD_KILL      |
| `pos_size`        | Float64  | Risk-based position sizing multiplier                  |

**Filter for this suite**: Only rows where `segment == "OOS"` and `side == "LONG"` and `symbol == "BTCUSD"`.

### 0.2 Secondary Input: `tuning_results_v2.parquet`

Produced by NB3 Celda 14. Each row is one parameter-combo result scored on IS data.

| Column      | Type    | Description                                         |
|-------------|---------|-----------------------------------------------------|
| `symbol`    | Utf8    |                                                     |
| `fold_id`   | Utf8    |                                                     |
| `sl_atr`    | Float64 | SL multiplier for this combo                        |
| `tp_atr`    | Float64 | TP multiplier                                       |
| `trail_atr` | Float64 | Trail multiplier (0 = no trail)                     |
| `time_stop` | Int64   | Time stop in bars                                   |
| `min_hold`  | Int64   | Min hold bars                                       |
| `n_trades`  | Int64   | IS trade count                                      |
| `sum_ret`   | Float64 | `sum(net_pnl_base)` on IS                           |
| `std_ret`   | Float64 | `std(net_pnl_base)` on IS                           |
| `mean_ret`  | Float64 | `mean(net_pnl_base)` on IS                          |
| `score`     | Float64 | `sum_ret / max(1e-12, std_ret)` (Sharpe-like on IS) |

### 0.3 Derived: Daily Returns Construction

The statistical suite operates primarily on **daily returns**, not per-trade returns.

```
INPUT:
    oos_trades      -- sorted by exit_time_utc
    pos_notional    -- USD notional per trade
    CAPITAL = 25_000

STEP 1 -- Compute pos_notional:
    sl_trades    = oos_trades.filter(exit_reason == "SL")
    sl_ret_median = median(abs(net_pnl_base)) of sl_trades
    sl_ret_median = max(sl_ret_median, 1e-8)
    pos_notional  = RISK_PER_TRADE_USD / sl_ret_median
    # RISK_PER_TRADE_USD = 75 (current NB3 Celda 16 setting)

STEP 2 -- Per-trade USD PnL:
    For each trade:
        pnl_usd = net_pnl_base * pos_notional   # (or net_pnl_stress)

STEP 3 -- Aggregate to daily:
    trade_date = exit_time_utc.date()
    daily_pnl_usd[date] = sum(pnl_usd) for all trades exiting on that date

STEP 4 -- Build equity curve and fractional daily returns:
    equity[0] = CAPITAL
    For each date in calendar-day order within OOS window:
        if date is a trading_day:
            daily_return[t] = daily_pnl_usd[date] / equity[t-1]
            equity[t]       = equity[t-1] + daily_pnl_usd[date]
        else:
            daily_return[t] = 0.0     # no position, no PnL
            equity[t]       = equity[t-1]
```

**Note on zero-return days**: Non-trading days within the OOS window are included as `daily_return = 0.0`. This is necessary for correct annualization. The OOS window spans `min(entry_time_utc).date()` to `max(exit_time_utc).date()`.

### 0.4 Annualization Convention

| Parameter        | Value | Rationale                                              |
|------------------|-------|--------------------------------------------------------|
| `ANNUAL_FACTOR`  | 365   | BTCUSD trades 24/7, 365 days/year.                    |
| `sqrt_annual`    | sqrt(365) = 19.105 | For Sharpe/Sortino annualization.          |
| `BARS_PER_DAY`   | 288   | 24h * 60min / 5min = 288 M5 bars per calendar day.    |

The existing `metrics.py` uses `BARS_PER_YEAR_5M = 252 * 288` (equity trading convention). This suite replaces that with `365 * 288` for crypto, but since all ratios are computed from daily returns (not per-bar), the factor is simply `sqrt(365)`.

### 0.5 Challenge Parameters (Prop-Firm)

| Parameter               | Value    | Source          |
|-------------------------|----------|-----------------|
| `CAPITAL`               | $25,000  | NB3 Celda 16    |
| `DAILY_MAX_LOSS`        | $1,250   | NB3 Celda 16    |
| `TOTAL_MAX_LOSS`        | $2,500   | NB3 Celda 16    |
| `PROFIT_TARGET`         | $1,250   | NB3 Celda 16    |
| `MIN_TRADING_DAYS`      | 2        | NB3 Celda 16    |
| `RISK_PER_TRADE_USD`    | $75      | NB3 Celda 16    |

---

## A. Investable Core Metrics

All metrics computed **twice**: once with `net_pnl_base` (suffix `_base`), once with `net_pnl_stress` (suffix `_stress`).

### A.1 Count and Coverage

| Metric          | Formula                                                         | Edge Case                                           |
|-----------------|-----------------------------------------------------------------|-----------------------------------------------------|
| `n_trades`      | `len(oos_trades)`                                               | If 0: all other metrics = NaN                       |
| `trading_days`  | `n_unique(exit_time_utc.date())`                                | 0 if no trades                                      |
| `calendar_days` | `(max(exit_time_utc) - min(entry_time_utc)).days + 1`           | 0 if no trades                                      |
| `exposure_pct`  | `sum(hold_bars) * 100 / total_bars_in_oos_window`               | `total_bars_in_oos_window` = `calendar_days * 288`. If 0: return 0. |

### A.2 Return Metrics

| Metric                  | Formula                                            | Edge Case                                 |
|-------------------------|----------------------------------------------------|-------------------------------------------|
| `total_pnl_usd`        | `equity[-1] - CAPITAL`                             | From daily equity curve.                  |
| `total_return_pct`      | `total_pnl_usd / CAPITAL * 100`                    |                                           |
| `CAGR`                  | `(equity[-1] / CAPITAL) ^ (365 / calendar_days) - 1` | If `calendar_days == 0`: return 0. If `equity[-1] <= 0`: return -1. |
| `mean_return_per_trade` | `mean(net_pnl_base)` (fractional)                  | If `n_trades == 0`: 0.                    |
| `mean_return_daily`     | `mean(daily_returns)` (fractional)                 | Includes zero-return non-trading days.    |
| `std_per_trade`         | `std(net_pnl_base, ddof=1)`                        | If `n < 2`: NaN.                          |
| `std_daily`             | `std(daily_returns, ddof=1)`                       | If `n < 2`: NaN.                          |

### A.3 Risk-Adjusted Ratios

**Sharpe Ratio** (annualized, from daily returns, Rf = 0):

```python
def sharpe_annual(daily_returns: np.ndarray) -> float:
    n = len(daily_returns)
    if n < 2:
        return float('nan')
    mu = daily_returns.mean()
    sigma = daily_returns.std(ddof=1)
    if sigma == 0.0:
        return 0.0 if mu == 0 else float('inf') * np.sign(mu)
    return (mu / sigma) * np.sqrt(365)
```

**Sortino Ratio** (annualized, from daily returns, target = 0):

```python
def sortino_annual(daily_returns: np.ndarray) -> float:
    n = len(daily_returns)
    if n < 2:
        return float('nan')
    mu = daily_returns.mean()
    # Downside deviation: sqrt of mean of squared negative deviations from 0
    # Uses ALL observations, not just negative ones (Sortino & Price 1994)
    downside_sq = np.minimum(daily_returns, 0.0) ** 2
    downside_dev = np.sqrt(downside_sq.mean())
    if downside_dev == 0.0:
        return float('inf') if mu > 0 else 0.0
    return (mu / downside_dev) * np.sqrt(365)
```

Implementation note: the denominator uses `mean(min(r_i, 0)^2)` over ALL N observations (not just the negative subset). This avoids sample-size bias from conditioning on sign.

**Calmar Ratio**:

```python
def calmar(cagr: float, max_dd_pct: float) -> float:
    # max_dd_pct is negative, e.g. -0.08
    if max_dd_pct == 0.0:
        return 0.0
    return cagr / abs(max_dd_pct)
```

### A.4 Drawdown Metrics

**MaxDD** (on daily cumulative USD equity curve):

```python
def max_drawdown(equity_curve: np.ndarray) -> tuple[float, float]:
    """Returns (max_dd_pct, max_dd_usd), both negative."""
    if len(equity_curve) < 2:
        return 0.0, 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd_usd = equity_curve - peak
    dd_pct = np.where(peak > 0, dd_usd / peak, 0.0)
    return float(dd_pct.min()), float(dd_usd.min())
```

**Ulcer Index**:

```python
def ulcer_index(equity_curve: np.ndarray) -> float:
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd_pct = np.where(peak > 0, (equity_curve - peak) / peak * 100, 0.0)
    return float(np.sqrt(np.mean(dd_pct ** 2)))
```

### A.5 Distribution Metrics (on daily returns)

**Skewness** (Fisher, bias-corrected):

```python
from scipy.stats import skew
skewness = skew(daily_returns, bias=False)
# If n < 3: return NaN
```

**Excess Kurtosis** (Fisher, bias-corrected):

```python
from scipy.stats import kurtosis
excess_kurt = kurtosis(daily_returns, bias=False)  # excess by default
# If n < 4: return NaN
```

**Tail Ratio**:

```python
def tail_ratio(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 20:
        return float('nan')
    p95 = np.percentile(daily_returns, 95)
    p05 = np.percentile(daily_returns, 5)
    if abs(p05) < 1e-12:
        return float('inf') if p95 > 0 else 0.0
    return abs(p95) / abs(p05)
```

Interpretation: > 1.0 means right tail is fatter than left (favorable for longs).

### A.6 VaR and CVaR (Historical, Non-Parametric)

Computed on daily returns (fractional). **No Gaussian assumption.**

```python
def var_cvar_historical(daily_returns: np.ndarray, alpha: float = 0.05):
    """
    Returns (VaR, CVaR) as fractional daily returns (negative = loss).
    alpha=0.05 => 95% confidence level.
    """
    if len(daily_returns) < 20:
        return float('nan'), float('nan')
    sorted_r = np.sort(daily_returns)  # ascending
    n = len(sorted_r)
    var_idx = int(np.floor(n * alpha))
    VaR = sorted_r[var_idx]
    tail = sorted_r[:var_idx + 1]
    CVaR = tail.mean() if len(tail) > 0 else VaR
    return float(VaR), float(CVaR)
```

USD equivalents:

```python
VaR_95_usd  = VaR_95 * CAPITAL   # approximate, using starting capital
CVaR_95_usd = CVaR_95 * CAPITAL
```

### A.7 Trade-Level Metrics

| Metric                    | Formula                                                      | Edge Case                         |
|---------------------------|--------------------------------------------------------------|-----------------------------------|
| `profit_factor`           | `sum(pnl_usd where pnl > 0) / abs(sum(pnl_usd where pnl < 0))` | No losses: +inf. No wins: 0.  |
| `win_rate`                | `count(pnl > 0) / n_trades`                                 | n=0: 0.                          |
| `payoff_ratio`            | `mean(pnl_usd where pnl > 0) / abs(mean(pnl_usd where pnl < 0))` | No losses: +inf. No wins: 0. |
| `expectancy_usd`          | `mean(net_pnl_base * pos_notional)`                         | n=0: 0.                          |
| `expectancy_frac`         | `mean(net_pnl_base)`                                        | n=0: 0.                          |
| `avg_duration_hours`      | `mean(hold_bars) * 5 / 60`                                  | n=0: 0.                          |
| `avg_duration_bars`       | `mean(hold_bars)`                                            | n=0: 0.                          |
| `max_consecutive_losses`  | Longest streak of `pnl < 0` in chronological order           | n=0: 0.                          |
| `max_consecutive_wins`    | Longest streak of `pnl > 0` in chronological order           | n=0: 0.                          |

---

## B. Significance Tests

### B.1 t-Test with Newey-West HAC Standard Errors

**Hypothesis**: H0: `mean(daily_return) <= 0` vs H1: `mean(daily_return) > 0` (one-sided).

```python
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def hac_ttest(daily_returns: np.ndarray) -> dict:
    T = len(daily_returns)
    if T < 10:
        return {"status": "INSUFFICIENT_DATA", "p_value": float('nan'),
                "t_stat": float('nan'), "n_obs": T}

    # Intercept-only OLS: r_t = alpha + epsilon_t
    X = add_constant(np.zeros(T))  # only intercept column
    model = OLS(daily_returns, X[:, :1]).fit(
        cov_type='HAC',
        cov_kwds={
            'maxlags': int(np.floor(4 * (T / 100) ** (2 / 9))),
            'kernel': 'bartlett',
        }
    )
    t_stat = float(model.tvalues[0])
    p_two = float(model.pvalues[0])

    # One-sided p-value
    if t_stat > 0:
        p_one = p_two / 2
    else:
        p_one = 1.0 - p_two / 2

    return {
        "t_stat": t_stat,
        "p_value": p_one,
        "n_obs": T,
        "hac_max_lag": int(np.floor(4 * (T / 100) ** (2 / 9))),
        "kernel": "bartlett",
        "low_power": T < 30,
    }
```

**Lag selection**: Newey-West (1994) automatic plug-in: `max_lag = floor(4 * (T/100)^(2/9))`. With T=8 this gives max_lag=2. With T=60 this gives max_lag=3.

**Thresholds**:

| p-value (one-sided) | Verdict |
|----------------------|---------|
| p < 0.05             | PASS    |
| 0.05 <= p < 0.10    | WARN    |
| p >= 0.10            | FAIL    |

Report `"LOW_POWER"` flag whenever T < 30.

### B.2 Bootstrap Confidence Intervals

**Method**: Stationary Bootstrap (Politis & Romano 1994).

**Why stationary over fixed-block**: Fixed-block bootstrap uses constant-length blocks, creating artificial discontinuities at block boundaries. The stationary bootstrap draws blocks of geometrically-distributed random length (mean = `avg_block_length`), producing strictly stationary resampled series. This is critical for financial returns that exhibit time-varying volatility clustering (GARCH effects in crypto). Random block lengths also reduce sensitivity to block-length misspecification.

**Optimal block length**: Politis & White (2004), corrected by Patton, Politis & White (2009):

```python
from arch.bootstrap import StationaryBootstrap, optimal_block_length

opt = optimal_block_length(daily_returns)
avg_block_length = max(1.0, float(opt.iloc[0]['stationary']))
# p = 1 / avg_block_length (geometric distribution parameter)
```

**Quantities to bootstrap** (N_BOOTSTRAP = 10,000 for each):

#### B.2.1 Mean Daily Return

```python
bs = StationaryBootstrap(avg_block_length, daily_returns, seed=42)

boot_means = []
for (data,) in bs.bootstrap(10000):
    boot_means.append(float(data.mean()))

ci_lower = np.percentile(boot_means, 2.5)
ci_upper = np.percentile(boot_means, 97.5)
p_value  = sum(1 for m in boot_means if m <= 0) / len(boot_means)
```

#### B.2.2 Sharpe Ratio (annualized)

```python
def _sharpe(x):
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(365)) if s > 0 else 0.0

boot_sharpes = []
for (data,) in bs.bootstrap(10000):
    boot_sharpes.append(_sharpe(data))

# Use BCa method for CI (handles skewness in Sharpe distribution)
ci_sharpe = bs.conf_int(_sharpe, reps=10000, method='bca', size=0.95)
p_sharpe  = sum(1 for s in boot_sharpes if s <= 0) / len(boot_sharpes)
```

#### B.2.3 Max Drawdown

```python
def _maxdd(x):
    cum = np.cumsum(x)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())

boot_mdds = []
for (data,) in bs.bootstrap(10000):
    boot_mdds.append(_maxdd(data))

# Report percentiles of bootstrap MDD distribution
mdd_p5  = np.percentile(boot_mdds, 5)
mdd_p50 = np.percentile(boot_mdds, 50)
mdd_p95 = np.percentile(boot_mdds, 95)
```

**Output per bootstrapped quantity**:

```json
{
    "statistic": "mean_daily_return",
    "observed": 0.00123,
    "ci_95_lower": -0.00045,
    "ci_95_upper": 0.00312,
    "p_value_bootstrap": 0.087,
    "n_bootstrap": 10000,
    "avg_block_length": 4.2,
    "method": "stationary_bootstrap_politis_romano"
}
```

**Thresholds** (applied to mean and Sharpe):

| Condition                       | Verdict |
|---------------------------------|---------|
| CI_lower > 0 AND p < 0.05      | PASS    |
| CI_lower > 0 OR p < 0.10       | WARN    |
| CI_lower <= 0 AND p >= 0.10    | FAIL    |

### B.3 White's Reality Check / SPA Test

**When applicable**: multiple strategy variants tested on same data (e.g. LONG + SHORT, or multiple symbols simultaneously deployed).

**Current status**: N/A -- single strategy variant (BTCUSD LONG).

```json
{
    "test": "white_spa",
    "status": "NOT_APPLICABLE",
    "reason": "single_strategy_variant",
    "n_strategies_tested": 1
}
```

**Interface for future use** (when K > 1 strategies):

```python
def white_spa_test(
    benchmark_returns: np.ndarray,       # (T,) daily returns of null model (buy-and-hold or 0)
    strategy_returns: np.ndarray,        # (T, K) daily returns for K strategies
    n_bootstrap: int = 10000,
    block_length: float | None = None,   # auto if None
    seed: int = 42,
) -> dict:
    """
    Hansen (2005) SPA test. H0: max_k E[d_kt] <= 0.
    Library: arch.bootstrap.SPA
    Returns: p_value_spa, best_strategy_idx, consistent (p < 0.05)
    """
```

### B.4 Benjamini-Hochberg FDR Control

**When applicable**: multiple hypothesis tests across strategies or parameter families.

**Current status**: N/A -- single primary hypothesis (mean return > 0).

```json
{
    "test": "benjamini_hochberg_fdr",
    "status": "NOT_APPLICABLE",
    "reason": "single_primary_hypothesis",
    "n_hypotheses": 1
}
```

**Interface for future use**:

```python
from statsmodels.stats.multitest import multipletests

def apply_fdr(p_values: list[float], alpha: float = 0.05) -> dict:
    reject, adj_p, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    return {
        "adjusted_p_values": adj_p.tolist(),
        "rejections": reject.tolist(),
        "n_rejected": int(reject.sum()),
        "fdr_level": alpha,
    }
```

---

## C. Monte Carlo Simulation

### C.1 Method: Stationary Bootstrap of Daily Returns

**Why daily, not per-trade**: Per-trade resampling destroys temporal clustering (multiple trades per day, gaps between trading days). Daily returns preserve calendar structure and PnL autocorrelation.

**Why stationary bootstrap, not IID**: IID bootstrap breaks volatility clustering (GARCH-like structure common in crypto), underestimating tail risk and drawdown severity. Stationary bootstrap preserves these dependencies.

```python
from arch.bootstrap import StationaryBootstrap, optimal_block_length
import numpy as np

def run_monte_carlo(
    daily_returns: np.ndarray,
    capital: float = 25_000,
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict:

    # Optimal block length (Politis & White 2004)
    opt = optimal_block_length(daily_returns)
    avg_bl = max(1.0, float(opt.iloc[0]['stationary']))

    bs = StationaryBootstrap(avg_bl, daily_returns, seed=seed)
    T = len(daily_returns)

    final_equities = []
    max_dd_pcts    = []
    max_dd_usds    = []
    sharpes        = []

    for (data,) in bs.bootstrap(n_sims):
        # Additive equity model (constant notional, prop-firm style)
        pnl_series = data * capital  # approximate: daily_return * current_equity
        eq = np.empty(T + 1)
        eq[0] = capital
        for t in range(T):
            eq[t + 1] = eq[t] * (1 + data[t])

        peak = np.maximum.accumulate(eq)
        dd_usd = eq - peak
        dd_pct = np.where(peak > 0, dd_usd / peak, 0.0)

        final_equities.append(float(eq[-1]))
        max_dd_pcts.append(float(dd_pct.min()))
        max_dd_usds.append(float(dd_usd.min()))

        s = data.std(ddof=1)
        sharpes.append(float(data.mean() / s * np.sqrt(365)) if s > 0 else 0.0)

    return {
        "n_sims": n_sims,
        "avg_block_length": avg_bl,
        "final_equities": final_equities,
        "max_dd_pcts": max_dd_pcts,
        "max_dd_usds": max_dd_usds,
        "sharpes": sharpes,
    }
```

### C.2 Report Quantities

**Percentiles of final equity**:

```python
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    equity_p = np.percentile(final_equities, p)
```

**Probabilities**:

```python
p_negative_pnl = np.mean(np.array(final_equities) < capital)

for X in [5, 10, 15, 20]:
    p_dd_X = np.mean(np.array(max_dd_pcts) < -X / 100)

p_sharpe_negative = np.mean(np.array(sharpes) < 0)
```

**Challenge-specific MC** (re-run with state machine):

```python
def challenge_mc_sim(daily_pnl_usd_samples: list[np.ndarray],
                     capital=25_000, daily_max_loss=1_250,
                     total_max_loss=2_500, profit_target=1_250):
    """
    For each MC simulation, replay daily PnL through challenge rules:
    - If cumulative daily loss >= daily_max_loss: skip remaining trades that day
    - If total DD from peak >= total_max_loss: challenge FAILED
    - If cumulative PnL >= profit_target: challenge PASSED
    Record: passed / failed / still_running
    """
    results = {"passed": 0, "failed": 0, "still_running": 0}
    for sim_daily_pnl in daily_pnl_usd_samples:
        equity = capital
        peak_eq = capital
        passed = False
        failed = False
        for pnl in sim_daily_pnl:
            equity += pnl
            peak_eq = max(peak_eq, equity)
            if peak_eq - equity >= total_max_loss:
                failed = True
                break
            if equity - capital >= profit_target:
                passed = True
                break
        if passed:
            results["passed"] += 1
        elif failed:
            results["failed"] += 1
        else:
            results["still_running"] += 1

    n = len(daily_pnl_usd_samples)
    return {
        "p_challenge_pass": results["passed"] / n,
        "p_challenge_fail": results["failed"] / n,
        "p_still_running":  results["still_running"] / n,
    }
```

### C.3 Thresholds

| Metric               | PASS                  | WARN                     | FAIL                  |
|----------------------|-----------------------|--------------------------|-----------------------|
| `p_negative_pnl`    | < 0.15                | 0.15 -- 0.35             | > 0.35                |
| `p_dd_10`            | < 0.20                | 0.20 -- 0.50             | > 0.50                |
| `p_dd_20`            | < 0.05                | 0.05 -- 0.20             | > 0.20                |
| `median_equity`      | > CAPITAL + $500      | CAPITAL to CAPITAL + $500| < CAPITAL             |
| `p5_equity`          | > CAPITAL - $1,000    | -$2,500 to -$1,000       | < CAPITAL - $2,500    |
| `p_sharpe_negative`  | < 0.20                | 0.20 -- 0.40             | > 0.40                |
| `p_challenge_pass`   | > 0.50                | 0.25 -- 0.50             | < 0.25                |
| `p_challenge_fail`   | < 0.10                | 0.10 -- 0.30             | > 0.30                |

---

## D. Alpha Decay Analysis

### D.1 Data Requirement Gate

```python
MIN_TRADING_DAYS_DECAY = 60

if trading_days < MIN_TRADING_DAYS_DECAY:
    return {
        "status": "INSUFFICIENT_DATA",
        "trading_days": trading_days,
        "minimum_required": MIN_TRADING_DAYS_DECAY,
    }
```

With the current ~8 OOS trading days, decay analysis will return `INSUFFICIENT_DATA`. This is correct and expected.

### D.2 Rolling Window Analysis

**Windows**: W = 30, 60, 90 trading days.

For each window size W:

```python
# daily_returns_trading: array of returns on actual trading days only (no zero-fill)
T = len(daily_returns_trading)
if T < W:
    # skip this window
    continue

rolling_mean   = []
rolling_sharpe = []
for i in range(W - 1, T):
    window = daily_returns_trading[i - W + 1 : i + 1]
    rm = window.mean()
    rs = (rm / window.std(ddof=1)) * np.sqrt(365) if window.std(ddof=1) > 0 else 0.0
    rolling_mean.append(rm)
    rolling_sharpe.append(rs)
```

### D.3 Slope of Rolling Mean (OLS)

```python
def decay_slope(rolling_values: np.ndarray) -> dict:
    n = len(rolling_values)
    if n < 3:
        return {"slope": float('nan'), "status": "INSUFFICIENT_DATA"}
    X = np.arange(n, dtype=float)
    y = rolling_values
    X_design = np.column_stack([np.ones(n), X])
    beta, residuals, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    slope = beta[1]
    # Standard error
    y_hat = X_design @ beta
    resid = y - y_hat
    mse = np.sum(resid ** 2) / (n - 2)
    se_slope = np.sqrt(mse / np.sum((X - X.mean()) ** 2))
    t_stat = slope / se_slope if se_slope > 0 else 0.0
    from scipy.stats import t as t_dist
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 2))
    return {"slope": slope, "se": se_slope, "t_stat": t_stat, "p_value": p_value}
```

### D.4 Half-Life Estimation

```python
from scipy.optimize import curve_fit

def exp_decay(t, a, lam, c):
    return a * np.exp(-lam * t) + c

def estimate_half_life(rolling_mean: np.ndarray) -> dict:
    X = np.arange(len(rolling_mean), dtype=float)
    y = rolling_mean.copy()
    try:
        a0 = y[0] - y[-1]
        c0 = y[-1]
        lam0 = 0.01
        popt, pcov = curve_fit(
            exp_decay, X, y,
            p0=[a0, lam0, c0],
            maxfev=5000,
            bounds=([-np.inf, -0.5, -np.inf], [np.inf, 2.0, np.inf]),
        )
        a, lam, c = popt
        y_hat = exp_decay(X, *popt)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if lam <= 0:
            return {"decay_detected": False, "lambda": lam,
                    "half_life_days": None, "r_squared": r_squared}
        half_life = np.log(2) / lam
        return {"decay_detected": True, "lambda": lam,
                "half_life_days": half_life, "r_squared": r_squared}
    except (RuntimeError, ValueError):
        return {"decay_detected": False, "lambda": None,
                "half_life_days": None, "r_squared": None}
```

### D.5 Decay Score

| Condition                                          | Verdict             |
|----------------------------------------------------|---------------------|
| `trading_days < 60`                                | INSUFFICIENT_DATA   |
| `slope >= 0` OR no decay detected                  | PASS                |
| Decay detected AND `half_life > 180` days          | PASS                |
| Decay detected AND `90 < half_life <= 180` days    | WARN                |
| Decay detected AND `half_life <= 90` days          | FAIL                |

---

## E. Overfitting / WFO Analysis

### E.1 Per-Fold OOS Metrics

For each unique `fold_id` in OOS trades:

```python
fold_trades = oos_trades.filter(pl.col("fold_id") == fid)
n      = fold_trades.height
pnls   = fold_trades.get_column("net_pnl_base").to_list()
wins   = [p for p in pnls if p > 0]
losses = [p for p in pnls if p < 0]

per_fold = {
    "fold_id":        fid,
    "n_trades":       n,
    "total_return":   sum(pnls),
    "mean_return":    np.mean(pnls) if n > 0 else 0.0,
    "std_return":     np.std(pnls, ddof=1) if n > 1 else float('nan'),
    "sharpe_fold":    (np.mean(pnls) / np.std(pnls, ddof=1) * np.sqrt(365))
                      if n > 1 and np.std(pnls, ddof=1) > 0 else 0.0,
    "win_rate":       len(wins) / n if n > 0 else 0.0,
    "profit_factor":  sum(wins) / abs(sum(losses))
                      if losses else (float('inf') if wins else 0.0),
    "is_positive":    sum(pnls) > 0,
}
```

### E.2 WFO Pass Rate

```python
n_folds_total    = number of unique fold_ids with OOS trades
n_folds_positive = count(folds where total_return > 0)
pass_rate        = n_folds_positive / n_folds_total
```

| Pass Rate   | Verdict |
|-------------|---------|
| >= 0.60     | PASS    |
| 0.40 -- 0.59 | WARN  |
| < 0.40      | FAIL    |

### E.3 PBO (Probability of Backtest Overfitting) -- Bailey et al. (2014)

**Requires**: For each fold and each parameter combo, both IS and OOS performance scores.

**Current data gap**: `tuning_results_v2.parquet` stores only IS scores per combo. To compute PBO, OOS performance per combo per fold is needed.

**Method** (when data is available):

```python
def compute_pbo(tuning_df: pl.DataFrame) -> dict:
    """
    tuning_df must have columns: fold_id, combo_id, score_is, score_oos

    For each fold:
        1. Rank combos by score_is (descending)
        2. best_IS_combo = combo with rank 1
        3. Get that combo's score_oos
        4. median_oos = median of all combos' score_oos for this fold
        5. is_overfit = (best_IS_combo's score_oos < median_oos)

    PBO = count(is_overfit) / n_folds
    """
    folds = tuning_df.get_column("fold_id").unique().to_list()
    n_overfit = 0
    for fid in folds:
        fold_data = tuning_df.filter(pl.col("fold_id") == fid)
        best_is = fold_data.sort("score_is", descending=True).head(1)
        best_oos = float(best_is.get_column("score_oos")[0])
        median_oos = float(fold_data.get_column("score_oos").median())
        if best_oos < median_oos:
            n_overfit += 1
    return {"PBO": n_overfit / len(folds), "n_folds": len(folds)}
```

**Until OOS-per-combo data is available**, report:

```json
{
    "test": "PBO",
    "status": "NOT_COMPUTABLE",
    "reason": "tuning_results lacks OOS performance per combo",
    "recommendation": "Extend Celda 14 to store oos_sum_ret, oos_mean_ret, oos_score per combo",
    "proxy": "Using wfo_pass_rate (Section E.2) as substitute"
}
```

**Thresholds** (when computable):

| PBO      | Verdict |
|----------|---------|
| < 0.50   | PASS    |
| 0.50 -- 0.74 | WARN |
| >= 0.75  | FAIL    |

### E.4 Deflated Sharpe Ratio (DSR) -- Bailey & Lopez de Prado (2014)

This IS computable now. It adjusts the observed Sharpe for the number of parameter combos tried during tuning.

```python
from scipy.stats import norm

def deflated_sharpe_ratio(
    sr_observed: float,       # annualized Sharpe on OOS
    n_trials: int,            # total parameter combos tested
    T: int,                   # number of daily OOS observations
    skew: float,              # skewness of daily returns
    kurt_excess: float,       # excess kurtosis of daily returns
) -> dict:
    if n_trials <= 1:
        return {"status": "NOT_APPLICABLE", "reason": "n_trials <= 1"}
    if T < 5:
        return {"status": "INSUFFICIENT_DATA", "n_obs": T}

    # Expected max Sharpe under null (Bailey & Lopez de Prado 2014, Eq. 10)
    euler_gamma = 0.5772156649
    log_n = np.log(n_trials)
    sr_expected = np.sqrt(2 * log_n) * (
        1 - euler_gamma / (2 * log_n)
    ) + euler_gamma / (2 * np.sqrt(2 * log_n))

    # Standard error of Sharpe (Lo 2002)
    se_sr = np.sqrt(
        (1 + 0.5 * sr_observed**2
         - skew * sr_observed
         + (kurt_excess / 4) * sr_observed**2
        ) / T
    )

    if se_sr <= 0:
        return {"status": "ERROR", "reason": "se_sr <= 0"}

    dsr_z = (sr_observed - sr_expected) / se_sr
    p_value = 1 - norm.cdf(dsr_z)

    return {
        "sr_observed": sr_observed,
        "sr_expected_null": sr_expected,
        "se_sr": se_sr,
        "dsr_z": dsr_z,
        "p_value": p_value,
        "n_trials": n_trials,
        "n_obs": T,
    }
```

`n_trials` = total unique combos from `tuning_results_v2.parquet` across all folds for this symbol. Read from: `tuning_df.filter(symbol == "BTCUSD").height` or `n_combos` from the tuning snapshot.

**Thresholds**:

| DSR p-value   | Verdict                              |
|---------------|--------------------------------------|
| < 0.05        | PASS (Sharpe survives deflation)     |
| 0.05 -- 0.10  | WARN                                 |
| >= 0.10       | FAIL (Sharpe likely inflated by selection) |

---

## F. Cost Sensitivity

### F.1 Method

For each `cost_multiplier` in `[1.0, 1.25, 1.5, 2.0]`:

```python
# Base roundtrip cost as fraction of notional:
# cost_base_rt = CostsConfig.total_roundtrip_dec
#              = 2 * (spread_bps/2 + commission_bps + slippage_bps) / 10_000

# The stored columns satisfy:
#   net_pnl_base = gross_pnl - cost_base_rt

# PnL at new cost level:
#   net_pnl_at_mult = gross_pnl - cost_base_rt * cost_multiplier
#                   = net_pnl_base + cost_base_rt * (1 - cost_multiplier)
#                   = net_pnl_base - cost_base_rt * (cost_multiplier - 1)

delta_cost = cost_base_rt * (cost_multiplier - 1.0)
oos_trades_adj = oos_trades.with_columns(
    (pl.col("net_pnl_base") - delta_cost).alias("net_pnl_adj")
)
```

For each multiplier, recompute ALL Section A metrics using `net_pnl_adj` in place of `net_pnl_base`.

### F.2 Investability Gate

```python
still_investable = (
    sharpe_annual > 0.0
    and total_return_pct > 0
    and max_dd_pct > -0.25   # MaxDD better than -25%
)
```

### F.3 Challenge Overlay per Cost Level

Re-run the NB3 Celda 16 challenge state machine with adjusted PnL at each multiplier. Output per level:

```json
{
    "cost_multiplier": 1.5,
    "challenge_passed": true,
    "final_equity": 25800.00,
    "max_daily_loss_seen": -420.00,
    "max_total_dd_seen": -980.00,
    "violated_daily": false,
    "violated_total": false
}
```

### F.4 Break-Even Cost Multiplier

```python
gross_total     = oos_trades.get_column("gross_pnl").sum()
n_trades        = oos_trades.height
cost_per_trade  = cost_base_rt  # roundtrip cost in fractional terms

if n_trades * cost_per_trade > 0:
    m_breakeven = gross_total / (n_trades * cost_per_trade)
else:
    m_breakeven = float('inf')
```

| m_breakeven  | Verdict                  |
|--------------|--------------------------|
| > 3.0        | PASS (very cost-resilient)|
| 2.0 -- 3.0   | PASS                     |
| 1.5 -- 1.99  | WARN                     |
| < 1.5        | FAIL (fragile edge)       |

---

## G. Thresholds Summary Table

| #  | Metric                  | PASS                    | WARN                     | FAIL                     | Section |
|----|-------------------------|-------------------------|--------------------------|--------------------------|---------|
| 1  | `n_trades`              | >= 30                   | 15 -- 29                 | < 15                     | A.1     |
| 2  | `total_return_pct`      | > 0%                    | -5% to 0%                | < -5%                    | A.2     |
| 3  | `sharpe_annual`         | > 0.50                  | 0.0 to 0.50              | < 0.0                    | A.3     |
| 4  | `sortino_annual`        | > 0.70                  | 0.0 to 0.70              | < 0.0                    | A.3     |
| 5  | `calmar`                | > 0.50                  | 0.0 to 0.50              | < 0.0                    | A.3     |
| 6  | `max_dd_pct`            | > -10%                  | -10% to -20%             | < -20%                   | A.4     |
| 7  | `max_dd_usd`            | > -$1,250               | -$1,250 to -$2,500       | < -$2,500                | A.4     |
| 8  | `ulcer_index`           | < 5.0                   | 5.0 to 10.0              | > 10.0                   | A.4     |
| 9  | `profit_factor`         | > 1.50                  | 1.0 to 1.50              | < 1.0                    | A.7     |
| 10 | `win_rate`              | > 0.40                  | 0.20 to 0.40             | < 0.20                   | A.7     |
| 11 | `expectancy_usd`        | > $50                   | $0 to $50                | < $0                     | A.7     |
| 12 | `skewness`              | > 0.50                  | -0.50 to 0.50            | < -0.50                  | A.5     |
| 13 | `excess_kurtosis`       | < 5.0                   | 5.0 to 10.0              | > 10.0                   | A.5     |
| 14 | `tail_ratio`            | > 1.20                  | 0.80 to 1.20             | < 0.80                   | A.5     |
| 15 | `VaR_95_usd`            | > -$500                 | -$500 to -$1,250         | < -$1,250                | A.6     |
| 16 | `CVaR_95_usd`           | > -$750                 | -$750 to -$1,500         | < -$1,500                | A.6     |
| 17 | `hac_ttest_p`           | < 0.05                  | 0.05 to 0.10             | >= 0.10                  | B.1     |
| 18 | `bootstrap_mean_ci_lo`  | > 0                     | --                        | <= 0                     | B.2     |
| 19 | `bootstrap_sharpe_p`    | < 0.05                  | 0.05 to 0.10             | >= 0.10                  | B.2     |
| 20 | `mc_p_negative_pnl`     | < 0.15                  | 0.15 to 0.35             | > 0.35                   | C.3     |
| 21 | `mc_p_dd_10`            | < 0.20                  | 0.20 to 0.50             | > 0.50                   | C.3     |
| 22 | `mc_p_dd_20`            | < 0.05                  | 0.05 to 0.20             | > 0.20                   | C.3     |
| 23 | `mc_median_equity`      | > CAPITAL + $500        | CAPITAL to +$500         | < CAPITAL                | C.3     |
| 24 | `mc_p5_equity`          | > CAPITAL - $1k         | -$2.5k to -$1k           | < CAPITAL - $2.5k        | C.3     |
| 25 | `mc_p_sharpe_neg`       | < 0.20                  | 0.20 to 0.40             | > 0.40                   | C.3     |
| 26 | `mc_p_challenge_pass`   | > 0.50                  | 0.25 to 0.50             | < 0.25                   | C.3     |
| 27 | `mc_p_challenge_fail`   | < 0.10                  | 0.10 to 0.30             | > 0.30                   | C.3     |
| 28 | `decay_score`           | slope>=0 or HL>180d     | HL 90--180d              | HL<90d                   | D.5     |
| 29 | `wfo_pass_rate`         | >= 0.60                 | 0.40 to 0.59             | < 0.40                   | E.2     |
| 30 | `PBO`                   | < 0.50                  | 0.50 to 0.74             | >= 0.75                  | E.3     |
| 31 | `DSR_p_value`           | < 0.05                  | 0.05 to 0.10             | >= 0.10                  | E.4     |
| 32 | `cost_breakeven_mult`   | > 2.0                   | 1.5 to 2.0               | < 1.5                    | F.4     |
| 33 | `cost_2x_investable`    | True                    | --                        | False                    | F.2     |

---

## H. Output Schema

### H.1 `stats_core_metrics.csv`

One row per scenario (`base`, `stress`):

```
scenario, n_trades, trading_days, calendar_days, exposure_pct,
total_pnl_usd, total_return_pct, CAGR,
mean_return_per_trade, mean_return_daily, std_per_trade, std_daily,
sharpe_annual, sortino_annual, calmar,
max_dd_pct, max_dd_usd, ulcer_index,
skewness, excess_kurtosis, tail_ratio,
VaR_95_pct, VaR_95_usd, CVaR_95_pct, CVaR_95_usd,
profit_factor, win_rate, payoff_ratio,
expectancy_usd, expectancy_frac,
avg_duration_hours, avg_duration_bars,
max_consec_wins, max_consec_losses,
verdict
```

`verdict` = overall PASS/WARN/FAIL based on worst individual metric verdict for that row.

### H.2 `stats_significance.csv`

One row per test:

```
test_name, statistic_name, observed_value,
t_stat, p_value, ci_95_lower, ci_95_upper,
n_obs, n_bootstrap, avg_block_length, method,
verdict, notes
```

Rows:
1. `hac_ttest_mean_daily`
2. `bootstrap_mean_daily`
3. `bootstrap_sharpe_annual`
4. `bootstrap_max_dd`
5. `white_spa` (NOT_APPLICABLE)
6. `benjamini_hochberg` (NOT_APPLICABLE)

### H.3 `stats_monte_carlo.csv`

Key-value format, one metric per row:

```
metric, value
n_sims, 10000
avg_block_length, 4.2
equity_p01, 23100.50
equity_p05, 23800.00
equity_p10, 24100.00
equity_p25, 24900.00
equity_p50, 25800.00
equity_p75, 26500.00
equity_p90, 27200.00
equity_p95, 27800.00
equity_p99, 28600.00
p_negative_pnl, 0.32
p_dd_gt_5pct, 0.45
p_dd_gt_10pct, 0.18
p_dd_gt_15pct, 0.06
p_dd_gt_20pct, 0.02
sharpe_p05, -0.30
sharpe_p25, 0.15
sharpe_p50, 0.45
sharpe_p75, 0.82
sharpe_p95, 1.35
p_sharpe_negative, 0.28
p_challenge_pass, 0.42
p_challenge_fail, 0.15
verdict, WARN
```

### H.4 `stats_mc_distributions.parquet`

Full simulation results (10,000 rows):

| Column              | Type    |
|---------------------|---------|
| `sim_id`            | Int64   |
| `final_equity`      | Float64 |
| `max_dd_pct`        | Float64 |
| `max_dd_usd`        | Float64 |
| `sharpe`            | Float64 |
| `total_return_pct`  | Float64 |
| `challenge_passed`  | Bool    |
| `challenge_failed`  | Bool    |

### H.5 `stats_alpha_decay.csv`

Key-value format:

```
metric, value
status, INSUFFICIENT_DATA
trading_days_available, 8
min_required, 60
W30_slope, NaN
W30_slope_tstat, NaN
W30_slope_pvalue, NaN
W30_n_points, 0
W60_slope, NaN
...
W90_slope, NaN
...
decay_detected, false
lambda, NaN
half_life_days, NaN
fit_r_squared, NaN
```

### H.6 `stats_wfo_folds.csv`

One row per fold:

```
fold_id, n_trades, total_return, mean_return, std_return,
sharpe_fold, win_rate, profit_factor, is_positive
```

### H.7 `stats_overfitting.csv`

Key-value format:

```
metric, value
wfo_pass_rate, 0.60
n_folds_positive, 6
n_folds_total, 10
PBO, NOT_COMPUTABLE
PBO_reason, tuning_results_lacks_oos_per_combo
DSR_sr_observed, 0.45
DSR_sr_expected, 0.82
DSR_se, 0.31
DSR_z, -1.19
DSR_p_value, 0.88
DSR_n_trials, 48
DSR_n_obs, 8
DSR_verdict, FAIL
```

### H.8 `stats_cost_sensitivity.csv`

One row per multiplier:

```
cost_multiplier, sharpe_annual, total_return_pct, max_dd_pct, win_rate,
expectancy_usd, profit_factor, still_investable,
challenge_passed, challenge_final_equity, challenge_violated_daily, challenge_violated_total
```

### H.9 `stats_summary_dashboard.json`

Aggregated JSON with all sections:

```json
{
    "version": "1.0.0",
    "generated_utc": "2026-02-19T12:00:00Z",
    "strategy": "BTCUSD_LONG_TREND_M5",
    "capital": 25000,
    "risk_per_trade_usd": 75,
    "pos_notional": 25000.0,
    "annualization_factor": 365,
    "core_metrics": {
        "base": {"...all A metrics..."},
        "stress": {"...all A metrics..."}
    },
    "significance": {
        "hac_ttest": {"..."},
        "bootstrap_mean": {"..."},
        "bootstrap_sharpe": {"..."},
        "bootstrap_maxdd": {"..."}
    },
    "monte_carlo": {
        "equity_percentiles": {"..."},
        "probabilities": {"..."},
        "challenge_simulation": {"..."}
    },
    "alpha_decay": {"..."},
    "overfitting": {
        "wfo_folds": ["..."],
        "pass_rate": 0.60,
        "pbo": {"..."},
        "dsr": {"..."}
    },
    "cost_sensitivity": {
        "multipliers": {"..."},
        "breakeven_multiplier": 2.8
    },
    "overall_verdict": "WARN",
    "verdict_breakdown": {
        "PASS": 22,
        "WARN": 8,
        "FAIL": 3,
        "NOT_APPLICABLE": 2,
        "INSUFFICIENT_DATA": 1
    }
}
```

**Overall verdict logic**:
```python
if any metric is FAIL:
    overall = "FAIL"
elif any metric is WARN:
    overall = "WARN"
else:
    overall = "PASS"
```

---

## I. Library Dependencies

| Library        | Version  | Usage                                                             |
|----------------|----------|-------------------------------------------------------------------|
| `numpy`        | >= 1.24  | Array operations, percentiles, linear algebra                     |
| `polars`       | >= 1.35  | DataFrame I/O, filtering (already in venv1)                      |
| `scipy`        | >= 1.11  | `skew`, `kurtosis`, `norm.cdf`, `curve_fit`, `t.cdf` (already)   |
| `statsmodels`  | >= 0.14  | `OLS.fit(cov_type='HAC')` for Newey-West t-test                  |
| `arch`         | >= 6.0   | `StationaryBootstrap`, `optimal_block_length`, future `SPA` test  |

**Installation** (in venv1):
```
pip install statsmodels arch
```

Both install cleanly on Windows with Python 3.11.

---

## J. Implementation Notes and Gotchas

### J.1 Division by Zero

Every ratio (Sharpe, Sortino, Calmar, profit_factor, payoff_ratio, tail_ratio) must check for zero denominators. Follow the `_safe_div` pattern in `metrics.py`: return 0.0 when denominator is 0, except where +inf is semantically correct (e.g., profit_factor with no losses).

### J.2 NaN Propagation

Filter NaN values from `net_pnl_base` before computation. Log a warning if any NaN trades are found.

```python
valid = oos_trades.filter(pl.col("net_pnl_base").is_not_nan())
if valid.height < oos_trades.height:
    warn(f"Dropped {oos_trades.height - valid.height} NaN trades")
```

### J.3 Empty OOS

If OOS has 0 trades, all metrics are NaN/0 and overall verdict is FAIL with `reason = "no_oos_trades"`.

### J.4 Single-Trade Folds

Some folds may have only 1 OOS trade. Per-fold Sharpe requires >= 2 observations (returns NaN for n=1). Count metrics (n_trades, total_return, win_rate) are reported regardless.

### J.5 Polars to Numpy Bridge

Construct daily returns in Polars (group_by date, aggregate), then convert to numpy for statistical tests:

```python
daily_series = (
    oos_trades
    .with_columns(pl.col("exit_time_utc").dt.date().alias("_date"))
    .group_by("_date")
    .agg(pl.col("pnl_usd").sum().alias("daily_pnl_usd"))
    .sort("_date")
)
daily_pnl = daily_series.get_column("daily_pnl_usd").to_numpy()
```

The `arch` and `statsmodels` libraries expect numpy arrays.

### J.6 Deterministic Seeds

All bootstrap and MC simulations use `seed=42` (configurable). The `arch.bootstrap.StationaryBootstrap` accepts a `seed` parameter. Document the seed in every output file.

### J.7 Additive vs Multiplicative Equity

- **Additive** (constant notional): for prop-firm challenge metrics (USD PnL, daily loss checks). This models fixed lot-size trading.
- **Multiplicative** (compounding): for percentage-based metrics (CAGR, Sharpe on fractional returns). This models reinvesting profits.

Document which model each metric uses:
- Sharpe, Sortino, CAGR, MaxDD%: multiplicative (`equity[t] = equity[t-1] * (1 + r[t])`)
- Challenge sim, MaxDD USD, VaR USD, expectancy USD: additive (`equity[t] = equity[t-1] + pnl_usd[t]`)

### J.8 Weekend Handling

With `mon_fri_only=True`, if a trade exits Friday and the next enters Monday, Saturday and Sunday have `daily_return = 0.0` (no position, no PnL). These zeros are included in the daily returns array for annualization consistency.

### J.9 Insufficient Data Warning

With ~20 OOS trades and ~8 trading days, most statistical tests will have low power. Always report effective sample size alongside p-values:
- HAC t-test with T=8 produces very wide confidence intervals.
- Bootstrap works mechanically but CIs will be very wide.
- This is expected and informative -- it signals "wait for more data before allocating."

### J.10 PBO Extension Recommendation

To make PBO fully computable, extend NB3 Celda 14 to also run each parameter combo on OOS and store the OOS score. Add columns to `tuning_results_v2.parquet`:
- `oos_n_trades` (Int64)
- `oos_sum_ret` (Float64)
- `oos_mean_ret` (Float64)
- `oos_score` (Float64)

This approximately doubles tuning runtime but is essential for genuine overfitting detection.

### J.11 pos_notional Stability

`pos_notional` depends on `sl_return_median` computed from SL-exit trades. If parameter selection changes SL distance across folds, pos_notional should ideally be recomputed per fold. For a single global analysis, use the median across all OOS SL trades. Document the value used in the output JSON.

### J.12 Crypto-Specific Assumptions

- Returns are fat-tailed (excess kurtosis >> 0) and often positively skewed for trend-following.
- All methods in this spec are non-parametric (historical VaR, bootstrap CIs, no Gaussian assumption).
- Volatility clustering is expected -- the stationary bootstrap handles this.
- 24/7 trading means 365-day annualization. The `mon_fri_only` engine flag reduces exposure but does not change the calendar.

### J.13 Compatibility with Existing Code

| Existing Module        | Status                  | Action Required                                     |
|------------------------|-------------------------|-----------------------------------------------------|
| `metrics.py`           | Basic, per-bar scaling  | Extend: add daily-return versions of Sharpe/Sortino; add Ulcer Index, skew, kurt, tail_ratio, VaR/CVaR |
| `monte_carlo.py`       | IID + block bootstrap   | Replace with stationary bootstrap (`arch` library); add challenge MC |
| `schema.py`            | `TRADES_SCHEMA` uses `net_pnl` | Add `net_pnl_base`, `net_pnl_stress`, `pos_size` to schema |
| `wfo.py`               | Fold structure OK       | No changes needed                                   |
| `config.py`            | Missing `StatsConfig`   | Add `StatsConfig` dataclass with all parameters     |
| `costs.py`             | OK                      | No changes needed                                   |

### J.14 Suggested Module Structure

```
03_STRATEGY_LAB/
  src/strategylab/
    institutional_stats.py     # NEW: main orchestrator
    stats_core.py              # NEW: Section A metrics
    stats_significance.py      # NEW: Section B tests
    stats_monte_carlo.py       # NEW: Section C (replaces/wraps monte_carlo.py)
    stats_decay.py             # NEW: Section D
    stats_overfitting.py       # NEW: Section E
    stats_cost_sensitivity.py  # NEW: Section F
    stats_config.py            # NEW: StatsConfig dataclass
    stats_report.py            # NEW: output formatters (CSV, JSON, parquet)
```

Each module exposes a single entry-point function that takes the trades DataFrame and config, and returns a typed result dict.

---

## End of Specification

This document is self-contained. A Python developer can implement every metric, test, threshold, and output format described above without additional clarification. All formulas are explicit, all edge cases are handled, all thresholds are defined, and all library dependencies are specified.
