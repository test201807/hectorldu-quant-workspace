"""
institutional_stats.py — Suite Estadística Institucional para NB3 TREND v2
===========================================================================
Genera un "institutional_pack" con métricas defendibles:
  - Core metrics (base + stress)
  - Significance tests (HAC t-test + stationary bootstrap)
  - Monte Carlo (equity distribution + challenge sim)
  - Alpha decay (rolling windows + half-life)
  - WFO overfitting (per-fold + PBO proxy + DSR)
  - Cost sensitivity (1x/1.25x/1.5x/2x)

Uso:
  cd C:\\Quant\\projects\\MT5_Data_Extraction
  .\\venv1\\Scripts\\python.exe -X utf8 tools\\institutional_stats.py [--run-id RUN_ID]

Output:
  outputs/trend_v2/<run_id>/institutional_pack/
    trades_oos.csv, trades_is.csv
    metrics_core.csv
    metrics_by_fold.csv
    significance_tests.csv
    bootstrap_ci.csv
    monte_carlo_distribution.csv
    alpha_decay_report.csv
    pbo_report.csv
    cost_sensitivity.csv
    institutional_manifest.json
    README.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.stats as ss
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CAPITAL = 25_000
RISK_PER_TRADE_USD = 75
ANNUAL_FACTOR = 365          # crypto 24/7
SQRT_ANNUAL = np.sqrt(ANNUAL_FACTOR)
COST_BASE_DEC = 0.0003       # 3 bps roundtrip (verified from trades)
COST_STRESS_DEC = 0.0006     # 6 bps roundtrip
N_BOOTSTRAP = 10_000
N_MC_SIMS = 10_000
MC_SEED = 42
CHALLENGE_DAILY_MAX_LOSS = 1_250
CHALLENGE_TOTAL_MAX_LOSS = 2_500
CHALLENGE_PROFIT_TARGET = 1_250
CHALLENGE_MIN_DAYS = 2

COST_MULTIPLIERS = [1.0, 1.25, 1.5, 2.0]

# Thresholds  {metric: (pass_threshold, warn_threshold, direction)}
# direction: "gt" = higher is better, "lt" = lower is better
THRESHOLDS = {
    "n_trades":          (30,    15,     "gt"),
    "total_return_pct":  (0.0,   -5.0,   "gt"),
    "sharpe_annual":     (0.50,  0.0,    "gt"),
    "sortino_annual":    (0.70,  0.0,    "gt"),
    "calmar":            (0.50,  0.0,    "gt"),
    "max_dd_pct":        (-10.0, -20.0,  "gt"),
    "ulcer_index":       (5.0,   10.0,   "lt"),
    "profit_factor":     (1.50,  1.0,    "gt"),
    "win_rate":          (0.40,  0.20,   "gt"),
    "expectancy_usd":    (50.0,  0.0,    "gt"),
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0 or not np.isfinite(b):
        return default
    return a / b


def _verdict(value: float, metric: str) -> str:
    if metric not in THRESHOLDS:
        return "N/A"
    pass_th, warn_th, direction = THRESHOLDS[metric]
    if not np.isfinite(value):
        return "FAIL"
    if direction == "gt":
        if value >= pass_th:
            return "PASS"
        elif value >= warn_th:
            return "WARN"
        return "FAIL"
    else:  # lt
        if value <= pass_th:
            return "PASS"
        elif value <= warn_th:
            return "WARN"
        return "FAIL"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def detect_run(project: Path, run_id: str | None = None) -> tuple[str, Path]:
    """Detect run_id and return (run_id, run_dir)."""
    trend_out = project / "outputs" / "trend_v2"
    if run_id:
        run_dir = trend_out / f"run_{run_id}"
    else:
        latest = trend_out / "_latest_run.txt"
        if not latest.exists():
            raise FileNotFoundError(f"No _latest_run.txt in {trend_out}")
        run_id = latest.read_text().strip()
        run_dir = trend_out / f"run_{run_id}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    return run_id, run_dir


def load_data(run_dir: Path) -> dict[str, Any]:
    """Load all required artifacts from the run directory."""
    data = {}
    data["overlay_trades"] = pl.read_parquet(run_dir / "overlay_trades_v2.parquet")
    data["tuning_results"] = pl.read_parquet(run_dir / "tuning_results_v2.parquet")
    data["wfo_folds"] = pl.read_parquet(run_dir / "wfo_folds_v2.parquet")
    data["equity_curve"] = pl.read_parquet(run_dir / "equity_curve_engine_v2.parquet")
    data["summary_engine"] = pl.read_parquet(run_dir / "summary_engine_v2.parquet")

    for name in ["engine_qa_report_v2", "overlay_snapshot_v2",
                  "challenge_dashboard_v2", "deploy_pack_v2",
                  "selection_snapshot_v2", "run_manifest_v2"]:
        p = run_dir / f"{name}.json"
        if p.exists():
            data[name.replace("_v2", "")] = json.loads(p.read_text(encoding="utf-8"))
    return data


# ══════════════════════════════════════════════════════════════════════════════
# RETURN SERIES CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_pos_notional(trades: pl.DataFrame) -> tuple[float, float]:
    """Compute position notional from SL-exit trades (median SL loss)."""
    sl = trades.filter(pl.col("exit_reason") == "SL")
    if sl.height > 0:
        sl_ret_med = float(sl["net_pnl_base"].abs().median())
    else:
        sl_ret_med = 0.003
    sl_ret_med = max(sl_ret_med, 1e-8)
    pos_notional = RISK_PER_TRADE_USD / sl_ret_med
    return pos_notional, sl_ret_med


def build_daily_returns(trades: pl.DataFrame, pos_notional: float,
                        pnl_col: str = "net_pnl_base") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build daily PnL series (USD) and daily fractional returns on equity.

    Returns: (daily_returns_frac, daily_pnl_usd, dates_list)
    """
    if trades.height == 0:
        return np.array([]), np.array([]), []

    df = (
        trades
        .with_columns(pl.col("exit_time_utc").cast(pl.Date).alias("trade_date"))
        .group_by("trade_date")
        .agg([
            (pl.col(pnl_col) * pos_notional).sum().alias("pnl_usd"),
            pl.len().alias("n_trades"),
        ])
        .sort("trade_date")
    )

    dates = df["trade_date"].to_list()
    pnl_usd = df["pnl_usd"].to_numpy()

    # Build equity curve and fractional returns
    equity = CAPITAL
    daily_ret = []
    for p in pnl_usd:
        r = p / equity if equity > 0 else 0.0
        daily_ret.append(r)
        equity += p

    return np.array(daily_ret), pnl_usd, [str(d) for d in dates]


def build_equity_curve_usd(daily_pnl_usd: np.ndarray) -> np.ndarray:
    """Build equity curve starting at CAPITAL."""
    eq = np.empty(len(daily_pnl_usd) + 1)
    eq[0] = CAPITAL
    for i, p in enumerate(daily_pnl_usd):
        eq[i + 1] = eq[i] + p
    return eq


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A: CORE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_core_metrics(trades: pl.DataFrame, daily_ret: np.ndarray,
                         daily_pnl: np.ndarray, pos_notional: float,
                         pnl_col: str = "net_pnl_base") -> dict[str, Any]:
    """Compute investable core metrics for a given scenario."""
    m: dict[str, Any] = {}
    n = trades.height
    m["n_trades"] = n

    if n == 0:
        return {k: 0.0 for k in ["n_trades", "trading_days", "total_return_pct",
                                   "sharpe_annual", "sortino_annual"]}

    # Counts
    trade_dates = trades["exit_time_utc"].cast(pl.Date).unique()
    m["trading_days"] = trade_dates.len()

    date_min = trades["entry_time_utc"].min()
    date_max = trades["exit_time_utc"].max()
    if date_min is not None and date_max is not None:
        m["calendar_days"] = (date_max - date_min).days + 1
    else:
        m["calendar_days"] = 0

    # Exposure
    total_hold_bars = int(trades["hold_bars"].sum())
    bars_per_day = 288  # M5
    total_oos_bars = m["calendar_days"] * bars_per_day if m["calendar_days"] > 0 else 1
    m["exposure_pct"] = round(total_hold_bars / total_oos_bars * 100, 2)

    # Returns
    eq = build_equity_curve_usd(daily_pnl)
    final_eq = eq[-1]
    m["total_pnl_usd"] = round(final_eq - CAPITAL, 2)
    m["total_return_pct"] = round((final_eq / CAPITAL - 1) * 100, 4)

    cal_days = max(m["calendar_days"], 1)
    if final_eq > 0:
        m["CAGR"] = round((final_eq / CAPITAL) ** (365 / cal_days) - 1, 6)
    else:
        m["CAGR"] = -1.0

    # Per-trade stats
    pnl_arr = trades[pnl_col].to_numpy()
    m["mean_return_per_trade"] = round(float(np.mean(pnl_arr)), 8)
    m["std_per_trade"] = round(float(np.std(pnl_arr, ddof=1)), 8) if n > 1 else 0.0

    # Daily stats
    T = len(daily_ret)
    m["mean_return_daily"] = round(float(np.mean(daily_ret)), 8) if T > 0 else 0.0
    m["std_daily"] = round(float(np.std(daily_ret, ddof=1)), 8) if T > 1 else 0.0

    # Sharpe (annualized from daily)
    if T > 1 and m["std_daily"] > 0:
        m["sharpe_annual"] = round(m["mean_return_daily"] / m["std_daily"] * SQRT_ANNUAL, 4)
    else:
        m["sharpe_annual"] = 0.0

    # Sortino (annualized from daily)
    if T > 1:
        downside_sq = np.array([min(r, 0) ** 2 for r in daily_ret])
        dd = np.sqrt(np.mean(downside_sq))
        m["sortino_annual"] = round(m["mean_return_daily"] / dd * SQRT_ANNUAL, 4) if dd > 0 else 0.0
    else:
        m["sortino_annual"] = 0.0

    # MaxDD
    peak = np.maximum.accumulate(eq)
    dd_pct = np.where(peak > 0, (eq - peak) / peak, 0.0)
    dd_usd = eq - peak
    m["max_dd_pct"] = round(float(np.min(dd_pct)) * 100, 4)
    m["max_dd_usd"] = round(float(np.min(dd_usd)), 2)

    # Calmar
    abs_mdd = abs(m["max_dd_pct"] / 100)
    m["calmar"] = round(_safe_div(m["CAGR"], abs_mdd), 4)

    # Ulcer index
    dd_sq = dd_pct ** 2 * 10000  # percentage points squared
    m["ulcer_index"] = round(float(np.sqrt(np.mean(dd_sq))), 4)

    # Distribution (on daily returns)
    if T >= 3:
        m["skewness"] = round(float(ss.skew(daily_ret, bias=False)), 4)
    else:
        m["skewness"] = float("nan")
    if T >= 4:
        m["excess_kurtosis"] = round(float(ss.kurtosis(daily_ret, bias=False)), 4)
    else:
        m["excess_kurtosis"] = float("nan")

    # Tail ratio
    if T >= 20:
        p95 = float(np.percentile(daily_ret, 95))
        p05 = float(np.percentile(daily_ret, 5))
        m["tail_ratio"] = round(_safe_div(abs(p95), abs(p05), 0.0), 4)
    else:
        m["tail_ratio"] = float("nan")

    # VaR/CVaR (historical)
    if T >= 20:
        sorted_ret = np.sort(daily_ret)
        idx5 = int(np.floor(T * 0.05))
        m["VaR_95_pct"] = round(float(sorted_ret[idx5]) * 100, 4)
        m["CVaR_95_pct"] = round(float(np.mean(sorted_ret[:idx5 + 1])) * 100, 4) if idx5 > 0 else m["VaR_95_pct"]
        m["VaR_95_usd"] = round(m["VaR_95_pct"] / 100 * CAPITAL, 2)
        m["CVaR_95_usd"] = round(m["CVaR_95_pct"] / 100 * CAPITAL, 2)
    else:
        m["VaR_95_pct"] = m["CVaR_95_pct"] = float("nan")
        m["VaR_95_usd"] = m["CVaR_95_usd"] = float("nan")

    # Trade-level metrics
    pnl_usd = pnl_arr * pos_notional
    wins = pnl_usd[pnl_usd > 0]
    losses = pnl_usd[pnl_usd < 0]
    m["profit_factor"] = round(_safe_div(float(np.sum(wins)), abs(float(np.sum(losses))), 0.0), 4)
    m["win_rate"] = round(len(wins) / n, 4) if n > 0 else 0.0
    m["payoff_ratio"] = round(
        _safe_div(float(np.mean(wins)) if len(wins) > 0 else 0.0,
                  abs(float(np.mean(losses))) if len(losses) > 0 else 1.0), 4)
    m["expectancy_usd"] = round(float(np.mean(pnl_usd)), 2)
    m["expectancy_frac"] = round(float(np.mean(pnl_arr)), 8)

    # Duration
    if "hold_bars" in trades.columns:
        m["avg_duration_bars"] = round(float(trades["hold_bars"].mean()), 1)
        m["avg_duration_hours"] = round(m["avg_duration_bars"] * 5 / 60, 2)
    else:
        m["avg_duration_bars"] = 0.0
        m["avg_duration_hours"] = 0.0

    # Consecutive wins/losses
    signs = np.sign(pnl_arr)
    max_cw = max_cl = cw = cl = 0
    for s in signs:
        if s > 0:
            cw += 1
            cl = 0
        elif s < 0:
            cl += 1
            cw = 0
        else:
            cw = cl = 0
        max_cw = max(max_cw, cw)
        max_cl = max(max_cl, cl)
    m["max_consec_wins"] = max_cw
    m["max_consec_losses"] = max_cl

    return m


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B: SIGNIFICANCE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def hac_ttest(daily_ret: np.ndarray) -> dict[str, Any]:
    """Newey-West HAC t-test for mean daily return > 0."""
    from statsmodels.regression.linear_model import OLS
    import statsmodels.tools as smtools

    T = len(daily_ret)
    result = {"test": "hac_ttest_mean_daily", "n_obs": T}

    if T < 5:
        result.update({"status": "INSUFFICIENT_DATA", "t_stat": float("nan"),
                       "p_value": float("nan"), "verdict": "INSUFFICIENT_DATA"})
        return result

    max_lags = max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    X = smtools.add_constant(np.ones(T))[:, :1]
    model = OLS(daily_ret, X).fit(cov_type="HAC",
                                   cov_kwds={"maxlags": max_lags, "kernel": "bartlett"})
    t_stat = float(model.tvalues[0])
    p_two = float(model.pvalues[0])
    p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2

    if p_one < 0.05:
        verdict = "PASS"
    elif p_one < 0.10:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    if T < 30:
        verdict += " (LOW_POWER)"

    result.update({"t_stat": round(t_stat, 4), "p_value": round(p_one, 6),
                   "max_lags": max_lags, "verdict": verdict})
    return result


def bootstrap_ci(daily_ret: np.ndarray) -> list[dict[str, Any]]:
    """Stationary bootstrap CIs for mean, Sharpe, MaxDD."""
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    T = len(daily_ret)
    results = []

    if T < 5:
        for stat in ["mean_daily", "sharpe_annual", "max_dd_pct"]:
            results.append({"statistic": stat, "status": "INSUFFICIENT_DATA",
                           "observed": float("nan"), "ci_95_lower": float("nan"),
                           "ci_95_upper": float("nan"), "p_value": float("nan"),
                           "verdict": "INSUFFICIENT_DATA"})
        return results

    # Optimal block length
    try:
        opt = optimal_block_length(daily_ret)
        avg_bl = max(1.0, float(opt.iloc[0]["stationary"]))
    except Exception:
        avg_bl = max(1.0, T ** (1 / 3))

    bs = StationaryBootstrap(avg_bl, daily_ret, seed=MC_SEED)

    # 1. Mean daily return
    boot_means = []
    for data in bs.bootstrap(N_BOOTSTRAP):
        boot_means.append(float(np.mean(data[0][0])))
    boot_means = np.array(boot_means)
    obs_mean = float(np.mean(daily_ret))
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    p_mean = float(np.mean(boot_means <= 0))
    v = "PASS" if ci_lo > 0 and p_mean < 0.05 else ("WARN" if p_mean < 0.10 else "FAIL")
    results.append({"statistic": "mean_daily", "observed": round(obs_mean, 8),
                    "ci_95_lower": round(ci_lo, 8), "ci_95_upper": round(ci_hi, 8),
                    "p_value": round(p_mean, 6), "n_bootstrap": N_BOOTSTRAP,
                    "avg_block_length": round(avg_bl, 2), "verdict": v})

    # 2. Sharpe (annualized)
    boot_sharpes = []
    for data in bs.bootstrap(N_BOOTSTRAP):
        d = data[0][0]
        s = np.std(d, ddof=1)
        boot_sharpes.append(float(np.mean(d) / s * SQRT_ANNUAL) if s > 0 else 0.0)
    boot_sharpes = np.array(boot_sharpes)
    obs_sharpe = float(np.mean(daily_ret) / np.std(daily_ret, ddof=1) * SQRT_ANNUAL) if np.std(daily_ret, ddof=1) > 0 else 0.0
    ci_lo_s, ci_hi_s = float(np.percentile(boot_sharpes, 2.5)), float(np.percentile(boot_sharpes, 97.5))
    p_sharpe = float(np.mean(boot_sharpes <= 0))
    v = "PASS" if ci_lo_s > 0 and p_sharpe < 0.05 else ("WARN" if p_sharpe < 0.10 else "FAIL")
    results.append({"statistic": "sharpe_annual", "observed": round(obs_sharpe, 4),
                    "ci_95_lower": round(ci_lo_s, 4), "ci_95_upper": round(ci_hi_s, 4),
                    "p_value": round(p_sharpe, 6), "n_bootstrap": N_BOOTSTRAP,
                    "avg_block_length": round(avg_bl, 2), "verdict": v})

    # 3. MaxDD (pct)
    boot_mdds = []
    for data in bs.bootstrap(N_BOOTSTRAP):
        d = data[0][0]
        cum = np.cumsum(d)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        boot_mdds.append(float(np.min(dd)) * 100)
    boot_mdds = np.array(boot_mdds)
    obs_cum = np.cumsum(daily_ret)
    obs_peak = np.maximum.accumulate(obs_cum)
    obs_mdd = float(np.min(obs_cum - obs_peak)) * 100
    ci_lo_m, ci_hi_m = float(np.percentile(boot_mdds, 2.5)), float(np.percentile(boot_mdds, 97.5))
    results.append({"statistic": "max_dd_pct", "observed": round(obs_mdd, 4),
                    "ci_95_lower": round(ci_lo_m, 4), "ci_95_upper": round(ci_hi_m, 4),
                    "p_value": float("nan"), "n_bootstrap": N_BOOTSTRAP,
                    "avg_block_length": round(avg_bl, 2), "verdict": "INFO"})

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C: MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════

def monte_carlo_sim(daily_ret: np.ndarray, daily_pnl: np.ndarray,
                    pos_notional: float) -> dict[str, Any]:
    """Monte Carlo via stationary bootstrap of daily returns."""
    from arch.bootstrap import StationaryBootstrap, optimal_block_length

    T = len(daily_ret)
    result: dict[str, Any] = {"n_sims": N_MC_SIMS, "n_days": T, "seed": MC_SEED}

    if T < 3:
        result["status"] = "INSUFFICIENT_DATA"
        return result

    try:
        opt = optimal_block_length(daily_ret)
        avg_bl = max(1.0, float(opt.iloc[0]["stationary"]))
    except Exception:
        avg_bl = max(1.0, T ** (1 / 3))

    result["avg_block_length"] = round(avg_bl, 2)

    bs = StationaryBootstrap(avg_bl, daily_pnl, seed=MC_SEED)

    final_eqs = []
    max_dds_pct = []
    max_dds_usd = []
    sharpes = []
    challenge_pass = 0
    challenge_fail = 0

    for data in bs.bootstrap(N_MC_SIMS):
        pnl_sim = data[0][0]  # resampled daily PnL USD
        eq = np.empty(len(pnl_sim) + 1)
        eq[0] = CAPITAL
        for i, p in enumerate(pnl_sim):
            eq[i + 1] = eq[i] + p

        final_eqs.append(eq[-1])

        # MaxDD
        peak = np.maximum.accumulate(eq)
        dd_usd = eq - peak
        dd_pct = np.where(peak > 0, dd_usd / peak, 0.0)
        max_dds_pct.append(float(np.min(dd_pct)))
        max_dds_usd.append(float(np.min(dd_usd)))

        # Sharpe
        ret_sim = pnl_sim / np.maximum(eq[:-1], 1)
        s = np.std(ret_sim, ddof=1) if len(ret_sim) > 1 else 0
        sharpes.append(float(np.mean(ret_sim) / s * SQRT_ANNUAL) if s > 0 else 0.0)

        # Challenge simulation
        ch_equity = CAPITAL
        ch_daily = {}
        passed = failed = False
        for i, p in enumerate(pnl_sim):
            day_idx = i
            ch_equity += p
            if day_idx not in ch_daily:
                ch_daily[day_idx] = 0.0
            ch_daily[day_idx] += p

            # Total stop
            if (ch_equity - CAPITAL) <= -CHALLENGE_TOTAL_MAX_LOSS:
                failed = True
                break
            # Target
            if (ch_equity - CAPITAL) >= CHALLENGE_PROFIT_TARGET:
                passed = True
                break

        if passed:
            challenge_pass += 1
        if failed:
            challenge_fail += 1

    final_eqs = np.array(final_eqs)
    max_dds_pct = np.array(max_dds_pct)
    sharpes = np.array(sharpes)

    pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in pctiles:
        result[f"equity_p{p:02d}"] = round(float(np.percentile(final_eqs, p)), 2)

    result["p_negative_pnl"] = round(float(np.mean(final_eqs < CAPITAL)), 4)

    for x in [5, 10, 15, 20]:
        result[f"p_dd_gt_{x}pct"] = round(float(np.mean(max_dds_pct < -x / 100)), 4)

    for p in [5, 25, 50, 75, 95]:
        result[f"sharpe_p{p:02d}"] = round(float(np.percentile(sharpes, p)), 4)
    result["p_sharpe_negative"] = round(float(np.mean(sharpes < 0)), 4)

    result["p_challenge_pass"] = round(challenge_pass / N_MC_SIMS, 4)
    result["p_challenge_fail"] = round(challenge_fail / N_MC_SIMS, 4)

    # Verdict
    v = "PASS"
    if result["p_negative_pnl"] > 0.35 or result["p_dd_gt_20pct"] > 0.20:
        v = "FAIL"
    elif result["p_negative_pnl"] > 0.15 or result["p_dd_gt_10pct"] > 0.50:
        v = "WARN"
    result["verdict"] = v

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D: ALPHA DECAY
# ══════════════════════════════════════════════════════════════════════════════

def alpha_decay_analysis(daily_ret: np.ndarray, dates: list[str]) -> dict[str, Any]:
    """Rolling window alpha decay analysis."""
    T = len(daily_ret)
    result: dict[str, Any] = {"trading_days_available": T, "min_required": 60}

    if T < 60:
        result["status"] = "INSUFFICIENT_DATA"
        return result

    windows = {}
    for W in [30, 60, 90]:
        if T < W:
            windows[f"W{W}"] = {"status": "INSUFFICIENT_DATA", "n_points": 0}
            continue

        rolling_mean = []
        for i in range(W - 1, T):
            window = daily_ret[i - W + 1: i + 1]
            rolling_mean.append(float(np.mean(window)))

        X = np.arange(len(rolling_mean))
        y = np.array(rolling_mean)

        if len(X) < 3:
            windows[f"W{W}"] = {"status": "INSUFFICIENT_DATA", "n_points": len(X)}
            continue

        # OLS slope
        X_d = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X_d, y, rcond=None)[0]
        slope = beta[1]
        residuals = y - X_d @ beta
        mse = np.sum(residuals ** 2) / max(len(y) - 2, 1)
        se_slope = np.sqrt(mse / max(np.sum((X - X.mean()) ** 2), 1e-12))
        t_stat = slope / se_slope if se_slope > 0 else 0.0
        p_val = 2 * (1 - ss.t.cdf(abs(t_stat), df=max(len(X) - 2, 1)))

        windows[f"W{W}"] = {
            "slope": round(slope, 10),
            "slope_t_stat": round(t_stat, 4),
            "slope_pvalue": round(p_val, 6),
            "n_points": len(X),
        }

    result["windows"] = windows

    # Half-life (exponential decay fit on W30 rolling mean if available)
    best_w = None
    for w in ["W90", "W60", "W30"]:
        if w in windows and "slope" in windows[w]:
            best_w = w
            break

    result["decay_detected"] = False
    result["half_life_days"] = None

    if best_w and T >= 30:
        W_int = int(best_w[1:])
        rolling_mean = []
        for i in range(W_int - 1, T):
            rolling_mean.append(float(np.mean(daily_ret[i - W_int + 1: i + 1])))
        X = np.arange(len(rolling_mean))
        y = np.array(rolling_mean)

        try:
            def exp_decay(t, a, lam, c):
                return a * np.exp(-lam * t) + c

            popt, _ = curve_fit(exp_decay, X, y,
                                p0=[y[0] - y[-1], 0.01, y[-1]],
                                maxfev=5000,
                                bounds=([-np.inf, -0.5, -np.inf], [np.inf, 2.0, np.inf]))
            lam = popt[1]
            if lam > 0:
                result["decay_detected"] = True
                result["half_life_days"] = round(np.log(2) / lam, 1)
                result["lambda"] = round(lam, 6)
        except Exception:
            pass

    # Verdict
    if result.get("status") == "INSUFFICIENT_DATA":
        pass
    elif not result["decay_detected"] or (result["half_life_days"] and result["half_life_days"] > 180):
        result["status"] = "PASS"
    elif result["half_life_days"] and result["half_life_days"] > 90:
        result["status"] = "WARN"
    else:
        result["status"] = "FAIL"

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E: WFO / OVERFITTING
# ══════════════════════════════════════════════════════════════════════════════

def wfo_fold_metrics(trades_oos: pl.DataFrame, pos_notional: float) -> list[dict]:
    """Per-fold OOS metrics."""
    folds = sorted(trades_oos["fold_id"].unique().to_list())
    rows = []
    for fid in folds:
        ft = trades_oos.filter(pl.col("fold_id") == fid)
        n = ft.height
        if n == 0:
            continue
        pnl = ft["net_pnl_base"].to_numpy()
        pnl_usd = pnl * pos_notional
        wins = pnl_usd[pnl_usd > 0]
        losses = pnl_usd[pnl_usd < 0]
        rows.append({
            "fold_id": fid,
            "n_trades": n,
            "total_return": round(float(np.sum(pnl)), 6),
            "total_pnl_usd": round(float(np.sum(pnl_usd)), 2),
            "mean_return": round(float(np.mean(pnl)), 8),
            "std_return": round(float(np.std(pnl, ddof=1)), 8) if n > 1 else 0.0,
            "win_rate": round(len(wins) / n, 4),
            "profit_factor": round(_safe_div(float(np.sum(wins)), abs(float(np.sum(losses)))), 4),
            "is_positive": float(np.sum(pnl)) > 0,
        })
    return rows


def pbo_analysis(tuning_results: pl.DataFrame, trades_oos: pl.DataFrame) -> dict[str, Any]:
    """PBO analysis. Since tuning_results only has IS scores, use approximate method."""
    result: dict[str, Any] = {"test": "PBO"}

    # Check if we have the data for proper PBO
    if "score" not in tuning_results.columns:
        result["status"] = "NOT_COMPUTABLE"
        result["reason"] = "no IS score in tuning_results"
        return result

    # Approximate PBO: for each fold, check if best IS combo leads to positive OOS
    folds = sorted(tuning_results["fold_id"].unique().to_list())
    n_overfit = 0
    n_folds_valid = 0

    for fid in folds:
        ft = tuning_results.filter(pl.col("fold_id") == fid)
        if ft.height == 0:
            continue

        # Best IS combo
        best_is = ft.sort("score", descending=True).head(1)
        best_score = float(best_is["score"][0])
        median_score = float(ft["score"].median())

        # Check OOS for this fold
        oos_fold = trades_oos.filter(pl.col("fold_id") == fid)
        if oos_fold.height == 0:
            continue

        oos_ret = float(oos_fold["net_pnl_base"].sum())
        n_folds_valid += 1

        # PBO proxy: best IS score is above median, but OOS is negative
        if oos_ret < 0 and best_score > median_score:
            n_overfit += 1

    if n_folds_valid == 0:
        result["status"] = "NOT_COMPUTABLE"
        result["reason"] = "no valid folds"
        return result

    pbo = n_overfit / n_folds_valid
    result["pbo"] = round(pbo, 4)
    result["n_folds_overfit"] = n_overfit
    result["n_folds_valid"] = n_folds_valid
    result["method"] = "proxy_is_rank_vs_oos_sign"
    result["note"] = "Approximate: full PBO requires OOS scores per combo (not stored)"

    if pbo < 0.50:
        result["verdict"] = "PASS"
    elif pbo < 0.75:
        result["verdict"] = "WARN"
    else:
        result["verdict"] = "FAIL"

    return result


def deflated_sharpe(daily_ret: np.ndarray, n_trials: int) -> dict[str, Any]:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014)."""
    T = len(daily_ret)
    result: dict[str, Any] = {"test": "DSR", "n_trials": n_trials, "n_obs": T}

    if T < 5 or n_trials < 1:
        result["status"] = "INSUFFICIENT_DATA"
        return result

    std_d = np.std(daily_ret, ddof=1)
    if std_d == 0:
        result["status"] = "ZERO_VARIANCE"
        return result

    sr_obs = float(np.mean(daily_ret) / std_d * SQRT_ANNUAL)

    # Expected SR under null (multiple testing)
    euler_gamma = 0.5772156649
    if n_trials > 1:
        log_n = np.log(n_trials)
        sr_expected = np.sqrt(2 * log_n) * (1 - euler_gamma / (2 * log_n))
    else:
        sr_expected = 0.0

    # SE of Sharpe (Lo 2002)
    skew = float(ss.skew(daily_ret, bias=False)) if T >= 3 else 0.0
    kurt = float(ss.kurtosis(daily_ret, bias=False)) if T >= 4 else 0.0
    se_sr = np.sqrt((1 + 0.5 * sr_obs ** 2 - skew * sr_obs + (kurt / 4) * sr_obs ** 2) / max(T, 1))

    if se_sr > 0:
        dsr_z = (sr_obs - sr_expected) / se_sr
        p_val = 1 - ss.norm.cdf(dsr_z)
    else:
        dsr_z = 0.0
        p_val = 1.0

    result["sr_observed"] = round(sr_obs, 4)
    result["sr_expected_null"] = round(sr_expected, 4)
    result["se_sr"] = round(se_sr, 4)
    result["dsr_z"] = round(dsr_z, 4)
    result["p_value"] = round(p_val, 6)

    if p_val < 0.05:
        result["verdict"] = "PASS"
    elif p_val < 0.10:
        result["verdict"] = "WARN"
    else:
        result["verdict"] = "FAIL"

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION F: COST SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

def cost_sensitivity(trades_oos: pl.DataFrame, pos_notional: float) -> list[dict]:
    """Recompute metrics at different cost multipliers."""
    rows = []
    for mult in COST_MULTIPLIERS:
        # Adjusted PnL
        adj_pnl_col = f"_pnl_cost_{mult}"
        trades_adj = trades_oos.with_columns(
            (pl.col("gross_pnl") - COST_BASE_DEC * mult).alias(adj_pnl_col)
        )

        pnl_arr = trades_adj[adj_pnl_col].to_numpy()
        pnl_usd = pnl_arr * pos_notional
        n = len(pnl_arr)

        # Daily aggregation
        df_daily = (
            trades_adj
            .with_columns(pl.col("exit_time_utc").cast(pl.Date).alias("td"))
            .group_by("td")
            .agg((pl.col(adj_pnl_col) * pos_notional).sum().alias("pnl_usd"))
            .sort("td")
        )
        daily_pnl_arr = df_daily["pnl_usd"].to_numpy()

        # Equity curve
        eq = np.empty(len(daily_pnl_arr) + 1)
        eq[0] = CAPITAL
        for i, p in enumerate(daily_pnl_arr):
            eq[i + 1] = eq[i] + p

        final_eq = eq[-1]
        total_ret = (final_eq / CAPITAL - 1) * 100

        # Daily returns
        daily_ret = []
        equity = CAPITAL
        for p in daily_pnl_arr:
            daily_ret.append(p / equity if equity > 0 else 0.0)
            equity += p
        daily_ret = np.array(daily_ret)

        T = len(daily_ret)
        std_d = float(np.std(daily_ret, ddof=1)) if T > 1 else 0.0
        mean_d = float(np.mean(daily_ret)) if T > 0 else 0.0
        sharpe = mean_d / std_d * SQRT_ANNUAL if std_d > 0 else 0.0

        # MaxDD
        peak = np.maximum.accumulate(eq)
        dd_pct = np.where(peak > 0, (eq - peak) / peak, 0.0)
        max_dd = float(np.min(dd_pct)) * 100

        wins = pnl_usd[pnl_usd > 0]
        losses = pnl_usd[pnl_usd < 0]

        still_investable = sharpe > 0 and total_ret > 0 and max_dd > -25

        # Challenge sim
        ch_eq = CAPITAL
        ch_pass = False
        ch_fail_daily = False
        ch_fail_total = False
        ch_daily_worst = 0.0
        ch_dd_worst = 0.0
        day_pnl = {}

        for i, p_usd in enumerate(pnl_usd):
            trade_date = str(trades_oos["exit_time_utc"][i].date()) if i < trades_oos.height else f"d{i}"
            if trade_date not in day_pnl:
                day_pnl[trade_date] = 0.0
            day_pnl[trade_date] += p_usd

            ch_eq += p_usd
            dd = ch_eq - CAPITAL
            ch_dd_worst = min(ch_dd_worst, dd)

            if dd <= -CHALLENGE_TOTAL_MAX_LOSS:
                ch_fail_total = True
                break
            if dd >= CHALLENGE_PROFIT_TARGET:
                ch_pass = True
                break

        for dpnl in day_pnl.values():
            ch_daily_worst = min(ch_daily_worst, dpnl)
        ch_fail_daily = ch_daily_worst <= -CHALLENGE_DAILY_MAX_LOSS

        rows.append({
            "cost_multiplier": mult,
            "cost_bps": round(COST_BASE_DEC * mult * 10000, 1),
            "sharpe_annual": round(sharpe, 4),
            "total_return_pct": round(total_ret, 4),
            "max_dd_pct": round(max_dd, 4),
            "win_rate": round(len(wins) / n, 4) if n > 0 else 0.0,
            "expectancy_usd": round(float(np.mean(pnl_usd)), 2),
            "profit_factor": round(_safe_div(float(np.sum(wins)), abs(float(np.sum(losses)))), 4),
            "still_investable": still_investable,
            "challenge_passed": ch_pass,
            "challenge_final_equity": round(ch_eq, 2),
            "challenge_violated_daily": ch_fail_daily,
            "challenge_violated_total": ch_fail_total,
            "verdict": "PASS" if still_investable else "FAIL",
        })

    # Breakeven multiplier
    gross_total = float(trades_oos["gross_pnl"].sum())
    n_trades = trades_oos.height
    if n_trades > 0 and COST_BASE_DEC > 0:
        m_breakeven = gross_total / (n_trades * COST_BASE_DEC)
    else:
        m_breakeven = float("inf")

    for r in rows:
        r["breakeven_multiplier"] = round(m_breakeven, 2)

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_pack(run_dir: Path, run_id: str, trades_oos: pl.DataFrame,
                trades_is: pl.DataFrame, all_results: dict[str, Any]) -> Path:
    """Export everything to institutional_pack/ folder."""
    pack_dir = run_dir / "institutional_pack"
    pack_dir.mkdir(exist_ok=True)

    # 1. Trades CSV
    trades_oos.write_csv(pack_dir / "trades_oos.csv")
    if trades_is.height > 0:
        trades_is.write_csv(pack_dir / "trades_is.csv")

    # 2. Core metrics
    rows = []
    for scenario in ["base", "stress"]:
        m = all_results["core_metrics"][scenario]
        m["scenario"] = scenario
        rows.append(m)
    pl.DataFrame(rows).write_csv(pack_dir / "metrics_core.csv")

    # 3. Fold metrics
    fold_rows = all_results["fold_metrics"]
    if fold_rows:
        pl.DataFrame(fold_rows).write_csv(pack_dir / "metrics_by_fold.csv")

    # 4. Significance tests
    sig_rows = [all_results["hac_ttest"]]
    pl.DataFrame(sig_rows).write_csv(pack_dir / "significance_tests.csv")

    # 5. Bootstrap CI
    boot_rows = all_results["bootstrap_ci"]
    pl.DataFrame(boot_rows).write_csv(pack_dir / "bootstrap_ci.csv")

    # 6. Monte Carlo
    mc = all_results["monte_carlo"]
    mc_rows = [{"metric": k, "value": v} for k, v in mc.items()]
    pl.DataFrame(mc_rows).write_csv(pack_dir / "monte_carlo_distribution.csv")

    # 7. Alpha decay
    decay = all_results["alpha_decay"]
    decay_flat = {"status": decay.get("status", "N/A"),
                  "trading_days": decay.get("trading_days_available", 0),
                  "min_required": decay.get("min_required", 60),
                  "decay_detected": decay.get("decay_detected", False),
                  "half_life_days": decay.get("half_life_days", "N/A")}
    if "windows" in decay:
        for wk, wv in decay["windows"].items():
            for k2, v2 in wv.items():
                decay_flat[f"{wk}_{k2}"] = v2
    pl.DataFrame([{"metric": k, "value": str(v)} for k, v in decay_flat.items()]).write_csv(
        pack_dir / "alpha_decay_report.csv")

    # 8. PBO report
    pbo = all_results["pbo"]
    dsr = all_results["dsr"]
    pbo_rows = [{"metric": k, "value": str(v)} for k, v in pbo.items()]
    pbo_rows += [{"metric": f"dsr_{k}", "value": str(v)} for k, v in dsr.items()]
    fold_data = all_results["fold_metrics"]
    n_pos = sum(1 for f in fold_data if f.get("is_positive"))
    n_total = len(fold_data)
    pbo_rows.insert(0, {"metric": "wfo_pass_rate", "value": str(round(n_pos / max(n_total, 1), 4))})
    pbo_rows.insert(1, {"metric": "n_folds_positive", "value": str(n_pos)})
    pbo_rows.insert(2, {"metric": "n_folds_total", "value": str(n_total)})
    pl.DataFrame(pbo_rows).write_csv(pack_dir / "pbo_report.csv")

    # 9. Cost sensitivity
    cost_rows = all_results["cost_sensitivity"]
    pl.DataFrame(cost_rows).write_csv(pack_dir / "cost_sensitivity.csv")

    # 10. Manifest
    manifest = {
        "version": "1.0.0",
        "generated_utc": _now_utc(),
        "run_id": run_id,
        "strategy": "BTCUSD_LONG_TREND_M5",
        "capital": CAPITAL,
        "risk_per_trade_usd": RISK_PER_TRADE_USD,
        "pos_notional": all_results["pos_notional"],
        "sl_return_median": all_results["sl_return_median"],
        "annualization_factor": ANNUAL_FACTOR,
        "cost_base_bps": COST_BASE_DEC * 10000,
        "cost_stress_bps": COST_STRESS_DEC * 10000,
        "n_bootstrap": N_BOOTSTRAP,
        "n_mc_sims": N_MC_SIMS,
        "oos_trades": trades_oos.height,
        "is_trades": trades_is.height,
        "symbols": sorted(trades_oos["symbol"].unique().to_list()),
        "oos_date_range": {
            "start": str(trades_oos["entry_time_utc"].min()),
            "end": str(trades_oos["exit_time_utc"].max()),
        },
        "overall_verdict": all_results["overall_verdict"],
        "verdict_breakdown": all_results["verdict_breakdown"],
        "files": {},
    }

    # Hash all output files
    for f in sorted(pack_dir.glob("*")):
        if f.name != "institutional_manifest.json":
            manifest["files"][f.name] = {"size_bytes": f.stat().st_size,
                                          "sha256_16": _file_hash(f)}

    (pack_dir / "institutional_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # 11. README
    readme = f"""INSTITUTIONAL STATISTICAL PACK — NB3 TREND v2
═══════════════════════════════════════════════

Run ID:    {run_id}
Generated: {_now_utc()}
Strategy:  BTCUSD LONG Trend-Following M5
Capital:   ${CAPITAL:,}
Risk/Trade: ${RISK_PER_TRADE_USD}

OVERALL VERDICT: {all_results["overall_verdict"]}

FILES:
  trades_oos.csv              — OOS trades with timestamps, PnL, exit reasons
  trades_is.csv               — IS trades (if available)
  metrics_core.csv            — Core metrics: Sharpe, Sortino, MaxDD, etc. (base + stress)
  metrics_by_fold.csv         — Per-fold OOS metrics (pass/fail by fold)
  significance_tests.csv      — HAC t-test for mean daily return > 0
  bootstrap_ci.csv            — Stationary bootstrap 95% CI: mean, Sharpe, MaxDD
  monte_carlo_distribution.csv — MC equity percentiles, P(negative), P(DD>X%)
  alpha_decay_report.csv      — Rolling window decay analysis + half-life
  pbo_report.csv              — WFO pass rate, PBO proxy, Deflated Sharpe Ratio
  cost_sensitivity.csv        — Performance at 1x/1.25x/1.5x/2x costs + challenge sim
  institutional_manifest.json — Metadata, hashes, parameters
  README.txt                  — This file

METRIC DEFINITIONS:
  - Returns: fractional per-trade (net_pnl_base = gross_pnl - 3bps roundtrip)
  - Daily returns: sum of trade PnL_USD / equity_t-1
  - Sharpe: annualized from daily returns, sqrt(365) factor (crypto 24/7)
  - Sortino: downside deviation using all observations, target=0
  - MaxDD: peak-to-trough on daily USD equity curve
  - VaR/CVaR: historical (non-parametric), 95% confidence
  - Bootstrap: Stationary (Politis & Romano 1994), automatic block length
  - Monte Carlo: {N_MC_SIMS:,} simulations via stationary bootstrap of daily PnL
  - PBO: Approximate (IS rank vs OOS sign per fold)
  - DSR: Bailey & López de Prado (2014), accounts for N trials tested

HOW TO USE:
  1. Open metrics_core.csv for executive summary (base vs stress)
  2. Check significance_tests.csv + bootstrap_ci.csv for statistical validity
  3. Review cost_sensitivity.csv for robustness to transaction costs
  4. Check pbo_report.csv for overfitting risk
  5. Upload this entire folder to ChatGPT for AI-assisted review
"""
    (pack_dir / "README.txt").write_text(readme, encoding="utf-8")

    return pack_dir


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Institutional Statistics Pack")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    project = Path(__file__).parent.parent
    run_id, run_dir = detect_run(project, args.run_id)
    print(f"[institutional_stats] Run: {run_id}")
    print(f"[institutional_stats] Dir: {run_dir}")

    # Load data
    data = load_data(run_dir)
    trades = data["overlay_trades"]

    # Filter to BTCUSD LONG (the GO symbol)
    trades = trades.filter(
        pl.col("symbol").is_in(["BTCUSD"]) & (pl.col("side") == "LONG")
    )
    trades_oos = trades.filter(pl.col("segment") == "OOS").sort("entry_time_utc")
    trades_is = trades.filter(pl.col("segment") == "IS").sort("entry_time_utc")

    print(f"[institutional_stats] Trades: OOS={trades_oos.height}, IS={trades_is.height}")

    # Sizing
    pos_notional, sl_ret_med = compute_pos_notional(trades_oos)
    print(f"[institutional_stats] Sizing: pos_notional=${pos_notional:,.0f}, "
          f"sl_ret_med={sl_ret_med:.4%}, 1SL=${sl_ret_med * pos_notional:,.2f}")

    all_results: dict[str, Any] = {
        "pos_notional": round(pos_notional, 2),
        "sl_return_median": round(sl_ret_med, 6),
    }

    # ── A: Core metrics (base + stress) ──
    print("[institutional_stats] Computing core metrics...")
    core = {}
    for scenario, pnl_col in [("base", "net_pnl_base"), ("stress", "net_pnl_stress")]:
        daily_ret, daily_pnl, dates = build_daily_returns(trades_oos, pos_notional, pnl_col)
        m = compute_core_metrics(trades_oos, daily_ret, daily_pnl, pos_notional, pnl_col)
        core[scenario] = m
        print(f"  [{scenario}] PnL=${m['total_pnl_usd']:+,.0f}, "
              f"Sharpe={m['sharpe_annual']:.3f}, MaxDD={m['max_dd_pct']:.1f}%, "
              f"WR={m['win_rate']:.1%}, PF={m['profit_factor']:.2f}")
    all_results["core_metrics"] = core

    # Use base scenario for downstream analysis
    daily_ret_base, daily_pnl_base, dates_base = build_daily_returns(
        trades_oos, pos_notional, "net_pnl_base")

    # ── B: Significance tests ──
    print("[institutional_stats] Running significance tests...")
    all_results["hac_ttest"] = hac_ttest(daily_ret_base)
    print(f"  HAC t-test: t={all_results['hac_ttest'].get('t_stat', 'N/A')}, "
          f"p={all_results['hac_ttest'].get('p_value', 'N/A')}, "
          f"verdict={all_results['hac_ttest'].get('verdict', 'N/A')}")

    print("[institutional_stats] Running bootstrap CIs (this may take ~30s)...")
    all_results["bootstrap_ci"] = bootstrap_ci(daily_ret_base)
    for b in all_results["bootstrap_ci"]:
        print(f"  Bootstrap {b['statistic']}: [{b.get('ci_95_lower','?')}, {b.get('ci_95_upper','?')}] "
              f"p={b.get('p_value','?')} -> {b.get('verdict','?')}")

    # ── C: Monte Carlo ──
    print(f"[institutional_stats] Running Monte Carlo ({N_MC_SIMS:,} sims)...")
    all_results["monte_carlo"] = monte_carlo_sim(daily_ret_base, daily_pnl_base, pos_notional)
    mc = all_results["monte_carlo"]
    print(f"  MC median equity: ${mc.get('equity_p50', 'N/A'):,.0f}, "
          f"P(neg)={mc.get('p_negative_pnl', 'N/A'):.1%}, "
          f"P(challenge pass)={mc.get('p_challenge_pass', 'N/A'):.1%}")

    # ── D: Alpha decay ──
    print("[institutional_stats] Analyzing alpha decay...")
    all_results["alpha_decay"] = alpha_decay_analysis(daily_ret_base, dates_base)
    print(f"  Alpha decay: {all_results['alpha_decay'].get('status', 'N/A')}")

    # ── E: WFO / Overfitting ──
    print("[institutional_stats] Computing WFO fold metrics...")
    all_results["fold_metrics"] = wfo_fold_metrics(trades_oos, pos_notional)
    n_pos = sum(1 for f in all_results["fold_metrics"] if f.get("is_positive"))
    n_total = len(all_results["fold_metrics"])
    pass_rate = n_pos / max(n_total, 1)
    print(f"  WFO pass rate: {n_pos}/{n_total} = {pass_rate:.0%}")

    print("[institutional_stats] Computing PBO...")
    tuning = data["tuning_results"].filter(pl.col("symbol") == "BTCUSD")
    all_results["pbo"] = pbo_analysis(tuning, trades_oos)
    print(f"  PBO: {all_results['pbo'].get('verdict', all_results['pbo'].get('status', 'N/A'))}")

    print("[institutional_stats] Computing Deflated Sharpe Ratio...")
    n_trials = tuning["fold_id"].n_unique() * 48  # combos per fold (approximate)
    # Actually get exact combo count
    n_combos = tuning.select(["sl_atr", "tp_atr", "trail_atr", "time_stop", "min_hold"]).unique().height
    all_results["dsr"] = deflated_sharpe(daily_ret_base, n_combos)
    print(f"  DSR: SR_obs={all_results['dsr'].get('sr_observed', 'N/A')}, "
          f"SR_null={all_results['dsr'].get('sr_expected_null', 'N/A')}, "
          f"verdict={all_results['dsr'].get('verdict', 'N/A')}")

    # ── F: Cost sensitivity ──
    print("[institutional_stats] Computing cost sensitivity...")
    all_results["cost_sensitivity"] = cost_sensitivity(trades_oos, pos_notional)
    for cs in all_results["cost_sensitivity"]:
        print(f"  Cost {cs['cost_multiplier']}x ({cs['cost_bps']}bps): "
              f"Sharpe={cs['sharpe_annual']:.3f}, Ret={cs['total_return_pct']:.1f}%, "
              f"Investable={'YES' if cs['still_investable'] else 'NO'}, "
              f"Challenge={'PASS' if cs['challenge_passed'] else 'FAIL'}")

    # ── Overall verdict ──
    verdicts = {"PASS": 0, "WARN": 0, "FAIL": 0, "N/A": 0, "INSUFFICIENT_DATA": 0}

    # Core metrics verdicts
    for scenario in ["base", "stress"]:
        for metric, val in core[scenario].items():
            v = _verdict(val, metric) if isinstance(val, (int, float)) else "N/A"
            if v in verdicts:
                verdicts[v] += 1

    # Other verdicts
    for item in [all_results["hac_ttest"]] + all_results["bootstrap_ci"]:
        v = item.get("verdict", "N/A").split(" ")[0]
        verdicts[v] = verdicts.get(v, 0) + 1

    mc_v = mc.get("verdict", "N/A")
    verdicts[mc_v] = verdicts.get(mc_v, 0) + 1

    decay_v = all_results["alpha_decay"].get("status", "N/A")
    verdicts[decay_v] = verdicts.get(decay_v, 0) + 1

    pbo_v = all_results["pbo"].get("verdict", all_results["pbo"].get("status", "N/A"))
    verdicts[pbo_v] = verdicts.get(pbo_v, 0) + 1

    dsr_v = all_results["dsr"].get("verdict", all_results["dsr"].get("status", "N/A"))
    verdicts[dsr_v] = verdicts.get(dsr_v, 0) + 1

    for cs in all_results["cost_sensitivity"]:
        verdicts[cs["verdict"]] = verdicts.get(cs["verdict"], 0) + 1

    if verdicts.get("FAIL", 0) > 0:
        overall = "FAIL"
    elif verdicts.get("WARN", 0) > 0:
        overall = "WARN"
    else:
        overall = "PASS"

    all_results["overall_verdict"] = overall
    all_results["verdict_breakdown"] = verdicts

    # ── Export ──
    print("[institutional_stats] Exporting institutional pack...")
    pack_dir = export_pack(run_dir, run_id, trades_oos, trades_is, all_results)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"INSTITUTIONAL PACK GENERATED")
    print(f"{'='*60}")
    print(f"  Directory: {pack_dir}")
    print(f"  Files:     {len(list(pack_dir.glob('*')))}")
    print(f"  Verdict:   {overall}")
    print(f"  Breakdown: {verdicts}")
    print(f"  Elapsed:   {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
