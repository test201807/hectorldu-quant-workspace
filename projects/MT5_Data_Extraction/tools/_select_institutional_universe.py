#!/usr/bin/env python3
"""
Selector Institucional de Universo LONG + SHORT.

Evalua TODOS los candidatos significativos del screener en ambas direcciones
(LONG + SHORT) usando WFO real con la estrategia TREND v2. Rankea por
estabilidad/consistencia per-fold (no rentabilidad bruta) y selecciona
top 10 LONG + top 10 SHORT mas estables.

Metricas de estabilidad per-fold:
  1. Fold consistency ratio (30%) — % de folds con return > 0
  2. Return stability (25%)      — inverse CV de returns por fold
  3. Worst fold score (20%)      — min fold return normalizado
  4. Profit factor stability (15%) — avg PF across folds
  5. Win rate consistency (10%)  — std de WR entre folds

Uso:
    cd C:\\Quant\\projects\\MT5_Data_Extraction
    venv1\\Scripts\\python.exe tools\\_select_institutional_universe.py [-v] [--save]
    venv1\\Scripts\\python.exe tools\\_select_institutional_universe.py --quick --save -v
    venv1\\Scripts\\python.exe tools\\_select_institutional_universe.py --symbols TSLA,NVDA,XAGUSD -v
    venv1\\Scripts\\python.exe tools\\_select_institutional_universe.py --top-long 10 --top-short 10 --save
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Paths — path_contract + strategylab
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "contracts"))
sys.path.insert(0, str(PROJECT_ROOT / "03_STRATEGY_LAB" / "src"))

from path_contract import m5_clean_dir, outputs_root  # noqa: E402
from strategylab.data_loader import load_bars_hive  # noqa: E402
from strategylab.signals_trend_v2 import compute_regime_gate  # noqa: E402
from strategylab.wfo import run_wfo, WFOResult  # noqa: E402
from strategylab.config import CostsConfig, RiskConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FEATURE_WIN = 288
ATR_WIN = 96
WARMUP_BARS = FEATURE_WIN + 10

# WFO config
WFO_IS_MONTHS = 18
WFO_OOS_MONTHS = 3
WFO_STEP_MONTHS = 3
WFO_EMBARGO_DAYS = 5

# Tuning grid — completo (18 combos)
PARAM_GRID: Dict[str, List[Any]] = {
    "sl_atr":             [1.5, 2.0, 2.5],
    "tp_atr":             [5.0, 7.0, 10.0],
    "trail_atr":          [0],
    "time_stop_bars":     [288, 576],
    "entry_confirm_bars": [6],
}

# Tuning grid — quick (1 combo)
PARAM_GRID_QUICK: Dict[str, List[Any]] = {
    "sl_atr":             [2.0],
    "tp_atr":             [7.0],
    "trail_atr":          [0],
    "time_stop_bars":     [576],
    "entry_confirm_bars": [6],
}

# Costs — 8 bps base (consistente con cost_model_snapshot_v2.json)
COSTS_CFG = CostsConfig(spread_bps=8.0, commission_bps=0.0, slippage_bps=0.0)

# Risk — relajado para evaluacion pura de edge
RISK_CFG = RiskConfig(
    max_drawdown_cap=-1.0,
    daily_loss_cap=-1.0,
    daily_profit_cap=1.0,
    max_trades_per_day=100,
)

# Screener filters
MIN_EDGE_BPS = 2.0
MAX_EDGE_BPS = 200.0
MIN_SIGNALS_SCREEN = 2000

# Stability thresholds
STABLE_SCORE = 55
MARGINAL_SCORE = 35
STABLE_MIN_FOLD_PCT = 0.5


# ---------------------------------------------------------------------------
# Features (replica NB3 + ATR para engine)
# ---------------------------------------------------------------------------

def compute_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula ER, momentum, volatilidad y ATR en precio."""
    w = FEATURE_WIN

    # Stage 1 — base
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret"),
        (pl.col("close") - pl.col("close").shift(1)).abs().alias("abs_diff"),
    ])

    # Stage 2 — ER/mom/vol (nombres que espera compute_regime_gate)
    df = df.with_columns([
        (
            (pl.col("close") - pl.col("close").shift(w)).abs()
            / pl.col("abs_diff").rolling_sum(w)
        ).alias("er_288"),
        ((pl.col("close") / pl.col("close").shift(w) - 1.0) * 10_000).alias("mom_bps_288"),
        (pl.col("ret").rolling_std(w) * 10_000).alias("vol_bps_288"),
    ])

    # Stage 3 — ATR en precio (para el engine)
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    df = df.with_columns(
        tr.rolling_mean(ATR_WIN).alias("atr_price")
    )

    return df


# ---------------------------------------------------------------------------
# Clasificacion de activos
# ---------------------------------------------------------------------------

def classify_asset(symbol: str) -> str:
    """Clasifica simbolo en asset class por heuristicas de nombre."""
    s = symbol.upper()
    if any(c in s for c in ("BTC", "ETH", "LTC", "XRP", "BNB", "SOL",
                            "DOGE", "ADA", "DOT", "AVAX", "LINK", "UNI",
                            "ICP", "DASH", "EOS", "NEAR", "APE", "SHIB")):
        return "crypto"
    if any(m in s for m in ("XAU", "XAG", "XPT", "XPD")):
        return "metal"
    if any(o in s for o in ("WTI", "BRENT", "OIL", "UKOIL", "USOIL",
                            "XTIUSD", "XBRUSD", "NGAS")):
        return "energy"
    if any(idx in s for idx in ("US500", "US100", "US30", "SPX", "NAS",
                                "NDX", "DAX", "FTSE", "NI225", "JP225",
                                "GER", "UK100", "FRA40", "AUS200",
                                "HK50", "STOXX")):
        return "index"
    forex_ccys = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF",
                  "SEK", "NOK", "DKK", "SGD", "HKD", "MXN", "ZAR", "TRY",
                  "PLN", "HUF", "CZK", "CNH", "CNY"}
    for ccy in forex_ccys:
        if s.startswith(ccy) or s.endswith(ccy):
            return "forex"
    return "other"


# ---------------------------------------------------------------------------
# Carga de candidatos del screener
# ---------------------------------------------------------------------------

def load_all_screener_candidates(screener_path: Path) -> pl.DataFrame:
    """Lee TODOS los candidatos significativos del screener (ambos sides)."""
    if not screener_path.exists():
        print(f"ERROR: no se encuentra screener output: {screener_path}")
        sys.exit(1)

    df = pl.read_parquet(screener_path)

    # Filtrar OOS significativos con edge razonable
    candidates = df.filter(
        (pl.col("segment") == "OOS")
        & (pl.col("significant") == True)  # noqa: E712
        & (pl.col("edge_bps") >= MIN_EDGE_BPS)
        & (pl.col("edge_bps") <= MAX_EDGE_BPS)
        & (pl.col("n_signals") >= MIN_SIGNALS_SCREEN)
    )

    return candidates.sort("edge_bps", descending=True)


def discover_symbols(m5_dir: Path) -> List[str]:
    """Descubre simbolos disponibles en hive partition."""
    return sorted(
        d.name.split("=", 1)[1]
        for d in m5_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith("symbol=")
    )


# ---------------------------------------------------------------------------
# Metricas de estabilidad per-fold
# ---------------------------------------------------------------------------

def extract_per_fold_metrics(wfo_result: WFOResult) -> List[Dict[str, float]]:
    """Extrae metricas per-fold de WFOResult.best_per_fold."""
    per_fold = []
    for fold_id in sorted(wfo_result.best_per_fold.keys()):
        grid_result = wfo_result.best_per_fold[fold_id]
        kpis = grid_result.kpis_oos
        per_fold.append({
            "fold_id":     fold_id,
            "fold_return": kpis.get("total_return", 0.0),
            "fold_mdd":    kpis.get("mdd", 0.0),
            "fold_wr":     kpis.get("hit_rate", 0.0),
            "fold_pf":     kpis.get("profit_factor", 0.0),
            "fold_sharpe": kpis.get("sharpe", 0.0),
            "fold_trades": grid_result.trades_oos,
        })
    return per_fold


def compute_stability_score(per_fold: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calcula stability score compuesto (0-100) a partir de metricas per-fold.

    Componentes:
      1. Fold consistency ratio (30%) — % de folds con return > 0
      2. Return stability (25%)      — inverse CV de returns por fold
      3. Worst fold score (20%)      — min fold return normalizado
      4. Profit factor stability (15%) — avg PF across folds
      5. Win rate consistency (10%)  — std de WR entre folds
    """
    n_folds = len(per_fold)
    if n_folds == 0:
        return {
            "stability_score": 0.0,
            "fold_consistency": 0.0,
            "return_stability": 0.0,
            "worst_fold_score": 0.0,
            "pf_score": 0.0,
            "wr_score": 0.0,
            "ret_avg": 0.0,
            "ret_cv": 0.0,
            "min_ret": 0.0,
            "avg_pf": 0.0,
            "wr_std": 0.0,
        }

    fold_returns = [f["fold_return"] for f in per_fold]
    fold_pfs = [f["fold_pf"] for f in per_fold]
    fold_wrs = [f["fold_wr"] for f in per_fold]

    # 1. Fold consistency ratio (30%)
    n_positive = sum(1 for r in fold_returns if r > 0)
    fold_consistency = n_positive / n_folds

    # 2. Return stability (25%) — inverse CV
    ret_mean = float(np.mean(fold_returns))
    ret_std = float(np.std(fold_returns, ddof=0))
    if abs(ret_mean) > 1e-9:
        cv = ret_std / abs(ret_mean)
    else:
        cv = 999.0
    return_stability = max(0.0, 1.0 - cv)

    # 3. Worst fold score (20%) — min return normalizado
    min_ret = min(fold_returns)
    # 0 si peor fold < -10%, 1 si peor fold >= 0%
    worst_fold_score = max(0.0, min(1.0, (min_ret + 0.10) / 0.10))

    # 4. Profit factor stability (15%) — avg PF
    avg_pf = float(np.mean(fold_pfs))
    # 0 si PF < 0.8, 1 si PF > 1.5
    pf_score = min(1.0, max(0.0, (avg_pf - 0.8) / 0.7))

    # 5. Win rate consistency (10%) — std de WR
    wr_std = float(np.std(fold_wrs, ddof=0))
    # 1 si std == 0, 0 si std >= 15%
    wr_score = max(0.0, 1.0 - wr_std / 0.15)

    stability_score = (
        0.30 * fold_consistency
        + 0.25 * return_stability
        + 0.20 * worst_fold_score
        + 0.15 * pf_score
        + 0.10 * wr_score
    ) * 100

    return {
        "stability_score": stability_score,
        "fold_consistency": fold_consistency,
        "return_stability": return_stability,
        "worst_fold_score": worst_fold_score,
        "pf_score": pf_score,
        "wr_score": wr_score,
        "ret_avg": ret_mean,
        "ret_cv": cv if cv < 999 else float("nan"),
        "min_ret": min_ret,
        "avg_pf": avg_pf,
        "wr_std": wr_std,
    }


def classify_stability(
    stability_score: float,
    fold_consistency: float,
    total_return: float,
) -> str:
    """Clasifica: STABLE, MARGINAL o UNSTABLE."""
    if (stability_score >= STABLE_SCORE
            and fold_consistency >= STABLE_MIN_FOLD_PCT
            and total_return > 0):
        return "STABLE"
    if stability_score >= MARGINAL_SCORE and total_return > 0:
        return "MARGINAL"
    return "UNSTABLE"


# ---------------------------------------------------------------------------
# Evaluar un candidato
# ---------------------------------------------------------------------------

def evaluate_candidate(
    m5_dir: Path,
    symbol: str,
    side: str,
    param_grid: Dict[str, List[Any]],
    verbose: bool = False,
) -> Optional[Dict]:
    """
    Ejecuta WFO real para un (symbol, side) y extrae metricas de estabilidad
    per-fold. Retorna dict con KPIs agregados + stability score, o None.
    """
    t0 = time.perf_counter()

    # Cargar datos
    try:
        df = load_bars_hive(m5_dir, symbols=[symbol])
    except FileNotFoundError:
        if verbose:
            print(f"  {symbol}: sin datos M5")
        return None

    if df.height < WARMUP_BARS + 1000:
        if verbose:
            print(f"  {symbol}: insuficientes barras ({df.height:,})")
        return None

    # Filtrar simbolo si hay columna symbol
    if "symbol" in df.columns:
        df = df.filter(pl.col("symbol") == symbol)

    df = df.sort("time_utc")

    # Features
    df = compute_features(df)

    # Descartar warmup
    df = df.slice(WARMUP_BARS)

    if df.height < 2000:
        if verbose:
            print(f"  {symbol}: pocas barras tras warmup ({df.height:,})")
        return None

    # Regime gate
    gate_long, gate_short, _gate_params = compute_regime_gate(
        df, q_er=0.60, q_mom_long=0.55, q_mom_short=0.45, q_vol=0.90,
    )

    # Filtrar por side
    n = df.height
    if side == "LONG":
        sig_long = gate_long
        sig_short = [False] * n
    else:
        sig_long = [False] * n
        sig_short = gate_short

    n_signals = sum(sig_long) + sum(sig_short)
    if n_signals < 100:
        if verbose:
            print(f"  {symbol} {side}: pocas senales gate ({n_signals})")
        return None

    # WFO
    wfo_result: WFOResult = run_wfo(
        df=df,
        signal_long=sig_long,
        signal_short=sig_short,
        param_grid=param_grid,
        costs_cfg=COSTS_CFG,
        risk_cfg=RISK_CFG,
        symbol=symbol,
        is_months=WFO_IS_MONTHS,
        oos_months=WFO_OOS_MONTHS,
        step_months=WFO_STEP_MONTHS,
        embargo_days=WFO_EMBARGO_DAYS,
        min_folds=0,
        max_combos=100,
        min_trades_is=10,
    )

    elapsed = time.perf_counter() - t0

    # Resultados agregados
    n_folds = len(wfo_result.folds)
    n_folds_ok = len(wfo_result.best_per_fold)
    kpis = wfo_result.oos_kpis
    n_oos = kpis.get("n_trades", 0)

    if n_oos == 0:
        if verbose:
            print(f"  {symbol} {side}: 0 trades OOS ({elapsed:.1f}s)")
        return None

    # Metricas per-fold
    per_fold = extract_per_fold_metrics(wfo_result)
    stability = compute_stability_score(per_fold)

    total_ret = kpis.get("total_return", 0.0)
    status = classify_stability(
        stability["stability_score"],
        stability["fold_consistency"],
        total_ret,
    )

    result = {
        # Identificacion
        "symbol":           symbol,
        "side":             side,
        "asset_class":      classify_asset(symbol),
        # KPIs agregados OOS
        "n_oos":            n_oos,
        "total_return":     total_ret,
        "mdd":              kpis.get("mdd", 0.0),
        "sharpe":           kpis.get("sharpe", 0.0),
        "hit_rate":         kpis.get("hit_rate", 0.0),
        "profit_factor":    kpis.get("profit_factor", 0.0),
        "expectancy":       kpis.get("expectancy", 0.0),
        # Folds
        "n_folds":          n_folds,
        "n_folds_ok":       n_folds_ok,
        # Stability score y componentes
        "stability_score":  stability["stability_score"],
        "fold_consistency": stability["fold_consistency"],
        "return_stability": stability["return_stability"],
        "worst_fold_score": stability["worst_fold_score"],
        "pf_score":         stability["pf_score"],
        "wr_score":         stability["wr_score"],
        # Metricas derivadas per-fold
        "ret_avg":          stability["ret_avg"],
        "ret_cv":           stability["ret_cv"],
        "min_ret":          stability["min_ret"],
        "avg_pf":           stability["avg_pf"],
        "wr_std":           stability["wr_std"],
        # Status y meta
        "status":           status,
        "elapsed_s":        elapsed,
    }

    if verbose:
        mark = {"STABLE": "+", "MARGINAL": "~", "UNSTABLE": " "}[status]
        print(f"  {symbol:<12} {side:<6} N={n_oos:<5} "
              f"Ret={total_ret:>+7.2%} Stab={stability['stability_score']:>5.1f} "
              f"Fold%={stability['fold_consistency']:.0%} "
              f"AvgPF={stability['avg_pf']:>4.2f} "
              f"MinR={stability['min_ret']:>+6.2%} "
              f"Folds={n_folds_ok}/{n_folds} {mark} ({elapsed:.1f}s)")

    return result


# ---------------------------------------------------------------------------
# Reportes
# ---------------------------------------------------------------------------

def print_top_table(
    results: List[Dict],
    side: str,
    top_n: int,
) -> List[Dict]:
    """Imprime tabla de top N candidatos para un side, rankeados por stability."""
    side_results = [r for r in results if r["side"] == side]
    side_results.sort(key=lambda r: r["stability_score"], reverse=True)
    top = side_results[:top_n]

    print(f"\n{'='*120}")
    print(f"TOP {top_n} {side} — Most Stable (Institutional)")
    print(f"{'='*120}")
    print(f"{'Rank':>4}  {'Symbol':<12} {'Class':<8} {'Stab':>5} {'Fold%':>5} "
          f"{'RetAvg':>8} {'RetCV':>6} {'MinRet':>8} {'AvgPF':>6} "
          f"{'WRstd':>6} {'N_OOS':>6} {'Status':<9}")
    print("-" * 120)

    for i, r in enumerate(top, 1):
        cv_str = f"{r['ret_cv']:.2f}" if not np.isnan(r.get("ret_cv", float("nan"))) else "  N/A"
        print(f"{i:>4}  {r['symbol']:<12} {r['asset_class']:<8} "
              f"{r['stability_score']:>5.1f} {r['fold_consistency']:>4.0%} "
              f"{r['ret_avg']:>+7.1%} {cv_str:>6} "
              f"{r['min_ret']:>+7.1%} {r['avg_pf']:>6.2f} "
              f"{r['wr_std']*100:>5.1f}% {r['n_oos']:>6} {r['status']:<9}")

    if not top:
        print(f"  (sin candidatos {side} evaluados)")

    return top


def print_portfolio_recommendation(
    top_long: List[Dict],
    top_short: List[Dict],
) -> None:
    """Imprime recomendacion de portfolio."""
    print(f"\n{'='*120}")
    print("PORTFOLIO RECOMENDADO")
    print(f"{'='*120}")

    stable_long = [r for r in top_long if r["status"] == "STABLE"]
    stable_short = [r for r in top_short if r["status"] == "STABLE"]
    marginal_long = [r for r in top_long if r["status"] == "MARGINAL"]
    marginal_short = [r for r in top_short if r["status"] == "MARGINAL"]

    def _sym_list(results: List[Dict]) -> str:
        if not results:
            return "(ninguno)"
        return ", ".join(r["symbol"] for r in results)

    print(f"  LONG  STABLE:   {_sym_list(stable_long)}")
    print(f"  LONG  MARGINAL: {_sym_list(marginal_long)}")
    print(f"  SHORT STABLE:   {_sym_list(stable_short)}")
    print(f"  SHORT MARGINAL: {_sym_list(marginal_short)}")

    # Diversificacion
    all_top = top_long + top_short
    viable = [r for r in all_top if r["status"] in ("STABLE", "MARGINAL")]
    unique_symbols = set(r["symbol"] for r in viable)
    classes = set(r["asset_class"] for r in viable)

    print(f"\n  Activos unicos (STABLE+MARGINAL): {len(unique_symbols)}")
    print(f"  Asset classes representadas:       {len(classes)} — {', '.join(sorted(classes))}")

    if stable_long or stable_short:
        avg_stab = np.mean([r["stability_score"] for r in (stable_long + stable_short)])
        print(f"  Stability score promedio STABLE:  {avg_stab:.1f}")


def print_full_ranking(results: List[Dict], side: str) -> None:
    """Imprime ranking completo de un side (todos los evaluados)."""
    side_results = [r for r in results if r["side"] == side]
    side_results.sort(key=lambda r: r["stability_score"], reverse=True)

    n_stable = sum(1 for r in side_results if r["status"] == "STABLE")
    n_marginal = sum(1 for r in side_results if r["status"] == "MARGINAL")
    n_unstable = sum(1 for r in side_results if r["status"] == "UNSTABLE")

    print(f"\n  {side} totals: {len(side_results)} evaluados | "
          f"STABLE={n_stable} MARGINAL={n_marginal} UNSTABLE={n_unstable}")


def save_results(results: List[Dict], output_dir: Path) -> None:
    """Guarda resultados completos en parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "select_institutional_universe.parquet"
    df = pl.DataFrame(results)
    df.write_parquet(out_path)
    print(f"\nResultados guardados en: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Selector Institucional de Universo LONG + SHORT"
    )
    parser.add_argument("--top-long", type=int, default=10,
                        help="Top N candidatos LONG a mostrar (default: 10)")
    parser.add_argument("--top-short", type=int, default=10,
                        help="Top N candidatos SHORT a mostrar (default: 10)")
    parser.add_argument("--quick", action="store_true",
                        help="Grid reducido para iteracion rapida (1 combo)")
    parser.add_argument("--save", action="store_true",
                        help="Guardar resultados en parquet")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Simbolos especificos (ej: TSLA,NVDA). "
                             "Bypass screener, evalua ambos sides.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Salida detallada")
    args = parser.parse_args()

    t0 = time.perf_counter()

    print("\n" + "=" * 120)
    print("SELECTOR INSTITUCIONAL — Universo LONG + SHORT (Stability Ranking)")
    print("=" * 120)

    m5_dir = m5_clean_dir(PROJECT_ROOT)
    if not m5_dir.exists():
        print(f"ERROR: directorio M5 no encontrado: {m5_dir}")
        sys.exit(1)

    # Seleccionar grid
    param_grid = PARAM_GRID_QUICK if args.quick else PARAM_GRID
    grid_combos = 1
    for v in param_grid.values():
        grid_combos *= len(v)

    # Construir lista de candidatos: (symbol, side)
    candidates: List[Tuple[str, str]] = []

    if args.symbols:
        # Bypass screener — evaluar ambos sides
        for sym in args.symbols.split(","):
            sym = sym.strip().upper()
            candidates.append((sym, "LONG"))
            candidates.append((sym, "SHORT"))
        source = "CLI --symbols"
    else:
        # Leer screener — todos los significativos
        screener_path = (outputs_root(PROJECT_ROOT) / "screening"
                         / "screen_trend_universe.parquet")
        screener_df = load_all_screener_candidates(screener_path)

        if screener_df.is_empty():
            print("ERROR: sin candidatos significativos. Ejecutar screener primero.")
            sys.exit(1)

        # Extraer simbolos unicos del screener
        screener_symbols = set()
        for row in screener_df.iter_rows(named=True):
            screener_symbols.add(row["symbol"])

        # Para cada simbolo, evaluar AMBOS sides
        for sym in sorted(screener_symbols):
            candidates.append((sym, "LONG"))
            candidates.append((sym, "SHORT"))

        n_screener_rows = screener_df.height
        source = f"screener ({n_screener_rows} OOS significativos, {len(screener_symbols)} simbolos unicos)"

    # Config summary
    print(f"Fuente:          {source}")
    print(f"Candidatos:      {len(candidates)} (symbol x side)")
    print(f"Grid:            {grid_combos} combo{'s' if grid_combos > 1 else ''} x folds"
          f"{' (QUICK mode)' if args.quick else ''}")
    print(f"WFO:             IS={WFO_IS_MONTHS}m, OOS={WFO_OOS_MONTHS}m, "
          f"step={WFO_STEP_MONTHS}m, embargo={WFO_EMBARGO_DAYS}d")
    print(f"Costs:           {COSTS_CFG.spread_bps} bps spread "
          f"(roundtrip {COSTS_CFG.total_roundtrip_dec * 10_000:.1f} bps)")
    print(f"Stability:       STABLE>={STABLE_SCORE}, MARGINAL>={MARGINAL_SCORE}")
    print(f"Top output:      {args.top_long} LONG + {args.top_short} SHORT")
    print()

    # Procesar candidatos
    results: List[Dict] = []
    n_errors = 0

    for i, (symbol, side) in enumerate(candidates, 1):
        pct = i / len(candidates) * 100
        if not args.verbose:
            print(f"\r  Evaluando {i}/{len(candidates)} ({pct:.0f}%) — "
                  f"{symbol:<12} {side:<6}", end="", flush=True)

        try:
            result = evaluate_candidate(
                m5_dir, symbol, side, param_grid, verbose=args.verbose,
            )
            if result is not None:
                results.append(result)
        except Exception as e:
            n_errors += 1
            if args.verbose:
                print(f"  {symbol} {side}: ERROR — {e}")

    if not args.verbose:
        print()  # newline tras progreso

    elapsed = time.perf_counter() - t0

    if not results:
        print(f"\nSin resultados WFO. Verificar datos M5 y screener. ({elapsed:.1f}s)")
        sys.exit(1)

    # Reportes
    top_long = print_top_table(results, "LONG", args.top_long)
    top_short = print_top_table(results, "SHORT", args.top_short)
    print_full_ranking(results, "LONG")
    print_full_ranking(results, "SHORT")
    print_portfolio_recommendation(top_long, top_short)

    # Guardar
    if args.save:
        out_dir = outputs_root(PROJECT_ROOT) / "screening"
        save_results(results, out_dir)

    # Resumen final
    print(f"\n{'='*120}")
    print(f"Tiempo total: {elapsed:.1f}s | "
          f"Evaluados: {len(results)}/{len(candidates)} | "
          f"Errores: {n_errors}")
    if args.quick:
        print(f"NOTA: modo --quick (1 combo). Para ranking definitivo, ejecutar sin --quick.")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
