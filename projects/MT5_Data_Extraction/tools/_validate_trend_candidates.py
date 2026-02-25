#!/usr/bin/env python3
"""
Validador WFO de candidatos TREND — Fase 2+3.

Lee resultados del screener (forward returns), filtra top N candidatos,
ejecuta WFO real con el engine de strategylab (SL/TP/trail, costos,
risk management) para obtener KPIs concluyentes.

Uso:
    cd C:\Quant\projects\MT5_Data_Extraction
    venv1\Scripts\python.exe tools\_validate_trend_candidates.py
    venv1\Scripts\python.exe tools\_validate_trend_candidates.py --top 20 --save
    venv1\Scripts\python.exe tools\_validate_trend_candidates.py --symbols XAUUSD,XAGUSD -v
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
from strategylab.metrics import compute_kpis  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FEATURE_WIN = 288
ATR_WIN = 96
WARMUP_BARS = FEATURE_WIN + 10

# WFO config
DEFAULT_TOP = 20
WFO_IS_MONTHS = 18
WFO_OOS_MONTHS = 3
WFO_STEP_MONTHS = 3
WFO_EMBARGO_DAYS = 5

# Tuning grid — subset eficiente del NB3 Cell 14
PARAM_GRID: Dict[str, List[Any]] = {
    "sl_atr":             [1.5, 2.0, 2.5],
    "tp_atr":             [5.0, 7.0, 10.0],
    "trail_atr":          [0],             # TRAIL=0 óptimo en NB3
    "time_stop_bars":     [288, 576],
    "entry_confirm_bars": [6],             # consistente con NB3 Cell 10
}
# → 18 combinaciones por fold

# Costs — 8 bps base (consistente con cost_model_snapshot_v2.json)
COSTS_CFG = CostsConfig(spread_bps=8.0, commission_bps=0.0, slippage_bps=0.0)

# Risk — relajado para validacion pura de edge.
# run_engine recorre todos los bars con un solo tracker; si IS acumula
# drawdown > cap, is_killed() bloquea todo OOS.  Desactivamos caps para
# evaluar edge sin interferencia de risk management (se aplica en live).
RISK_CFG = RiskConfig(
    max_drawdown_cap=-1.0,
    daily_loss_cap=-1.0,
    daily_profit_cap=1.0,
    max_trades_per_day=100,
)

# Shortlist filters (del screener)
MIN_EDGE_BPS = 2.0
MAX_EDGE_BPS = 200.0
MIN_SIGNALS_SCREEN = 2000

# WFO pass/fail
MIN_TRADES_PASS = 30
MIN_SHARPE_WARN = 0.5
MIN_PF_WARN = 1.1


# ---------------------------------------------------------------------------
# Features (réplica NB3 + ATR para engine)
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
# Carga de shortlist del screener
# ---------------------------------------------------------------------------

def load_screener_shortlist(
    screener_path: Path,
    top_n: int,
) -> pl.DataFrame:
    """Lee resultados del screener y filtra shortlist."""
    if not screener_path.exists():
        print(f"ERROR: no se encuentra screener output: {screener_path}")
        sys.exit(1)

    df = pl.read_parquet(screener_path)

    # Filtrar OOS significativos
    shortlist = df.filter(
        (pl.col("segment") == "OOS")
        & (pl.col("significant") == True)  # noqa: E712
        & (pl.col("edge_bps") >= MIN_EDGE_BPS)
        & (pl.col("edge_bps") <= MAX_EDGE_BPS)
        & (pl.col("n_signals") >= MIN_SIGNALS_SCREEN)
    )

    # Ordenar por edge descendente, tomar top N
    shortlist = shortlist.sort("edge_bps", descending=True).head(top_n)

    return shortlist


# ---------------------------------------------------------------------------
# Descubrir símbolos (para bypass screener)
# ---------------------------------------------------------------------------

def discover_symbols(m5_dir: Path) -> List[str]:
    """Descubre símbolos disponibles en hive partition."""
    return sorted(
        d.name.split("=", 1)[1]
        for d in m5_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith("symbol=")
    )


# ---------------------------------------------------------------------------
# Validar un candidato via WFO
# ---------------------------------------------------------------------------

def validate_candidate(
    m5_dir: Path,
    symbol: str,
    side: str,
    verbose: bool = False,
) -> Optional[Dict]:
    """Ejecuta WFO real para un (symbol, side). Retorna dict con KPIs o None."""
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

    # Filtrar símbolo si hay columna symbol
    if "symbol" in df.columns:
        df = df.filter(pl.col("symbol") == symbol)

    # Ordenar
    df = df.sort("time_utc")

    # Features
    df = compute_features(df)

    # Descartar warmup (NaN de rolling windows)
    df = df.slice(WARMUP_BARS)

    if df.height < 2000:
        if verbose:
            print(f"  {symbol}: pocas barras tras warmup ({df.height:,})")
        return None

    # Regime gate
    gate_long, gate_short, gate_params = compute_regime_gate(
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
            print(f"  {symbol} {side}: pocas señales gate ({n_signals})")
        return None

    # WFO
    wfo_result: WFOResult = run_wfo(
        df=df,
        signal_long=sig_long,
        signal_short=sig_short,
        param_grid=PARAM_GRID,
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

    # Extraer resultados
    n_folds = len(wfo_result.folds)
    n_folds_ok = len(wfo_result.best_per_fold)
    kpis = wfo_result.oos_kpis
    n_oos = kpis.get("n_trades", 0)

    if n_oos == 0:
        if verbose:
            print(f"  {symbol} {side}: 0 trades OOS ({elapsed:.1f}s)")
        return None

    # Clasificar PASS/WARN/FAIL
    total_ret = kpis.get("total_return", 0.0)
    sharpe = kpis.get("sharpe", 0.0)
    pf = kpis.get("profit_factor", 0.0)
    wr = kpis.get("hit_rate", 0.0)

    if total_ret > 0 and sharpe > 0 and n_oos >= MIN_TRADES_PASS and pf > 1.0:
        if sharpe < MIN_SHARPE_WARN or pf < MIN_PF_WARN:
            status = "WARN"
        else:
            status = "PASS"
    else:
        status = "FAIL"

    result = {
        "symbol": symbol,
        "side": side,
        "n_oos": n_oos,
        "total_return": total_ret,
        "mdd": kpis.get("mdd", 0.0),
        "sharpe": sharpe,
        "hit_rate": wr,
        "profit_factor": pf,
        "expectancy": kpis.get("expectancy", 0.0),
        "n_folds": n_folds,
        "n_folds_ok": n_folds_ok,
        "status": status,
        "elapsed_s": elapsed,
    }

    if verbose:
        mark = "*" if status == "PASS" else ("~" if status == "WARN" else " ")
        print(f"  {symbol:<12} {side:<6} N={n_oos:<5} "
              f"Ret={total_ret:>+7.2%} MDD={kpis['mdd']:>+7.2%} "
              f"Sharpe={sharpe:>5.2f} WR={wr:.0%} PF={pf:>4.2f} "
              f"Folds={n_folds_ok}/{n_folds} {mark} ({elapsed:.1f}s)")

    return result


# ---------------------------------------------------------------------------
# Reportes
# ---------------------------------------------------------------------------

def _classify_asset(symbol: str) -> str:
    """Clasifica símbolo en asset class (reutiliza lógica del screener)."""
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
                                "GER", "UK100", "FRA40", "AUS200")):
        return "index"
    forex_ccys = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF",
                  "SEK", "NOK", "DKK", "SGD", "HKD", "MXN", "ZAR", "TRY",
                  "PLN", "HUF", "CZK", "CNH", "CNY"}
    for ccy in forex_ccys:
        if s.startswith(ccy) or s.endswith(ccy):
            return "forex"
    return "other"


def print_validated_table(results: List[Dict]) -> None:
    """Imprime tabla rankeada de candidatos validados."""
    passed = [r for r in results if r["status"] in ("PASS", "WARN")]
    failed = [r for r in results if r["status"] == "FAIL"]

    passed.sort(key=lambda r: r["sharpe"], reverse=True)

    print(f"\n{'='*110}")
    print("VALIDATED CANDIDATES — WFO OOS Results")
    print(f"{'='*110}")
    print(f"{'Rank':>4}  {'Symbol':<12} {'Side':<6} {'N_OOS':>6} {'Return%':>9} "
          f"{'MDD%':>8} {'Sharpe':>7} {'WR%':>5} {'PF':>6} {'Folds':>7} {'PASS':>5}")
    print("-" * 110)

    for i, r in enumerate(passed, 1):
        mark = "*" if r["status"] == "PASS" else "~"
        print(f"{i:>4}  {r['symbol']:<12} {r['side']:<6} {r['n_oos']:>6} "
              f"{r['total_return']:>+8.1%} {r['mdd']:>+7.1%} "
              f"{r['sharpe']:>7.2f} {r['hit_rate']:>4.0%} "
              f"{r['profit_factor']:>6.2f} "
              f"{r['n_folds_ok']:>3}/{r['n_folds']:<3} {mark:>4}")

    if not passed:
        print("  (ningun candidato paso la validacion WFO)")

    if failed:
        print(f"\nREJECTED (edge negativo en WFO OOS):")
        failed.sort(key=lambda r: r["total_return"], reverse=True)
        for r in failed:
            print(f"  {r['symbol']:<12} {r['side']:<6}  ->  "
                  f"Return {r['total_return']:>+.1%}, MDD {r['mdd']:>+.1%}, "
                  f"Sharpe {r['sharpe']:.2f}")


def print_portfolio_summary(results: List[Dict]) -> None:
    """Resumen de portfolio de candidatos que pasan."""
    passed = [r for r in results if r["status"] in ("PASS", "WARN")]

    print(f"\n{'='*110}")
    print("PORTFOLIO SUMMARY (candidatos PASS/WARN)")
    print(f"{'='*110}")

    if len(passed) == 0:
        print("  Sin candidatos viables.")
        return

    print(f"  N candidatos:        {len(passed)}")

    avg_ret = float(np.mean([r["total_return"] for r in passed]))
    sum_ret = sum(r["total_return"] for r in passed)
    avg_sharpe = float(np.mean([r["sharpe"] for r in passed]))
    worst_mdd = min(r["mdd"] for r in passed)

    print(f"  Return promedio OOS: {avg_ret:>+.2%}")
    print(f"  Return suma OOS:     {sum_ret:>+.2%}")
    print(f"  Sharpe promedio:     {avg_sharpe:.2f}")
    print(f"  Peor MDD:            {worst_mdd:>+.2%}")

    # Por asset class
    by_class: Dict[str, List[str]] = {}
    for r in passed:
        cls = _classify_asset(r["symbol"])
        by_class.setdefault(cls, []).append(f"{r['symbol']} {r['side']}")

    print(f"\n  Por asset class:")
    for cls in sorted(by_class):
        print(f"    {cls:<10}: {', '.join(by_class[cls])}")

    # Correlacion OOS returns (si >1 candidato)
    if len(passed) > 1:
        print(f"\n  (Correlacion OOS requiere --save + analisis offline)")


def save_results(results: List[Dict], output_dir: Path) -> None:
    """Guarda resultados en parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "validate_trend_candidates.parquet"
    df = pl.DataFrame(results)
    df.write_parquet(out_path)
    print(f"\nResultados guardados en: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validador WFO de candidatos TREND — Fase 2+3"
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP,
                        help=f"Top N candidatos del screener (default: {DEFAULT_TOP})")
    parser.add_argument("--save", action="store_true",
                        help="Guardar resultados en parquet")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Simbolos especificos (ej: XAUUSD,XAGUSD). "
                             "Bypass screener, valida directamente ambos sides.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Salida detallada")
    args = parser.parse_args()

    t0 = time.perf_counter()

    print("\n" + "=" * 110)
    print("VALIDADOR WFO — Candidatos TREND (Fase 2+3)")
    print("=" * 110)

    m5_dir = m5_clean_dir(PROJECT_ROOT)
    if not m5_dir.exists():
        print(f"ERROR: directorio M5 no encontrado: {m5_dir}")
        sys.exit(1)

    # Construir lista de candidatos
    candidates: List[Tuple[str, str]] = []  # (symbol, side)

    if args.symbols:
        # Bypass screener — validar directamente ambos sides
        for sym in args.symbols.split(","):
            sym = sym.strip().upper()
            candidates.append((sym, "LONG"))
            candidates.append((sym, "SHORT"))
        source = "CLI --symbols"
    else:
        # Leer screener
        screener_path = (outputs_root(PROJECT_ROOT) / "screening"
                         / "screen_trend_universe.parquet")
        shortlist = load_screener_shortlist(screener_path, args.top)

        if shortlist.is_empty():
            print("ERROR: shortlist vacia. Ejecutar screener primero con --save.")
            sys.exit(1)

        for row in shortlist.iter_rows(named=True):
            candidates.append((row["symbol"], row["side"]))
        source = f"screener top {len(candidates)}"

    # Config summary
    grid_combos = 1
    for v in PARAM_GRID.values():
        grid_combos *= len(v)

    print(f"Fuente:          {source}")
    print(f"Candidatos:      {len(candidates)}")
    print(f"Grid:            {grid_combos} combos x folds")
    print(f"WFO:             IS={WFO_IS_MONTHS}m, OOS={WFO_OOS_MONTHS}m, "
          f"step={WFO_STEP_MONTHS}m, embargo={WFO_EMBARGO_DAYS}d")
    print(f"Costs:           {COSTS_CFG.spread_bps} bps spread "
          f"(roundtrip {COSTS_CFG.total_roundtrip_dec * 10_000:.1f} bps)")
    print(f"PASS criteria:   return>0, sharpe>0, PF>1.0, trades>={MIN_TRADES_PASS}")
    print()

    # Procesar candidatos
    results: List[Dict] = []

    for i, (symbol, side) in enumerate(candidates, 1):
        pct = i / len(candidates) * 100
        if not args.verbose:
            print(f"\r  Validando {i}/{len(candidates)} ({pct:.0f}%) — "
                  f"{symbol:<12} {side:<6}", end="", flush=True)

        result = validate_candidate(m5_dir, symbol, side, verbose=args.verbose)
        if result is not None:
            results.append(result)

    if not args.verbose:
        print()  # newline tras progreso

    elapsed = time.perf_counter() - t0

    if not results:
        print(f"\nSin resultados WFO. Verificar datos M5 y screener. ({elapsed:.1f}s)")
        sys.exit(1)

    # Reportes
    print_validated_table(results)
    print_portfolio_summary(results)

    # Guardar
    if args.save:
        out_dir = outputs_root(PROJECT_ROOT) / "screening"
        save_results(results, out_dir)

    print(f"\n{'='*110}")
    print(f"Tiempo total: {elapsed:.1f}s | "
          f"Candidatos evaluados: {len(results)}/{len(candidates)}")
    print(f"{'='*110}\n")


if __name__ == "__main__":
    main()
