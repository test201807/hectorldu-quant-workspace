#!/usr/bin/env python3
"""
Screener TREND — Universo completo de símbolos M5.

Evalúa edge via forward returns (sin backtest) para cada símbolo con datos
M5 limpios.  Réplica exacta de features NB3 Cell 05 + regime gate.

Uso:
    cd C:\Quant\projects\MT5_Data_Extraction
    venv1\Scripts\python.exe tools\_screen_trend_universe.py
    venv1\Scripts\python.exe tools\_screen_trend_universe.py --q-er 0.60 --save
    venv1\Scripts\python.exe tools\_screen_trend_universe.py --symbols BTCUSD,XAUUSD -v
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Path contract
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "contracts"))
from path_contract import m5_clean_dir, outputs_root  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
HORIZONS = [12, 24, 48, 96, 288]           # barras M5 → 1h, 2h, 4h, 8h, 24h
HORIZON_LABELS = {12: "1h", 24: "2h", 48: "4h", 96: "8h", 288: "24h"}
FEATURE_WIN = 288                            # ventana ER / momentum / volatilidad
IS_FRAC = 0.80                               # fracción In-Sample
MIN_SIGNALS = 30                             # mínimo señales por (symbol, side)
BOOTSTRAP_ITERS = 500                        # iteraciones bootstrap
BOOTSTRAP_SUBSAMPLE = 5_000                  # submuestreo si hay más señales
WARMUP_BARS = FEATURE_WIN + 10               # barras de warmup (descartar NaN)

# Clasificación de asset class por patrón de nombre
ASSET_CLASSES: Dict[str, str] = {}           # llenado dinámicamente


def classify_asset(symbol: str) -> str:
    """Clasifica un símbolo en asset class por heurísticas de nombre."""
    s = symbol.upper()
    if any(c in s for c in ("BTC", "ETH", "LTC", "XRP", "BNB", "SOL",
                            "DOGE", "ADA", "DOT", "AVAX", "LINK", "UNI")):
        return "crypto"
    if any(m in s for m in ("XAU", "XAG", "XPT", "XPD")):
        return "metal"
    if any(o in s for o in ("WTI", "BRENT", "OIL", "UKOIL", "USOIL", "XTIUSD", "XBRUSD", "NGAS")):
        return "energy"
    if any(idx in s for idx in ("US500", "US100", "US30", "SPX", "NAS", "NDX",
                                "DAX", "FTSE", "NI225", "JP225", "GER", "UK100",
                                "FRA40", "AUS200", "HK50", "STOXX")):
        return "index"
    # Forex: contiene dos divisas de 3 letras
    forex_ccys = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF",
                  "SEK", "NOK", "DKK", "SGD", "HKD", "MXN", "ZAR", "TRY",
                  "PLN", "HUF", "CZK", "CNH", "CNY"}
    for ccy in forex_ccys:
        if s.startswith(ccy) or s.endswith(ccy):
            return "forex"
    return "other"


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def discover_symbols(m5_dir: Path) -> List[str]:
    """Descubre todos los símbolos en hive partition m5_clean/symbol=XXX/."""
    symbol_dirs = sorted(
        d for d in m5_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith("symbol=")
    )
    return [d.name.split("=", 1)[1] for d in symbol_dirs]


def load_symbol(m5_dir: Path, symbol: str) -> Optional[pl.DataFrame]:
    """Carga datos M5 para un símbolo desde hive partition."""
    sym_dir = m5_dir / f"symbol={symbol}"
    if not sym_dir.exists():
        return None

    parquet_files = list(sym_dir.rglob("*.parquet"))
    if not parquet_files:
        return None

    try:
        df = pl.read_parquet(parquet_files)
    except Exception:
        return None

    if df.is_empty():
        return None

    # Normalizar timestamp
    if "timestamp_utc" in df.columns:
        ts_col = "timestamp_utc"
    elif "time_utc" in df.columns:
        ts_col = "time_utc"
    else:
        return None

    # Si es Int64 (epoch ms), convertir a datetime
    if df.schema[ts_col] == pl.Int64:
        df = df.with_columns(
            pl.from_epoch(pl.col(ts_col), time_unit="ms").alias("time_utc")
        )
    elif ts_col != "time_utc":
        df = df.rename({ts_col: "time_utc"})

    # Verificar columna close
    if "close" not in df.columns:
        return None

    # Ordenar y dedup
    df = df.sort("time_utc").unique(subset=["time_utc"], keep="last")

    return df


# ---------------------------------------------------------------------------
# Features (réplica exacta NB3 Cell 05)
# ---------------------------------------------------------------------------

def compute_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula ER, momentum y volatilidad — mismas fórmulas que NB3."""
    w = FEATURE_WIN

    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret"),
        (pl.col("close") - pl.col("close").shift(1)).abs().alias("abs_diff"),
    ])

    df = df.with_columns([
        # Volatilidad (bps)
        (pl.col("ret").rolling_std(w) * 10_000).alias("vol_bps"),
        # Momentum (bps)
        ((pl.col("close") / pl.col("close").shift(w) - 1.0) * 10_000).alias("mom_bps"),
        # ER = |net move| / sum(|bar moves|)
        (
            (pl.col("close") - pl.col("close").shift(w)).abs()
            / pl.col("abs_diff").rolling_sum(w)
        ).alias("er"),
    ])

    return df


# ---------------------------------------------------------------------------
# Señales + forward returns
# ---------------------------------------------------------------------------

def generate_signals_and_fwd_returns(
    df: pl.DataFrame,
    er_threshold: float,
    is_frac: float,
) -> Optional[pl.DataFrame]:
    """
    Genera señales LONG/SHORT y calcula forward returns index-based.

    Regime gate: er >= er_threshold (calculado como quantile sobre IS).
    Señales: trending + mom>0 → LONG, trending + mom<0 → SHORT.
    Forward returns: bar offset [12,24,48,96,288].
    """
    # Descartar warmup
    df = df.slice(WARMUP_BARS).drop_nulls(subset=["er", "mom_bps", "vol_bps"])
    if len(df) < MIN_SIGNALS * 4:
        return None

    n = len(df)
    is_end = int(n * is_frac)

    # Quantile de ER sobre IS
    is_data = df.slice(0, is_end)
    er_q = float(is_data.select(pl.col("er").quantile(er_threshold, interpolation="linear")).item())

    # Columna de segmento
    idx = np.arange(n)
    segments = np.where(idx < is_end, "IS", "OOS")

    close_arr = df["close"].to_numpy()
    er_arr = df["er"].to_numpy()
    mom_arr = df["mom_bps"].to_numpy()

    # Mask trending
    trending = er_arr >= er_q

    # Señales
    long_mask = trending & (mom_arr > 0)
    short_mask = trending & (mom_arr < 0)

    # Forward returns (index-based)
    fwd = {}
    for h in HORIZONS:
        fwd_ret = np.full(n, np.nan)
        valid = np.arange(n - h)
        fwd_ret[valid] = (close_arr[valid + h] - close_arr[valid]) / close_arr[valid]
        fwd[h] = fwd_ret

    # Construir resultados
    rows = []
    for i in range(n):
        if not (long_mask[i] or short_mask[i]):
            continue
        side = "LONG" if long_mask[i] else "SHORT"
        sign = 1.0 if side == "LONG" else -1.0
        row = {
            "side": side,
            "segment": segments[i],
            "er": er_arr[i],
        }
        all_nan = True
        for h in HORIZONS:
            val = fwd[h][i]
            if not np.isnan(val):
                all_nan = False
            row[f"fwd_{h}"] = val * sign
        if not all_nan:
            rows.append(row)

    if not rows:
        return None

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    n_iters: int = BOOTSTRAP_ITERS,
    max_n: int = BOOTSTRAP_SUBSAMPLE,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Bootstrap percentile CI para la media. Retorna (mean, ci_lo, ci_hi)."""
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(42)

    # Submuestreo si hay demasiados puntos
    if len(values) > max_n:
        values = rng.choice(values, size=max_n, replace=False)

    obs_mean = float(np.mean(values))
    n = len(values)
    means = np.empty(n_iters)
    for i in range(n_iters):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)

    ci_lo = float(np.percentile(means, alpha / 2 * 100))
    ci_hi = float(np.percentile(means, (1 - alpha / 2) * 100))
    return (obs_mean, ci_lo, ci_hi)


# ---------------------------------------------------------------------------
# Análisis por símbolo
# ---------------------------------------------------------------------------

def analyze_symbol(
    m5_dir: Path,
    symbol: str,
    er_quantile: float,
    verbose: bool = False,
) -> List[dict]:
    """Analiza un símbolo completo. Retorna lista de dicts (uno por side)."""
    df = load_symbol(m5_dir, symbol)
    if df is None:
        if verbose:
            print(f"  {symbol}: sin datos")
        return []

    n_bars = len(df)
    if n_bars < WARMUP_BARS + MIN_SIGNALS * 4:
        if verbose:
            print(f"  {symbol}: insuficientes barras ({n_bars:,})")
        return []

    df = compute_features(df)
    signals_df = generate_signals_and_fwd_returns(df, er_quantile, IS_FRAC)

    if signals_df is None or len(signals_df) == 0:
        if verbose:
            print(f"  {symbol}: sin señales tras regime gate")
        return []

    results = []
    asset_class = classify_asset(symbol)

    for side in ["LONG", "SHORT"]:
        for segment in ["IS", "OOS"]:
            subset = signals_df.filter(
                (pl.col("side") == side) & (pl.col("segment") == segment)
            )
            n_sig = len(subset)
            if n_sig < MIN_SIGNALS:
                continue

            # Encontrar mejor horizonte y calcular bootstrap
            best_h = None
            best_mean = -999.0
            best_ci_lo = np.nan
            best_ci_hi = np.nan

            for h in HORIZONS:
                col = f"fwd_{h}"
                vals = subset.select(col).drop_nulls().to_series().to_numpy()
                if len(vals) < MIN_SIGNALS:
                    continue
                m, lo, hi = bootstrap_ci(vals)
                if m > best_mean:
                    best_mean = m
                    best_ci_lo = lo
                    best_ci_hi = hi
                    best_h = h

            if best_h is None:
                continue

            results.append({
                "symbol": symbol,
                "side": side,
                "segment": segment,
                "asset_class": asset_class,
                "n_signals": n_sig,
                "n_bars": n_bars,
                "edge_bps": best_mean * 10_000,
                "ci_lo_bps": best_ci_lo * 10_000,
                "ci_hi_bps": best_ci_hi * 10_000,
                "significant": best_ci_lo > 0,
                "best_horizon": HORIZON_LABELS.get(best_h, f"{best_h}b"),
            })

    if verbose and results:
        oos_results = [r for r in results if r["segment"] == "OOS"]
        for r in oos_results:
            sig = "*" if r["significant"] else " "
            print(f"  {symbol:<12} {r['side']:<6} OOS  n={r['n_signals']:<6,} "
                  f"edge={r['edge_bps']:>+7.1f}bps  CI[{r['ci_lo_bps']:>+7.1f}, "
                  f"{r['ci_hi_bps']:>+7.1f}] {sig}  H={r['best_horizon']}")

    return results


# ---------------------------------------------------------------------------
# Reportes
# ---------------------------------------------------------------------------

def print_ranked_table(all_results: List[dict]) -> None:
    """Imprime tabla rankeada por edge OOS descendente."""
    oos = [r for r in all_results if r["segment"] == "OOS"]
    if not oos:
        print("\nSin resultados OOS para rankear.")
        return

    oos.sort(key=lambda r: r["edge_bps"], reverse=True)

    print(f"\n{'='*100}")
    print("RANKED RESULTS — OOS Edge (mejores primero)")
    print(f"{'='*100}")
    print(f"{'Rank':>4}  {'Symbol':<12} {'Side':<6} {'Class':<8} {'N_OOS':>7} "
          f"{'Edge_bps':>9} {'CI_lo':>8} {'CI_hi':>8} {'Sig':>4} {'H(best)':>8}")
    print("-" * 100)

    for i, r in enumerate(oos, 1):
        sig = "*" if r["significant"] else ""
        print(f"{i:>4}  {r['symbol']:<12} {r['side']:<6} {r['asset_class']:<8} "
              f"{r['n_signals']:>7,} {r['edge_bps']:>+9.1f} {r['ci_lo_bps']:>+8.1f} "
              f"{r['ci_hi_bps']:>+8.1f} {sig:>4} {r['best_horizon']:>8}")


def print_top_candidates(all_results: List[dict]) -> None:
    """Imprime candidatos con CI > 0 en OOS."""
    oos_sig = [r for r in all_results
               if r["segment"] == "OOS" and r["significant"]]
    oos_sig.sort(key=lambda r: r["edge_bps"], reverse=True)

    print(f"\n{'='*100}")
    print(f"TOP CANDIDATES — CI > 0 en OOS (edge estadísticamente significativo)")
    print(f"{'='*100}")

    if not oos_sig:
        print("  Ningún símbolo con edge significativo en OOS.")
        return

    print(f"{'Symbol':<12} {'Side':<6} {'Class':<8} {'N_OOS':>7} "
          f"{'Edge_bps':>9} {'CI_lo':>8} {'CI_hi':>8} {'H(best)':>8}")
    print("-" * 80)

    for r in oos_sig:
        print(f"{r['symbol']:<12} {r['side']:<6} {r['asset_class']:<8} "
              f"{r['n_signals']:>7,} {r['edge_bps']:>+9.1f} {r['ci_lo_bps']:>+8.1f} "
              f"{r['ci_hi_bps']:>+8.1f} {r['best_horizon']:>8}")

    print(f"\nTotal candidatos significativos: {len(oos_sig)}")


def print_asset_class_summary(all_results: List[dict]) -> None:
    """Resumen por asset class."""
    oos = [r for r in all_results if r["segment"] == "OOS"]
    if not oos:
        return

    print(f"\n{'='*100}")
    print("ASSET CLASS SUMMARY (OOS)")
    print(f"{'='*100}")

    classes: Dict[str, List[dict]] = {}
    for r in oos:
        cls = r["asset_class"]
        classes.setdefault(cls, []).append(r)

    print(f"{'Class':<10} {'N_sym':>6} {'N_sig':>5} {'Avg_edge':>10} "
          f"{'Best_sym':<12} {'Best_edge':>10} {'Best_side':<6}")
    print("-" * 80)

    for cls in sorted(classes.keys()):
        rows = classes[cls]
        symbols = set(r["symbol"] for r in rows)
        n_sig = sum(1 for r in rows if r["significant"])
        avg_edge = np.mean([r["edge_bps"] for r in rows])
        best = max(rows, key=lambda r: r["edge_bps"])
        print(f"{cls:<10} {len(symbols):>6} {n_sig:>5} {avg_edge:>+10.1f} "
              f"{best['symbol']:<12} {best['edge_bps']:>+10.1f} {best['side']:<6}")


def print_is_vs_oos_comparison(all_results: List[dict]) -> None:
    """Compara IS vs OOS para detectar overfitting."""
    print(f"\n{'='*100}")
    print("IS vs OOS COMPARISON (top 20 por edge OOS)")
    print(f"{'='*100}")

    # Emparejar IS y OOS por (symbol, side)
    by_key: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for r in all_results:
        key = (r["symbol"], r["side"])
        by_key.setdefault(key, {})[r["segment"]] = r

    pairs = []
    for key, segs in by_key.items():
        if "IS" in segs and "OOS" in segs:
            pairs.append((segs["IS"], segs["OOS"]))

    if not pairs:
        print("  Sin pares IS/OOS disponibles.")
        return

    pairs.sort(key=lambda p: p[1]["edge_bps"], reverse=True)

    print(f"{'Symbol':<12} {'Side':<6} {'IS_edge':>9} {'OOS_edge':>9} "
          f"{'Decay%':>8} {'IS_n':>7} {'OOS_n':>7}")
    print("-" * 80)

    for is_r, oos_r in pairs[:20]:
        is_e = is_r["edge_bps"]
        oos_e = oos_r["edge_bps"]
        decay = ((oos_e - is_e) / abs(is_e) * 100) if abs(is_e) > 0.01 else 0.0
        print(f"{oos_r['symbol']:<12} {oos_r['side']:<6} {is_e:>+9.1f} {oos_e:>+9.1f} "
              f"{decay:>+8.1f} {is_r['n_signals']:>7,} {oos_r['n_signals']:>7,}")


def save_results(all_results: List[dict], output_dir: Path) -> Path:
    """Guarda resultados en parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "screen_trend_universe.parquet"
    df = pl.DataFrame(all_results)
    df.write_parquet(out_path)
    print(f"\nResultados guardados en: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Screener TREND — forward-return edge en universo M5 completo"
    )
    parser.add_argument("--q-er", type=float, default=0.60,
                        help="Quantile ER para regime gate (default: 0.60)")
    parser.add_argument("--save", action="store_true",
                        help="Guardar resultados en parquet")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Símbolos específicos separados por coma (ej: BTCUSD,XAUUSD)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Salida detallada por símbolo")
    args = parser.parse_args()

    t0 = time.perf_counter()

    print("\n" + "=" * 80)
    print("SCREENER TREND — Forward-Return Edge en Universo M5")
    print("=" * 80)

    # Descubrir datos
    m5_dir = m5_clean_dir(PROJECT_ROOT)
    if not m5_dir.exists():
        print(f"ERROR: directorio M5 no encontrado: {m5_dir}")
        sys.exit(1)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = discover_symbols(m5_dir)

    print(f"Directorio M5: {m5_dir}")
    print(f"Símbolos: {len(symbols)}")
    print(f"ER quantile gate: {args.q_er}")
    print(f"IS/OOS split: {IS_FRAC:.0%} / {1-IS_FRAC:.0%}")
    print(f"Horizontes: {', '.join(HORIZON_LABELS[h] for h in HORIZONS)}")
    print(f"Bootstrap: {BOOTSTRAP_ITERS} iters, subsample {BOOTSTRAP_SUBSAMPLE:,}")
    print(f"Min señales: {MIN_SIGNALS}")
    print()

    # Procesar cada símbolo
    all_results: List[dict] = []
    n_ok = 0
    n_skip = 0

    for i, symbol in enumerate(symbols, 1):
        if not args.verbose:
            # Progreso compacto
            pct = i / len(symbols) * 100
            print(f"\r  Procesando {i}/{len(symbols)} ({pct:.0f}%) — {symbol:<12}", end="", flush=True)

        results = analyze_symbol(m5_dir, symbol, args.q_er, verbose=args.verbose)
        if results:
            all_results.extend(results)
            n_ok += 1
        else:
            n_skip += 1

    if not args.verbose:
        print()  # newline tras progreso

    elapsed = time.perf_counter() - t0

    # Resumen de ejecución
    print(f"\nCompletado en {elapsed:.1f}s")
    print(f"Símbolos procesados: {n_ok} con señales, {n_skip} sin señales/datos")
    print(f"Total registros (symbol × side × segment): {len(all_results)}")

    if not all_results:
        print("\nSin resultados. Verificar datos M5.")
        sys.exit(1)

    # Reportes
    print_ranked_table(all_results)
    print_top_candidates(all_results)
    print_asset_class_summary(all_results)
    print_is_vs_oos_comparison(all_results)

    # Guardar si se pidió
    if args.save:
        out_dir = outputs_root(PROJECT_ROOT) / "screening"
        save_results(all_results, out_dir)

    print(f"\n{'='*80}")
    print(f"Tiempo total: {elapsed:.1f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
